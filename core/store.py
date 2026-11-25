from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from .affection import (
    AUTO_AFFECTION_DAILY_ABS_LIMIT,
    AUTO_AFFECTION_DAILY_MAX_COUNT,
    AUTO_AFFECTION_REVIEW_THRESHOLD,
)
from .contract import (
    MEMBER_PROFILES,
    digest,
    encode,
    entry_key,
    is_free_text_typing_habit,
    llm_approved,
    policy_allowed,
)
from .review_evidence import DEFER_FINGERPRINT, pending_evidence_update


class Conflict(ValueError):
    pass


DEFAULT_AUDIT_RETENTION_DAYS = 90
JOURNAL_SIZE_LIMIT_BYTES = 67_108_864
INCREMENTAL_VACUUM_PAGES = 256
RELATION_RECEIPT_RETENTION_DAYS = 30


def _compact_evidence_list(evidence):
    if not isinstance(evidence, list):
        return evidence, False
    compacted = []
    changed = False
    for item in evidence:
        if isinstance(item, dict):
            if set(item) <= {"id", "message_id", "member", "scope", "trimmed"} and item.get("trimmed") is True:
                compacted.append(item)
            else:
                compacted.append(
                    {
                        "id": item.get("id", ""),
                        "message_id": item.get("message_id", ""),
                        "member": item.get("member", ""),
                        "scope": item.get("scope", ""),
                        "trimmed": True,
                    }
                )
                changed = True
        else:
            compacted.append(item)
    return compacted, changed


class Store:
    """All state transitions are SQLite transactions with compare-and-swap revisions."""

    def __init__(self, path):
        self.path = Path(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        current_version = self.db.execute("PRAGMA user_version").fetchone()[0]
        self.db.executescript(f"""
        PRAGMA auto_vacuum=INCREMENTAL;
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA busy_timeout=5000;
        PRAGMA journal_size_limit={JOURNAL_SIZE_LIMIT_BYTES};
        CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, scope TEXT, persona TEXT, created REAL, body TEXT);
        CREATE INDEX IF NOT EXISTS events_scope ON events(scope, persona, created);
        CREATE INDEX IF NOT EXISTS events_scope_kind_created ON events(scope, json_extract(body,'$.kind'), created);
        CREATE INDEX IF NOT EXISTS events_turn ON events(scope, persona, json_extract(body,'$.turn_id'));
        CREATE TABLE IF NOT EXISTS proposals(id TEXT PRIMARY KEY, scope TEXT, persona TEXT, state TEXT,
          created REAL, updated REAL, body TEXT);
        CREATE TABLE IF NOT EXISTS entries(id TEXT PRIMARY KEY, body TEXT);
        CREATE INDEX IF NOT EXISTS entries_scope_persona ON entries(json_extract(body,'$.scope'), json_extract(body,'$.persona'), json_extract(body,'$.category'), json_extract(body,'$.active'));
        CREATE INDEX IF NOT EXISTS entries_proposal_id ON entries(json_extract(body,'$.proposal_id'));
        CREATE INDEX IF NOT EXISTS proposals_entry ON proposals(json_extract(body,'$.entry_id'),created);
        CREATE INDEX IF NOT EXISTS proposals_queue ON proposals(state,scope,created,id);
        CREATE INDEX IF NOT EXISTS proposals_cat_key ON proposals(scope,persona,json_extract(body,'$.category'),created);
        CREATE TABLE IF NOT EXISTS proposal_evidence_sources(proposal TEXT,message_id TEXT,event_id TEXT,
          PRIMARY KEY(proposal,message_id));
        CREATE TABLE IF NOT EXISTS review_claims(proposal TEXT PRIMARY KEY,scope TEXT,job TEXT);
        CREATE INDEX IF NOT EXISTS review_claims_scope ON review_claims(scope);
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, proposal TEXT, action TEXT, actor TEXT, created REAL, body TEXT);
        CREATE INDEX IF NOT EXISTS audit_created ON audit(created);
        CREATE TABLE IF NOT EXISTS checkpoints(scope TEXT, persona TEXT, cursor REAL, PRIMARY KEY(scope, persona));
        CREATE TABLE IF NOT EXISTS social_bindings(scope TEXT,persona TEXT,legacy_key TEXT,PRIMARY KEY(scope,persona));
        CREATE TABLE IF NOT EXISTS legacy_group_owners(scope TEXT PRIMARY KEY,legacy_key TEXT);
        CREATE TABLE IF NOT EXISTS legacy_social(scope TEXT,source_key TEXT,category TEXT,member TEXT,item_key TEXT,body TEXT,
          PRIMARY KEY(scope,source_key,category,member,item_key));
        CREATE TABLE IF NOT EXISTS social_imports(import_key TEXT PRIMARY KEY,created REAL,body TEXT);
        CREATE TABLE IF NOT EXISTS affection_source_claims(scope TEXT,persona TEXT,member TEXT,message_id TEXT,proposal TEXT,
          PRIMARY KEY(scope,persona,member,message_id));
        """)
        if current_version < 9:
            self.db.execute("PRAGMA user_version=9")
        from .profiles import Profiles

        self.profiles = Profiles(self)
        from .native import NativeStore

        self.native = NativeStore(self)
        from .scheduling import Scheduling

        self.scheduling = Scheduling(self)
        from .source_withdrawals import SourceWithdrawals

        self.source_withdrawals = SourceWithdrawals(self)
        from .console import Console

        self.console = Console(self)
        from .automatic_review_store import AutomaticReviewStore

        self.automatic_reviews = AutomaticReviewStore(self)
        from .scene_store import SceneStore

        self.scenes = SceneStore(self)
        from .expression_store import ExpressionStore

        self.expressions = ExpressionStore(self)

    @contextmanager
    def transaction(self):
        with self.lock:
            nested = self.db.in_transaction
            if not nested:
                self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                if not nested:
                    self.db.commit()
            except BaseException:
                if not nested:
                    self.db.rollback()
                raise

    def close(self):
        with self.lock:
            self.db.close()

    def enforce_personality_policy(self):
        """Retain historical evidence but revoke incompatible learning atomically."""
        changed = dict(entries=0, proposals=0)
        with self.lock, self.db:
            for entry in self.entries():
                if entry["active"] and not policy_allowed(entry):
                    entry.update(active=False, revision=entry["revision"] + 1, disabled_reason="fixed_personality")
                    self.db.execute("UPDATE entries SET body=? WHERE id=?", (encode(entry), entry["id"]))
                    self._audit(entry, "policy_disabled", "system:fixed_personality")
                    changed["entries"] += 1
            for row in self.db.execute(
                "SELECT body FROM proposals WHERE state IN ('pending','reviewed','applied')"
            ).fetchall():
                p = json.loads(row[0])
                if not policy_allowed(p):
                    p["state"] = "rejected"
                    p["reviews"].append(
                        dict(
                            actor="system:fixed_personality",
                            verdict="reject",
                            reason="自身性格固定；旧人格增量或未按成员观察规范审查的表达资料停用。",
                            risks=[],
                            created=time.time(),
                        )
                    )
                    self._save(p)
                    self._audit(p, "policy_disabled", "system:fixed_personality")
                    changed["proposals"] += 1
        return changed

    def _ingest_one(self, event):
        self.db.execute(
            "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?)",
            (event["id"], event["scope"], event["persona"], event["received_at"], encode(event)),
        )
        if self.source_withdrawals.has_withdrawn(event["scope"], [event]):
            return
        self.profiles.bind(event)
        self.native.register_scope(event)
        self.scheduling.ingest(event)

    def ingest(self, event):
        with self.transaction():
            self._ingest_one(event)

    def ingest_batch(self, events):
        if not events:
            return
        with self.transaction():
            for event in events:
                self._ingest_one(event)

    def events(self, scope, persona, limit=80):
        with self.lock:
            rows = self.db.execute(
                "SELECT body FROM events WHERE scope=? AND persona=? ORDER BY created DESC LIMIT ?",
                (scope, persona, limit),
            ).fetchall()
            return [json.loads(r[0]) for r in reversed(rows)]

    def batches(self, minimum):
        return self.scheduling.batches(minimum)

    def checkpoint(self, scope, persona, cursor):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO checkpoints VALUES(?,?,?)", (scope, persona, cursor))
            self.db.execute(
                "UPDATE analysis_inbox SET processed=1 WHERE scope=? AND persona=? AND created<=?",
                (scope, persona, cursor),
            )

    def entries(self):
        with self.lock:
            result = []
            for row in self.db.execute(
                "SELECT e.body,p.body AS proposal FROM entries e LEFT JOIN proposals p ON p.id=json_extract(e.body,'$.proposal_id')"
            ):
                entry = json.loads(row[0])
                if "source_uids" not in entry and row[1]:
                    entry.update(self.provenance(json.loads(row[1])))
                result.append(entry)
            return result

    @staticmethod
    def provenance(proposal):
        evidence = proposal.get("evidence", [])
        authors = sorted({e["member"] for e in evidence if e.get("member") and e.get("scope") == proposal["scope"]})
        return {
            "source_uids": authors,
            "provenance_complete": bool(evidence)
            and all(e.get("member") and e.get("scope") == proposal["scope"] for e in evidence),
        }

    def review_claim(self, pid):
        with self.lock:
            row = self.db.execute("SELECT job FROM review_claims WHERE proposal=?", (pid,)).fetchone()
            return row[0] if row else None

    def review_material(self, proposal, target_category):
        with self.lock:
            base = self.db.execute(
                "SELECT body FROM entries WHERE id=?", (proposal["entry_id"],)
            ).fetchone()
            target = None
            if target_category:
                target = self.db.execute(
                    """SELECT body FROM entries WHERE json_extract(body,'$.proposal_id')=?
                    AND json_extract(body,'$.scope')=? AND json_extract(body,'$.persona')=?
                    AND json_extract(body,'$.category')=?""",
                    (proposal["value"]["target_id"], proposal["scope"], proposal["persona"], target_category),
                ).fetchone()
            return (json.loads(base[0]) if base else None, json.loads(target[0]) if target else None)

    def latest_personas(self):
        with self.lock:
            rows = self.db.execute("""SELECT e.scope,e.persona FROM events e JOIN
                (SELECT scope,max(created) AS stamp FROM events WHERE json_extract(body,'$.kind')='observed' GROUP BY scope) q
                ON e.scope=q.scope AND e.created=q.stamp WHERE json_extract(e.body,'$.kind')='observed'""").fetchall()
            return dict(rows)

    def latest_character_context(self, scope, persona):
        with self.lock:
            row = self.db.execute(
                """SELECT body FROM events
                WHERE scope=? AND persona=?
                AND json_extract(body,'$.context.persona_summary') IS NOT NULL
                AND length(trim(json_extract(body,'$.context.persona_summary')))>0
                ORDER BY created DESC LIMIT 1""",
                (scope, persona),
            ).fetchone()
            if row is None and persona:
                row = self.db.execute(
                    """SELECT body FROM events
                    WHERE persona=?
                    AND json_extract(body,'$.context.persona_summary') IS NOT NULL
                    AND length(trim(json_extract(body,'$.context.persona_summary')))>0
                    ORDER BY created DESC LIMIT 1""",
                    (persona,),
                ).fetchone()
            return json.loads(row[0]) if row else None

    def get(self, proposal_id):
        with self.lock:
            row = self.db.execute("SELECT body FROM proposals WHERE id=?", (proposal_id,)).fetchone()
            if not row:
                raise ValueError("proposal not found")
            return json.loads(row[0])

    def proposals(self, scope="", limit=100):
        with self.lock:
            rows = self.db.execute(
                "SELECT body FROM proposals WHERE (?='' OR scope=?) ORDER BY created DESC LIMIT ?",
                (scope, scope, limit),
            )
            return [json.loads(r[0]) for r in rows]

    def review_groups(self):
        with self.lock:
            return [
                dict(r)
                for r in self.db.execute("""SELECT scope,persona,json_extract(body,'$.member') AS member,
                count(*) AS count,sum(state='reviewed') AS reviewed,min(created) AS oldest,
                group_concat(DISTINCT json_extract(body,'$.category')) AS categories
                FROM proposals WHERE state IN ('pending','reviewed') GROUP BY scope,persona,member ORDER BY oldest""")
            ]

    def proposal_counts(self):
        with self.lock:
            return dict(self.db.execute("SELECT state,count(*) FROM proposals GROUP BY state").fetchall())

    def review_group(self, scope, persona, member, page=1):
        page = max(1, min(10000, int(page)))
        with self.lock:
            args = (scope, persona, member)
            condition = "scope=? AND persona=? AND json_extract(body,'$.member')=? AND state IN ('pending','reviewed')"
            total = self.db.execute("SELECT count(*) FROM proposals WHERE " + condition, args).fetchone()[0]
            rows = self.db.execute(
                "SELECT body FROM proposals WHERE " + condition + " ORDER BY created LIMIT 30 OFFSET ?",
                (*args, (page - 1) * 30),
            ).fetchall()
            return {"total": total, "page": page, "page_size": 30, "proposals": [json.loads(r[0]) for r in rows]}

    def apply_batch(self, prepared, actor):
        with self.transaction():
            return [self.apply(pid, actor, stamp, automatic_affection=automatic) for pid, stamp, automatic in prepared]

    def reject_batch(self, prepared, actor, reason):
        with self.transaction():
            return [
                self.review(pid, {"verdict": "reject", "reason": reason, "risks": []}, actor, stamp)
                for pid, stamp in prepared
            ]

    def _audit(self, p, action, actor, body=None):
        self.db.execute(
            "INSERT INTO audit(proposal,action,actor,created,body) VALUES(?,?,?,?,?)",
            (p["id"], action, actor, time.time(), encode(body or {})),
        )

    def _save(self, p):
        if p["state"] != "pending":
            p.pop(DEFER_FINGERPRINT, None)
        p["updated"] = max(time.time(), math.nextafter(p["updated"], math.inf))
        self.db.execute(
            "UPDATE proposals SET state=?,updated=?,body=? WHERE id=?", (p["state"], p["updated"], encode(p), p["id"])
        )

    def _refresh_pending_evidence(self, old, data, before, source):
        previous_sources = dict(self.db.execute(
            """SELECT message_id,event_id FROM proposal_evidence_sources
            WHERE proposal=? AND message_id IN (SELECT value FROM json_each(?))""",
            (old["id"], encode([row["message_id"] for row in data["evidence"]])),
        ))
        updated = pending_evidence_update(old, data, before, source, previous_sources)
        if updated is None:
            return

        previous = old["evidence"]
        previous_reason = old["reason"]
        retained = {row["message_id"] for row in updated["evidence"]}
        self.db.executemany(
            "INSERT OR REPLACE INTO proposal_evidence_sources VALUES(?,?,?)",
            [(old["id"], row["message_id"], row["id"]) for row in previous if row["message_id"] not in retained],
        )
        old.update(updated)
        old.pop(DEFER_FINGERPRINT, None)
        self._save(old)
        current = {row["id"]: row for row in old["evidence"]}
        self._audit(old, "evidence_updated", source, {
            "previous": [row["id"] for row in previous],
            "current": list(current),
            "previous_reason": previous_reason,
            "superseded": [row["id"] for row in previous if current.get(row["id"]) != row],
        })

    def propose(self, data, mode, source):
        if data.get("category") in {
            "scene_pattern",
            "scene_adjustment",
            "expression_style",
            "expression_adjustment",
        } and mode not in {"llm", "llm_then_human"}:
            raise ValueError("scene learning requires independent LLM review")
        if mode == "auto" and data.get("category") != "affection":
            raise ValueError("automatic mode is only supported for affection")
        if not policy_allowed(data):
            raise ValueError("fixed personality policy rejects proposal")
        with self.transaction():
            if self.source_withdrawals.has_withdrawn(data["scope"], data["evidence"]):
                raise Conflict("proposal cites a withdrawn source; regenerate from remaining evidence")
            key = entry_key(*(data[k] for k in ("scope", "persona", "member", "category", "key")))
            row = self.db.execute("SELECT body FROM entries WHERE id=?", (key,)).fetchone()
            before = json.loads(row[0]) if row else None
            if data["category"] == "affection":
                identity = data["value"].get("character_key") or data["persona"]
                if any(
                    self.db.execute(
                        "SELECT 1 FROM affection_source_claims WHERE scope=? AND persona=? AND member=? AND message_id=?",
                        (data["scope"], identity, data["member"], e["message_id"]),
                    ).fetchone()
                    for e in data["evidence"]
                ):
                    raise Conflict("affection evidence was already used; collect fresh sources")
            # Repeated batches cannot create identical pending or active proposals.
            for record in self.db.execute(
                "SELECT body FROM proposals WHERE json_extract(body,'$.entry_id')=? ORDER BY created DESC", (key,)
            ):
                old = json.loads(record[0])
                if (
                    (mode == "auto" or bool(data.get("auto_guard_reason")))
                    and data["category"] == "affection"
                    and old["entry_id"] == key
                    and old["state"] in {"rejected", "rolled_back", "archived"}
                    and {e["message_id"] for e in old["evidence"]} & {e["message_id"] for e in data["evidence"]}
                ):
                    raise Conflict("automatic affection requires fresh evidence after rejection or rollback")
                same_base = data["category"] != "participation" or (old["baseline"], old["host_version"]) == (
                    data["baseline"],
                    data["host_version"],
                )
                still_pending = old["state"] in {"pending", "reviewed"} and old["base_revision"] == (
                    before["revision"] if before else 0
                )
                still_active = (
                    old["state"] == "applied"
                    and before
                    and before["active"]
                    and before["proposal_id"] == old["id"]
                    and (before.get("category") == "affection" or before["expires"] > time.time())
                )
                if (
                    old["entry_id"] == key
                    and old["value"] == data["value"]
                    and same_base
                    and (still_pending or still_active)
                ):
                    if old["state"] == "pending" and still_pending and not llm_approved(old) and mode != "auto":
                        self._refresh_pending_evidence(old, data, before, source)
                    return old
            now = time.time()
            p = {
                **data,
                "id": uuid.uuid4().hex,
                "entry_id": key,
                "before": before,
                "base_revision": before["revision"] if before else 0,
                "state": "pending",
                "mode": mode,
                "source": source,
                "created": now,
                "updated": now,
                "reviews": [],
                "evaluation": {"status": "not_validated", "note": "审查意见不是效果验证；应用后需观察真实互动。"},
            }
            self.db.execute(
                "INSERT INTO proposals VALUES(?,?,?,?,?,?,?)",
                (p["id"], p["scope"], p["persona"], p["state"], now, now, encode(p)),
            )
            self._audit(p, "proposed", source)
            return p

    def review(self, proposal_id, result, actor, expected_updated, *, evidence_fingerprint=None, owner=None):
        with self.transaction():
            p = self.get(proposal_id)
            if p["state"] not in {"pending", "reviewed"} or p["updated"] != expected_updated:
                raise Conflict("proposal changed while review was running")
            is_llm = actor.startswith("llm:")
            if is_llm and self.review_claim(proposal_id) not in (None, owner):
                raise Conflict("proposal is reserved by a review batch")
            p["reviews"].append({**result, "actor": actor, "created": time.time()})
            p["state"] = (
                "rejected"
                if result["verdict"] == "reject"
                else "reviewed"
                if result["verdict"] == "approve"
                else "pending"
            )
            if p["state"] == "rejected":
                self.db.execute("DELETE FROM review_claims WHERE proposal=?", (p["id"],))
            p.pop(DEFER_FINGERPRINT, None)
            if is_llm and result["verdict"] == "defer" and evidence_fingerprint:
                p[DEFER_FINGERPRINT] = evidence_fingerprint
            self._save(p)
            audit = {**result, "evidence_fingerprint": evidence_fingerprint} if p.get(DEFER_FINGERPRINT) else result
            self._audit(p, "review", actor, audit)
            return p

    def invalidate_review_defer(self, proposal_id, fingerprint):
        if not fingerprint:
            return
        with self.transaction():
            self.db.execute(
                """UPDATE proposals SET body=json_remove(body,'$.review_defer_fingerprint')
                WHERE id=? AND state='pending' AND json_extract(body,'$.review_defer_fingerprint')=?""",
                (proposal_id, fingerprint),
            )

    def pending_affection(self):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute("""SELECT p.body FROM proposals p
                WHERE p.state IN ('pending','reviewed') AND json_extract(p.body,'$.category')='affection'
                AND NOT EXISTS (SELECT 1 FROM review_claims WHERE proposal=p.id)
                ORDER BY p.created LIMIT 500""")
            ]

    def pending_profiles(self):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute("""SELECT body FROM proposals WHERE state IN ('pending','reviewed')
                AND json_extract(body,'$.category') IN ('style','typing_habit','preference','expression_example') ORDER BY created DESC,rowid DESC""")
            ]

    def pending_catchup_proposals(self, limit=200, *, after=None):
        cursor = ""
        args = ()
        if after is not None:
            cursor = "AND (p.created,p.rowid)>(?,coalesce((SELECT rowid FROM proposals WHERE id=?),0))"
            args = after
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.db.execute(
                    f"""SELECT p.body FROM proposals p
                    WHERE (p.state='pending' OR (p.state='reviewed' AND json_extract(p.body,'$.mode')='llm'))
                    AND NOT EXISTS (SELECT 1 FROM review_claims WHERE proposal=p.id)
                    {cursor}
                    ORDER BY p.created ASC, p.rowid ASC LIMIT ?""",
                    (*args, int(limit)),
                )
            ]

    def affection_daily_usage(self, scope, identity, member, *, window_seconds=86400, now=None):
        cutoff = (time.time() if now is None else float(now)) - max(1.0, float(window_seconds))
        total_abs = Decimal(0)
        count = 0
        with self.lock:
            rows = self.db.execute(
                """SELECT json_extract(body,'$.value.experience_delta'),
                          json_extract(body,'$.value.character_key'),
                          persona
                   FROM proposals
                   WHERE scope=? AND state='applied'
                     AND json_extract(body,'$.category')='affection'
                     AND json_extract(body,'$.member')=?
                     AND coalesce(json_extract(body,'$.applied_at'), updated) >= ?""",
                (scope, member, cutoff),
            ).fetchall()
        for raw_delta, char_key, persona in rows:
            row_identity = char_key or persona
            if row_identity != identity:
                continue
            if raw_delta is None:
                continue
            try:
                total_abs += abs(Decimal(str(raw_delta)))
            except Exception:
                continue
            count += 1
        return total_abs, count

    def escalate_affection_review(self, proposal_id, expected_updated, reason):
        with self.transaction():
            p = self.get(proposal_id)
            if p["category"] != "affection" or p["state"] not in {"pending", "reviewed"} or p["updated"] != expected_updated:
                raise Conflict("affection proposal changed")
            changed = False
            if p.get("auto_guard_reason") != reason:
                p["auto_guard_reason"] = reason
                changed = True
            if p["mode"] == "auto":
                p["previous_mode"] = p["mode"]
                p["mode"] = "llm_then_human"
                changed = True
            if changed:
                self._save(p)
                self._audit(p, "auto_guard_escalated", "system:affection-guard", {"reason": reason, "mode": p["mode"]})
            return p

    def adopt_profile_llm_mode(self, proposal_id, expected_updated):
        with self.transaction():
            p = self.get(proposal_id)
            if (
                p["category"] not in MEMBER_PROFILES
                or p["state"] not in {"pending", "reviewed"}
                or p["updated"] != expected_updated
            ):
                raise Conflict("member proposal changed")
            if is_free_text_typing_habit(p["category"], p.get("value")):
                if p["mode"] == "llm":
                    previous = p["mode"]
                    p["mode"] = "llm_then_human"
                    self._save(p)
                    self._audit(
                        p,
                        "review_mode_changed",
                        "system:member-review-policy",
                        {"before": previous, "after": "llm_then_human"},
                    )
                return p
            if p["mode"] != "llm":
                previous = p["mode"]
                p["mode"] = "llm"
                self._save(p)
                self._audit(
                    p, "review_mode_changed", "system:member-review-policy", {"before": previous, "after": "llm"}
                )
            return p

    def apply(
        self,
        proposal_id,
        actor,
        expected_updated,
        *,
        automatic_affection=False,
        owner=None,
        auto_review_threshold=AUTO_AFFECTION_REVIEW_THRESHOLD,
        auto_daily_limit=AUTO_AFFECTION_DAILY_ABS_LIMIT,
        auto_daily_max_count=AUTO_AFFECTION_DAILY_MAX_COUNT,
    ):
        with self.transaction():
            p = self.get(proposal_id)
            claim = self.review_claim(proposal_id)
            if claim is not None and claim != owner:
                raise Conflict("proposal is reserved by a review batch")
            if automatic_affection and p["category"] != "affection":
                raise Conflict("automatic mode is only supported for affection")
            if p["mode"] == "auto" and not automatic_affection:
                raise Conflict("automatic affection requires current configuration authorization")
            if not policy_allowed(p):
                raise Conflict("fixed personality policy rejects proposal")
            if p["updated"] != expected_updated or p["state"] not in {"pending", "reviewed"}:
                raise Conflict("proposal is no longer applicable")
            if self.source_withdrawals.has_withdrawn(p["scope"], p["evidence"]):
                raise Conflict("proposal cites a withdrawn source")
            approved = llm_approved(p)
            if is_free_text_typing_habit(p["category"], p.get("value")) and not actor.startswith("human:"):
                raise Conflict("human approval required")
            if not automatic_affection and p["mode"] in {"llm", "llm_then_human"} and not approved:
                raise Conflict("LLM approval required")
            if not automatic_affection and p["mode"] != "llm" and not actor.startswith("human:"):
                raise Conflict("human approval required")
            old_row = self.db.execute("SELECT body FROM entries WHERE id=?", (p["entry_id"],)).fetchone()
            old = json.loads(old_row[0]) if old_row else None
            if (old["revision"] if old else 0) != p["base_revision"]:
                raise Conflict("learned value changed; regenerate proposal")
            if p["category"] == "affection":
                sources = sorted({e["message_id"] for e in p["evidence"]})
                affection_identity = p["value"].get("character_key") or p["persona"]
                if automatic_affection:
                    if p.get("auto_guard_reason"):
                        raise Conflict("affection proposal requires review")
                    try:
                        delta_abs = abs(Decimal(str(p["value"].get("experience_delta", 0))))
                    except Exception:
                        raise Conflict("invalid affection delta") from None
                    if delta_abs >= Decimal(str(auto_review_threshold)):
                        raise Conflict("large affection delta requires review")
                    used_abs, used_count = self.affection_daily_usage(p["scope"], affection_identity, p["member"])
                    if used_abs + delta_abs > Decimal(str(auto_daily_limit)) or used_count >= int(auto_daily_max_count):
                        raise Conflict("daily automatic affection limit reached")
                for message_id in sources:
                    if self.db.execute(
                        "SELECT 1 FROM affection_source_claims WHERE scope=? AND persona=? AND member=? AND message_id=?",
                        (p["scope"], affection_identity, p["member"], message_id),
                    ).fetchone():
                        raise Conflict("affection evidence was already used; regenerate proposal with fresh sources")
                for message_id in sources:
                    self.db.execute(
                        "INSERT INTO affection_source_claims VALUES(?,?,?,?,?)",
                        (p["scope"], affection_identity, p["member"], message_id, p["id"]),
                    )
            entry = {
                k: p[k] for k in ("scope", "persona", "member", "category", "key", "value", "baseline", "host_version")
            }
            if "profile_policy" in p:
                entry["profile_policy"] = p["profile_policy"]
            if "scene_policy" in p:
                entry["scene_policy"] = p["scene_policy"]
            if "expression_policy" in p:
                entry["expression_policy"] = p["expression_policy"]
            entry.update(self.provenance(p))
            entry.update(
                id=p["entry_id"],
                revision=p["base_revision"] + 1,
                proposal_id=p["id"],
                expires=time.time() + p["days"] * 86400,
                active=True,
                state_updated_at=time.time(),
            )
            if p["category"] == "affection":
                entry["tone_updated_at"] = time.time()
            self.db.execute("INSERT OR REPLACE INTO entries VALUES(?,?)", (entry["id"], encode(entry)))
            p["state"], p["applied_revision"], p["applied_at"] = "applied", entry["revision"], time.time()
            if automatic_affection:
                p["previous_mode"] = p["mode"]
                p["mode"] = "auto"
            p["applied_by"] = actor
            self.db.execute("DELETE FROM review_claims WHERE proposal=?", (p["id"],))
            self._save(p)
            self._audit(p, "applied", actor)
            return p

    def rollback(self, proposal_id, actor, expected_updated=None):
        with self.transaction():
            p = self.get(proposal_id)
            if expected_updated is not None and p["updated"] != expected_updated:
                raise Conflict("proposal changed; refresh before rollback")
            row = self.db.execute("SELECT body FROM entries WHERE id=?", (p["entry_id"],)).fetchone()
            current = json.loads(row[0]) if row else {}
            if (
                p["state"] != "applied"
                or current.get("revision") != p["applied_revision"]
                or current.get("proposal_id") != p["id"]
            ):
                raise Conflict("a later change exists; rollback that change first")
            restored = dict(p["before"] or current)
            restored.update(revision=current["revision"] + 1, active=bool(p["before"] and p["before"].get("active")))
            if p["category"] == "affection" and (p["before"] is None or not restored["active"]):
                from .affection import state as affection_state

                prev_val = restored.get("value")
                char_key = prev_val.get("character_key") if isinstance(prev_val, dict) else None
                restored["value"] = affection_state({"character_key": char_key} if char_key else None)
            if not policy_allowed(restored):
                restored["active"] = False
            self.db.execute("UPDATE entries SET body=? WHERE id=?", (encode(restored), p["entry_id"]))
            # Re-link the prior version so stacked rollbacks remain possible.
            if p["before"] and not p["before"]["proposal_id"].startswith("legacy:"):
                previous = self.get(p["before"]["proposal_id"])
                previous["applied_revision"] = restored["revision"]
                self._save(previous)
            p["state"] = "rolled_back"
            self.db.execute("DELETE FROM affection_source_claims WHERE proposal=?", (p["id"],))
            p["rolled_back_by"], p["rolled_back_at"] = actor, time.time()
            self._save(p)
            self._audit(p, "rolled_back", actor)
            return p

    def purge(self, scope):
        with self.transaction():
            self.automatic_reviews.purge(scope)
            self.scenes.purge(scope)
            self.expressions.purge(scope)
            ids = [r[0] for r in self.db.execute("SELECT id FROM proposals WHERE scope=?", (scope,))]
            for pid in ids:
                self.db.execute("DELETE FROM audit WHERE proposal=?", (pid,))
                self.db.execute("DELETE FROM proposal_evidence_sources WHERE proposal=?", (pid,))
            for e in self.entries():
                if e["scope"] == scope:
                    self.db.execute("DELETE FROM entries WHERE id=?", (e["id"],))
            for table in (
                "events",
                "proposals",
                "withdrawn_review_sources",
                "checkpoints",
                "profile_raw",
                "profile_targets",
                "profile_messages",
                "native_scope_registry",
                "native_receipts",
                "legacy_archive",
                "analysis_inbox",
                "analysis_runs",
            ):
                self.db.execute(f"DELETE FROM {table} WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM analysis_sources WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM reply_usage WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM reply_usage_refs WHERE receipt NOT IN (SELECT id FROM reply_usage)")
            self.db.execute("DELETE FROM console_tasks WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM review_batches WHERE scope=?", (scope,))
            self.db.execute("DELETE FROM review_claims WHERE scope=?", (scope,))

    def latest_baselines(self):
        with self.lock:
            rows = self.db.execute("""SELECT e.scope,e.persona,
                json_extract(e.body,'$.baseline') AS baseline,
                json_extract(e.body,'$.host_version') AS host_version
                FROM events e JOIN
                (SELECT scope,persona,max(created) AS stamp FROM events GROUP BY scope,persona) q
                ON e.scope=q.scope AND e.persona=q.persona AND e.created=q.stamp""").fetchall()
            return {
                (r[0], r[1]): (r[2], r[3])
                for r in rows
                if r[2] is not None and r[3] is not None
            }

    def archive_stale_proposals(
        self,
        *,
        current_personas=None,
        latest_baselines=None,
        now=None,
        actor="system:auto-archive",
        scope="",
        include_policy_invalid=False,
        include_revision_changed=False,
        require_known_baseline=True,
    ):
        """Transition stale pending/reviewed proposals to archived with audit trail."""
        stamp = time.time() if now is None else float(now)
        archived = []
        reasons: dict[str, int] = {}
        with self.transaction():
            personas = (
                dict(self.latest_personas())
                if current_personas is None
                else {**self.latest_personas(), **current_personas}
            )
            baselines = (
                dict(self.latest_baselines())
                if latest_baselines is None
                else {**self.latest_baselines(), **latest_baselines}
            )
            active_claims = {
                r[0]
                for r in self.db.execute(
                    """SELECT c.proposal FROM review_claims c
                    JOIN review_batches j ON j.id=c.job
                    WHERE j.status IN ('queued','running','pausing','cancelling')"""
                ).fetchall()
            }
            if scope:
                rows = self.db.execute(
                    "SELECT body FROM proposals WHERE scope=? AND state IN ('pending','reviewed')",
                    (scope,),
                ).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT body FROM proposals WHERE state IN ('pending','reviewed')"
                ).fetchall()
            for row in rows:
                p = json.loads(row[0])
                if p["id"] in active_claims:
                    continue
                reason = ""
                if include_policy_invalid and not policy_allowed(p):
                    reason = "policy_invalid"
                elif personas.get(p["scope"]) is not None and personas.get(p["scope"]) != p["persona"]:
                    reason = "persona_changed"
                elif p["created"] + p["days"] * 86400 <= stamp:
                    reason = "expired"
                elif include_revision_changed:
                    entry_row = self.db.execute(
                        "SELECT json_extract(body,'$.revision') FROM entries WHERE id=?",
                        (p["entry_id"],),
                    ).fetchone()
                    if (entry_row[0] if entry_row else 0) != p["base_revision"]:
                        reason = "revision_changed"
                if not reason and p["category"] == "participation":
                    pair = (p["scope"], p["persona"])
                    if pair in baselines:
                        if baselines[pair] != (p["baseline"], p["host_version"]):
                            reason = "baseline_changed"
                    elif not require_known_baseline:
                        reason = "baseline_changed"
                if not reason:
                    continue
                p.update(
                    previous_state=p["state"],
                    state="archived",
                    archive_reason=reason,
                    archived_at=stamp,
                    archived_by=actor,
                )
                self.db.execute("DELETE FROM review_claims WHERE proposal=?", (p["id"],))
                self._save(p)
                self._audit(p, "archive", actor, {"reason": reason})
                archived.append(p["id"])
                reasons[reason] = reasons.get(reason, 0) + 1
        return {"count": len(archived), "archived": archived, "reasons": reasons}

    def _protected_applied_proposal_ids(self):
        protected = set()
        frontier = set()
        for row in self.db.execute("SELECT body FROM entries").fetchall():
            try:
                entry = json.loads(row[0])
            except (TypeError, ValueError):
                continue
            pid = entry.get("proposal_id")
            if isinstance(pid, str) and pid and not pid.startswith("legacy:"):
                protected.add(pid)
                frontier.add(pid)
        for _ in range(32):
            if not frontier:
                break
            next_frontier = set()
            placeholders = ",".join("?" for _ in frontier)
            for row in self.db.execute(
                f"SELECT body FROM proposals WHERE id IN ({placeholders})", tuple(frontier)
            ).fetchall():
                try:
                    p = json.loads(row[0])
                except (TypeError, ValueError):
                    continue
                before = p.get("before")
                if isinstance(before, dict):
                    prev_pid = before.get("proposal_id")
                    if (
                        isinstance(prev_pid, str)
                        and prev_pid
                        and not prev_pid.startswith("legacy:")
                        and prev_pid not in protected
                    ):
                        protected.add(prev_pid)
                        next_frontier.add(prev_pid)
            frontier = next_frontier
        return protected

    def prune(
        self,
        days,
        *,
        current_personas=None,
        latest_baselines=None,
        audit_retention_days=None,
        allowed_scopes=None,
        inbox_expire_days=None,
    ):
        archived = self.archive_stale_proposals(
            current_personas=current_personas,
            latest_baselines=latest_baselines,
            actor="system:auto-archive",
            include_policy_invalid=False,
            include_revision_changed=False,
            require_known_baseline=True,
        )
        self.automatic_reviews.prune(days)
        self.console.prune(days)
        self.scenes.prune(days)
        self.expressions.prune(days)
        audit_days = (
            max(1, int(audit_retention_days))
            if audit_retention_days is not None
            else max(DEFAULT_AUDIT_RETENTION_DAYS, max(1, int(days)))
        )
        proposals_compacted = 0
        proposals_pruned = 0
        inbox_expired = 0
        receipts_pruned = 0
        with self.lock, self.db:
            now = time.time()
            cutoff = now - days * 86400
            audit_cutoff = now - audit_days * 86400
            if allowed_scopes is not None:
                inbox_expired += self.db.execute(
                    "UPDATE analysis_inbox SET processed=1 WHERE processed=0 AND scope NOT IN (SELECT value FROM json_each(?))",
                    (encode(list(allowed_scopes)),),
                ).rowcount
            if current_personas is not None:
                for sc, active_persona in current_personas.items():
                    inbox_expired += self.db.execute(
                        "UPDATE analysis_inbox SET processed=1 WHERE processed=0 AND scope=? AND persona!=? AND created<?",
                        (sc, active_persona, cutoff),
                    ).rowcount
            if inbox_expire_days is not None:
                inbox_cutoff = now - max(1, int(inbox_expire_days)) * 86400
                inbox_expired += self.db.execute(
                    "UPDATE analysis_inbox SET processed=1 WHERE processed=0 AND created<?",
                    (inbox_cutoff,),
                ).rowcount
            self.db.execute(
                "DELETE FROM events WHERE created<? AND NOT EXISTS (SELECT 1 FROM analysis_inbox q WHERE q.event_id=events.id AND q.processed=0)",
                (cutoff,),
            )
            self.db.execute("DELETE FROM analysis_inbox WHERE processed=1 AND created<?", (cutoff,))
            self.db.execute("DELETE FROM analysis_runs WHERE created<?", (cutoff,))
            self.db.execute(
                "DELETE FROM analysis_sources WHERE NOT EXISTS (SELECT 1 FROM analysis_inbox q WHERE q.event_id=analysis_sources.event_id AND q.processed=0)"
            )
            for table in ("profile_raw", "profile_targets", "profile_messages"):
                self.db.execute(
                    "DELETE FROM "
                    + table
                    + " WHERE created<? AND NOT EXISTS (SELECT 1 FROM analysis_sources s WHERE s.scope="
                    + table
                    + ".scope AND s.source_id="
                    + table
                    + ".source_id)",
                    (cutoff,),
                )
            self.db.execute("DELETE FROM native_receipts WHERE created<?", (cutoff,))

            protected_applied = self._protected_applied_proposal_ids()
            for row in self.db.execute(
                "SELECT id, state, updated, body FROM proposals WHERE state IN ('rejected','archived','rolled_back','applied') AND updated<?",
                (cutoff,),
            ).fetchall():
                pid, state, updated, raw_body = row[0], row[1], row[2], row[3]
                if pid in protected_applied:
                    continue
                if state in {"rejected", "archived", "rolled_back"} and updated < audit_cutoff:
                    self.db.execute("DELETE FROM proposals WHERE id=?", (pid,))
                    proposals_pruned += 1
                    continue
                try:
                    p = json.loads(raw_body)
                except (TypeError, ValueError):
                    continue
                changed = False
                compacted_ev, ev_changed = _compact_evidence_list(p.get("evidence"))
                if ev_changed:
                    p["evidence"] = compacted_ev
                    changed = True
                if state in {"rejected", "archived", "rolled_back"} and isinstance(p.get("before"), dict):
                    slim_before = {
                        k: p["before"][k]
                        for k in ("id", "revision", "active", "proposal_id")
                        if k in p["before"]
                    }
                    if slim_before != p["before"]:
                        p["before"] = slim_before
                        changed = True
                if changed:
                    self.db.execute("UPDATE proposals SET body=? WHERE id=?", (encode(p), pid))
                    proposals_compacted += 1

            for row in self.db.execute(
                "SELECT id, body FROM audit WHERE action='evidence_updated' AND instr(body,'\"superseded\":[{')>0"
            ).fetchall():
                try:
                    abody = json.loads(row[1])
                except (TypeError, ValueError):
                    continue
                sup = abody.get("superseded")
                if isinstance(sup, list) and any(isinstance(item, dict) for item in sup):
                    abody["superseded"] = [item.get("id", "") if isinstance(item, dict) else item for item in sup]
                    self.db.execute("UPDATE audit SET body=? WHERE id=?", (encode(abody), row[0]))

            self.db.execute(
                "DELETE FROM proposal_evidence_sources WHERE proposal NOT IN (SELECT id FROM proposals WHERE state IN ('pending','reviewed'))"
            )
            self.db.execute(
                "DELETE FROM affection_source_claims WHERE proposal NOT IN (SELECT id FROM proposals WHERE state='applied')"
            )
            if self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='relation_source_receipts'"
            ).fetchone():
                cols = {r[1] for r in self.db.execute("PRAGMA table_info(relation_source_receipts)").fetchall()}
                if "created" in cols:
                    receipt_cutoff = int(now - RELATION_RECEIPT_RETENTION_DAYS * 86400)
                    receipts_pruned = self.db.execute(
                        "DELETE FROM relation_source_receipts WHERE created<?",
                        (receipt_cutoff,),
                    ).rowcount

            audit_pruned = self.db.execute("DELETE FROM audit WHERE created<?", (audit_cutoff,)).rowcount

        with self.lock:
            self.db.execute(f"PRAGMA journal_size_limit={JOURNAL_SIZE_LIMIT_BYTES}")
            self.db.execute(f"PRAGMA incremental_vacuum({INCREMENTAL_VACUUM_PAGES})")
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return {
            "archived": archived["count"],
            "archived_ids": archived["archived"],
            "audit_pruned": audit_pruned,
            "proposals_compacted": proposals_compacted,
            "proposals_pruned": proposals_pruned,
            "inbox_expired": inbox_expired,
            "receipts_pruned": receipts_pruned,
        }

    def observations(self, proposal):
        """Operational observations, never inferred quality labels or counterfactual scores."""
        rows = self.events(proposal["scope"], proposal["persona"], 500)
        groups = {"before": {}, "after": {}}
        for row in rows:
            versions = row.get("policy", {}).get("versions", [])
            if proposal["id"] in versions:
                side = "after"
            elif row["received_at"] < proposal.get("applied_at", proposal["created"]):
                side = "before"
            else:
                continue
            turn = groups[side].setdefault(row["turn_id"], {"decision": None, "delivered": False, "failure": False})
            if isinstance(row.get("decision"), dict):
                turn["decision"] = row["decision"].get("action")
            outcome = row.get("outcome", {})
            turn["delivered"] |= outcome.get("delivered") is True or row.get("reply", {}).get("delivered") is True
            turn["failure"] |= outcome.get("final_outcome") in {"delivery_failed", "generation_failed"}

        def summary(turns):
            values = list(turns.values())
            return {
                "turns": len(values),
                "admitted": sum(v["decision"] not in (None, "ignore") for v in values),
                "silent": sum(v["decision"] == "ignore" for v in values),
                "delivered": sum(v["delivered"] for v in values),
                "failed_without_delivery": sum(v["failure"] and not v["delivered"] for v in values),
            }

        return {k: summary(v) for k, v in groups.items()}

    def bind_legacy_social(self, scope, persona, legacy_key):
        from .affection import state

        with self.lock, self.db:
            if not isinstance(legacy_key, str) or not legacy_key.startswith("narrative:v1:"):
                raise ValueError("verified character scope required")
            native_keys = {
                r[0]
                for r in self.db.execute(
                    "SELECT DISTINCT source_key FROM legacy_social WHERE scope=? AND category='affection' AND source_key LIKE 'narrative:v1:%'",
                    (scope,),
                )
            }
            owner = self.db.execute("SELECT legacy_key FROM legacy_group_owners WHERE scope=?", (scope,)).fetchone()
            allow_bare = (owner is not None and owner[0] == legacy_key) or (
                owner is None and (not native_keys or legacy_key in native_keys)
            )
            if owner is None and allow_bare:
                self.db.execute("INSERT INTO legacy_group_owners VALUES(?,?)", (scope, legacy_key))
            bare = scope.rsplit(":", 1)[-1] if allow_bare else ""
            rows = self.db.execute(
                """SELECT source_key,category,member,item_key,body FROM legacy_social
                WHERE scope=? AND (category='jargon' OR source_key IN (?,?))
                ORDER BY (source_key=?) DESC,(source_key=?) DESC,source_key DESC""",
                (scope, legacy_key, bare, legacy_key, scope.rsplit(":", 1)[-1]),
            ).fetchall()
            seen = set()
            aliases = {
                r[0]
                for r in self.db.execute(
                    "SELECT persona FROM social_bindings WHERE scope=? AND legacy_key=?", (scope, legacy_key)
                )
            }
            previous_affection = {}
            for old in self.entries():
                if (
                    old["scope"] == scope
                    and old["persona"] in aliases
                    and old["category"] == "affection"
                    and old["active"]
                ):
                    current = previous_affection.get(old["member"])
                    if current is None or old.get("state_updated_at", 0) > current.get("state_updated_at", 0):
                        previous_affection[old["member"]] = old
            for row in rows:
                source_key, category, member, key, body = row
                identity = (category, member, key)
                if identity in seen:
                    continue
                seen.add(identity)
                eid = entry_key(scope, persona, member, category, key)
                if self.db.execute("SELECT 1 FROM entries WHERE id=?", (eid,)).fetchone():
                    continue
                source = json.loads(body)
                value = state(source["value"]) if category == "affection" else source["value"]
                if category == "affection":
                    if member in previous_affection:
                        value = dict(previous_affection[member]["value"])
                    value["character_key"] = legacy_key
                entry = dict(
                    id=eid,
                    scope=scope,
                    persona=persona,
                    member=member,
                    category=category,
                    key=key,
                    value=value,
                    revision=1,
                    proposal_id="legacy:" + digest([scope, legacy_key, source_key, identity, source]),
                    expires=time.time() + 10 * 365 * 86400,
                    active=True,
                    baseline="",
                    host_version="",
                    provenance=source.get("provenance", {}),
                    imported=True,
                )
                if category == "affection" and member in previous_affection:
                    entry["state_updated_at"] = previous_affection[member].get("state_updated_at", 0)
                self.db.execute("INSERT INTO entries VALUES(?,?)", (eid, encode(entry)))
            for member, previous in previous_affection.items():
                eid = entry_key(scope, persona, member, "affection", "bot_affection")
                if self.db.execute("SELECT 1 FROM entries WHERE id=?", (eid,)).fetchone():
                    continue
                entry = {
                    **previous,
                    "id": eid,
                    "persona": persona,
                    "revision": 1,
                    "proposal_id": "legacy:" + digest([previous["id"], previous["revision"], persona]),
                    "provenance": {"previous_entry": previous["id"], "previous_proposal": previous["proposal_id"]},
                    "imported": True,
                }
                self.db.execute("INSERT INTO entries VALUES(?,?)", (eid, encode(entry)))
            self.db.execute("INSERT OR REPLACE INTO social_bindings VALUES(?,?,?)", (scope, persona, legacy_key))
