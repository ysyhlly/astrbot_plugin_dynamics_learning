"""Bounded administrative views and body-free task / reply-use receipts."""

import json
import time
import uuid

from .contract import digest, encode


class Console:
    def __init__(self, store):
        self.store = store
        store.db.executescript("""
        CREATE INDEX IF NOT EXISTS proposals_console ON proposals(state,json_extract(body,'$.category'),created);
        CREATE INDEX IF NOT EXISTS proposals_member ON proposals(scope,persona,json_extract(body,'$.member'),state,created);
        CREATE INDEX IF NOT EXISTS profiles_member_time ON profile_messages(scope,persona,member,created);
        CREATE TABLE IF NOT EXISTS console_tasks(id TEXT PRIMARY KEY,scope TEXT,phase TEXT,status TEXT,queued REAL,started REAL,finished REAL,error TEXT,retry REAL);
        CREATE INDEX IF NOT EXISTS console_tasks_time ON console_tasks(queued);
        CREATE TABLE IF NOT EXISTS reply_usage(id TEXT PRIMARY KEY,scope TEXT,member TEXT,turn_id TEXT,created REAL,body TEXT);
        CREATE INDEX IF NOT EXISTS reply_usage_scope ON reply_usage(scope,created);
        CREATE INDEX IF NOT EXISTS reply_usage_created ON reply_usage(created);
        CREATE TABLE IF NOT EXISTS reply_usage_refs(receipt TEXT,source_id TEXT,PRIMARY KEY(receipt,source_id));
        CREATE INDEX IF NOT EXISTS reply_usage_refs_source ON reply_usage_refs(source_id,receipt);
        CREATE TABLE IF NOT EXISTS review_batches(id TEXT PRIMARY KEY,scope TEXT,status TEXT,created REAL,body TEXT);
        CREATE INDEX IF NOT EXISTS review_batches_scope ON review_batches(scope,created);
        CREATE INDEX IF NOT EXISTS review_batches_status_created ON review_batches(status,created);
        CREATE TABLE IF NOT EXISTS analysis_meta(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS llm_usage_stats(id INTEGER PRIMARY KEY AUTOINCREMENT,scope TEXT,phase TEXT,ok INTEGER,
          prompt_chars INTEGER,completion_chars INTEGER,prompt_tokens INTEGER,completion_tokens INTEGER,elapsed_ms INTEGER,created REAL);
        CREATE INDEX IF NOT EXISTS llm_usage_stats_created ON llm_usage_stats(created);
        CREATE INDEX IF NOT EXISTS llm_usage_stats_phase ON llm_usage_stats(phase,created);
        """)
        with store.transaction():
            if not store.db.execute("SELECT 1 FROM analysis_meta WHERE key='reply_usage_refs_backfilled_v1'").fetchone():
                for field in ("profiles", "learning"):
                    store.db.execute(
                        "INSERT OR IGNORE INTO reply_usage_refs SELECT u.id,json_extract(r.value,'$.source_id') FROM reply_usage u,json_each(u.body,?) r WHERE json_extract(r.value,'$.source_id') IS NOT NULL",
                        ("$.{}".format(field),),
                    )
                store.db.execute(
                    "INSERT OR REPLACE INTO analysis_meta VALUES('reply_usage_refs_backfilled_v1',?)",
                    (str(time.time()),),
                )
            store.db.execute(
                "UPDATE console_tasks SET status='interrupted',finished=?,error='Restarted' WHERE status IN ('queued','running')",
                (time.time(),),
            )

    def task(self, scope, phase):
        identity = uuid.uuid4().hex
        with self.store.transaction():
            self.store.db.execute(
                "INSERT INTO console_tasks VALUES(?,?,?,?,?,?,?,?,?)",
                (identity, scope, phase, "queued", time.time(), None, None, "", None),
            )
        return identity

    def task_started(self, identity):
        with self.store.transaction():
            self.store.db.execute(
                "UPDATE console_tasks SET status='running',started=? WHERE id=?", (time.time(), identity)
            )

    def task_finished(self, identity, status, error="", retry=None):
        with self.store.transaction():
            self.store.db.execute(
                "UPDATE console_tasks SET status=?,finished=?,error=?,retry=? WHERE id=?",
                (status, time.time(), error, retry, identity),
            )

    def delete_task(self, identity):
        with self.store.transaction():
            self.store.db.execute("DELETE FROM console_tasks WHERE id=?", (identity,))

    def tasks(self):
        with self.store.lock:
            rows = [dict(r) for r in self.store.db.execute("SELECT * FROM console_tasks ORDER BY queued DESC LIMIT 60")]
            latest = {}
            for row in rows:
                key = (row["scope"], row["phase"])
                if row["status"] == "succeeded":
                    latest[key] = max(latest.get(key, 0), row["finished"] or 0)
                row["recovered"] = latest.get(key, 0) > (row["finished"] or time.time())
                row["queue_ms"] = round(1000 * ((row["started"] or time.time()) - row["queued"]))
                row["elapsed_ms"] = (
                    round(1000 * ((row["finished"] or time.time()) - row["started"])) if row["started"] else None
                )
            return rows

    @staticmethod
    def summary(p):
        result = {
            k: p[k]
            for k in (
                "id",
                "scope",
                "persona",
                "member",
                "category",
                "key",
                "value",
                "state",
                "mode",
                "created",
                "updated",
                "days",
                "reason",
                "evaluation",
            )
            if k in p
        }
        result["before"] = {k: p["before"].get(k) for k in ("active", "value", "revision")} if p.get("before") else None
        result["reviews"] = p.get("reviews", [])[-1:]
        result["evidence_count"] = len(p.get("evidence", []))
        result["evidence"] = []
        result["details_lazy"] = True
        risks: list[str] = []
        seen_risks: set[str] = set()
        for raw_risk in p.get("risks") or ():
            clean = str(raw_risk or "").strip()
            if clean and clean not in seen_risks:
                seen_risks.add(clean)
                risks.append(clean)
        for rev in reversed(p.get("reviews") or ()):
            if not isinstance(rev, dict):
                continue
            for raw_risk in rev.get("risks") or ():
                clean = str(raw_risk or "").strip()
                if clean and clean not in seen_risks:
                    seen_risks.add(clean)
                    risks.append(clean)
        result["risks"] = risks
        for key in ("applied_by", "rolled_back_by", "auto_guard_reason", "applied_revision", "applied_at", "source_kind", "submitted_by", "restored_at"):
            if key in p:
                result[key] = p[key]
        return result

    def proposals(self, scope="", limit=20):
        return [self.summary(p) for p in self.store.proposals(scope, limit)]

    def entries(
        self,
        scope="",
        category="",
        state="active",
        query="",
        page=1,
        *,
        affection_view=None,
        current_personas=None,
        allowed_scopes=None,
    ):
        page = max(1, min(10000, int(page)))
        with self.store.lock:
            view = ("current" if state == "active" else "all") if affection_view is None else affection_view
            if not isinstance(view, str) or view not in {"current", "history", "all"}:
                raise ValueError("invalid affection view")
            if affection_view is not None and category != "affection":
                raise ValueError("affection view requires affection category")
            selected, metadata = set(), {}
            if category in {"", "affection"}:
                from .affection_view import current_records

                selected, metadata = current_records(self.store, scope, current_personas)
            condition = "(?='' OR json_extract(body,'$.scope')=?) AND (?='' OR json_extract(body,'$.category')=?) AND (?='' OR instr(lower(body),lower(?))>0)"
            args = [scope, scope, category, category, query, query]
            if allowed_scopes is not None:
                condition += " AND json_extract(body,'$.scope') IN (SELECT value FROM json_each(?))"
                args.append(encode(list(allowed_scopes)))
            if state == "active":
                condition += " AND json_extract(body,'$.active')=1 AND (json_extract(body,'$.category')='affection' OR json_extract(body,'$.expires')>?)"
                args.append(time.time())
            elif state == "expired":
                condition += " AND json_extract(body,'$.active')=1 AND json_extract(body,'$.category')!='affection' AND json_extract(body,'$.expires')<=?"
                args.append(time.time())
            elif state == "inactive":
                condition += " AND json_extract(body,'$.active')!=1"
            elif state != "all":
                raise ValueError("invalid entry state")
            if category in {"", "affection"} and view == "current":
                condition += (
                    " AND (json_extract(body,'$.category')!='affection' OR id IN (SELECT value FROM json_each(?)))"
                )
                args.append(encode(sorted(selected)))
            elif category == "affection" and view == "history":
                condition += " AND id NOT IN (SELECT value FROM json_each(?))"
                args.append(encode(sorted(selected)))
            total = self.store.db.execute("SELECT count(*) FROM entries WHERE " + condition, args).fetchone()[0]
            ordering = (
                "json_extract(body,'$.value.affection_level') DESC,id" if category == "affection" else "rowid DESC"
            )
            rows = self.store.db.execute(
                "SELECT body FROM entries WHERE " + condition + " ORDER BY " + ordering + " LIMIT 25 OFFSET ?",
                (*args, (page - 1) * 25),
            ).fetchall()
            entries = [json.loads(r[0]) for r in rows]
            for entry in entries:
                if entry["category"] == "affection":
                    entry.update(metadata.get(entry["id"], {}))
            return {
                "total": total,
                "page": page,
                "page_size": 25,
                "entries": entries,
                "affection_view": view if category == "affection" else None,
            }

    def record_usage(self, raw, allowed):
        if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("scope") not in allowed:
            raise ValueError("usage scope not allowed")
        if raw.get("stage") != "provider_kwargs_after_boundaries":
            raise ValueError("provider proof required")
        if any(
            not isinstance(raw.get(k, []), list) for k in ("profiles", "learning", "memory_refs", "omitted")
        ) or not isinstance(raw.get("rhythm", {}), dict):
            raise ValueError("invalid usage fields")
        for key in ("member", "turn_id", "material_hash"):
            if not isinstance(raw.get(key), str) or not 1 <= len(raw[key]) <= 256:
                raise ValueError("invalid usage identity")
        # Only identifiers and counters, never source text or model prompts.
        clean = {k: raw[k] for k in ("scope", "member", "turn_id", "material_hash", "stage")}
        clean["profiles"] = []
        for row in raw.get("profiles", [])[:30]:
            if not isinstance(row, dict):
                continue
            clean["profiles"].append(
                {k: str(row[k])[:256] for k in ("source_id", "category", "revision", "source_umo") if k in row}
            )
        clean["learning"] = [
            {k: str(row[k])[:256] for k in ("source_id", "category", "revision", "source_umo") if k in row}
            for row in raw.get("learning", [])[:30]
            if isinstance(row, dict)
        ]
        clean["memory_refs"] = [str(v)[:256] for v in raw.get("memory_refs", [])[:8]]
        clean["omitted"] = [
            {"source_id": str(r.get("source_id", ""))[:128], "reason": str(r.get("reason", ""))[:80]}
            for r in raw.get("omitted", [])[:30]
            if isinstance(r, dict)
        ]
        clean["rhythm"] = {
            k: str(v)[:80]
            for k, v in raw.get("rhythm", {}).items()
            if k in ("action", "mode", "length", "continuation_kind")
        }
        clean["dialogue"] = raw.get("dialogue") is True
        clean["personality_locked"] = raw.get("personality_locked") is True
        identity = digest([clean["scope"], clean["turn_id"], clean["material_hash"]])
        with self.store.transaction():
            inserted = self.store.db.execute(
                "INSERT OR IGNORE INTO reply_usage VALUES(?,?,?,?,?,?)",
                (identity, clean["scope"], clean["member"], clean["turn_id"], time.time(), encode(clean)),
            ).rowcount
            for row in clean["profiles"] + clean["learning"] if inserted else []:
                if row.get("source_id"):
                    self.store.db.execute(
                        "INSERT OR IGNORE INTO reply_usage_refs VALUES(?,?)", (identity, row["source_id"])
                    )

    def usage(self, scope, page=1):
        page = max(1, min(10000, int(page)))
        with self.store.lock:
            total = self.store.db.execute("SELECT count(*) FROM reply_usage WHERE scope=?", (scope,)).fetchone()[0]
            rows = self.store.db.execute(
                "SELECT id,created,body FROM reply_usage WHERE scope=? ORDER BY created DESC LIMIT 20 OFFSET ?",
                (scope, (page - 1) * 20),
            ).fetchall()
            records = [json.loads(row["body"]) for row in rows]
            source_ids = list(
                dict.fromkeys(
                    profile.get("source_id", "")
                    for record in records
                    for profile in record.get("profiles", []) + record.get("learning", [])
                    if profile.get("source_id")
                )
            )
            labels = {}
            for offset in range(0, len(source_ids), 400):
                batch = source_ids[offset : offset + 400]
                placeholders = ",".join("?" for _ in batch)
                for pid, key in self.store.db.execute(
                    f"SELECT json_extract(body,'$.proposal_id'),json_extract(body,'$.key') FROM entries WHERE json_extract(body,'$.proposal_id') IN ({placeholders})",
                    batch,
                ):
                    if pid is not None and pid not in labels:
                        labels[pid] = key or ""
            result = []
            for row, record in zip(rows, records):
                for profile in record["profiles"] + record.get("learning", []):
                    profile["label"] = labels.get(profile.get("source_id", ""), "历史版本")
                result.append({"id": row["id"], "created": row["created"], **record})
            return {"total": total, "page": page, "page_size": 20, "records": result}

    def daily(self):
        cutoff = time.time() - 86400
        with self.store.lock:
            return {
                "window": "last_24h",
                "applied": self.store.db.execute(
                    "SELECT count(*) FROM audit WHERE action='applied' AND created>?", (cutoff,)
                ).fetchone()[0],
                "provider_uses": self.store.db.execute(
                    "SELECT count(*) FROM reply_usage WHERE created>?", (cutoff,)
                ).fetchone()[0],
            }

    def record_llm_usage(
        self,
        *,
        scope: str = "",
        phase: str = "analysis",
        ok: bool = True,
        prompt_chars: int = 0,
        completion_chars: int = 0,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        elapsed_ms: int = 0,
    ) -> None:
        clean_phase = str(phase or "analysis").strip().lower()
        if clean_phase not in {"analysis", "review", "maintenance"}:
            clean_phase = "analysis"
        p_chars = max(0, int(prompt_chars or 0))
        c_chars = max(0, int(completion_chars or 0))
        p_tok = (
            max(0, int(prompt_tokens))
            if isinstance(prompt_tokens, (int, float)) and not isinstance(prompt_tokens, bool) and prompt_tokens > 0
            else (max(1, round(p_chars / 2.2)) if p_chars > 0 else 0)
        )
        c_tok = (
            max(0, int(completion_tokens))
            if isinstance(completion_tokens, (int, float))
            and not isinstance(completion_tokens, bool)
            and completion_tokens > 0
            else (max(1, round(c_chars / 2.2)) if c_chars > 0 else 0)
        )
        with self.store.transaction():
            self.store.db.execute(
                "INSERT INTO llm_usage_stats(scope,phase,ok,prompt_chars,completion_chars,prompt_tokens,completion_tokens,elapsed_ms,created) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(scope or "")[:512],
                    clean_phase,
                    1 if ok else 0,
                    p_chars,
                    c_chars,
                    p_tok,
                    c_tok,
                    max(0, int(elapsed_ms or 0)),
                    time.time(),
                ),
            )

    def llm_usage_summary(self, hours: int = 24) -> dict:
        window = max(1, min(24 * 30, int(hours or 24)))
        cutoff = time.time() - window * 3600
        phases = {
            name: {
                "calls": 0,
                "ok": 0,
                "failed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "avg_ms": 0,
            }
            for name in ("analysis", "review", "maintenance")
        }
        with self.store.lock:
            rows = self.store.db.execute(
                "SELECT phase, count(*) AS calls, sum(ok) AS ok_calls, "
                "coalesce(sum(prompt_tokens),0) AS p_tok, coalesce(sum(completion_tokens),0) AS c_tok, "
                "coalesce(avg(elapsed_ms),0) AS avg_ms "
                "FROM llm_usage_stats WHERE created>=? GROUP BY phase",
                (cutoff,),
            ).fetchall()
        total_calls = ok_calls = p_sum = c_sum = 0
        for row in rows:
            ph = str(row["phase"] or "analysis")
            if ph not in phases:
                continue
            calls = int(row["calls"] or 0)
            ok = int(row["ok_calls"] or 0)
            pt = int(row["p_tok"] or 0)
            ct = int(row["c_tok"] or 0)
            phases[ph] = {
                "calls": calls,
                "ok": ok,
                "failed": max(0, calls - ok),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
                "avg_ms": round(float(row["avg_ms"] or 0)),
            }
            total_calls += calls
            ok_calls += ok
            p_sum += pt
            c_sum += ct
        return {
            "window_hours": window,
            "total_calls": total_calls,
            "ok_calls": ok_calls,
            "failed_calls": max(0, total_calls - ok_calls),
            "estimated_prompt_tokens": p_sum,
            "estimated_completion_tokens": c_sum,
            "estimated_total_tokens": p_sum + c_sum,
            "by_phase": phases,
        }

    def prune(self, days):
        with self.store.transaction():
            self.store.db.execute(
                "DELETE FROM review_batches WHERE status NOT IN ('queued','running','pausing','paused','cancelling','interrupted') AND coalesce(json_extract(body,'$.updated'),created)<?",
                (time.time() - days * 86400,),
            )
            for table, column in [("console_tasks", "queued"), ("reply_usage", "created"), ("llm_usage_stats", "created")]:
                self.store.db.execute("DELETE FROM " + table + " WHERE " + column + "<?", (time.time() - days * 86400,))
            self.store.db.execute("DELETE FROM reply_usage_refs WHERE receipt NOT IN (SELECT id FROM reply_usage)")
