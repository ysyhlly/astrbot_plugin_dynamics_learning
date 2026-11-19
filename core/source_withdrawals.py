"""Keep withdrawn native evidence out of Learning without changing raw history."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping

from .contract import encode


def recall_source(event):
    if event.get_platform_name() != "aiocqhttp":
        return None
    raw = getattr(event.message_obj, "raw_message", None)
    if not isinstance(raw, Mapping) or raw.get("post_type") != "notice":
        return None
    scope = str(event.unified_msg_origin)
    scope_parts = scope.rsplit(":", 2)
    if len(scope_parts) != 3:
        return None
    platform, scope_type, session = scope_parts
    if platform != str(event.get_platform_id()):
        return None
    member, target = str(raw.get("user_id", "")), str(raw.get("message_id", ""))
    if not re.fullmatch(r"[1-9][0-9]{0,31}", member) or member != str(
        event.get_sender_id()
    ):
        return None
    if not re.fullmatch(r"-?[0-9]{1,32}", target):
        return None
    kind = raw.get("notice_type")
    if (
        kind == "group_recall"
        and scope_type == "GroupMessage"
        and session == str(raw.get("group_id")) == str(event.get_group_id())
        and re.fullmatch(r"[1-9][0-9]{0,31}", session)
    ):
        return scope, member, target
    if (
        kind == "friend_recall"
        and scope_type == "FriendMessage"
        and session == member
        and not raw.get("group_id")
    ):
        return scope, member, target
    return None


def _references(row):
    context = row.get("context", {})
    grouped = context.get("message_ids", []) if isinstance(context, dict) else []
    ids = [row.get("message_id"), *(grouped if isinstance(grouped, list) else [])]
    return {
        (row.get("member"), str(identity))
        for identity in ids
        if isinstance(identity, (str, int)) and identity
    }


class SourceWithdrawals:
    def __init__(self, store):
        self.store = store
        store.db.execute("""CREATE TABLE IF NOT EXISTS withdrawn_review_sources(
            scope TEXT, member TEXT, message_id TEXT, PRIMARY KEY(scope,member,message_id))""")

    def has_withdrawn(self, scope, rows):
        with self.store.lock:
            if not self.store.db.execute(
                "SELECT 1 FROM withdrawn_review_sources WHERE scope=? LIMIT 1", (scope,)
            ).fetchone():
                return False
            references = {
                reference
                for row in rows
                if isinstance(row, dict)
                for reference in _references(row)
            }
            return bool(
                references
                and self.store.db.execute(
                    """SELECT 1 FROM json_each(?) AS source JOIN withdrawn_review_sources AS w
                ON w.scope=? AND w.member=json_extract(source.value,'$[0]')
                AND w.message_id=json_extract(source.value,'$[1]') LIMIT 1""",
                    (encode(list(references)), scope),
                ).fetchone()
            )

    def _authoritative_source(self, row):
        scope, persona, member = (row[key] for key in ("scope", "persona", "member"))
        db = self.store.db
        direct = db.execute(
            "SELECT body FROM events WHERE id=? AND scope=? AND persona=?",
            (row["id"], scope, persona),
        ).fetchone()
        if direct:
            return json.loads(direct[0])
        if not row.get("context", {}).get("source_verified"):
            return None
        verified = db.execute(
            "SELECT body FROM profile_messages WHERE scope=? AND persona=? AND member=? AND source_id=?",
            (scope, persona, member, row["message_id"]),
        ).fetchone()
        original = json.loads(verified[0]) if verified else None
        return original if original and original["id"] == row["id"] else None

    def _requeue(self, proposal, remaining):
        db = self.store.db
        source_rows_durable = False
        for evidence in remaining:
            if not isinstance(evidence, dict) or (
                evidence.get("scope"),
                evidence.get("persona"),
            ) != (
                proposal["scope"],
                proposal["persona"],
            ):
                continue
            event = self._authoritative_source(evidence)
            if (
                not event
                or event["member"] != evidence["member"]
                or self.has_withdrawn(proposal["scope"], [event])
            ):
                continue
            source_rows_durable = True
            if proposal["category"] not in {
                "scene_pattern",
                "scene_adjustment",
                "expression_style",
                "expression_adjustment",
            }:
                self.store.scheduling.requeue(event)
        if source_rows_durable and proposal["category"] == "expression_style":
            db.execute(
                "DELETE FROM expression_analysis WHERE scope=? AND persona=?",
                (proposal["scope"], proposal["persona"]),
            )

    def record(self, scope, member, message_id, *, authorized=True):
        db = self.store.db
        with self.store.transaction():
            if db.execute(
                "SELECT 1 FROM withdrawn_review_sources WHERE scope=? AND member=? AND message_id=?",
                (scope, member, message_id),
            ).fetchone():
                return False
            target = (member, message_id)
            pending = [
                (row[0],)
                for row in self.store.scheduling.pending_with_sources(scope)
                if row[1] and target in _references(json.loads(row[1]))
            ]
            affected = []
            for row in db.execute(
                "SELECT body FROM proposals WHERE scope=? AND state IN ('pending','reviewed')",
                (scope,),
            ):
                proposal = json.loads(row[0])
                evidence = proposal.get("evidence", [])
                remaining = [
                    event
                    for event in evidence
                    if not isinstance(event, dict) or target not in _references(event)
                ]
                if len(remaining) != len(evidence):
                    affected.append((proposal, remaining))
            # A scope that was disabled can still revoke an exact source already on file.
            if (
                not authorized
                and not pending
                and not affected
                and not db.execute(
                    "SELECT 1 FROM profile_raw WHERE scope=? AND source_id=? AND json_extract(body,'$.member')=?",
                    (scope, message_id, member),
                ).fetchone()
            ):
                return False
            db.execute(
                "INSERT INTO withdrawn_review_sources VALUES(?,?,?)",
                (scope, member, message_id),
            )
            db.executemany("UPDATE analysis_inbox SET processed=1 WHERE seq=?", pending)
            actor = "system:source-withdrawal"
            for proposal, remaining in affected:
                proposal.update(
                    previous_state=proposal["state"],
                    state="archived",
                    archive_reason="source_withdrawn",
                    archived_at=time.time(),
                    archived_by=actor,
                )
                db.execute(
                    "DELETE FROM review_claims WHERE proposal=?", (proposal["id"],)
                )
                self.store._save(proposal)
                self.store._audit(
                    proposal,
                    "archive",
                    actor,
                    {"reason": "source_withdrawn", "message_id": message_id},
                )
                self._requeue(proposal, remaining)
            return True
