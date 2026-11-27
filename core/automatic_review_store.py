"""Persistence for automatic first-review work and review provider attempts."""

from __future__ import annotations

import json
import time

from .contract import encode


class AutomaticReviewStore:
    def __init__(self, store):
        self.store = store
        with store.lock:
            store.db.executescript("""
                CREATE TABLE IF NOT EXISTS automatic_review_queue(
                    proposal TEXT PRIMARY KEY, scope TEXT NOT NULL, updated REAL NOT NULL,
                    ready REAL NOT NULL, retry REAL NOT NULL DEFAULT 0,
                    singleton INTEGER NOT NULL DEFAULT 0,
                    lease TEXT, lease_until REAL NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS automatic_review_scope ON automatic_review_queue(scope);
                CREATE TABLE IF NOT EXISTS review_provider_attempts(
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, provider TEXT NOT NULL,
                    model TEXT NOT NULL, kind TEXT NOT NULL, items TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_provider_attempts_created ON review_provider_attempts(created);
            """)

    def enqueue(self, proposals, now):
        accepted = set()
        with self.store.transaction():
            for p in proposals:
                row = self.store.db.execute(
                    "SELECT updated,state,json_array_length(body,'$.reviews') FROM proposals WHERE id=?", (p["id"],)
                ).fetchone()
                if row is None or row["state"] != "pending" or row["updated"] != p["updated"] or row[2]:
                    continue
                self.store.db.execute(
                    """INSERT INTO automatic_review_queue(proposal,scope,updated,ready) VALUES(?,?,?,?)
                    ON CONFLICT(proposal) DO UPDATE SET updated=excluded.updated,lease=NULL,lease_until=0
                    WHERE automatic_review_queue.updated!=excluded.updated""",
                    (p["id"], p["scope"], p["updated"], now),
                )
                accepted.add(p["id"])
        return accepted

    def undiscovered(self, scopes, now):
        with self.store.lock:
            return [
                json.loads(row[0])
                for row in self.store.db.execute(
                    """SELECT p.body FROM proposals p
                    WHERE p.state='pending' AND p.scope IN (SELECT value FROM json_each(?))
                    AND json_extract(p.body,'$.mode') IN ('llm','llm_then_human')
                    AND json_extract(p.body,'$.source') NOT LIKE 'human:%'
                    AND json_extract(p.body,'$.category') NOT IN ('expression_style','expression_adjustment','scene_adjustment')
                    AND json_array_length(p.body,'$.reviews')=0
                    AND json_extract(p.body,'$.created')+json_extract(p.body,'$.days')*86400>?
                    AND json_extract(p.body,'$.base_revision')=coalesce((
                        SELECT json_extract(e.body,'$.revision') FROM entries e WHERE e.id=json_extract(p.body,'$.entry_id')
                    ),0)
                    AND NOT EXISTS (SELECT 1 FROM review_claims c WHERE c.proposal=p.id)
                    AND NOT EXISTS (SELECT 1 FROM automatic_review_queue q WHERE q.proposal=p.id AND q.updated=p.updated)
                    ORDER BY p.created,p.id""",
                    (encode(scopes), now),
                )
            ]

    def waiting(self, now, *, include_leased=False):
        with self.store.transaction():
            self.store.db.execute(
                """DELETE FROM automatic_review_queue AS q WHERE NOT EXISTS (
                SELECT 1 FROM proposals p WHERE p.id=q.proposal AND p.state='pending'
                AND json_array_length(p.body,'$.reviews')=0
                AND json_extract(p.body,'$.created')+json_extract(p.body,'$.days')*86400>?
                AND json_extract(p.body,'$.base_revision')=coalesce((
                    SELECT json_extract(e.body,'$.revision') FROM entries e WHERE e.id=json_extract(p.body,'$.entry_id')
                ),0))""",
                (now,),
            )
            return [
                ({k: row[k] for k in row.keys() if k != "body"}, json.loads(row["body"]))
                for row in self.store.db.execute(
                    """SELECT q.*,p.body FROM automatic_review_queue q
                    JOIN proposals p ON p.id=q.proposal AND p.updated=q.updated
                    WHERE (? OR q.lease_until<=?) ORDER BY q.ready,q.proposal""",
                    (include_leased, now),
                )
            ]

    def claim(self, proposal, updated, lease, now, lease_until):
        cursor = self.store.db.execute(
            """UPDATE automatic_review_queue SET lease=?,lease_until=? WHERE proposal=? AND updated=?
            AND lease_until<=? AND retry<=?""",
            (lease, lease_until, proposal, updated, now, now),
        )
        return cursor.rowcount == 1

    def owns(self, proposal, lease):
        with self.store.lock:
            return self.store.db.execute(
                "SELECT 1 FROM automatic_review_queue WHERE proposal=? AND lease=? AND lease_until>?",
                (proposal, lease, time.time()),
            ).fetchone() is not None

    def complete(self, proposal, lease):
        self.store.db.execute("DELETE FROM automatic_review_queue WHERE proposal=? AND lease=?", (proposal, lease))

    def release(self, lease, retry):
        with self.store.transaction():
            self.store.db.execute(
                "UPDATE automatic_review_queue SET lease=NULL,lease_until=0,retry=? WHERE lease=?",
                (retry, lease),
            )

    def fallback(self, proposal, lease):
        with self.store.transaction():
            self.store.db.execute(
                """UPDATE automatic_review_queue SET singleton=1,retry=0,lease=NULL,lease_until=0
                WHERE proposal=? AND lease=?""",
                (proposal, lease),
            )

    def record_dispatch(self, dispatch):
        with self.store.transaction():
            self.store.db.execute(
                "INSERT INTO review_provider_attempts VALUES(?,?,?,?,?,?,?)",
                (
                    dispatch.operation_id, dispatch.scope, dispatch.provider_id, dispatch.model,
                    dispatch.kind, encode(dispatch.proposals), time.time(),
                ),
            )

    def stats(self, *, since=0, until=float("inf")):
        with self.store.lock:
            totals = [
                dict(row)
                for row in self.store.db.execute("""
                    SELECT provider,model,kind,COUNT(*) AS provider_invocations,
                    SUM(json_array_length(items)) AS submitted_items
                    FROM review_provider_attempts WHERE created>=? AND created<?
                    GROUP BY provider,model,kind ORDER BY provider,model,kind
                """, (since, until))
            ]
            logical, automatic = self.store.db.execute("""
                SELECT COUNT(DISTINCT i.value),COUNT(DISTINCT CASE
                    WHEN a.kind IN ('automatic_first','automatic_batch','automatic_item_retry') THEN i.value END)
                FROM review_provider_attempts a, json_each(a.items) i WHERE a.created>=? AND a.created<?
            """, (since, until)).fetchone()
            waiting = self.store.db.execute("SELECT COUNT(*) FROM automatic_review_queue").fetchone()[0]
        return {
            "logical_review_items": logical,
            "logical_automatic_first_reviews": automatic,
            "queued_first_reviews": waiting,
            "provider_invocations": sum(row["provider_invocations"] for row in totals),
            "by_provider_model_and_kind": totals,
        }

    def purge(self, scope):
        with self.store.transaction():
            self.store.db.execute("DELETE FROM automatic_review_queue WHERE scope=?", (scope,))
            self.store.db.execute("DELETE FROM review_provider_attempts WHERE scope=?", (scope,))

    def prune(self, days, *, batch_size=500):
        cutoff = time.time() - max(1, int(days)) * 86400
        self.store._prune_batched("review_provider_attempts", "created<?", (cutoff,), batch_size=batch_size)
