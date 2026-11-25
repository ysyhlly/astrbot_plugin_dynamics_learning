"""Batch 2 regression tests: P0-1 storage bloat, P0-2 indexes/batched prune/throttle, P1-1/2/3/6 integrity."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, encode, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.social import character_key
from astrbot_plugin_dynamics_learning.core.store import (
    DEFAULT_EVENTS_RETENTION_DAYS,
    INCREMENTAL_VACUUM_PAGES,
    Store,
)
from astrbot_plugin_dynamics_learning.vendor.group_relations.store import RelationStore, _SharedConnection
from test_group_review import OTHER, SCOPE, items, make
from test_social import CHARACTER, SCOPE as SOCIAL_SCOPE, legacy


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "batch2.sqlite3")
    yield s
    s.close()


def _live_event(
    *,
    kind="observed",
    scope=SCOPE,
    persona="persona-a",
    member="alice",
    turn_id="turn-1",
    message_id="m1",
    received_at=None,
    persona_summary="人设核心摘要" * 600,
    policy_context="策略完整上下文" * 400,
    decision=None,
    outcome=None,
    reply=None,
    versions=None,
):
    raw = {
        "schema_version": 1,
        "kind": kind,
        "scope": scope,
        "persona": persona,
        "member": member,
        "turn_id": turn_id,
        "message_id": message_id,
        "host_version": "host1",
        "baseline": "base1",
        "baseline_threshold": 65,
        "mandatory": False,
        "text": "今天群里聊得很开心",
        "timestamp": received_at or time.time(),
        "context": {
            "persona_summary": persona_summary,
            "character_scope": CHARACTER,
            "message_ids": [message_id],
        },
        "policy": {
            "threshold": 65,
            "versions": list(versions or []),
            "context": policy_context,
        },
    }
    if decision is not None:
        raw["decision"] = decision
    if outcome is not None:
        raw["outcome"] = outcome
    if reply is not None:
        raw["reply"] = reply
    ev = event_record(raw)
    if received_at is not None:
        ev["received_at"] = float(received_at)
    return ev


# ---------------------------------------------------------------------------
# P0-1: Event body slimming, per-turn coalescing, native_receipts, tiered TTL, auto_vacuum
# ---------------------------------------------------------------------------


def test_p0_1_events_body_slimmed_under_1kb_and_persona_context_preserved(store):
    ev = _live_event(versions=["prop-1"])
    raw_bytes = len(encode(ev).encode("utf-8"))
    assert raw_bytes > 10_000

    store.ingest(ev)
    stored_body = store.db.execute("SELECT body FROM events WHERE id=?", (ev["id"],)).fetchone()[0]
    stored_bytes = len(stored_body.encode("utf-8"))
    assert stored_bytes < 1024, f"expected slimmed event < 1KB, got {stored_bytes} bytes"

    parsed = json.loads(stored_body)
    assert "persona_summary" not in parsed.get("context", {})
    assert "context" not in parsed.get("policy", {})
    assert parsed["policy"]["threshold"] == 65
    assert parsed["policy"]["versions"] == ["prop-1"]

    # Read compatibility: latest_character_context resolves from persona_contexts (same & cross scope)
    same = store.latest_character_context(SCOPE, "persona-a")
    assert same is not None and same["context"]["persona_summary"] == "人设核心摘要" * 600
    cross = store.latest_character_context(OTHER, "persona-a")
    assert cross is not None and cross["context"]["persona_summary"] == "人设核心摘要" * 600


def test_p0_1_per_turn_three_phases_coalesce_to_single_event_row(store):
    now = time.time()
    e_obs = _live_event(kind="observed", turn_id="t1", message_id="m1", received_at=now - 3, versions=["p1"])
    e_dec = _live_event(
        kind="decision",
        turn_id="t1",
        message_id="m1",
        received_at=now - 2,
        decision={"action": "reply", "reason": "relevant"},
    )
    e_out = _live_event(
        kind="outcome",
        turn_id="t1",
        message_id="m1",
        received_at=now - 1,
        outcome={"delivered": True, "final_outcome": "sent"},
        reply={"delivered": True, "message_id": "bot-1", "text": "收到啦"},
    )
    store.ingest(e_obs)
    store.ingest(e_dec)
    store.ingest(e_out)

    # Only 1 row persisted in events for all 3 phases of turn t1
    rows = store.db.execute("SELECT id, body FROM events").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == e_out["id"]
    merged = json.loads(rows[0][1])
    assert merged["kind"] == "outcome"
    assert merged["decision"]["action"] == "reply"
    assert merged["outcome"]["delivered"] is True
    assert merged["reply"]["delivered"] is True
    assert "p1" in merged["policy"]["versions"]

    # latest_personas() still detects observed persona via observed_personas table
    assert store.latest_personas()[SCOPE] == "persona-a"

    # Out-of-order late observed event must not create a 2nd row or clobber outcome
    e_late_obs = _live_event(kind="observed", turn_id="t1", message_id="m1", received_at=now, versions=["p2"])
    store.ingest(e_late_obs)
    rows_after = store.db.execute("SELECT id, body FROM events").fetchall()
    assert len(rows_after) == 1
    assert rows_after[0][0] == e_out["id"]
    merged_after = json.loads(rows_after[0][1])
    assert merged_after["kind"] == "outcome"
    assert "p2" in merged_after["policy"]["versions"]

    # observations() accurately reports decision + delivery from coalesced turn row
    obs = store.observations({"id": "p1", "scope": SCOPE, "persona": "persona-a", "created": now - 10})
    assert obs["after"]["turns"] == 1
    assert obs["after"]["admitted"] == 1
    assert obs["after"]["delivered"] == 1


def test_p0_1_native_receipts_strips_body_and_tiered_ttl_prunes_at_7_days(store):
    now = time.time()
    payload = {
        "schema_version": 1,
        "type": "message.received",
        "event_id": "native-1",
        "scope": CHARACTER,
        "blob": "x" * 4000,
    }
    assert store.native.observe([payload], {SOCIAL_SCOPE}) == 1
    # Duplicate delivery is idempotent; tampered payload still raises
    assert store.native.observe([payload], {SOCIAL_SCOPE}) == 0
    with pytest.raises(ValueError, match="identity changed"):
        store.native.observe([{**payload, "blob": "tampered"}], {SOCIAL_SCOPE})

    # Persisted body is empty string
    stored_body = store.db.execute("SELECT body FROM native_receipts WHERE event_id='native-1'").fetchone()[0]
    assert stored_body == ""

    # Tiered 7-day TTL: 10-day-old processed events & native_receipts pruned even when days=30
    assert DEFAULT_EVENTS_RETENTION_DAYS == 7
    assert INCREMENTAL_VACUUM_PAGES >= 2048

    ten_days_ago = now - 10 * 86400
    ev_old = _live_event(kind="outcome", turn_id="t-old", message_id="m-old", received_at=ten_days_ago)
    ev_new = _live_event(kind="outcome", turn_id="t-new", message_id="m-new", received_at=now - 2 * 86400)
    store.ingest(ev_old)
    store.ingest(ev_new)
    with store.transaction():
        store.db.execute("UPDATE analysis_inbox SET processed=1")
        store.db.execute("UPDATE native_receipts SET created=?, body='legacy-blob' WHERE event_id='native-1'", (ten_days_ago,))
        store.db.execute(
            "INSERT INTO native_receipts VALUES(?,?,?,?,?)",
            ("native-fresh", SCOPE, now - 2 * 86400, "fp-fresh", "legacy-fresh-blob"),
        )
        store.db.execute("INSERT INTO analysis_runs VALUES(?,?,?,?,?)", ("run-10d", SCOPE, "persona-a", ten_days_ago, "{}"))

    store.prune(30)
    remaining_events = {r[0] for r in store.db.execute("SELECT id FROM events").fetchall()}
    assert ev_old["id"] not in remaining_events
    assert ev_new["id"] in remaining_events

    receipts = dict(store.db.execute("SELECT event_id, body FROM native_receipts").fetchall())
    assert "native-1" not in receipts
    assert receipts == {"native-fresh": ""}  # Legacy fresh body stripped to ""

    # 30-day table (analysis_runs) retains 10-day-old row
    assert store.db.execute("SELECT COUNT(*) FROM analysis_runs WHERE id='run-10d'").fetchone()[0] == 1


def test_p0_1_auto_vacuum_none_detection_and_safe_migration(tmp_path):
    legacy_path = tmp_path / "legacy_none.sqlite3"
    conn = sqlite3.connect(legacy_path)
    conn.execute("PRAGMA auto_vacuum=NONE")
    conn.execute("CREATE TABLE legacy_dummy(id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO legacy_dummy VALUES(1)")
    conn.commit()
    assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 0
    conn.close()

    store = Store(legacy_path)
    try:
        status = store.auto_vacuum_status()
        assert status["mode"] == 0
        assert status["incremental"] is False
        assert status["needs_migration"] is True

        prune_res = store.prune(30)
        assert prune_res["auto_vacuum_needs_migration"] is True

        mig = store.enable_incremental_vacuum(vacuum=True)
        assert mig["migrated"] is True
        assert mig["before"] == 0
        assert mig["mode"] == 2
        assert store.auto_vacuum_status()["needs_migration"] is False
        assert store.prune(30)["auto_vacuum_needs_migration"] is False
    finally:
        store.close()


# ---------------------------------------------------------------------------
# P0-2: EXPLAIN QUERY PLAN index verification, batched prune, throttled relation prune
# ---------------------------------------------------------------------------


def _explain(db, sql, params=()):
    return " | ".join(str(row[3]) for row in db.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall())


def test_p0_2_explain_query_plan_uses_indexes_across_all_tables(store):
    rel_store = RelationStore(store.path, shared_store=store)
    assert rel_store is not None

    queries = [
        ("events", "SELECT rowid FROM events WHERE created < ? LIMIT 500", (1000.0,)),
        ("events", "SELECT body FROM events WHERE persona=? ORDER BY created DESC LIMIT 1", ("persona-a",)),
        ("persona_contexts", "SELECT summary FROM persona_contexts WHERE persona=? ORDER BY updated DESC LIMIT 1", ("persona-a",)),
        ("analysis_inbox", "SELECT rowid FROM analysis_inbox WHERE processed=1 AND created < ? LIMIT 500", (1000.0,)),
        ("analysis_runs", "SELECT rowid FROM analysis_runs WHERE created < ? LIMIT 500", (1000.0,)),
        ("profile_raw", "SELECT rowid FROM profile_raw WHERE created < ? LIMIT 500", (1000.0,)),
        ("profile_targets", "SELECT rowid FROM profile_targets WHERE created < ? LIMIT 500", (1000.0,)),
        ("profile_messages", "SELECT rowid FROM profile_messages WHERE created < ? LIMIT 500", (1000.0,)),
        ("native_receipts", "SELECT rowid FROM native_receipts WHERE created < ? LIMIT 500", (1000.0,)),
        ("proposals", "SELECT id FROM proposals WHERE state IN ('rejected','archived','rolled_back','applied') AND updated < ?", (1000.0,)),
        ("audit", "SELECT id FROM audit WHERE proposal=?", ("p1",)),
        ("audit", "SELECT id FROM audit WHERE action='evidence_updated'", ()),
        ("audit", "SELECT rowid FROM audit WHERE created < ? LIMIT 500", (1000.0,)),
        ("reply_usage", "SELECT id FROM reply_usage WHERE created < ?", (1000.0,)),
        ("expression_deliveries", "SELECT rowid FROM expression_deliveries WHERE created < ?", (1000.0,)),
        ("scene_receipts", "SELECT id FROM scene_receipts WHERE created < ?", (1000.0,)),
        ("scene_deliveries", "SELECT rowid FROM scene_deliveries WHERE created < ?", (1000.0,)),
        ("scene_feedback", "SELECT id FROM scene_feedback WHERE created < ?", (1000.0,)),
        ("affection_source_claims", "SELECT rowid FROM affection_source_claims WHERE proposal=?", ("p1",)),
        ("directed_events", "SELECT rowid FROM directed_events WHERE ts < ? LIMIT 500", (1000,)),
        ("relation_daily", "SELECT rowid FROM relation_daily WHERE day < ? LIMIT 500", ("2026-01-01",)),
        ("member_daily", "SELECT rowid FROM member_daily WHERE day < ? LIMIT 500", ("2026-01-01",)),
        ("relation_source_receipts", "SELECT rowid FROM relation_source_receipts WHERE created < ? LIMIT 500", (1000,)),
    ]
    for table, sql, params in queries:
        plan = _explain(store.db, sql, params)
        assert f"SCAN {table}" not in plan, f"Unexpected full table SCAN on {table}: {plan}"
        assert "SEARCH" in plan, f"Expected indexed SEARCH on {table}: {plan}"


def test_p0_2_batched_prune_releases_lock_between_chunks(store):
    old_ts = time.time() - 20 * 86400
    for i in range(12):
        ev = _live_event(kind="outcome", turn_id=f"tb-{i}", message_id=f"mb-{i}", received_at=old_ts + i)
        store.ingest(ev)
    with store.transaction():
        store.db.execute("UPDATE analysis_inbox SET processed=1")

    tx_count = 0
    orig_tx = store.transaction

    def counting_tx():
        nonlocal tx_count
        tx_count += 1
        return orig_tx()

    store.transaction = counting_tx
    store.prune(30, batch_size=3)
    assert store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    # 12 events with batch_size=3 requires at least 4 chunked transactions (plus inbox/other tables)
    assert tx_count >= 8


def test_p0_2_relation_store_throttles_per_message_prune_and_supports_batched_prune(store):
    async def run():
        rel = RelationStore(store.path, shared_store=store)
        rel._history_prune_interval_seconds = 3600
        now = int(time.time())
        old_day = "2020-01-01"
        old_ts = now - 200 * 86400

        # First message triggers initial prune
        await rel.record_message(
            SCOPE, "alice", "Alice", "hello bob", window_seconds=300, max_recent=60,
            pair_weight=0.25, mention_weight=5, same_topic_weight=0.8, directed=[("bob", "reply", 8.0)], source_id="r1",
        )
        # Seed old rows right after initial prune
        with store.transaction():
            for i in range(5):
                store.db.execute(
                    "INSERT INTO relation_daily(group_id,day,a,b,raw_evidence,evidence) VALUES(?,?,?,?,1,1)",
                    (SCOPE, old_day, f"u{i}", f"v{i}"),
                )
                store.db.execute(
                    "INSERT INTO member_daily(group_id,day,user_id,message_count) VALUES(?,?,?,1)",
                    (SCOPE, old_day, f"u{i}"),
                )
                store.db.execute(
                    "INSERT INTO directed_events(group_id,src,dst,kind,weight,ts) VALUES(?,?,?,?,?,?)",
                    (SCOPE, f"u{i}", f"v{i}", "reply", 1.0, old_ts),
                )

        # Second message within throttle window must NOT scan/delete history immediately
        await rel.record_message(
            SCOPE, "bob", "Bob", "hi alice", window_seconds=300, max_recent=60,
            pair_weight=0.25, mention_weight=5, same_topic_weight=0.8, directed=[("alice", "reply", 8.0)], source_id="r2",
        )
        assert store.db.execute("SELECT COUNT(*) FROM relation_daily WHERE day=?", (old_day,)).fetchone()[0] == 5

        # Explicit batched prune_history clears all 5 rows across batch_size=2 chunks
        pruned = await rel.prune_history(batch_size=2)
        assert pruned["relation_daily"] == 5
        assert pruned["member_daily"] == 5
        assert pruned["directed_events"] == 5
        assert store.db.execute("SELECT COUNT(*) FROM relation_daily WHERE day=?", (old_day,)).fetchone()[0] == 0

    asyncio.run(run())


# ---------------------------------------------------------------------------
# P1-1, P1-2, P1-3, P1-6: Data integrity fixes
# ---------------------------------------------------------------------------


def test_p1_1_purge_clears_affection_claims_entry_audits_and_persona_contexts(store):
    legacy(store)
    store.bind_legacy_social(SOCIAL_SCOPE, "fingerprint", character_key(CHARACTER))
    ev = _live_event(scope=SOCIAL_SCOPE, persona="fingerprint", turn_id="t-aff", message_id="aff-m1")
    store.ingest(ev)
    p = store.propose(
        candidate(
            {
                "category": "affection",
                "key": "bot_affection",
                "member": "alice",
                "value": {"experience_delta": 2, "tone": 0.1},
                "reason": "真实互动",
                "evidence_ids": [ev["id"]],
                "days": 30,
            },
            [ev],
            store.entries()[0],
        ),
        "human",
        "test",
    )
    store.apply(p["id"], "human:admin", p["updated"])
    entry_id = p["entry_id"]

    # Seed an entry_id-keyed audit row (like policy_disabled) AND another scope that must survive
    other_ev = _live_event(scope=OTHER, persona="persona-b", turn_id="t-other", message_id="m-other")
    store.ingest(other_ev)
    with store.transaction():
        store._audit({"id": entry_id}, "policy_disabled", "system:fixed_personality")
        store.db.execute(
            "INSERT INTO affection_source_claims VALUES(?,?,?,?,?)",
            (OTHER, "persona-b", "bob", "m-other", "other-prop"),
        )

    assert store.db.execute("SELECT COUNT(*) FROM affection_source_claims WHERE scope=?", (SOCIAL_SCOPE,)).fetchone()[0] == 1
    assert store.db.execute("SELECT COUNT(*) FROM audit WHERE proposal=?", (entry_id,)).fetchone()[0] == 1

    store.purge(SOCIAL_SCOPE)

    assert store.db.execute("SELECT COUNT(*) FROM affection_source_claims WHERE scope=?", (SOCIAL_SCOPE,)).fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM audit WHERE proposal=?", (entry_id,)).fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM persona_contexts WHERE scope=?", (SOCIAL_SCOPE,)).fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM observed_personas WHERE scope=?", (SOCIAL_SCOPE,)).fetchone()[0] == 0
    # Other scope untouched
    assert store.db.execute("SELECT COUNT(*) FROM affection_source_claims WHERE scope=?", (OTHER,)).fetchone()[0] == 1
    assert store.db.execute("SELECT COUNT(*) FROM persona_contexts WHERE scope=?", (OTHER,)).fetchone()[0] == 1


def test_p1_2_rollback_skips_terminal_and_pruned_predecessors(store):
    e1 = _live_event(turn_id="rb1", message_id="rb-m1")
    e2 = _live_event(turn_id="rb2", message_id="rb-m2")
    e3 = _live_event(turn_id="rb3", message_id="rb-m3")

    # Case A: P1 applied -> P1 rolled_back -> P2 applied (before points to inactive P1) -> P2 rolled_back
    p1 = store.propose(
        candidate(
            {"category": "style", "key": "k1", "member": "alice", "value": "风格一", "reason": "r1", "evidence_ids": [e1["id"]], "days": 30},
            [e1],
            None,
        ),
        "human",
        "test",
    )
    p1 = store.apply(p1["id"], "human:admin", p1["updated"])
    p1 = store.rollback(p1["id"], "human:admin")
    p1_updated_after_rb = p1["updated"]
    p1_rev_after_rb = p1["applied_revision"]

    p2 = store.propose(
        candidate(
            {"category": "style", "key": "k1", "member": "alice", "value": "风格二", "reason": "r2", "evidence_ids": [e2["id"]], "days": 30},
            [e2],
            store.entries()[0],
        ),
        "human",
        "test",
    )
    p2 = store.apply(p2["id"], "human:admin", p2["updated"])
    p2 = store.rollback(p2["id"], "human:admin")
    assert p2["state"] == "rolled_back"
    # Terminal P1 must NOT have been mutated!
    p1_reloaded = store.get(p1["id"])
    assert p1_reloaded["state"] == "rolled_back"
    assert p1_reloaded["updated"] == p1_updated_after_rb
    assert p1_reloaded["applied_revision"] == p1_rev_after_rb

    # Case B: Active predecessor pruned (>32 hops) -> rollback must NOT raise ValueError
    p3 = store.propose(
        candidate(
            {"category": "style", "key": "k2", "member": "alice", "value": "第一版", "reason": "r3", "evidence_ids": [e1["id"]], "days": 30},
            [e1],
            None,
        ),
        "human",
        "test",
    )
    p3 = store.apply(p3["id"], "human:admin", p3["updated"])
    entry_k2 = next(e for e in store.entries() if e["key"] == "k2")
    p4 = store.propose(
        candidate(
            {"category": "style", "key": "k2", "member": "alice", "value": "第二版", "reason": "r4", "evidence_ids": [e3["id"]], "days": 30},
            [e3],
            entry_k2,
        ),
        "human",
        "test",
    )
    p4 = store.apply(p4["id"], "human:admin", p4["updated"])
    # Simulate >32 hop prune deleting predecessor P3 from proposals
    with store.transaction():
        store.db.execute("DELETE FROM proposals WHERE id=?", (p3["id"],))
    p4_rb = store.rollback(p4["id"], "human:admin")
    assert p4_rb["state"] == "rolled_back"
    restored_k2 = next(e for e in store.entries() if e["key"] == "k2")
    assert restored_k2["value"] == "第一版" and restored_k2["active"] is True


def test_p1_3_batch_actions_shield_commit_and_mark_cache_refresh_pending(store):
    async def run():
        p1 = make(store, "ba1", approved=True)
        p2 = make(store, "ba2", approved=True)
        p3 = make(store, "ba3", approved=True)
        engine = Engine(store, {"sessions": [SCOPE]}, None)
        await engine.refresh()

        # 1) group_batch_action (partial=False): cancelled during refresh still completes commit + refresh
        entered = asyncio.Event()
        release = asyncio.Event()
        orig_refresh = engine.refresh

        async def slow_refresh():
            entered.set()
            await release.wait()
            await orig_refresh()

        engine.refresh = slow_refresh
        task = asyncio.create_task(engine.group_batch_action(SCOPE, items(p1), "human:admin", partial=False))
        await entered.wait()
        assert engine.cache_refresh_pending is True
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Because finish_operation shielded commit+refresh, cache is up to date!
        assert engine.cache_refresh_pending is False
        assert any(e["id"] == p1["entry_id"] and e["active"] for e in engine.cache)

        # 2) group_batch_action (partial=True) and batch_action set cache_refresh_pending if refresh fails
        async def failing_refresh():
            raise RuntimeError("refresh failed")

        engine.refresh = failing_refresh
        with pytest.raises(RuntimeError, match="refresh failed"):
            await engine.group_batch_action(SCOPE, items(p2), "human:admin", partial=True)
        assert engine.cache_refresh_pending is True

        engine.cache_refresh_pending = False
        with pytest.raises(RuntimeError, match="refresh failed"):
            await engine.batch_action(SCOPE, p3["persona"], p3["member"], items(p3), "human:admin")
        assert engine.cache_refresh_pending is True

    asyncio.run(run())


def test_p1_6_shared_connection_reentrancy_preserves_outer_transaction(store):
    rel = RelationStore(store.path, shared_store=store)
    assert rel is not None

    # Inner _SharedConnection exiting normally must NOT prematurely commit outer Store.transaction()
    with pytest.raises(RuntimeError, match="outer abort"):
        with store.transaction():
            store.db.execute("INSERT INTO checkpoints VALUES('s1','p1',1.0)")
            with _SharedConnection(store) as conn:
                conn.execute("INSERT INTO checkpoints VALUES('s2','p2',2.0)")
            # Outer transaction must still be open so raising rolls back BOTH writes!
            assert store.db.in_transaction is True
            raise RuntimeError("outer abort")

    assert store.db.execute("SELECT COUNT(*) FROM checkpoints WHERE scope IN ('s1','s2')").fetchone()[0] == 0

    # Nested _SharedConnection inside _SharedConnection also respects outer rollback
    with pytest.raises(RuntimeError, match="nested abort"):
        with _SharedConnection(store) as outer:
            outer.execute("INSERT INTO checkpoints VALUES('s3','p3',3.0)")
            with _SharedConnection(store) as inner:
                inner.execute("INSERT INTO checkpoints VALUES('s4','p4',4.0)")
            assert outer.in_transaction is True
            raise RuntimeError("nested abort")

    assert store.db.execute("SELECT COUNT(*) FROM checkpoints WHERE scope IN ('s3','s4')").fetchone()[0] == 0
