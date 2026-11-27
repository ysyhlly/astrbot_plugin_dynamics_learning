"""Focused unit tests for 2GB RAM performance & memory optimizations in dynamics_learning."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

import pytest

from astrbot_plugin_dynamics_learning.core.automatic_reviews import AutomaticReviews
from astrbot_plugin_dynamics_learning.core.contract import encode, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.review_queue import query_queue
from astrbot_plugin_dynamics_learning.core.social import Social
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.vendor.group_relations.store import RelationStore

SCOPE = "aiocqhttp:GroupMessage:8801"


def _ev(idx: int, *, scope: str = SCOPE, persona: str = "安安", created: float | None = None):
    now = time.time() if created is None else float(created)
    row = event_record(
        {
            "schema_version": 1,
            "kind": "observed",
            "scope": scope,
            "persona": persona,
            "turn_id": f"turn-{scope}-{idx}",
            "message_id": f"msg-{scope}-{idx}",
            "member": f"u{idx % 5}",
            "text": f"hello world {idx}",
            "timestamp": now,
            "baseline": "b1",
            "baseline_threshold": 50.0,
            "backend": "llm",
            "mandatory": False,
            "host_version": "1.0.0",
            "context": {},
        }
    )
    row["received_at"] = now
    return row


def test_relation_store_pair_latencies_respects_window_cutoff_and_pragmas(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ASTRBOT_RESOURCE_PROFILE", "low_memory")
    db_path = tmp_path / "rel.sqlite3"
    rel = RelationStore(db_path, analysis_window_days=7)

    now = int(time.time())
    old_ts = now - 30 * 86400  # Outside 7-day analysis window
    recent_ts = now - 100

    with rel._connect() as conn:
        # Verify SQLite PRAGMAs applied in _connect()
        assert conn.execute("PRAGMA cache_size").fetchone()[0] == -1024
        assert conn.execute("PRAGMA temp_store").fetchone()[0] == 1
        assert conn.execute("PRAGMA mmap_size").fetchone()[0] == 0

        # Old pair outside window (dt=10s) must NOT appear in latencies
        conn.execute(
            "INSERT INTO directed_events(group_id, src, dst, kind, weight, ts) VALUES (?,?,?,?,?,?)",
            (SCOPE, "oldA", "oldB", "reply", 8.0, old_ts),
        )
        conn.execute(
            "INSERT INTO directed_events(group_id, src, dst, kind, weight, ts) VALUES (?,?,?,?,?,?)",
            (SCOPE, "oldB", "oldA", "reply", 8.0, old_ts + 10),
        )
        # Recent pair inside window (dt=25s) MUST appear
        conn.execute(
            "INSERT INTO directed_events(group_id, src, dst, kind, weight, ts) VALUES (?,?,?,?,?,?)",
            (SCOPE, "newA", "newB", "reply", 8.0, recent_ts),
        )
        conn.execute(
            "INSERT INTO directed_events(group_id, src, dst, kind, weight, ts) VALUES (?,?,?,?,?,?)",
            (SCOPE, "newB", "newA", "reply", 8.0, recent_ts + 25),
        )

        lat = rel._pair_latencies_map(conn, SCOPE, 300, now_ts=now)
        assert ("oldA", "oldB") not in lat
        assert lat[("newA", "newB")] == pytest.approx(25.0)

    # Verify prune_history defaults directed_event_retain_days from profile (30d for low_memory, 14d for aggressive)
    monkeypatch.setenv("ASTRBOT_RESOURCE_PROFILE", "low_memory_aggressive")
    with rel._connect() as conn:
        conn.execute(
            "INSERT INTO directed_events(group_id, src, dst, kind, weight, ts) VALUES (?,?,?,?,?,?)",
            (SCOPE, "midA", "midB", "reply", 8.0, now - 20 * 86400),
        )
    pruned = asyncio.run(rel.prune_history(now_ts=now))
    assert pruned["directed_events"] >= 3  # old (30d) + mid (20d) both > 14d
    with rel._connect() as conn:
        rem = conn.execute("SELECT count(*) FROM directed_events").fetchone()[0]
    assert rem == 2  # recent pair kept


def test_store_pragmas_entries_sql_opt_latest_personas_and_profile_prune(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ASTRBOT_RESOURCE_PROFILE", "low_memory_aggressive")
    store = Store(tmp_path / "learning.sqlite3")
    try:
        # PRAGMAs applied for low_memory_aggressive: cache -512, temp_store=1, mmap=0, journal_limit=4MiB
        assert store.db.execute("PRAGMA cache_size").fetchone()[0] == -512
        assert store.db.execute("PRAGMA temp_store").fetchone()[0] == 1
        assert store.db.execute("PRAGMA mmap_size").fetchone()[0] == 0
        assert store.db.execute("PRAGMA journal_size_limit").fetchone()[0] == 4 * 1024 * 1024

        # Verify readonly_connection & review_queue work cleanly with PRAGMAs
        with store.readonly_connection() as ro:
            assert ro.execute("PRAGMA cache_size").fetchone()[0] == -256
            assert ro.execute("PRAGMA temp_store").fetchone()[0] == 1
            assert ro.execute("PRAGMA mmap_size").fetchone()[0] == 0

        qres = query_queue(store, [SCOPE], {"jargon": "human"}, {}, {}, [])
        assert qres["total"] == 0

        # entries() skips p.body when source_uids already present on entry
        now = time.time()
        with store.transaction():
            store.db.execute(
                "INSERT INTO proposals VALUES(?,?,?,?,?,?,?)",
                (
                    "p-fast",
                    SCOPE,
                    "安安",
                    "applied",
                    now,
                    now,
                    encode({"id": "p-fast", "scope": SCOPE, "evidence": [{"member": "u999", "scope": SCOPE}]}),
                ),
            )
            entry_with_uids = {
                "id": "e-fast",
                "scope": SCOPE,
                "persona": "安安",
                "member": "",
                "category": "jargon",
                "key": "kk",
                "value": "meaning",
                "revision": 1,
                "proposal_id": "p-fast",
                "expires": now + 86400,
                "active": True,
                "source_uids": ["u1"],
                "provenance_complete": True,
            }
            store.db.execute("INSERT INTO entries VALUES(?,?)", ("e-fast", encode(entry_with_uids)))
        row_check = store.db.execute(
            """SELECT CASE WHEN json_extract(e.body,'$.source_uids') IS NULL THEN p.body ELSE NULL END
            FROM entries e LEFT JOIN proposals p ON p.id=json_extract(e.body,'$.proposal_id')
            WHERE e.id='e-fast'"""
        ).fetchone()
        assert row_check[0] is None
        loaded = store.entries_for_scope(SCOPE)
        assert len(loaded) == 1
        assert loaded[0]["source_uids"] == ["u1"]

        # latest_personas() skips scanning events once observed_personas_backfilled_v1 is populated
        store.ingest(_ev(1, persona="安安", created=now - 10))
        assert store.latest_personas()[SCOPE] == "安安"
        # Inject raw event bypassing ingest; since backfilled marker is set, latest_personas avoids scanning events
        with store.transaction():
            store.db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?)",
                ("raw-bypass", SCOPE, "bypassed_persona", now + 100, encode({"kind": "observed", "scope": SCOPE, "persona": "bypassed_persona"})),
            )
        assert store.latest_personas()[SCOPE] == "安安"

        # Profile-aware prune: low_memory_aggressive uses 2d events, 7d profiles, 14d audit, 3d terminal proposal compaction
        old_3d = now - 3.5 * 86400
        old_15d = now - 15 * 86400
        with store.transaction():
            store.db.execute(
                "INSERT INTO events VALUES(?,?,?,?,?)",
                ("ev-old-3d", SCOPE, "安安", old_3d, encode({"id": "ev-old-3d", "kind": "observed", "scope": SCOPE, "persona": "安安"})),
            )
            store.db.execute(
                "INSERT INTO audit(proposal,action,actor,created,body) VALUES(?,?,?,?,?)",
                ("p-old", "review", "llm:test", old_15d, "{}"),
            )
            rejected_prop = {
                "id": "p-rej-4d",
                "scope": SCOPE,
                "persona": "安安",
                "state": "rejected",
                "created": old_3d,
                "updated": old_3d,
                "evidence": [{"id": "e1", "message_id": "m1", "member": "u1", "scope": SCOPE, "text": "HUGE" * 200}],
            }
            store.db.execute(
                "INSERT INTO proposals VALUES(?,?,?,?,?,?,?)",
                ("p-rej-4d", SCOPE, "安安", "rejected", old_3d, old_3d, encode(rejected_prop)),
            )
            # Active affection entry must NEVER be pruned
            aff_entry = {
                "id": "e-aff",
                "scope": SCOPE,
                "persona": "安安",
                "member": "u1",
                "category": "affection",
                "key": "bot_affection",
                "value": {"affection_level": 88.0},
                "revision": 1,
                "proposal_id": "p-aff",
                "expires": now + 365 * 86400,
                "active": True,
                "source_uids": ["u1"],
            }
            store.db.execute("INSERT INTO entries VALUES(?,?)", ("e-aff", encode(aff_entry)))
            # Seed console / scene / expression / auto-review old rows for batched prune check
            store.db.execute("INSERT INTO reply_usage VALUES('ru1',?,?,?,?,'{}')", (SCOPE, "u1", "t1", old_15d))
            store.db.execute("INSERT INTO scene_receipts VALUES('sr1',?,?,'t1','provider',?,'{}')", (SCOPE, "安安", old_15d))
            store.db.execute("INSERT INTO expression_deliveries VALUES(?,?,'m1',?,'[]')", (SCOPE, "安安", old_15d))
            store.db.execute("INSERT INTO review_provider_attempts VALUES('rpa1',?,'p','m','k','[]',?)", (SCOPE, old_15d))

        res = store.prune(30)
        assert res["audit_pruned"] >= 1
        assert res["proposals_compacted"] >= 1
        assert store.db.execute("SELECT 1 FROM events WHERE id='ev-old-3d'").fetchone() is None
        assert store.db.execute("SELECT 1 FROM reply_usage WHERE id='ru1'").fetchone() is None
        assert store.db.execute("SELECT 1 FROM scene_receipts WHERE id='sr1'").fetchone() is None
        assert store.db.execute("SELECT 1 FROM expression_deliveries WHERE message_id='m1'").fetchone() is None
        assert store.db.execute("SELECT 1 FROM review_provider_attempts WHERE id='rpa1'").fetchone() is None
        # Terminal rejected proposal evidence compacted after 3d
        compacted_row = json.loads(store.db.execute("SELECT body FROM proposals WHERE id='p-rej-4d'").fetchone()[0])
        assert compacted_row["evidence"][0].get("trimmed") is True
        # Active entries (including affection) untouched
        assert store.db.execute("SELECT 1 FROM entries WHERE id='e-aff'").fetchone() is not None
        assert store.db.execute("SELECT 1 FROM entries WHERE id='e-fast'").fetchone() is not None
        # ExpressionStore sample limit respects profile
        assert store.expressions._sample_limit() == 1000
    finally:
        store.close()


def test_social_lru_groups_profile_limits_and_batched_consume(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ASTRBOT_RESOURCE_PROFILE", "low_memory_aggressive")
    store = Store(tmp_path / "social.sqlite3")
    scopes = [f"aiocqhttp:GroupMessage:{9000 + i}" for i in range(6)]

    async def _noop_model(*_a, **_kw):
        return {}

    engine = Engine(store, {"sessions": scopes, "group_relations_enabled": True}, _noop_model)
    social = Social(engine)
    try:
        assert isinstance(social.groups, OrderedDict)
        assert social._max_groups() == 4
        assert social._top_limit() == 200
        assert social._refresh_interval_seconds() == 600.0
        assert social._effective_max_recent() == 25

        # Refresh 6 groups: LRU cap = 4 evicts oldest 2
        async def _test_lru():
            for sc in scopes:
                await social.refresh_group(sc, force=True)
            assert len(social.groups) == 4
            assert list(social.groups.keys()) == scopes[2:]
            # Access scopes[2] to make it MRU, then refresh scopes[0]: scopes[3] should be evicted
            _ = social._get_group(scopes[2])
            await social.refresh_group(scopes[0], force=True)
            assert len(social.groups) == 4
            assert scopes[2] in social.groups
            assert scopes[3] not in social.groups

        asyncio.run(_test_lru())

        # Batch consume drains burst into a single record_messages_batch call
        batch_sizes = []
        orig_batch = social.graph.record_messages_batch

        async def _spy_batch(items):
            batch_sizes.append(len(items))
            return await orig_batch(items)

        monkeypatch.setattr(social.graph, "record_messages_batch", _spy_batch)

        async def _test_batch_consume():
            for i in range(10):
                assert social.observe(
                    {
                        "scope": scopes[0],
                        "member": f"u{i % 3}",
                        "name": f"User{i % 3}",
                        "text": f"hello burst {i}",
                        "directed": [],
                        "source_id": f"src-{i}",
                        "bot_id": "bot",
                    }
                )
            consumer = asyncio.create_task(social.consume())
            await asyncio.wait_for(social.queue.join(), timeout=5.0)
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)

        asyncio.run(_test_batch_consume())
        assert sum(batch_sizes) == 10
        assert len(batch_sizes) == 1  # All 10 burst messages written in 1 SQLite tx!
    finally:
        store.close()


def test_engine_cache_sweeps_sequential_analysis_and_auto_vacuum_migration(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ASTRBOT_RESOURCE_PROFILE", "low_memory")

    # Create a legacy auto_vacuum=0 DB with freelist pages
    db_path = tmp_path / "legacy-learning.sqlite3"
    raw = sqlite3.connect(db_path)
    raw.execute("PRAGMA auto_vacuum=NONE")
    raw.execute("CREATE TABLE legacy_junk (id INTEGER PRIMARY KEY, data TEXT)")
    raw.executemany("INSERT INTO legacy_junk(data) VALUES (?)", [("z" * 4000,) for _ in range(200)])
    raw.commit()
    raw.execute("DELETE FROM legacy_junk")
    raw.commit()
    raw.close()

    store = Store(db_path)
    try:
        assert store.needs_auto_vacuum_migration is True
        assert store.can_run_auto_vacuum_migration(min_freelist_pages=10) is True

        async def _noop_model(*_a, **_kw):
            return {}

        scopes = [f"aiocqhttp:GroupMessage:{7000 + i}" for i in range(3)]
        engine = Engine(store, {"sessions": scopes, "batch_events": 1}, _noop_model)

        # Test sweep_runtime_caches & _character_context_cache LRU cap (64)
        now = time.time()
        engine.review_retry_at["expired"] = now - 10
        engine.review_retry_at["active"] = now + 100
        engine.scope_vibes["old_vibe"] = {"expires_at": now - 10}
        engine.scope_vibes["new_vibe"] = {"expires_at": now + 100}
        engine.active_games["old_game"] = {"expires_at": now - 10}
        engine.active_games["new_game"] = {"expires_at": now + 100}
        for i in range(80):
            engine._set_character_context_cache((f"s:{i}", "p"), {"persona_summary": f"sum-{i}"})
        assert len(engine._character_context_cache) == 64

        swept = engine.sweep_runtime_caches(now=now)
        assert swept["review_retry_at"] == 1
        assert swept["scope_vibes"] == 1
        assert swept["active_games"] == 1
        assert "expired" not in engine.review_retry_at and "active" in engine.review_retry_at
        assert "old_vibe" not in engine.scope_vibes and "new_vibe" in engine.scope_vibes
        assert "old_game" not in engine.active_games and "new_game" in engine.active_games

        # Test sequential analysis_cycle in low_memory profile (max concurrency == 1)
        for idx, sc in enumerate(scopes):
            store.ingest(_ev(idx, scope=sc, persona="安安", created=now - 1000))

        active_now = 0
        max_concurrent = 0

        async def _fake_analyze(scope, persona):
            nonlocal active_now, max_concurrent
            active_now += 1
            max_concurrent = max(max_concurrent, active_now)
            await asyncio.sleep(0.02)
            active_now -= 1
            store.checkpoint(scope, persona, now)

        monkeypatch.setattr(engine, "analyze", _fake_analyze)
        asyncio.run(engine.analysis_cycle())
        assert max_concurrent == 1

        # Test AutomaticReviews disabled sleep delay == 60s
        recorded_timeouts = []
        ar = AutomaticReviews(engine)
        engine.config["automatic_review_batch_enabled"] = False
        orig_wait_for = asyncio.wait_for

        async def _spy_wait_for(coro, timeout=None):
            recorded_timeouts.append(timeout)
            engine.closed = True
            if hasattr(coro, "close"):
                coro.close()
            raise TimeoutError

        monkeypatch.setattr(asyncio, "wait_for", _spy_wait_for)
        asyncio.run(ar.run())
        monkeypatch.setattr(asyncio, "wait_for", orig_wait_for)
        assert recorded_timeouts == [60]

        # Test maintenance() wires enable_incremental_vacuum migration
        engine.closed = False
        monkeypatch.setattr(
            store,
            "can_run_auto_vacuum_migration",
            lambda: store.auto_vacuum_status()["needs_migration"],
        )

        async def _run_one_maintenance_iter():
            async def _stop_after_vacuum():
                engine.closed = True
                raise asyncio.CancelledError

            monkeypatch.setattr(engine, "apply_pending_affection", _stop_after_vacuum)
            with pytest.raises(asyncio.CancelledError):
                await engine.maintenance()

        asyncio.run(_run_one_maintenance_iter())
        assert store.needs_auto_vacuum_migration is False
        assert store.auto_vacuum_status()["mode"] == 2
    finally:
        store.close()
