"""Batch 3 regression tests: P1-4 optimistic concurrency, P1-5 provider backoff/telemetry, P1-6 deblocking, P2-1/2/3."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_dynamics_learning.core import automatic_reviews as ar_mod
from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.model_projection import project_current_entry_for_model
from astrbot_plugin_dynamics_learning.core.review_batches import ReviewDeferred
from astrbot_plugin_dynamics_learning.core.scenes import sanitize
from astrbot_plugin_dynamics_learning.core.social import Social
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.vendor.group_relations.store import RelationStore
from test_group_review import OTHER, SCOPE
from test_security_p0_3 import _load_plugin_class_offline


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "batch3.sqlite3")
    yield s
    s.close()


def _ev(mid, *, member="alice", scope=SCOPE, persona="persona-a", text="今天聊得很开心，确实如此"):
    return event_record(
        {
            "schema_version": 1,
            "kind": "observed",
            "scope": scope,
            "persona": persona,
            "member": member,
            "turn_id": f"t:{mid}",
            "message_id": mid,
            "host_version": "host1",
            "baseline": "base1",
            "baseline_threshold": 65,
            "mandatory": False,
            "text": text,
            "timestamp": time.time(),
            "context": {"source_verified": True},
        }
    )


# ---------------------------------------------------------------------------
# P1-4: Per-entry optimistic concurrency in Engine.analyze()
# ---------------------------------------------------------------------------


def test_p1_4_analyze_survives_unrelated_entry_change_and_drops_only_conflicting_candidate(store):
    async def run():
        e_alice = _ev("m-alice", member="alice", text="我说话比较直接")
        e_bob = _ev("m-bob", member="bob", text="我喜欢简短回复")
        e_charlie = _ev("m-charlie", member="charlie", text="我平时喜欢用表情")
        store.ingest(e_alice)
        store.ingest(e_bob)

        entered = asyncio.Event()
        release = asyncio.Event()

        async def model(role, instruction, payload):
            entered.set()
            await release.wait()
            return {
                "proposals": [
                    {
                        "category": "style",
                        "key": "directness",
                        "member": "alice",
                        "value": "表达直接干脆",
                        "reason": "真实对话",
                        "evidence_ids": [e_alice["id"]],
                        "days": 30,
                    },
                    {
                        "category": "style",
                        "key": "brevity",
                        "member": "bob",
                        "value": "偏好简短回复",
                        "reason": "真实对话",
                        "evidence_ids": [e_bob["id"]],
                        "days": 30,
                    },
                ]
            }

        engine = Engine(store, {"sessions": [SCOPE], "style_review": "human", "analysis_provider": "ap"}, model)
        await engine.refresh()
        engine.current_personas[SCOPE] = "persona-a"

        # Part 1: Unrelated member (charlie) changes while analyze() is waiting on LLM -> ZERO proposals wasted!
        task = asyncio.create_task(engine.analyze(SCOPE, "persona-a"))
        await entered.wait()
        p_charlie = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "emoji",
                    "member": "charlie",
                    "value": "常用表情",
                    "reason": "并发自动审核落地",
                    "evidence_ids": [e_charlie["id"]],
                    "days": 30,
                },
                [e_charlie],
                None,
            ),
            "human",
            "test",
        )
        store.apply(p_charlie["id"], "human:admin", p_charlie["updated"])
        await engine.refresh(scope=SCOPE)
        release.set()

        res = await task
        assert res["proposals"] == 2
        assert engine.stale_results == 0

        # Part 2: Same target entry for alice changes mid-flight -> ONLY alice dropped, bob still commits!
        e_alice2 = _ev("m-alice2", member="alice", text="依然直接")
        e_bob2 = _ev("m-bob2", member="bob", text="依然简短")
        store.ingest(e_alice2)
        store.ingest(e_bob2)
        entered.clear()
        release.clear()

        async def model2(role, instruction, payload):
            entered.set()
            await release.wait()
            return {
                "proposals": [
                    {
                        "category": "style",
                        "key": "directness_v2",
                        "member": "alice",
                        "value": "直接风格v2",
                        "reason": "真实对话",
                        "evidence_ids": [e_alice2["id"]],
                        "days": 30,
                    },
                    {
                        "category": "style",
                        "key": "brevity_v2",
                        "member": "bob",
                        "value": "简短风格v2",
                        "reason": "真实对话",
                        "evidence_ids": [e_bob2["id"]],
                        "days": 30,
                    },
                ]
            }

        engine.call_model = model2
        task2 = asyncio.create_task(engine.analyze(SCOPE, "persona-a"))
        await entered.wait()
        # Concurrently apply alice's exact key directness_v2
        p_conflict = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "directness_v2",
                    "member": "alice",
                    "value": "并发写入版本",
                    "reason": "并发写入",
                    "evidence_ids": [e_alice2["id"]],
                    "days": 30,
                },
                [e_alice2],
                None,
            ),
            "human",
            "test",
        )
        store.apply(p_conflict["id"], "human:admin", p_conflict["updated"])
        await engine.refresh(scope=SCOPE)
        release.set()

        res2 = await task2
        assert res2["proposals"] == 1
        assert res2["invalid_candidates"] == 1
        assert engine.stale_results == 1
        latest_keys = {p["key"] for p in store.proposals() if p["state"] == "pending"}
        assert "brevity_v2" in latest_keys

    asyncio.run(run())


# ---------------------------------------------------------------------------
# P1-5: Per-provider backoff isolation & series_llm_call model/group_id telemetry
# ---------------------------------------------------------------------------


def test_p1_5_backoff_isolated_by_provider_id_and_shared_when_same_provider(store):
    engine = Engine(store, {"sessions": [SCOPE], "analysis_provider": "prov-a", "review_provider": "prov-r"}, None)
    ar = engine.automatic_reviews
    t0 = 10_000.0

    # Backoff on review_provider must NOT pollute analysis_provider
    ar.note_budget_busy(ReviewDeferred(retry_after=30.0), now=t0, role="review_provider")
    assert ar.is_budget_backed_off(now=t0 + 1, role="review_provider") is True
    assert ar.is_budget_backed_off(now=t0 + 1, role="analysis_provider") is False

    # Backoff on analysis_provider is independent and resets independently
    ar.note_budget_busy(ReviewDeferred(retry_after=45.0), now=t0, role="analysis_provider")
    assert ar.is_budget_backed_off(now=t0 + 1, role="analysis_provider") is True
    ar.reset_budget_backoff(role="analysis_provider")
    assert ar.is_budget_backed_off(now=t0 + 1, role="analysis_provider") is False
    assert ar.is_budget_backed_off(now=t0 + 1, role="review_provider") is True

    # Shared provider_id shares the same bucket
    shared_engine = Engine(store, {"sessions": [SCOPE], "analysis_provider": "same", "review_provider": "same"}, None)
    shared_ar = shared_engine.automatic_reviews
    shared_ar.note_budget_busy(ReviewDeferred(retry_after=20.0), now=t0, role="analysis_provider")
    assert shared_ar.is_budget_backed_off(now=t0 + 1, role="review_provider") is True


def test_p1_5_main_model_passes_model_and_group_id_to_series_llm_call(store, monkeypatch):
    main_mod = _load_plugin_class_offline(monkeypatch)
    seen = {}

    async def fake_series_llm_call(ctx, provider_id, plugin_id, purpose, invoke, **kwargs):
        seen["provider_id"] = provider_id
        seen["plugin_id"] = plugin_id
        seen["purpose"] = purpose
        seen["kwargs"] = kwargs
        return await invoke()

    monkeypatch.setattr(main_mod, "series_llm_call", fake_series_llm_call)

    async def fake_text_chat(**_kwargs):
        return SimpleNamespace(completion_text='{"proposals":[]}')

    provider = SimpleNamespace(
        provider_config={"id": "ap", "model": "gemini-flash-test"},
        meta=lambda: SimpleNamespace(model="gemini-flash-test"),
        text_chat=fake_text_chat,
        is_native_decision_provider=False,
    )
    ctx = SimpleNamespace(get_provider_by_id=lambda pid: provider if pid == "ap" else None)
    plugin = main_mod.DynamicsLearning(ctx, {"sessions": [SCOPE], "analysis_provider": "ap"})
    plugin.engine = Engine(store, plugin.config, plugin._model, host_context=ctx)

    async def run():
        out = await plugin._model("analysis_provider", "sys", {"scope": SCOPE, "events": [{"scope": SCOPE}]})
        assert out == {"proposals": []}
        assert seen["provider_id"] == "ap"
        assert seen["purpose"] == "analysis"
        assert seen["kwargs"]["model"] == "gemini-flash-test"
        assert seen["kwargs"]["group_id"] == SCOPE

    asyncio.run(run())


# ---------------------------------------------------------------------------
# P1-6 & Event-loop deblocking
# ---------------------------------------------------------------------------


def test_p1_6_incremental_refresh_and_scoped_category_pending_query(store):
    async def run():
        e1 = _ev("inc-1", scope=SCOPE)
        e2 = _ev("inc-2", scope=OTHER)
        p1 = store.propose(
            candidate(
                {"category": "style", "key": "k1", "member": "alice", "value": "A", "reason": "r", "evidence_ids": [e1["id"]], "days": 30},
                [e1],
                None,
            ),
            "human",
            "test",
        )
        p2 = store.propose(
            candidate(
                {"category": "style", "key": "k2", "member": "alice", "value": "B", "reason": "r", "evidence_ids": [e2["id"]], "days": 30},
                [e2],
                None,
            ),
            "human",
            "test",
        )
        store.apply(p1["id"], "human:admin", p1["updated"])
        store.apply(p2["id"], "human:admin", p2["updated"])

        engine = Engine(store, {"sessions": [SCOPE, OTHER], "style_review": "human"}, None)
        await engine.refresh()
        assert len(engine.cache) == 2

        full_calls = 0
        scoped_calls = 0
        orig_entries = store.entries
        orig_scoped = store.entries_for_scope

        def count_full():
            nonlocal full_calls
            full_calls += 1
            return orig_entries()

        def count_scoped(sc):
            nonlocal scoped_calls
            scoped_calls += 1
            return orig_scoped(sc)

        store.entries = count_full
        store.entries_for_scope = count_scoped

        # Incremental refresh for SCOPE must NOT trigger full-DB store.entries()
        p3 = store.propose(
            candidate(
                {"category": "style", "key": "k3", "member": "alice", "value": "C", "reason": "r", "evidence_ids": [e1["id"]], "days": 30},
                [e1],
                None,
            ),
            "human",
            "test",
        )
        await engine.action(p3["id"], "approve", "human:admin", p3["updated"])
        assert full_calls == 0
        assert scoped_calls == 1
        assert len(engine.cache) == 3
        assert len(engine._entries_for_scope(OTHER)) == 1

        # pending_proposals_by_categories returns only matching unclaimed pending proposals
        p_pend = store.propose(
            candidate(
                {"category": "style", "key": "k4", "member": "alice", "value": "D", "reason": "r", "evidence_ids": [e1["id"]], "days": 30},
                [e1],
                None,
            ),
            "human",
            "test",
        )
        found = store.pending_proposals_by_categories(SCOPE, "persona-a", {"style"}, 16)
        assert [p["id"] for p in found] == [p_pend["id"]]
        assert store.pending_proposals_by_categories(SCOPE, "persona-a", {"jargon"}, 16) == []

    asyncio.run(run())


def test_p1_6_sync_cross_plugin_hooks_do_not_block_on_store_lock(store, monkeypatch):
    main_mod = _load_plugin_class_offline(monkeypatch)
    e1 = _ev("hook-1", scope=SCOPE, text="赛博浇花很有意思")
    store.ingest(e1)
    store.propose(
        candidate(
            {
                "category": "jargon",
                "key": "赛博浇花",
                "member": "",
                "value": "打开3D小屋看角色浇水",
                "reason": "群聊黑话",
                "evidence_ids": [e1["id"]],
                "days": 30,
            },
            [e1],
            None,
        ),
        "human",
        "test",
    )

    plugin = main_mod.DynamicsLearning(SimpleNamespace(get_config=lambda: {}), {"sessions": [SCOPE], "jargon_review": "human"})
    plugin.engine = Engine(store, plugin.config, None)
    plugin.engine.current_personas[SCOPE] = "persona-a"
    plugin.ready = True

    # Hold store.lock in a background thread to prove sync hooks never block on it!
    locked = threading.Event()
    release = threading.Event()

    def hold_lock():
        with store.lock:
            locked.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert locked.wait(timeout=2.0)
    try:
        summary = plugin.get_series_summary()
        assert summary["ready_count"] == 1
        health = plugin.get_series_health()
        assert health["metrics"]["ready_count"] == 1
        scope_status = plugin.get_series_scope_status(SCOPE)
        assert scope_status["enabled"] is True and "待审 1" in scope_status["summary"]
        search = plugin.series_search_items("赛博浇花", scope=SCOPE)
        assert len(search) == 1
        journal = plugin.get_journal_events(scope=SCOPE, persona="persona-a", date="all")
        assert len(journal) == 1
    finally:
        release.set()
        holder.join(timeout=2.0)


# ---------------------------------------------------------------------------
# P2-1, P2-2, P2-3: Schema keys, single-char nickname filter, dead code cleanup
# ---------------------------------------------------------------------------


def test_p2_1_schema_declares_all_missing_keys():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "_conf_schema.json").read_text(encoding="utf-8"))
    assert schema["reply_projection_budget"]["default"] == 6000
    assert schema["max_jargon_per_turn"]["default"] == 4
    assert schema["audit_retention_days"]["default"] == 90


def test_p2_2_sanitize_filters_single_char_and_short_ascii_nicknames():
    # Single-char CJK ("我", "啊") and 1-2 char ASCII ("a", "hi") must NOT corrupt text
    text = "我啊，hi bobby，小明和bob刚才都在讨论这个"
    cleaned = sanitize(text, identities=["我", "啊", "a", "hi", "小明", "bob"])
    assert cleaned.startswith("我啊，hi bobby，")
    assert "[成员]和[成员]刚才都在讨论这个" in cleaned
    assert "小明" not in cleaned
    assert "bobby" in cleaned  # Word boundary prevents corrupting longer ASCII words


def test_p2_3_dead_code_removed_and_social_drain_closes_sentiment(store):
    # 1) BATCH_OUTPUT removed
    assert not hasattr(ar_mod, "BATCH_OUTPUT")
    # 2) unified_reply_context removed
    assert not hasattr(Engine, "unified_reply_context")
    # 3) created_at not emitted by project_current_entry_for_model
    proj = project_current_entry_for_model({"id": "e1", "category": "style", "key": "k", "created_at": 123})
    assert "created_at" not in proj
    # 4) Unused upstream methods removed from RelationStore
    for dead_method in ("user_relations", "pair_detail", "export_group", "group_report", "import_external_memory", "sentiment_rank"):
        assert not hasattr(RelationStore, dead_method)
    # 5) Social.drain() awaits sentiment.close()
    async def run():
        engine = Engine(store, {"sessions": [SCOPE]}, None)
        social = Social(engine)
        closed = []

        async def fake_close():
            closed.append(True)

        social.sentiment = SimpleNamespace(close=fake_close)
        await social.drain()
        assert closed == [True]

    asyncio.run(run())
