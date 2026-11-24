"""Focused regression tests for R1 (P0 budget backoff), R2 (P0 storage), and R3 (P1 LLM cost/cache)."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning import series_panels
from astrbot_plugin_dynamics_learning.core.contract import candidate, encode, event_record
from astrbot_plugin_dynamics_learning.core.engine import (
    Engine,
    build_review_instruction,
)
from astrbot_plugin_dynamics_learning.core.expressions import REVIEW_STYLE
from astrbot_plugin_dynamics_learning.core.model_projection import encode_prompt_payload
from astrbot_plugin_dynamics_learning.core.review_batches import ReviewDeferred
from astrbot_plugin_dynamics_learning.core.scenes import REVIEW_SCENES
from astrbot_plugin_dynamics_learning.core.store import JOURNAL_SIZE_LIMIT_BYTES, Store
from astrbot_plugin_dynamics_learning.main import DynamicsLearning, _cache_anchor_contexts
from astrbot_plugin_dynamics_learning.vendor.group_relations.store import RelationStore
from test_group_review import SCOPE, make


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "r1_r2_r3.sqlite3")
    yield s
    s.close()


def _make_ev(
    *,
    scope=SCOPE,
    persona="persona-a",
    member="alice",
    mid="m1",
    threshold=60.0,
    summary="角色核心人设摘要" * 20,
    text="群聊正常讨论内容测试",
):
    return event_record(
        {
            "schema_version": 1,
            "kind": "outcome",
            "scope": scope,
            "persona": persona,
            "member": member,
            "turn_id": f"0:{mid}",
            "message_id": mid,
            "host_version": "host1",
            "baseline": "base1",
            "baseline_threshold": threshold,
            "text": text,
            "mandatory": False,
            "timestamp": time.time(),
            "outcome": {"delivered": True, "final_outcome": "sent"},
            "context": {"persona_summary": summary, "padding": "x" * 1200},
        }
    )


def test_r1_budget_busy_raises_review_deferred_and_avoids_junk_rows(store, monkeypatch):
    async def run():
        provider = SimpleNamespace(
            provider_config={"id": "rp", "model": "deepseek-flash"},
            meta=lambda: SimpleNamespace(model="deepseek-flash"),
            text_chat=lambda **kw: None,
            is_native_decision_provider=False,
        )

        class BusyErr(RuntimeError):
            is_provider_budget_busy = True
            reason = "circuit_open"
            retry_after = 47.5

        async def fake_series_llm_call(*_args, **_kwargs):
            raise BusyErr("circuit open")

        monkeypatch.setattr("astrbot_plugin_dynamics_learning.main.series_llm_call", fake_series_llm_call)
        ctx = SimpleNamespace(get_provider_by_id=lambda pid: provider if pid == "rp" else None)
        plugin = DynamicsLearning(ctx, {"sessions": [SCOPE], "review_provider": "rp"})
        plugin.engine = Engine(store, plugin.config, plugin._model, host_context=ctx)
        plugin.engine.current_personas[SCOPE] = "persona-a"

        p = make(store, "k1", member="alice", approved=False)
        item = plugin.engine.review_item(p)
        from astrbot_plugin_dynamics_learning.core.automatic_reviews import ReviewDispatch

        dispatch = ReviewDispatch.for_items([item], "automatic_first", "op-1")
        with pytest.raises(ReviewDeferred) as exc_info:
            await plugin._model("review_provider", "sys", {"proposal": p}, review_dispatch=dispatch)
        assert exc_info.value.retry_after == pytest.approx(47.5)
        assert exc_info.value.reason == "circuit_open"
        # No junk review_provider_attempts row was recorded on pre-flight budget rejection
        assert store.db.execute("SELECT COUNT(*) FROM review_provider_attempts").fetchone()[0] == 0

        # Pre-flight provider_circuit_remaining short-circuits BEFORE inserting console_tasks
        monkeypatch.setattr(series_panels, "provider_circuit_remaining", lambda _c, _p: 35.0, raising=False)
        with pytest.raises(ReviewDeferred) as exc2:
            await plugin.engine.invoke_model("review_provider", "sys", {"proposal": p})
        assert exc2.value.retry_after == pytest.approx(35.0)
        assert store.db.execute("SELECT COUNT(*) FROM console_tasks").fetchone()[0] == 0

    asyncio.run(run())


def test_r1_automatic_reviews_exponential_backoff_and_global_pause(store):
    async def run():
        calls = []

        async def model(role, instruction, data):
            calls.append(data)
            raise ReviewDeferred("circuit open", retry_after=15.0, reason="circuit_open")

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model)
        engine.current_personas[SCOPE] = "persona-a"
        ar = engine.automatic_reviews
        ar.running = True
        ar.wait_seconds = 0

        p = make(store, "k_auto", member="alice", approved=False)
        await ar.enqueue([p])
        groups, _ = ar.ready(time.time(), set(), discover=False)
        assert len(groups) == 1

        before = time.time()
        await ar.dispatch(groups[0])
        assert len(calls) == 1
        # Retry must be >= 60s (never 0ms spin-loop) and global backoff must be active
        row = store.db.execute(
            "SELECT retry, lease FROM automatic_review_queue WHERE proposal=?", (p["id"],)
        ).fetchone()
        assert row["lease"] is None
        assert row["retry"] >= before + 59.0
        assert ar.is_budget_backed_off() is True
        # No junk console_tasks rows left behind on ReviewDeferred
        assert store.db.execute("SELECT COUNT(*) FROM console_tasks").fetchone()[0] == 0

        # Exponential ladder: 60s -> 120s -> 300s (and honors larger retry_after)
        t0 = 10_000.0
        ar.reset_budget_backoff()
        assert ar.note_budget_busy(ReviewDeferred(retry_after=10.0), now=t0) == pytest.approx(t0 + 60.0)
        assert ar.note_budget_busy(ReviewDeferred(retry_after=10.0), now=t0) == pytest.approx(t0 + 120.0)
        assert ar.note_budget_busy(ReviewDeferred(retry_after=10.0), now=t0) == pytest.approx(t0 + 300.0)
        assert ar.note_budget_busy(ReviewDeferred(retry_after=450.0), now=t0) == pytest.approx(t0 + 450.0)

        # Global pause in AutomaticReviews.run() dispatches 0 tasks even when wake fires
        ar._budget_backoff_until = time.time() + 120.0
        calls.clear()
        runner = asyncio.create_task(ar.run())
        await asyncio.sleep(0.05)
        ar.wake.set()
        await asyncio.sleep(0.05)
        assert calls == []
        engine.closed = True
        ar.wake.set()
        await asyncio.wait_for(runner, timeout=2.0)

    asyncio.run(run())


def test_r1_review_batch_and_durable_job_pause_instead_of_skipping(store):
    async def run():
        calls = []

        async def busy_model(role, instruction, data):
            calls.append(data)
            raise ReviewDeferred("daily budget", retry_after=90.0, reason="daily_budget_exhausted")

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, busy_model)
        engine.current_personas[SCOPE] = "persona-a"
        props = [make(store, f"bk{i}", member=f"m{i}", approved=False) for i in range(3)]

        # 1) Engine.review_batch aborts immediately on first ReviewDeferred (1 call, 0 failed)
        failed = set()
        res = await engine.review_batch([p["id"] for p in props], automatic=True, failed_out=failed)
        assert res == {}
        assert failed == set()
        assert len(calls) == 1
        assert all(engine.review_retry_at.get(p["id"], 0) > time.time() for p in props)

        # 2) ReviewBatches.run pauses job and preserves items as queued (never "skipped")
        calls.clear()
        engine.automatic_reviews.reset_budget_backoff()
        engine.review_retry_at.clear()
        job = engine.review_batches.create(SCOPE, props, "human:admin")
        await engine.review_batches.run(job)
        saved = engine.review_batches.get(job["id"])
        assert saved["status"] == "paused"
        assert saved["pause_reason"] == "daily_budget_exhausted"
        assert saved["retry_after"] == pytest.approx(90.0)
        assert all(item["status"] == "queued" for item in saved["items"])
        assert len(calls) == 1

    asyncio.run(run())


def test_r2_storage_auto_vacuum_compaction_and_inbox_expiry(store):
    # 1) PRAGMA auto_vacuum = INCREMENTAL (2) and journal_size_limit = 64 MiB
    assert store.db.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
    assert store.db.execute("PRAGMA journal_size_limit").fetchone()[0] == JOURNAL_SIZE_LIMIT_BYTES

    now = time.time()
    old_ts = now - 40 * 86400
    ancient_ts = now - 120 * 86400

    # 2) Active applied proposal & its rollback predecessor are protected; unreferenced aged applied + rejected are compacted
    ev1 = _make_ev(mid="ev-app-1")
    ev2 = _make_ev(mid="ev-app-2")
    p_prev = store.propose(
        candidate(
            {
                "category": "style",
                "key": "tone",
                "value": "第一版风格",
                "member": "alice",
                "reason": "v1",
                "evidence_ids": [ev1["id"]],
                "days": 30,
            },
            [ev1],
            None,
        ),
        "human",
        "test",
    )
    p_prev = store.apply(p_prev["id"], "human:admin", p_prev["updated"])
    p_curr = store.propose(
        candidate(
            {
                "category": "style",
                "key": "tone",
                "value": "第二版风格",
                "member": "alice",
                "reason": "v2",
                "evidence_ids": [ev2["id"]],
                "days": 30,
            },
            [ev2],
            store.entries()[0],
        ),
        "human",
        "test",
    )
    p_curr = store.apply(p_curr["id"], "human:admin", p_curr["updated"])

    # Rejected proposal aged 40d -> evidence compacted to minimal stubs
    ev_rej = _make_ev(mid="ev-rej")
    p_rej = store.propose(
        candidate(
            {
                "category": "style",
                "key": "rej_key",
                "value": "被拒风格",
                "member": "alice",
                "reason": "rej",
                "evidence_ids": [ev_rej["id"]],
                "days": 30,
            },
            [ev_rej],
            None,
        ),
        "human",
        "test",
    )
    p_rej = store.review(p_rej["id"], {"verdict": "reject", "reason": "no", "risks": []}, "human:admin", p_rej["updated"])

    # Ancient rejected proposal aged 120d -> pruned completely
    ev_anc = _make_ev(mid="ev-anc")
    p_anc = store.propose(
        candidate(
            {
                "category": "style",
                "key": "anc_key",
                "value": "远古被拒",
                "member": "alice",
                "reason": "old",
                "evidence_ids": [ev_anc["id"]],
                "days": 30,
            },
            [ev_anc],
            None,
        ),
        "human",
        "test",
    )
    p_anc = store.review(p_anc["id"], {"verdict": "reject", "reason": "no", "risks": []}, "human:admin", p_anc["updated"])

    # 3) _refresh_pending_evidence stores only IDs in audit.superseded, never full event dicts
    ev_r1 = event_record(
        {
            "schema_version": 1,
            "kind": "observed",
            "scope": SCOPE,
            "persona": "persona-a",
            "member": "alice",
            "turn_id": "0:ev-ref-1",
            "message_id": "ev-ref-1",
            "host_version": "host1",
            "baseline": "base1",
            "baseline_threshold": 60.0,
            "text": "原证据",
            "mandatory": False,
            "timestamp": now,
            "context": {"padding": "x" * 1000},
        }
    )
    ev_r2 = event_record(
        {
            "schema_version": 1,
            "kind": "outcome",
            "scope": SCOPE,
            "persona": "persona-a",
            "member": "alice",
            "turn_id": "0:ev-ref-1",
            "message_id": "ev-ref-1",
            "host_version": "host1",
            "baseline": "base1",
            "baseline_threshold": 60.0,
            "text": "原证据",
            "mandatory": False,
            "timestamp": now + 1,
            "outcome": {"delivered": True, "final_outcome": "sent"},
            "context": {"padding": "x" * 1000},
        }
    )
    ev_r2["received_at"] = ev_r1["received_at"] + 5.0
    p_pend = store.propose(
        candidate(
            {
                "category": "style",
                "key": "ref_key",
                "value": "刷新风格",
                "member": "alice",
                "reason": "r1",
                "evidence_ids": [ev_r1["id"]],
                "days": 30,
            },
            [ev_r1],
            None,
        ),
        "llm_then_human",
        "llm:analysis",
    )
    store.propose(
        candidate(
            {
                "category": "style",
                "key": "ref_key",
                "value": "刷新风格",
                "member": "alice",
                "reason": "r2",
                "evidence_ids": [ev_r2["id"]],
                "days": 30,
            },
            [ev_r2],
            None,
        ),
        "llm_then_human",
        "llm:analysis",
    )
    upd_audit = store.db.execute(
        "SELECT body FROM audit WHERE proposal=? AND action='evidence_updated'", (p_pend["id"],)
    ).fetchone()
    assert upd_audit is not None
    sup_list = json.loads(upd_audit[0])["superseded"]
    assert sup_list == [ev_r1["id"]]

    # 4) Seed orphaned proposal_evidence_sources, affection_source_claims, relation_source_receipts, and pinned inbox
    rel_store = RelationStore(store.path, shared_store=store)
    disabled_ev = _make_ev(scope="qq:GroupMessage:disabled", mid="dis-1")
    disabled_ev["received_at"] = old_ts
    store.ingest(disabled_ev)
    aged_ev = _make_ev(scope=SCOPE, mid="aged-1")
    aged_ev["received_at"] = old_ts
    store.ingest(aged_ev)

    with store.transaction():
        store.db.execute("UPDATE proposals SET updated=? WHERE id IN (?,?,?)", (old_ts, p_prev["id"], p_curr["id"], p_rej["id"]))
        store.db.execute("UPDATE proposals SET updated=? WHERE id=?", (ancient_ts, p_anc["id"]))
        store.db.execute("INSERT INTO proposal_evidence_sources VALUES(?,?,?)", (p_rej["id"], "m-orph", "e-orph"))
        store.db.execute("INSERT INTO affection_source_claims VALUES(?,?,?,?,?)", (SCOPE, "persona-a", "alice", "m-orph", p_rej["id"]))
        store.db.execute(
            "INSERT INTO relation_source_receipts(group_id,source_id,fingerprint,created) VALUES(?,?,?,?)",
            (SCOPE, "old-src", "fp1", int(now - 35 * 86400)),
        )
        store.db.execute(
            "INSERT INTO relation_source_receipts(group_id,source_id,fingerprint,created) VALUES(?,?,?,?)",
            (SCOPE, "fresh-src", "fp2", int(now - 5 * 86400)),
        )
        # Legacy audit row with full dict blob in superseded
        store.db.execute(
            "INSERT INTO audit(proposal,action,actor,created,body) VALUES(?,?,?,?,?)",
            (p_pend["id"], "evidence_updated", "legacy", now, encode({"superseded": [{"id": "legacy-id", "blob": "x" * 2000}]})),
        )

    res = store.prune(
        30,
        current_personas={SCOPE: "persona-a"},
        allowed_scopes=[SCOPE],
        inbox_expire_days=30,
    )
    assert res["proposals_compacted"] >= 1
    assert res["proposals_pruned"] >= 1
    assert res["inbox_expired"] >= 2
    assert res["receipts_pruned"] == 1

    # Protected applied proposals (active + rollback predecessor) retained full evidence
    assert "padding" in store.get(p_curr["id"])["evidence"][0]["context"]
    assert "padding" in store.get(p_prev["id"])["evidence"][0]["context"]
    # Rejected aged proposal evidence compacted while preserving message_id/member/scope
    compacted_rej = store.get(p_rej["id"])
    assert compacted_rej["evidence"][0]["trimmed"] is True
    assert compacted_rej["evidence"][0]["message_id"] == "ev-rej"
    assert "context" not in compacted_rej["evidence"][0]
    # Ancient rejected proposal pruned
    with pytest.raises(ValueError):
        store.get(p_anc["id"])
    # Orphaned rows pruned
    assert store.db.execute("SELECT COUNT(*) FROM proposal_evidence_sources WHERE proposal=?", (p_rej["id"],)).fetchone()[0] == 0
    assert store.db.execute("SELECT COUNT(*) FROM affection_source_claims WHERE proposal=?", (p_rej["id"],)).fetchone()[0] == 0
    # Old relation receipt pruned, fresh kept
    receipt_ids = [r[0] for r in store.db.execute("SELECT source_id FROM relation_source_receipts").fetchall()]
    assert receipt_ids == ["fresh-src"]
    # Disabled-scope & aged inbox events were unpinned and pruned
    assert store.db.execute("SELECT COUNT(*) FROM events WHERE id IN (?,?)", (disabled_ev["id"], aged_ev["id"])).fetchone()[0] == 0
    # Legacy audit blob compacted
    legacy_audit = store.db.execute(
        "SELECT body FROM audit WHERE actor='legacy'"
    ).fetchone()
    assert json.loads(legacy_audit[0])["superseded"] == ["legacy-id"]
    assert rel_store is not None


def test_r3_disabled_modules_omitted_and_cross_group_different_threshold_shares_system_prompt(store):
    async def run():
        scope_a = "qq:GroupMessage:11111"
        scope_b = "qq:GroupMessage:22222"
        ev_a = _make_ev(scope=scope_a, mid="ta", threshold=42.0, text="群A聊游戏很开心")
        ev_b = _make_ev(scope=scope_b, mid="tb", threshold=78.0, text="群B聊电影很开心")
        store.ingest(ev_a)
        store.ingest(ev_b)

        p_a = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "词A",
                    "value": "义A",
                    "member": "",
                    "reason": "r",
                    "evidence_ids": [ev_a["id"]],
                    "days": 30,
                },
                [ev_a],
                None,
            ),
            "llm_then_human",
            "llm:analysis",
        )
        p_b = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "词B",
                    "value": "义B",
                    "member": "",
                    "reason": "r",
                    "evidence_ids": [ev_b["id"]],
                    "days": 30,
                },
                [ev_b],
                None,
            ),
            "llm_then_human",
            "llm:analysis",
        )

        seen = []

        async def model(role, instruction, data):
            seen.append((role, instruction, data))
            if role == "analysis_provider":
                return {"proposals": []}
            return {"verdict": "approve", "reason": "ok", "risks": []}

        engine = Engine(
            store,
            {
                "sessions": [scope_a, scope_b],
                "analysis_provider": "ap",
                "review_provider": "rp",
                "scene_learning_enabled": False,
                "expression_style_enabled": False,
            },
            model,
        )
        engine.current_personas[scope_a] = "persona-a"
        engine.current_personas[scope_b] = "persona-a"

        await engine.analyze(scope_a, "persona-a")
        await engine.analyze(scope_b, "persona-a")
        await engine.review(p_a["id"])
        await engine.review(p_b["id"])

        analyze_calls = [(ins, d) for r, ins, d in seen if r == "analysis_provider"]
        review_calls = [(ins, d) for r, ins, d in seen if r == "review_provider"]

        # 1) REVIEW_SCENES and REVIEW_STYLE omitted when disabled
        assert REVIEW_SCENES not in review_calls[0][0]
        assert REVIEW_STYLE not in review_calls[0][0]
        assert build_review_instruction(include_scenes=False, include_style=False) in review_calls[0][0]

        # 2) Different per-group baseline_threshold (42.0 vs 78.0) NEVER breaks system_prompt cache!
        assert analyze_calls[0][0] == analyze_calls[1][0]
        assert review_calls[0][0] == review_calls[1][0]
        assert _cache_anchor_contexts(analyze_calls[0][0]) == _cache_anchor_contexts(analyze_calls[1][0])
        assert "baseline_threshold" not in analyze_calls[0][0]
        assert "baseline_threshold" not in review_calls[0][0]

        # 3) Per-group baseline_threshold lives in user JSON payload with static-first key ordering
        assert analyze_calls[0][1]["baseline_threshold"] == 42.0
        assert analyze_calls[1][1]["baseline_threshold"] == 78.0
        assert review_calls[0][1]["baseline_threshold"] == 42.0
        assert review_calls[1][1]["baseline_threshold"] == 78.0
        raw_payload = encode_prompt_payload(analyze_calls[0][1])
        assert raw_payload.startswith('{"current":')
        assert raw_payload.index('"current":') < raw_payload.index('"baseline_threshold":') < raw_payload.index('"events":')

    asyncio.run(run())
