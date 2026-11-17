import asyncio
import sqlite3
import time
from contextlib import closing

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.core.workbench import govern


SCOPE = "qq:GroupMessage:audit"
APPROVE = {"verdict": "approve", "reason": "Reviewed evidence", "risks": []}


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "audit.sqlite3")
    yield result
    result.close()


def proposal(store, category, mode):
    now = time.time()
    evidence = event_record({
        "scope": SCOPE, "persona": "persona-a", "member": "alice", "kind": "observed",
        "turn_id": "turn-1", "message_id": "message-1", "schema_version": 1, "mandatory": False,
        "baseline": "baseline-a", "host_version": "1", "baseline_threshold": 40,
        "text": "Synthetic interaction", "timestamp": now,
        "context": {"character_scope": {"platform": "qq", "umo": SCOPE,
                                          "persona_id": "test", "instance_id": "test"}},
    })
    store.ingest(evidence)
    value = {"experience_delta": 2, "tone": 0.1} if category == "affection" else "Enjoys discussing books"
    key = "bot_affection" if category == "affection" else "books"
    change = candidate({
        "category": category, "member": "alice", "value": value, "key": key,
        "reason": "Synthetic test", "evidence_ids": [evidence["id"]], "days": 30,
    }, [evidence], None)
    return store.propose(change, mode, "test")


def test_commit_failure_rolls_back_nested_work_and_next_write_commits(store):
    fail_once = True

    def authorize(action, operation, *_args):
        nonlocal fail_once
        if fail_once and action == sqlite3.SQLITE_TRANSACTION and operation == "COMMIT":
            fail_once = False
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.db.set_authorizer(authorize)
    with pytest.raises(sqlite3.DatabaseError):
        with store.transaction():
            failed = proposal(store, "relationship", "llm")

    assert not store.db.in_transaction
    assert store.proposals() == []
    recovered = proposal(store, "relationship", "llm")
    assert recovered["id"] != failed["id"]
    with closing(sqlite3.connect(store.path)) as reader:
        assert reader.execute("SELECT id FROM proposals").fetchall() == [(recovered["id"],)]


def test_affection_mode_change_respects_paused_claim_until_released(store):
    async def run():
        p = proposal(store, "affection", "llm_then_human")
        config = {"sessions": [SCOPE], "affection_review": "llm_then_human"}
        engine = Engine(store, config, None)
        await engine.refresh()
        manager = engine.review_batches
        job = await manager.prepare(SCOPE, [{"id": p["id"], "updated": p["updated"]}], "human:a")
        await manager.control(job["id"], "pause", "human:a")
        config["affection_review"] = "auto"

        engine = Engine(store, config, None)
        await engine.refresh()
        manager = engine.review_batches
        assert (await engine.apply_pending_affection()) == {"applied": 0, "stale": 0}
        assert store.review_claim(p["id"]) == job["id"]
        assert store.get(p["id"]) == p
        assert manager.get(job["id"])["status"] == "paused"
        with pytest.raises(ValueError, match="reserved"):
            await engine.action(p["id"], "approve", "human:a", p["updated"])

        await manager.control(job["id"], "cancel", "human:a")
        assert (await engine.apply_pending_affection()) == {"applied": 0, "stale": 0}
        govern(engine, SCOPE, [p], "release", "human:a")
        assert (await engine.apply_pending_affection()) == {"applied": 1, "stale": 0}
        assert store.get(p["id"])["state"] == "applied"

    asyncio.run(run())


@pytest.mark.parametrize("restart", [False, True])
def test_batch_retry_reuses_non_profile_approval(store, monkeypatch, restart):
    async def run():
        p = proposal(store, "relationship", "llm")
        calls = []

        async def review_model(*args):
            calls.append(args)
            return APPROVE

        config = {"sessions": [SCOPE], "relationship_review": "llm"}
        engine = Engine(store, config, review_model)
        await engine.refresh()
        manager = engine.review_batches
        parent = await manager.prepare(SCOPE, [{"id": p["id"], "updated": p["updated"]}], "human:a")
        original_apply = store.apply

        def failed_apply(*args, **kwargs):
            original_apply(*args, **kwargs)
            raise OSError("simulated storage failure after writing")

        monkeypatch.setattr(store, "apply", failed_apply)
        await manager.run(parent)
        saved = manager.get(parent["id"])
        approved = store.get(p["id"])
        assert saved["items"][0]["auto_apply_pending"] and saved["items"][0]["error"]
        assert approved["state"] == "reviewed" and store.entries() == []
        assert len(calls) == len(approved["reviews"]) == 1
        if restart:
            engine = Engine(store, config, None)
            await engine.refresh()
            manager = engine.review_batches

        if restart:
            config["relationship_review"] = "llm_then_human"
            with pytest.raises(ValueError, match="没有可重试"):
                await manager.control(parent["id"], "retry", "human:a", require_provider=True)
            assert not manager.get(parent["id"]).get("retries")
            config["relationship_review"] = "llm"

        child = await manager.control(parent["id"], "retry", "human:a", require_provider=True)
        await manager.run(child)
        assert child["status"] == "completed_with_issues"
        assert child["items"][0]["status"] == "failed"
        assert store.get(p["id"]) == approved

        monkeypatch.setattr(store, "apply", original_apply)
        recovered = await manager.control(child["id"], "retry", "human:a")
        await manager.run(recovered)
        assert recovered["status"] == "completed"
        assert recovered["items"][0]["status"] == "applied"
        assert manager.get(parent["id"])["items"] == saved["items"]
        assert store.get(p["id"])["reviews"] == approved["reviews"]
        assert store.entries()[0]["proposal_id"] == p["id"]
        assert len(calls) == 1

    asyncio.run(run())


def test_individual_retry_reuses_non_profile_approval(store):
    async def run():
        p = proposal(store, "relationship", "llm")
        approved = store.review(p["id"], APPROVE, "llm:test", p["updated"])
        engine = Engine(store, {"sessions": [SCOPE], "relationship_review": "llm_then_human"}, None)
        await engine.refresh()
        assert await engine.review(p["id"]) == approved
        engine.config["relationship_review"] = "llm"
        assert await engine.review(p["id"], defer_apply=True) == approved
        recovered = await engine.review(p["id"])
        assert recovered["state"] == "applied"
        assert recovered["reviews"] == approved["reviews"]

    asyncio.run(run())
