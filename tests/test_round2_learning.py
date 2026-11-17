import asyncio
import json
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning import main
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.core.workbench import govern
from test_audit_review_recovery import APPROVE, SCOPE, proposal
from test_affection_view import put


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "round2.sqlite3")
    yield s
    s.close()


@pytest.mark.parametrize("view", ["active", "all"])
def test_all_groups_entries_only_reads_enabled_sessions(store, monkeypatch, view):
    put(store, "current", scope=SCOPE)
    distinct_scope = "qq:GroupMessage:inactive"
    put(store, "current", scope=distinct_scope)
    plugin = main.DynamicsLearning(SimpleNamespace(), {"sessions": [SCOPE]})
    plugin.engine = SimpleNamespace(store=store, current_personas={})

    async def authorize():
        return "human:audit"

    async def request_body():
        return json.dumps({"scope": "", "state": view}).encode()

    monkeypatch.setattr(plugin, "_authorized", authorize)
    monkeypatch.setattr(main, "request", SimpleNamespace(_request=SimpleNamespace(body=request_body)))
    response = asyncio.run(plugin.entries())
    data = json.loads(response.body)["data"]
    assert data["total"] == 1
    assert [item["scope"] for item in data["entries"]] == [SCOPE]


def test_amend_does_not_leave_replacement_after_rejection_write_fails(store, monkeypatch):
    p = proposal(store, "relationship", "llm")
    engine = Engine(store, {"sessions": [SCOPE]}, None)
    review = store.review

    def reject_then_fail(*args, **kwargs):
        review(*args, **kwargs)
        raise OSError("simulated rejection write failure")

    monkeypatch.setattr(store, "review", reject_then_fail)
    with pytest.raises(OSError):
        asyncio.run(engine.amend(p["id"], "updated value", "explanation", "human:audit"))
    assert store.proposals() == [p]


def test_amend_endpoint_rejects_editor_snapshot_changed_by_review(store, monkeypatch):
    p = proposal(store, "relationship", "llm")
    approved = store.review(p["id"], APPROVE, "llm:test", p["updated"])
    plugin = main.DynamicsLearning(SimpleNamespace(), {"sessions": [SCOPE]})
    plugin.engine = Engine(store, plugin.config, None)

    async def authorize():
        return "human:audit"

    body = {"id": p["id"], "updated": p["updated"], "value": "updated value", "reason": "explanation"}

    async def request_body():
        return json.dumps(body).encode()

    monkeypatch.setattr(plugin, "_authorized", authorize)
    monkeypatch.setattr(main, "request", SimpleNamespace(_request=SimpleNamespace(body=request_body)))
    response = asyncio.run(plugin.amend())
    assert response.status_code == 400
    assert "proposal changed" in json.loads(response.body)["message"]
    assert store.proposals() == [approved]
    body["updated"] = approved["updated"]
    response = asyncio.run(plugin.amend())
    assert response.status_code == 200
    assert store.get(p["id"])["state"] == "rejected"
    assert json.loads(response.body)["data"]["value"] == "updated value"


def test_approved_batch_item_stays_reserved_when_apply_failed(store, monkeypatch):
    async def run():
        from test_group_review import make, items

        p = make(store, "profile", scope=SCOPE, approved=False, category="style")
        engine = Engine(store, {"sessions": [SCOPE]}, None)
        engine.current_personas[SCOPE] = p["persona"]
        calls = []

        async def model(*args):
            calls.append(args)
            return APPROVE

        monkeypatch.setattr(engine, "invoke_model", model)
        await engine.refresh()
        manager = engine.review_batches
        batch = await manager.prepare(SCOPE, items(p), "human:a")
        apply = store.apply

        def fail(*args, **kwargs):
            raise OSError("one-time application error")

        monkeypatch.setattr(store, "apply", fail)
        await manager.run(batch)
        approved = store.get(p["id"])
        assert batch["items"][0]["auto_apply_pending"] and approved["state"] == "reviewed"
        monkeypatch.setattr(store, "apply", apply)

        assert (await engine.apply_approved_profiles())["applied"] == 0
        assert store.review_claim(p["id"]) == batch["id"]
        assert store.get(p["id"]) == approved

        govern(engine, SCOPE, [approved], "release", "human:a")
        assert (await engine.apply_approved_profiles())["applied"] == 1
        assert store.get(p["id"])["state"] == "applied"
        assert len(calls) == 1

    asyncio.run(run())


def test_cycle_cancel_waits_for_active_usage_callback_write(store, monkeypatch):
    import threading

    async def run():
        engine = Engine(store, {"sessions": [SCOPE]}, None)
        entered = threading.Event()
        finish = threading.Event()
        real_write = store.console.record_usage
        sample = {
            "schema_version": 1, "scope": SCOPE, "member": "alice", "turn_id": "one",
            "material_hash": "hash", "stage": "provider_kwargs_after_boundaries",
        }

        def write(*args, **kwargs):
            entered.set()
            assert finish.wait(5)
            real_write(*args, **kwargs)

        monkeypatch.setattr(store.console, "record_usage", write)
        engine.queue.put_nowait((engine.scope_token(SCOPE), {"kind": "reply_usage", "scope": SCOPE, "record": sample}))
        consumer = asyncio.create_task(engine.consume())
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            consumer.cancel()
            await asyncio.sleep(0.01)
            assert not consumer.done()
        finally:
            finish.set()
            await asyncio.gather(consumer, return_exceptions=True)
        assert store.console.usage(SCOPE)["total"] == 1

    asyncio.run(run())
