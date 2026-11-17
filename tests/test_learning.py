import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record, review_result, decode_model_object
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Conflict, Store


def event(**changes):
    return event_record(dict(schema_version=1, kind="decision", scope="qq:GroupMessage:1", persona="persona-a",
        member="alice", turn_id="0:m1", message_id="m1", host_version="host1", baseline="config1",
        baseline_threshold=70, text="请简短一些", mandatory=False, backend="jev", **changes))


def raw(**changes):
    return dict(category="style", key="length", value="回答偏好简短", member="alice", reason="用户明确要求",
                evidence_ids=[event()["id"]], days=30, **changes)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "learning.sqlite3")
    yield s
    s.close()


def proposal(store, mode="human", value="回答偏好简短"):
    e = event()
    r = raw()
    r["value"] = value
    return store.propose(candidate(r, [e], None), mode, "test")


def test_event_idempotence_and_scope(store):
    e = event()
    store.ingest(e)
    store.ingest(e)
    assert len(store.events(e["scope"], e["persona"])) == 1
    assert not store.events("other", e["persona"])


@pytest.mark.parametrize("field,value", [("schema_version", 9), ("text", "a"*4001), ("mandatory", 1),
                                        ("baseline_threshold", float("nan")), ("scope", "")])
def test_bad_events(field, value):
    e = event()
    e.pop("id")
    e.pop("received_at")
    e[field] = value
    with pytest.raises(ValueError):
        event_record(e)


def test_member_cannot_be_forged():
    r = raw()
    r["member"] = "bob"
    with pytest.raises(ValueError, match="member evidence"):
        candidate(r, [event()], None)


def test_foreign_evidence_rejected():
    r = raw()
    r["evidence_ids"] = ["invented"]
    with pytest.raises(ValueError):
        candidate(r, [event()], None)


def test_state_expires_within_day():
    r = raw()
    r["category"] = "state"
    assert candidate(r, [event()], None)["days"] == 1


def test_threshold_bounds_and_sample_count():
    events = []
    for i in range(10):
        e = event()
        for k in ("id", "received_at"):
            e.pop(k)
        e["turn_id"], e["message_id"] = str(i), str(i)
        events.append(event_record(e))
    r = dict(category="participation", key="reply_probability_threshold", value=75, member="",
             reason="ten interactions", evidence_ids=[e["id"] for e in events], days=30)
    assert candidate(r, events, None)["value"] == 75
    r["value"] = 76
    with pytest.raises(ValueError, match="bounded range"):
        candidate(r, events, None)
    r["value"] = 75
    r["evidence_ids"] = r["evidence_ids"][:9]
    with pytest.raises(ValueError, match="ten distinct"):
        candidate(r, events, None)


def test_human_apply_and_rollback(store):
    p = proposal(store)
    with pytest.raises(Conflict):
        store.apply(p["id"], "llm:x", p["updated"])
    p = store.apply(p["id"], "human:admin", p["updated"])
    assert store.entries()[0]["active"]
    store.rollback(p["id"], "human:admin")
    assert not store.entries()[0]["active"]


@pytest.mark.parametrize("mode", ["llm", "llm_then_human"])
def test_llm_review_required(store, mode):
    p = proposal(store, mode)
    with pytest.raises(Conflict, match="LLM approval"):
        store.apply(p["id"], "human:admin", p["updated"])
    p = store.review(p["id"], {"verdict": "approve", "reason": "ok", "risks": []}, "llm:test", p["updated"])
    p = store.apply(p["id"], "llm:test" if mode == "llm" else "human:admin", p["updated"])
    assert p["state"] == "applied"


def test_stale_review_rejected(store):
    p = proposal(store)
    store.review(p["id"], {"verdict": "reject", "reason": "no", "risks": []}, "human:admin", p["updated"])
    with pytest.raises(Conflict):
        store.review(p["id"], {"verdict": "approve", "reason": "yes", "risks": []}, "llm:test", p["updated"])


def test_new_defer_invalidates_previous_llm_approval(store):
    p = proposal(store, "llm_then_human")
    p = store.review(p["id"], {"verdict": "approve", "reason": "ok", "risks": []}, "llm:a", p["updated"])
    p = store.review(p["id"], {"verdict": "defer", "reason": "uncertain", "risks": []}, "llm:b", p["updated"])
    with pytest.raises(Conflict, match="LLM approval"):
        store.apply(p["id"], "human:admin", p["updated"])


@pytest.mark.parametrize("payload", ['{"proposals":[]}', '```json\n{"proposals":[]}\n```',
                                    '\ufeff {"proposals":[]} ', '```JSON\r\n{"proposals":[]}\r\n```'])
def test_model_response_json_fence_compatibility(payload):
    assert decode_model_object(payload) == {"proposals": []}


@pytest.mark.parametrize("payload", ['结果如下 {"proposals":[]}', '```python\n{}\n```', '', '[]', '{}\nextra'])
def test_response_prose_and_invalid_json_stay_rejected(payload):
    with pytest.raises(ValueError):
        decode_model_object(payload)


def test_cas_and_stacked_rollback(store):
    p1 = proposal(store)
    p2 = proposal(store, value="回答应适当详细")
    store.apply(p1["id"], "human:admin", p1["updated"])
    with pytest.raises(Conflict, match="changed"):
        store.apply(p2["id"], "human:admin", p2["updated"])
    # A new proposal based on the current revision is independently reviewed.
    p3 = proposal(store, value="先给结论，再解释")
    store.apply(p3["id"], "human:admin", p3["updated"])
    with pytest.raises(Conflict):
        store.rollback(p1["id"], "human:admin")
    store.rollback(p3["id"], "human:admin")
    assert store.entries()[0]["value"] == p1["value"]
    store.rollback(p1["id"], "human:admin")
    assert not store.entries()[0]["active"]


def test_duplicate_proposal(store):
    assert proposal(store)["id"] == proposal(store)["id"]


def test_purge_removes_evidence_and_content(store):
    p = proposal(store)
    store.ingest(event())
    store.apply(p["id"], "human:admin", p["updated"])
    store.purge(p["scope"])
    assert not store.entries() and not store.proposals() and not store.events(p["scope"], p["persona"])


@pytest.mark.parametrize("result", [{"verdict": "approve"}, {"verdict": "approve", "reason": "ok", "risks": [], "extra": 1},
                                   {"verdict": "yes", "reason": "ok", "risks": []}])
def test_invalid_review(result):
    with pytest.raises(ValueError):
        review_result(result)


def test_context_isolation_and_expiry(store):
    async def run():
        p = proposal(store)
        store.apply(p["id"], "human:admin", p["updated"])
        engine = Engine(store, {"sessions": [p["scope"]]}, None)
        await engine.refresh()
        args = dict(scope=p["scope"], persona=p["persona"], member=p["member"], baseline="config1", host_version="host1")
        assert p["value"] in engine.context(**args)["context"]
        assert not engine.context(**{**args, "member": "bob"})["context"]
        assert not engine.context(**{**args, "persona": "other"})["context"]
        assert not engine.context(**{**args, "scope": "other"})["context"]
        engine.cache[0]["expires"] = time.time()-1
        assert not engine.context(**args)["context"]
    asyncio.run(run())


@pytest.mark.parametrize("mode,state", [("human", "pending"), ("llm_then_human", "reviewed"), ("llm", "applied")])
def test_full_learning_review_flow(store, mode, state):
    async def run():
        calls = []
        async def model(role, instruction, data):
            calls.append(role)
            if role == "analysis_provider":
                return {"proposals": [raw()]}
            return {"verdict": "approve", "reason": "explicit preference", "risks": []}
        e = event()
        store.ingest(e)
        engine = Engine(store, {"sessions": [e["scope"]], "style_review": mode, "review_provider": "review-model"}, model)
        await engine.analyze(e["scope"], e["persona"])
        p = store.proposals()[0]
        assert p["state"] == state
        if mode == "llm_then_human":
            await engine.action(p["id"], "approve", "human:admin", p["updated"])
            assert engine.cache[0]["active"]
        assert len(calls) == (1 if mode == "human" else 2)
    asyncio.run(run())


def test_llm_failure_never_approves(store):
    async def run():
        async def model(*args):
            raise TimeoutError()
        p = proposal(store, "llm")
        engine = Engine(store, {"sessions": [p["scope"]]}, model)
        with pytest.raises(TimeoutError):
            await engine.review(p["id"])
        assert store.get(p["id"])["state"] == "pending"
        assert not store.entries()
    asyncio.run(run())


def test_purge_invalidates_queued_records(store):
    async def run():
        e = event()
        engine = Engine(store, {"sessions": [e["scope"]]}, None)
        raw_event = {k: v for k, v in e.items() if k not in {"id", "received_at"}}
        assert engine.ingest(raw_event)
        await engine.purge(e["scope"])
        consumer = asyncio.create_task(engine.consume())
        await engine.queue.join()
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        assert not store.events(e["scope"], e["persona"])
    asyncio.run(run())


def test_amend_resets_review(store):
    async def run():
        p = proposal(store, "llm_then_human")
        p = store.review(p["id"], {"verdict": "approve", "reason": "ok", "risks": []}, "llm:x", p["updated"])
        engine = Engine(store, {"sessions": [p["scope"]]}, None)
        new = await engine.amend(p["id"], "先给简短结论", "保留解释空间", "human:admin")
        assert new["state"] == "pending" and not new["reviews"]
        assert store.get(p["id"])["state"] == "rejected"
    asyncio.run(run())


def test_restart_preserves_applied_values(tmp_path):
    path = tmp_path / "persist.sqlite3"
    s = Store(path)
    p = proposal(s)
    s.apply(p["id"], "human:admin", p["updated"])
    s.close()
    s = Store(path)
    assert s.entries()[0]["value"] == p["value"]
    assert json.loads(s.db.execute("SELECT body FROM audit WHERE action='applied'").fetchone()[0]) == {}
    s.close()


def test_tightening_review_mode_blocks_automatic_apply(store):
    async def run():
        p = proposal(store, "llm")
        config = {"sessions": [p["scope"]], "style_review": "llm", "review_provider": "review"}
        async def model(*args):
            config["style_review"] = "human"
            return {"verdict": "approve", "reason": "ok", "risks": []}
        engine = Engine(store, config, model)
        await engine.review(p["id"])
        assert store.get(p["id"])["state"] == "reviewed"
        assert not store.entries()
    asyncio.run(run())


def test_processed_scope_remains_available_for_manual_reanalysis(store):
    e = event()
    store.ingest(e)
    store.checkpoint(e["scope"], e["persona"], e["received_at"])
    assert not store.batches(1)
    assert store.batches(0)[0]["count"] == 0


def test_observation_counts_delivery_as_fact_not_quality(store):
    p = proposal(store)
    store.apply(p["id"], "human:admin", p["updated"])
    p = store.get(p["id"])
    raw_event = {k: v for k, v in event().items() if k not in {"id", "received_at"}}
    raw_event["policy"] = {"versions": [p["id"]]}
    raw_event["decision"] = {"action": "reply"}
    raw_event["reply"] = {"message_id": "sent1", "text": "好的", "delivered": True}
    store.ingest(event_record(raw_event))
    raw_event["kind"] = "outcome"
    raw_event["outcome"] = {"final_outcome": "delivery_failed", "delivered": False}
    store.ingest(event_record(raw_event))
    stats = store.observations(p)
    assert stats["after"] == {"turns": 1, "admitted": 1, "silent": 0, "delivered": 1, "failed_without_delivery": 0}
