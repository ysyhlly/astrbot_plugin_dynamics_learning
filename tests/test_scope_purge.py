import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning.core.contract import event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.social import Social
from astrbot_plugin_dynamics_learning.core.store import Store


SCOPE_A = "account-a:GroupMessage:123"
SCOPE_B = "account-b:GroupMessage:123"
SCOPE_C = "account-a:GroupMessage:456"
SCOPES = [SCOPE_A, SCOPE_B, SCOPE_C]
PERSONA = "persona"


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "learning.sqlite3")
    yield value
    value.close()


def event(scope, message_id):
    return {
        "schema_version": 1,
        "kind": "observed",
        "scope": scope,
        "persona": PERSONA,
        "message_id": message_id,
        "turn_id": message_id,
        "member": "alice",
        "host_version": "test-host",
        "baseline": "test-baseline",
        "baseline_threshold": 70,
        "mandatory": False,
        "text": "谢谢你认真陪我讨论",
    }


def observation(scope, message_id, *, directed=False):
    return {
        "scope": scope,
        "source_id": message_id,
        "member": "alice",
        "name": "Alice",
        "text": "谢谢你认真陪我讨论",
        "directed": [("bob", "reply", 8.0)] if directed else [],
    }


def message_ids(store, scopes=SCOPES):
    return {scope: [row["message_id"] for row in store.events(scope, PERSONA)] for scope in scopes}


def graph_source_ids(store):
    with store.lock:
        return {
            scope: [row[0] for row in store.db.execute(
                "SELECT source_id FROM relation_source_receipts WHERE group_id=? ORDER BY source_id", (scope,)
            )]
            for scope in SCOPES
        }


async def join(queue):
    await asyncio.wait_for(queue.join(), 5)


async def shutdown(engine, *consumers):
    engine.closed = True
    engine.generation += 1
    for task in consumers:
        task.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)
    if engine.social:
        await engine.social.drain()


def test_learning_purge_keeps_other_queued_scopes_and_accepts_new_events(store):
    async def run():
        engine = Engine(store, {"sessions": SCOPES}, None)
        for scope in SCOPES:
            assert engine.ingest(event(scope, "old"))
            assert engine.capture_message({
                **observation(scope, "old"), "created": time.time(), "eligible": True, "reply_to": "",
            })

        await engine.purge(SCOPE_A)
        assert engine.ingest(event(SCOPE_A, "new"))
        assert engine.capture_message({
            **observation(SCOPE_A, "new"), "created": time.time(), "eligible": True, "reply_to": "",
        })
        consumer = asyncio.create_task(engine.consume())
        try:
            await join(engine.queue)
        finally:
            await shutdown(engine, consumer)

        assert not engine.last_error
        assert message_ids(store) == {SCOPE_A: ["new"], SCOPE_B: ["old"], SCOPE_C: ["old"]}
        for scope, ids in message_ids(store).items():
            assert [row["message_id"] for row in store.profiles.sources(scope, PERSONA)] == ids
            assert engine.current_personas[scope] == PERSONA
    asyncio.run(run())


def test_purge_clears_learning_receipts_only_for_target_scope(store):
    pytest.importorskip("astrbot.api.star")
    from astrbot_plugin_dynamics_learning.main import DynamicsLearning

    async def run():
        engine = Engine(store, {"sessions": SCOPES, "scene_learning_enabled": True}, None)
        plugin = SimpleNamespace(engine=engine, ready=True, closed=False)
        for scope in SCOPES:
            assert engine.ingest(event(scope, "observed"))
            receipt = dict(scope=scope, member="alice", turn_id="observed", persona=PERSONA)
            assert DynamicsLearning.report_scene_event(plugin, {**receipt, "stage": "selected"})
            assert DynamicsLearning.report_reply_usage(plugin, {
                **receipt, "schema_version": 1, "material_hash": "hash", "scene_trace": receipt,
                "stage": "provider_kwargs_after_boundaries",
            })
        await engine.purge(SCOPE_A)
        consumer = asyncio.create_task(engine.consume())
        try:
            await join(engine.queue)
        finally:
            await shutdown(engine, consumer)

        assert not engine.last_error
        with store.lock:
            assert [tuple(row) for row in store.db.execute("SELECT scope,stage FROM scene_receipts ORDER BY scope,stage")] == [
                (scope, stage) for scope in [SCOPE_C, SCOPE_B] for stage in ["provider", "selected"]
            ]
        assert {scope: store.console.usage(scope)["total"] for scope in SCOPES} == {
            SCOPE_A: 0, SCOPE_B: 1, SCOPE_C: 1,
        }
    asyncio.run(run())


def test_purge_invalidates_only_target_scope_inflight_analysis(store):
    async def run():
        started = {scope: asyncio.Event() for scope in SCOPES[:2]}
        release = asyncio.Event()

        async def model(_role, _instruction, data):
            started[data["events"][0]["scope"]].set()
            await release.wait()
            return {"proposals": []}

        engine = Engine(store, {"sessions": SCOPES}, model)
        consumer = asyncio.create_task(engine.consume())
        try:
            for scope in started:
                assert engine.ingest(event(scope, "old"))
            await join(engine.queue)
            analyses = {scope: asyncio.create_task(engine.analyze(scope, PERSONA)) for scope in started}
            await asyncio.wait_for(asyncio.gather(*(signal.wait() for signal in started.values())), 5)
            await engine.purge(SCOPE_A)
            release.set()
            with pytest.raises(ValueError, match="analysis invalidated"):
                await asyncio.wait_for(analyses[SCOPE_A], 5)
            assert (await asyncio.wait_for(analyses[SCOPE_B], 5))["turns"] == 1
        finally:
            release.set()
            await shutdown(engine, consumer)

        assert not engine.last_error
        assert message_ids(store) == {SCOPE_A: [], SCOPE_B: ["old"], SCOPE_C: []}
        with store.lock:
            assert [row[0] for row in store.db.execute("SELECT scope FROM console_tasks")] == [SCOPE_B]
            assert [row[0] for row in store.db.execute("SELECT scope FROM analysis_runs")] == [SCOPE_B]
    asyncio.run(run())


def test_purge_keeps_other_scopes_inflight_sentiment_and_queued_observations(store):
    async def run():
        engine = Engine(store, {"sessions": SCOPES, "shared_sentiment_enabled": True}, None)
        social = engine.social = Social(engine)
        started, release = asyncio.Event(), asyncio.Event()

        async def classify(_):
            started.set()
            await release.wait()
            return None

        social.sentiment = SimpleNamespace(classify=classify)
        assert social.observe(observation(SCOPE_B, "inflight", directed=True))
        consumer = asyncio.create_task(social.consume())
        try:
            await asyncio.wait_for(started.wait(), 5)
            for scope in SCOPES:
                assert social.observe(observation(scope, "queued"))
            await engine.purge(SCOPE_A)
            assert social.observe(observation(SCOPE_A, "new"))
            release.set()
            await join(social.queue)
        finally:
            await shutdown(engine, consumer)

        assert not social.last_error
        assert graph_source_ids(store) == {SCOPE_A: ["new"], SCOPE_B: ["inflight", "queued"], SCOPE_C: ["queued"]}
    asyncio.run(run())


def test_same_scope_purge_discards_inflight_sentiment_result(store):
    async def run():
        engine = Engine(store, {"sessions": SCOPES, "shared_sentiment_enabled": True}, None)
        social = engine.social = Social(engine)
        started, release = asyncio.Event(), asyncio.Event()

        async def classify(_):
            started.set()
            await release.wait()
            return None

        social.sentiment = SimpleNamespace(classify=classify)
        assert social.observe(observation(SCOPE_A, "inflight", directed=True))
        assert social.observe(observation(SCOPE_B, "queued"))
        consumer = asyncio.create_task(social.consume())
        try:
            await asyncio.wait_for(started.wait(), 5)
            await engine.purge(SCOPE_A)
            release.set()
            await join(social.queue)
        finally:
            await shutdown(engine, consumer)

        assert not social.last_error
        assert graph_source_ids(store) == {SCOPE_A: [], SCOPE_B: ["queued"], SCOPE_C: []}
    asyncio.run(run())


def test_purge_orders_reset_after_inflight_graph_write(store, monkeypatch):
    async def run():
        engine = Engine(store, {"sessions": SCOPES}, None)
        social = engine.social = Social(engine)
        started, release, resetting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        record_message = social.graph.record_message
        purge = social.purge

        async def delayed_record(scope, *args, **kwargs):
            if scope == SCOPE_A:
                started.set()
                await release.wait()
            return await record_message(scope, *args, **kwargs)

        async def tracked_purge(scope):
            resetting.set()
            return await purge(scope)

        monkeypatch.setattr(social.graph, "record_message", delayed_record)
        monkeypatch.setattr(social, "purge", tracked_purge)
        assert social.observe(observation(SCOPE_A, "inflight"))
        assert social.observe(observation(SCOPE_B, "queued"))
        consumer = asyncio.create_task(social.consume())
        try:
            await asyncio.wait_for(started.wait(), 5)
            purging = asyncio.create_task(engine.purge(SCOPE_A))
            await asyncio.wait_for(resetting.wait(), 5)
            assert not purging.done()
            assert graph_source_ids(store)[SCOPE_A] == []
            release.set()
            await asyncio.wait_for(purging, 5)
            await join(social.queue)
        finally:
            release.set()
            await shutdown(engine, consumer)

        assert not social.last_error
        assert graph_source_ids(store) == {SCOPE_A: [], SCOPE_B: ["queued"], SCOPE_C: []}
    asyncio.run(run())


def test_cancelled_purge_keeps_lock_until_sqlite_and_cache_complete(store, monkeypatch):
    async def run():
        engine = Engine(store, {"sessions": SCOPES}, None)
        loop = asyncio.get_running_loop()
        started, waiting = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        observations = []
        purge = store.purge
        store.ingest(event_record(event(SCOPE_A, "persisted")))
        await engine.refresh()
        assert engine.current_personas[SCOPE_A] == PERSONA

        def delayed_purge(scope):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(5):
                raise TimeoutError("test did not release the SQLite purge")
            purge(scope)

        async def check_after_purge():
            waiting.set()
            async with engine.lock:
                observations.append((message_ids(store)[SCOPE_A], engine.current_personas.get(SCOPE_A)))

        monkeypatch.setattr(store, "purge", delayed_purge)
        purging = asyncio.create_task(engine.purge(SCOPE_A))
        try:
            await asyncio.wait_for(started.wait(), 5)
            purging.cancel()
            waiter = asyncio.create_task(check_after_purge())
            await asyncio.wait_for(waiting.wait(), 5)
            purging.cancel()
            assert not purging.done()
            assert not waiter.done()
            assert message_ids(store)[SCOPE_A] == ["persisted"]
        finally:
            release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(purging, 5)
        await asyncio.wait_for(waiter, 5)
        assert observations == [([], None)]
        assert engine.ingest(event(SCOPE_A, "new"))
        consumer = asyncio.create_task(engine.consume())
        try:
            await join(engine.queue)
        finally:
            await shutdown(engine, consumer)
        assert message_ids(store)[SCOPE_A] == ["new"]
    asyncio.run(run())


def test_scope_tokens_also_respect_global_lifecycle_generation(store):
    async def run():
        engine = Engine(store, {"sessions": SCOPES}, None)
        social = engine.social = Social(engine)
        for scope in SCOPES:
            assert engine.ingest(event(scope, "old"))
            assert social.observe(observation(scope, "old"))

        engine.generation += 1
        consumers = [asyncio.create_task(consume()) for consume in (engine.consume, social.consume)]
        try:
            await join(engine.queue)
            await join(social.queue)
        finally:
            await shutdown(engine, *consumers)

        for scope in SCOPES:
            assert not engine.ingest(event(scope, "closed"))
            assert not social.observe(observation(scope, "closed"))
        assert not engine.last_error
        assert not social.last_error
        assert message_ids(store) == {scope: [] for scope in SCOPES}
        assert graph_source_ids(store) == {scope: [] for scope in SCOPES}
    asyncio.run(run())
