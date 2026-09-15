from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY
from astrbot_plugin_dynamics_learning.core.query_cache import QueryCache
from .factories import annotated_sessions, export_payload, evidenced_policy


@pytest.mark.asyncio
async def test_confirmed_clear_invalidates_evidence_but_failed_import_preserves_it(plugin):
    rows = annotated_sessions(sessions=2, per_session=3)
    await plugin.ingest(source="export", payload=export_payload(rows))
    await plugin.store.save_policies([evidenced_policy().with_status("validated")])
    plugin._last_report = {"summary": "previous"}
    count = len(await plugin.load_samples())
    failed = await plugin.ingest(source="export", payload={"schema": "unknown"})
    assert failed["ok"] is False
    assert len(await plugin.load_samples()) == count
    assert not plugin._last_report.get("stale")
    session = rows[0][0]
    await plugin.ingest(source="export", payload={"sessions": [
        {"session_key": session, "records": []}]})
    assert all(row.session_key != session for row in await plugin.load_samples())
    assert plugin._last_report["stale"]
    assert (await plugin.store.load_policies())[0].evidence["stale"]
    assert not (await plugin.candidate_payload())["policies"]


@pytest.mark.asyncio
async def test_bare_record_import_preserves_other_messages(plugin):
    rows = annotated_sessions(sessions=1, per_session=4)
    await plugin.ingest(source="export", payload=export_payload(rows))
    before = {row.sample_id for row in await plugin.load_samples()}
    session, record = rows[0]
    await plugin.ingest(source="export", payload=[{**record, "session_key": session}])
    assert {row.sample_id for row in await plugin.load_samples()} == before


@pytest.mark.asyncio
async def test_two_policy_writes_from_same_revision_conflict(plugin):
    candidate = evidenced_policy()
    await plugin.store.save_policies([candidate])
    revision = await plugin.store.policy_revision()
    results = await asyncio.gather(
        plugin.update_policy(candidate.version, "validate", expected_revision=revision),
        plugin.update_policy(candidate.version, "ignore", expected_revision=revision),
        return_exceptions=True)
    assert sum(isinstance(value, dict) for value in results) == 1
    assert any(isinstance(value, ValueError) and "revision conflict" in str(value)
               for value in results)
    stored = await plugin.store.load_candidate()
    assert stored["revision"] == await plugin.store.policy_revision()


@pytest.mark.asyncio
async def test_effective_host_baseline_and_fallback_are_distinguished(plugin):
    values = {**BASE_POLICY, "strong_addressivity_threshold": 0.73}
    plugin.context.get_all_stars = lambda: [SimpleNamespace(
        author="ysyhlly", name="astrbot_plugin_chat_dynamics", activated=True,
        version="v1.8.0", star_cls=SimpleNamespace(
            _learning_policy_effective_config=lambda: values))]
    assert await plugin._analysis_baseline() == (values, "host_effective", "v1.8.0")
    values["strong_addressivity_threshold"] = float("nan")
    baseline, source, _ = await plugin._analysis_baseline()
    assert baseline == BASE_POLICY and source == "default_reference"


@pytest.mark.asyncio
async def test_query_cache_merges_work_and_discards_old_generation():
    cache = QueryCache()
    started, release = threading.Event(), threading.Event()
    calls = []
    loop_thread = threading.get_ident()

    def compute():
        calls.append(threading.get_ident())
        started.set()
        assert release.wait(3)
        return {"rows": [1]}

    first = asyncio.create_task(cache.compute("same", compute))
    second = asyncio.create_task(cache.compute("same", compute))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set() and calls == [calls[0]] and calls[0] != loop_thread
        cache.invalidate()
    finally:
        release.set()
    await asyncio.gather(first, second)
    assert not cache.values
    value = await cache.compute("same", lambda: {"rows": [2]})
    value["rows"].clear()
    assert await cache.compute("same", lambda: None) == {"rows": [2]}
    await cache.close()
