"""Reading the host contract and persisting samples without unbounded growth."""
from __future__ import annotations

import sys
import types

import pytest

from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.ingest import (
    IngestResult, collect_from_host, parse_export, parse_preferences,
)
from astrbot_plugin_dynamics_learning.core.samples import build_dataset, session_hash
from astrbot_plugin_dynamics_learning.core.store import (
    POLICY_KEY, SAMPLE_INDEX_KEY, SAMPLE_KEY_PREFIX, LearningStore, MemoryBackend,
)

from .factories import ambient_record, make_record, make_trace


def _preference(key, value):
    return {"key": key, "value": {"val": value}}


def test_runtime_snapshot_recovers_the_session_behind_a_hashed_key():
    session = "umo:group:42"
    rows = [
        _preference("panel_runtime_v1", {"version": 1, "sessions": [{"session_key": session}]}),
        _preference("topic_annotations_v1_" + session_hash(session),
                    [make_record("m1", trace=make_trace(topic_id="t1", topic_confidence=0.5),
                                 predicted_topic="t1", expected_topic="t1")]),
    ]
    result = parse_preferences(rows)
    assert result.records == 1
    assert result.annotations[0][0] == session
    assert result.diagnostics["runtime_present"] is True
    assert result.diagnostics["unknown_sessions"] == 0


def test_annotations_without_a_known_session_are_reported_not_guessed():
    rows = [_preference("topic_annotations_v1_" + "a" * 64,
                        [make_record("m1", trace=make_trace())])]
    result = parse_preferences(rows)
    assert result.records == 0
    assert result.diagnostics["unknown_sessions"] == 1


def test_malformed_preference_rows_are_counted_and_skipped():
    session = "s"
    rows = [
        {"key": "panel_runtime_v1", "value": {"val": "not-a-mapping"}},
        {"key": "topic_annotations_v1_" + session_hash(session), "value": {"val": "not-a-list"}},
        {"key": "topic_annotations_v1_" + session_hash(session) + "extra", "value": {"val": []}},
        {"key": 5, "value": {"val": []}},
        "not-a-row",
        {"key": "panel_runtime_v1", "value": {"val": {"version": 1, "sessions": [{"session_key": session}]}}},
        {"key": "topic_annotations_v1_" + session_hash(session),
         "value": {"val": [{"msg_id": ""}, "junk", {"msg_id": "m1",
                                                    "decision_trace": make_trace()}]}},
    ]
    result = parse_preferences(rows)
    assert result.records == 1
    assert result.diagnostics["malformed"] >= 3


def test_a_key_that_looks_like_an_annotation_but_is_not_is_ignored():
    result = parse_preferences([_preference("topic_annotations_v1_short", [])])
    assert result.diagnostics["annotation_keys"] == 0


def test_export_import_accepts_the_documented_shapes():
    record = make_record("m1", trace=make_trace(topic_id="t1", topic_confidence=0.5),
                         predicted_topic="t1", expected_topic="t1")
    grouped = parse_export({"sessions": [{"session_key": "s", "records": [record]}]})
    assert grouped.records == 1
    listed = parse_export([{"session_key": "s", "records": [record]}])
    assert listed.records == 1
    bare = parse_export([dict(record, session_key="s")])
    assert bare.records == 1
    assert parse_export("nope").diagnostics["error"] == "unsupported export shape"
    assert parse_export([None, 5]).diagnostics["malformed"] == 2


@pytest.mark.asyncio
async def test_store_shards_per_session_and_prunes_oldest_first():
    backend = MemoryBackend()
    store = LearningStore(backend)
    config = LearningConfig(max_samples=10)
    for index in range(4):
        session = f"umo:group:{index}"
        rows = [s for s in build_dataset(
            [(session, ambient_record(f"m{index}", seed=index, annotated_at=float(index)))]
        )]
        await store.replace_session(session, rows, config=config, now=float(index))
    index_payload = backend.data[SAMPLE_INDEX_KEY]
    assert index_payload["total"] <= 10
    assert len([key for key in backend.data if key.startswith(SAMPLE_KEY_PREFIX)]) == \
        len(index_payload["sessions"])
    # The newest session must survive; the oldest is the one evicted.
    assert "umo:group:3" in {row["session_key"] for row in index_payload["sessions"].values()}


@pytest.mark.asyncio
async def test_store_round_trips_samples_and_index_rows():
    backend = MemoryBackend()
    store = LearningStore(backend)
    rows = build_dataset([("umo:group:1", ambient_record("m1", seed=1))])
    await store.replace_session("umo:group:1", rows, now=1.0)
    restored = await store.load_samples()
    assert [row.sample_id for row in restored] == [row.sample_id for row in rows]
    assert restored[0].features == rows[0].features
    assert restored[0].trace.get("routing_schema_version") == 2
    assert len(await store.index_rows()) == 1


@pytest.mark.asyncio
async def test_replacing_a_session_with_nothing_removes_its_shard():
    backend = MemoryBackend()
    store = LearningStore(backend)
    rows = build_dataset([("s", ambient_record("m1", seed=1))])
    await store.replace_session("s", rows)
    assert SAMPLE_KEY_PREFIX + session_hash("s") in backend.data
    await store.replace_session("s", [])
    assert SAMPLE_KEY_PREFIX + session_hash("s") not in backend.data
    assert (await store.load_samples()) == []


@pytest.mark.asyncio
async def test_policies_are_versioned_and_immutable_by_copy():
    from astrbot_plugin_dynamics_learning.core.policy import candidate_from
    backend = MemoryBackend()
    store = LearningStore(backend)
    candidate = candidate_from({"topic_commit_threshold": 0.62})
    await store.append_policy(candidate)
    assert (await store.load_policies())[0].status == "candidate"
    updated = await store.update_policy_status(candidate.version, "accepted")
    assert updated is not None and updated.status == "accepted"
    assert (await store.load_policies())[0].status == "accepted"
    assert await store.update_policy_status("policy_v99", "accepted") is None
    assert POLICY_KEY in backend.data


@pytest.mark.asyncio
async def test_state_patch_is_bounded_and_survives_a_reload():
    backend = MemoryBackend()
    store = LearningStore(backend)
    await store.patch_state(last_analysis_at=123.0, note="x")
    state = await store.load_state()
    assert state["last_analysis_at"] == 123.0
    assert state["store_schema_version"] == 1
    await store.save_state({"bad": object()})
    assert "bad" not in await store.load_state()


@pytest.mark.asyncio
async def test_clear_samples_removes_every_shard():
    backend = MemoryBackend()
    store = LearningStore(backend)
    await store.replace_session("a", build_dataset([("a", ambient_record("m1", seed=1))]))
    await store.replace_session("b", build_dataset([("b", ambient_record("m2", seed=2))]))
    removed = await store.clear_samples()
    assert removed >= 0
    assert await store.load_samples() == []
    assert backend.data[SAMPLE_INDEX_KEY]["total"] == 0


@pytest.mark.asyncio
async def test_a_missing_host_sdk_is_a_diagnostic_not_an_exception():
    """The host SDK is optional and looked up by name, at call time.

    This exercises the real fallback rather than a monkeypatched one, because
    `astrbot.core` is deliberately absent from the SDK double in conftest: a
    companion plugin must never break the bot it observes just because the host
    is not there. Resolving the import by name is also what lets the module
    type-check identically whether or not the host is installed.
    """
    result = await collect_from_host("ysyhlly/astrbot_plugin_chat_dynamics")
    assert isinstance(result, IngestResult)
    assert result.diagnostics["source"] == "shared_preferences"
    host = sys.modules.get("astrbot")
    if not hasattr(host, "core"):
        assert result.diagnostics["available"] is False
        assert result.records == 0


@pytest.mark.asyncio
async def test_a_host_module_without_the_contract_is_reported_not_used():
    """Something that merely is not the host is refused instead of half-read."""
    result = await collect_from_host("ysyhlly/astrbot_plugin_chat_dynamics",
                                     sp_module=types.SimpleNamespace())
    assert result.records == 0
    assert result.diagnostics["available"] is False
    assert result.diagnostics["error"] == "astrbot shared preferences unavailable"
