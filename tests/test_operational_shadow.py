from __future__ import annotations

import time

import pytest

from astrbot_plugin_dynamics_learning.main import DynamicsLearningPlugin


@pytest.mark.asyncio
async def test_operational_reads_independent_host_key_without_creating_samples():
    plugin = DynamicsLearningPlugin(None, {})
    before = list(await plugin.load_samples())
    calls = []
    class Preferences:
        async def get_async(self, **kwargs):
            calls.append(kwargs)
            return {"schema_version": 1, "updated_at": time.time(),
                    "retention_seconds": 2592000, "max_records": 20000,
                    "observations": []}
    payload = await plugin.operational_shadow_payload(sp_module=Preferences())
    assert payload["source_status"] == "ok"
    assert payload["comparisons"] == 0
    assert calls[0]["key"] == "shadow_telemetry_v1"
    assert calls[0]["scope_id"] == plugin.runtime_config().source_plugin_id
    assert await plugin.load_samples() == before


@pytest.mark.asyncio
async def test_operational_read_failure_is_distinct_from_zero_comparisons():
    plugin = DynamicsLearningPlugin(None, {})
    class Preferences:
        async def get_async(self, **kwargs):
            raise OSError("offline")
    payload = await plugin.operational_shadow_payload(sp_module=Preferences())
    assert payload["source_status"] == "unavailable"
    assert payload["comparisons"] == 0


def test_real_host_telemetry_round_trip_preserves_denominator_and_buckets():
    from astrbot_plugin_chat_dynamics.core.learning_policy import shadow_decision
    from astrbot_plugin_chat_dynamics.core.shadow_telemetry import ShadowTelemetry
    from astrbot_plugin_dynamics_learning.core.shadow_coverage import evaluate_shadow_coverage

    store = ShadowTelemetry()
    for index, score in enumerate((0.6, 0.8)):
        row = shadow_decision(score=score, level="strong" if score >= .7 else "weak",
                              evidence_codes=[], has_prior_bot=True, baseline_threshold=.7,
                              params={"strong_addressivity_threshold": .5}, policy_id="p", now=100)
        store.record(row, session="private session", message_id=str(index), host_version="v1.8.0")
    report = evaluate_shadow_coverage(store.export(100), now=100)
    assert report["comparisons"] == 2
    assert report["disagreements"] == 1
    assert report["disagreement_rate"] == .5
    assert report["buckets"][0]["host_version"] == "v1.8.0"
    assert "private session" not in str(report)
