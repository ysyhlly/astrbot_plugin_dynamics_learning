from dataclasses import fields
import pytest
from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.outcome import parse_outcome, parse_record_outcome
from astrbot_plugin_dynamics_learning.core.ingest import parse_preferences
from astrbot_plugin_dynamics_learning.core.samples import session_hash
from astrbot_plugin_dynamics_learning.core.window import window_payload


def test_record_fallback_keeps_provenance_and_explicit_stage():
    raw = {"outcome": {"value": "suppressed", "stage": "gate", "reason": "new_gate"}}
    outcome = parse_outcome({}, raw)
    assert outcome.source == "record"
    assert outcome.stage == "gate"
    assert parse_record_outcome(raw).source == "record"
    assert parse_outcome(raw).source == "trace"


def test_future_runtime_version_keeps_structurally_valid_session_mapping():
    rows = [{"key": "panel_runtime_v1", "value": {"version": 2, "sessions": [{"session_key": "s"}]}},
            {"key": "topic_annotations_v1_" + session_hash("s"), "value": [{"msg_id": "m"}]}]
    result = parse_preferences(rows)
    assert result.records == 1
    assert result.diagnostics["runtime_version"] == 2
    assert result.diagnostics["runtime_version_supported"] is False


def test_absent_limits_do_not_invent_a_deadline():
    runtime = {"saved_wall": 2000, "saved_clock": 1000, "sessions": [
        {"session_key": "s", "nodes": [{"msg_id": "m", "timestamp": 900}]}]}
    row = window_payload(runtime, {}, now_wall=2000)["sessions"][0]
    assert row["expires_in_seconds"] is None
    assert row["cap_pressure"] is None


def test_effective_config_exports_every_resolved_field():
    cfg = LearningConfig()
    assert set(cfg.as_dict()) == {field.name for field in fields(cfg)}


def test_host_shadow_trace_preserves_observation_time():
    host = pytest.importorskip("astrbot_plugin_chat_dynamics.core.routing_trace")
    block = host._shadow_block({"policy_id": "p", "recorded_at": 1800000000.0})
    assert block["recorded_at"] == 1800000000.0
