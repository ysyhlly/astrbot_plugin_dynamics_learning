"""Contract parsing and the frozen feature vector."""
from __future__ import annotations

from astrbot_plugin_dynamics_learning.core.features import FEATURE_NAMES, build_features, feature_summary, vector
from astrbot_plugin_dynamics_learning.core.trace import (
    DecisionTrace, EXPLICIT_CODES, parse_decision_trace, parse_legacy_routing,
    trace_from_sample_record,
)

from .factories import make_record, make_trace


def test_parses_the_host_schema_two_shape():
    trace = parse_decision_trace(make_trace(
        evidence=[("ambient_baseline", "baseline", 0.2), ("continuation_cue", "dialogue", 0.15)],
        bot_targeted=True, topic_id="t1", topic_confidence=0.66,
        participation_score=0.35, participation_level="hover",
        recipient_confidence=0.8,
    ))
    assert trace.trace_schema_version == 2
    assert trace.topic_id == "t1"
    assert trace.topic_confidence == 0.66
    assert trace.bot_targeted is True
    assert trace.participation_level == "hover"
    assert trace.codes == frozenset({"ambient_baseline", "continuation_cue"})
    assert trace.contribution_total == 0.35
    assert trace.degraded is False


def test_malformed_input_degrades_without_raising():
    for value in (None, [], "nope", 7, {"topic": 3, "recipient": "x"}):
        trace = parse_decision_trace(value)
        assert isinstance(trace, DecisionTrace)
    assert parse_decision_trace(None).degraded is True
    # A missing schema version is tolerated: old records simply omit it.
    assert parse_decision_trace({"topic": {"topic_id": "t"}}).degraded is False
    # An unknown schema version is flagged rather than silently accepted.
    assert parse_decision_trace({"routing_schema_version": 99}).degraded is True


def test_unknown_evidence_family_is_kept_and_flagged():
    trace = parse_decision_trace(make_trace(evidence=[("brand_new_code", "brand_new_family", 0.3)]))
    assert [item.code for item in trace.evidence] == ["brand_new_code"]
    assert trace.degraded is True


def test_round_trips_through_the_storage_contract():
    original = parse_decision_trace(make_trace(
        evidence=[("temporal_gap", "temporal", 0.25), ("pending_hover", "dialogue", 0.2)],
        bot_targeted=True, topic_id="t9", topic_confidence=0.71, topic_ambiguous=False,
        participation_score=0.65, participation_level="hover", recipient_confidence=0.55,
        state={"intervening_users": 2, "pending_hover": True},
        identity={"bot_reference": "mention", "mention": True, "vocative": False, "subject": False},
    ))
    restored = parse_decision_trace(original.to_contract())
    assert restored.bot_targeted is True
    assert restored.topic_id == "t9"
    assert restored.codes == original.codes
    assert restored.contribution_total == original.contribution_total
    assert restored.state["intervening_users"] == 2
    assert restored.identity["mention"] is True
    assert restored.participation_level == "hover"


def test_prior_bot_proxy_matches_the_host_early_return_signature():
    early = parse_decision_trace(make_trace(evidence=[("ambient_baseline", "baseline", 0.2)]))
    assert early.prior_bot_proxy == 0
    with_quote = parse_decision_trace(make_trace(
        evidence=[("ambient_baseline", "baseline", 0.2), ("human_quote", "recipient", -0.15)]))
    assert with_quote.prior_bot_proxy == 0
    scored = parse_decision_trace(make_trace(
        evidence=[("ambient_baseline", "baseline", 0.2), ("continuation_cue", "dialogue", 0.15)]))
    assert scored.prior_bot_proxy == 1


def test_explicit_detection_only_uses_structural_codes():
    explicit = parse_decision_trace(make_trace(
        evidence=[("bot_mention", "recipient", 1.0), ("ambient_baseline", "baseline", 0.2)]))
    assert explicit.is_explicit is True
    assert explicit.explicit_code == "bot_mention"
    ambient = parse_decision_trace(make_trace(
        evidence=[("ambient_baseline", "baseline", 0.2), ("platform_wake", "platform", 0.15)]))
    assert ambient.is_explicit is False
    assert EXPLICIT_CODES.isdisjoint({"ambient_baseline", "platform_wake"})


def test_legacy_routing_mapping_is_flagged_degraded():
    trace = parse_legacy_routing({"topic_id": "t1", "topic_confidence": 0.5,
                                  "addressee_ids": ["bot"], "bot_is_addressee": True})
    assert trace.degraded is True
    assert trace.topic_id == "t1"
    assert parse_legacy_routing(None).degraded is True


def test_trace_from_record_prefers_schema_two():
    record = make_record("m1", trace=make_trace(topic_id="t2", topic_confidence=0.4))
    assert trace_from_sample_record(record).topic_id == "t2"
    legacy = trace_from_sample_record({"routing": {"topic_id": "old"}})
    assert legacy.topic_id == "old" and legacy.degraded is True


def test_feature_vector_is_ordered_dense_and_finite():
    trace = parse_decision_trace(make_trace(
        evidence=[("temporal_gap", "temporal", 0.25), ("lexical_overlap", "topic", 0.15)],
        topic_id="t1", topic_confidence=0.5, participation_level="hover"))
    features = build_features(trace)
    values = vector(features)
    assert len(values) == len(FEATURE_NAMES)
    assert all(isinstance(item, float) for item in values)
    assert features["ev_temporal_gap"] == 1.0
    assert features["st_temporal_gap"] == 0.25
    assert features["ev_pending_hover"] == 0.0
    # Diagnostic-only keys never enter the model vector.
    assert "base_score" in features and "ctx_bot_targeted" in features
    assert "base_score" not in FEATURE_NAMES
    assert "ctx_bot_targeted" not in FEATURE_NAMES


def test_feature_summary_exposes_no_identifiers():
    trace = parse_decision_trace(make_trace(
        evidence=[("bot_reply", "recipient", 0.98), ("ambient_baseline", "baseline", 0.2)],
        bot_targeted=True))
    summary = feature_summary(trace)
    assert summary["explicit_code"] == "bot_reply"
    assert summary["codes"] == ["ambient_baseline", "bot_reply"]
    assert "ids" not in summary
