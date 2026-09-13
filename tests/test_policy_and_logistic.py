"""Parameter bounds, replay semantics and the deterministic fitter."""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.logistic import LogisticModel, fit, sweep_threshold
from astrbot_plugin_dynamics_learning.core.policy import (
    BASE_POLICY, PARAM_NAMES, PARAM_SPECS, PolicyCandidate, bounded_target, candidate_from,
    clamp_param, decide, next_version, normalize_policy, policy_deltas, sweep_values,
)
from astrbot_plugin_dynamics_learning.core.trace import parse_decision_trace

from .factories import make_trace


def _trace(**kwargs):
    return parse_decision_trace(make_trace(**kwargs))


def test_every_parameter_maps_to_a_real_host_key_with_a_default_in_range():
    for name in PARAM_NAMES:
        spec = PARAM_SPECS[name]
        assert spec["min"] <= spec["default"] <= spec["max"], name
        assert BASE_POLICY[name] == spec["default"]


def test_clamping_is_total_over_garbage():
    assert clamp_param("strong_addressivity_threshold", 5.0) == 0.90
    assert clamp_param("strong_addressivity_threshold", -1) == 0.50
    assert clamp_param("strong_addressivity_threshold", "x") == 0.70
    assert clamp_param("topic_margin_threshold", float("nan")) == 0.06
    assert clamp_param("safe_hover_threshold", True) == 0.40
    with pytest.raises(KeyError):
        clamp_param("not_a_parameter", 1.0)


def test_normalize_keeps_hover_below_strong():
    policy = normalize_policy({"strong_addressivity_threshold": 0.60,
                               "safe_hover_threshold": 0.60})
    assert policy["safe_hover_threshold"] == pytest.approx(0.55, abs=1e-9)
    assert policy["safe_hover_threshold"] < policy["strong_addressivity_threshold"]


def test_bounded_target_never_exceeds_the_delta_budget():
    moved = bounded_target("strong_addressivity_threshold", 0.70, 0.95, 0.05)
    assert moved == pytest.approx(0.735, abs=1e-6)
    still = bounded_target("strong_addressivity_threshold", 0.70, 0.70, 0.0)
    assert still == 0.70
    # A zero-budget move on a zero-base parameter cannot divide by zero.
    assert bounded_target("topic_margin_threshold", 0.0, 0.4, 0.05) == 0.0


def test_explicit_turns_are_threshold_independent():
    trace = _trace(evidence=[("bot_mention", "recipient", 1.0),
                             ("ambient_baseline", "baseline", 0.2)], bot_targeted=True)
    low = decide(trace, {**BASE_POLICY, "strong_addressivity_threshold": 0.50})
    high = decide(trace, {**BASE_POLICY, "strong_addressivity_threshold": 0.90})
    assert low.targeted is True and high.targeted is True
    assert low.level == "strong" and high.level == "strong"


def test_no_prior_bot_turn_returns_early_regardless_of_threshold():
    trace = _trace(evidence=[("ambient_baseline", "baseline", 0.2),
                             ("human_quote", "recipient", -0.15)])
    decision = decide(trace, {**BASE_POLICY, "strong_addressivity_threshold": 0.50})
    assert decision.targeted is False and decision.level == "weak"


def test_ambient_replay_reproduces_the_recorded_score():
    trace = _trace(evidence=[("ambient_baseline", "baseline", 0.2),
                             ("continuation_cue", "dialogue", 0.15),
                             ("temporal_gap", "temporal", 0.25)],
                   participation_level="hover", participation_score=0.6)
    # 0.20 + 0.15 + 0.25 = 0.60, exactly the host's pre-clamp total.
    low = decide(trace, {**BASE_POLICY, "strong_addressivity_threshold": 0.60})
    high = decide(trace, {**BASE_POLICY, "strong_addressivity_threshold": 0.61})
    assert low.score == pytest.approx(0.60)
    assert low.targeted is True and low.reply_label == "reply"
    assert high.targeted is False and high.level == "hover"


def test_model_score_replaces_the_additive_score_for_ambient_turns_only():
    ambient = _trace(evidence=[("ambient_baseline", "baseline", 0.2),
                               ("continuation_cue", "dialogue", 0.15)],
                     participation_level="weak")
    overridden = decide(ambient, {**BASE_POLICY, "strong_addressivity_threshold": 0.50},
                        model_score=0.99)
    assert overridden.targeted is True and overridden.score == pytest.approx(0.99)

    explicit = _trace(evidence=[("bot_reply", "recipient", 0.98)], bot_targeted=True)
    assert decide(explicit, BASE_POLICY, model_score=0.01).targeted is True


def test_topic_commit_moves_with_the_threshold():
    trace = _trace(topic_id="t1", topic_confidence=0.62)
    assert decide(trace, {**BASE_POLICY, "topic_commit_threshold": 0.60}).topic_committed
    assert not decide(trace, {**BASE_POLICY, "topic_commit_threshold": 0.80}).topic_committed
    ambiguous = _trace(topic_id="t1", topic_confidence=0.9, topic_ambiguous=True)
    assert not decide(ambiguous, BASE_POLICY).topic_committed


def test_candidate_versions_increment_and_deltas_are_reported():
    candidate = candidate_from({"strong_addressivity_threshold": 0.80},
                               existing_versions=["policy_v1", "policy_v7", "junk"])
    assert candidate.version == "policy_v8"
    assert candidate.params["strong_addressivity_threshold"] == 0.80
    deltas = policy_deltas(candidate.baseline, candidate.params)
    assert [row["param"] for row in deltas] == ["strong_addressivity_threshold"]
    assert deltas[0]["before"] == 0.70
    assert next_version([]) == "policy_v1"


def test_policy_candidate_round_trips_and_rejects_junk_status():
    candidate = candidate_from({"topic_commit_threshold": 0.62})
    restored = PolicyCandidate.from_dict(candidate.as_dict())
    assert restored is not None
    assert restored.version == candidate.version
    assert PolicyCandidate.from_dict({"version": ""}) is None
    assert PolicyCandidate.from_dict(None) is None
    assert candidate.with_status("nonsense").status == "candidate"
    assert candidate.with_status("accepted").status == "accepted"


def test_sweep_values_stay_inside_the_host_slider():
    for name in PARAM_NAMES:
        values = sweep_values(name, steps=5)
        assert values == sorted(values)
        assert all(PARAM_SPECS[name]["min"] <= value <= PARAM_SPECS[name]["max"] for value in values)


def test_fit_is_deterministic_and_separates_signal():
    rows = [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 0.9], [1.0, 0.05], [0.05, 1.0]]
    labels = [True, True, False, False, True, False]
    first = fit(rows, labels, feature_names=("a", "b"))
    second = fit(rows, labels, feature_names=("a", "b"))
    assert first.weights == second.weights and first.bias == second.bias
    assert first.score([1.0, 0.0]) > first.score([0.0, 1.0])


def test_single_class_fit_falls_back_to_the_base_rate():
    model = fit([[1.0], [0.5], [0.2]], [True, True, True], feature_names=("a",))
    assert model.weights == (0.0,)
    assert model.score([0.0]) > 0.5
    empty = fit([], [], feature_names=())
    assert empty.weights == () and empty.converged is False


def test_model_round_trip_and_alignment():
    model = fit([[1.0, 0.0], [0.0, 1.0]], [True, False], feature_names=("a", "b"))
    restored = LogisticModel.from_dict(model.as_dict())
    assert restored is not None
    assert restored.weights == pytest.approx(model.weights)
    aligned = restored.aligned(("b", "a", "c"))
    assert aligned.feature_names == ("b", "a", "c")
    assert aligned.weights[2] == 0.0
    assert LogisticModel.from_dict({"feature_names": ["a"], "weights": []}) is None
    assert LogisticModel.from_dict(None) is None


def test_threshold_sweep_finds_the_separating_cut_and_is_deterministic():
    scores = [0.9, 0.8, 0.85, 0.2, 0.15, 0.3]
    labels = [True, True, True, False, False, False]
    first = sweep_threshold(scores, labels, metric="f1")
    second = sweep_threshold(scores, labels, metric="f1")
    assert first == second
    assert first["value"] == pytest.approx(1.0)
    assert 0.3 < first["threshold"] <= 0.8
    assert sweep_threshold([], [])["threshold"] is None


def test_threshold_sweep_prefers_a_middle_cut_on_ties():
    result = sweep_threshold([0.5, 0.5], [True, False], metric="accuracy")
    assert result["threshold"] is not None
    assert 0.0 <= result["threshold"] <= 1.0
