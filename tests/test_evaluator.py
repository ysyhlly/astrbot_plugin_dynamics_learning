"""v0.4 evaluation: the split, the gate and the verdicts."""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.evaluator import (
    VERDICT_ACCEPTED, VERDICT_INSUFFICIENT, VERDICT_REJECTED, evaluate_dataset,
    split_by_session,
)
from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY, PolicyCandidate
from astrbot_plugin_dynamics_learning.core.samples import build_dataset

from .factories import (
    annotated_sessions, strict_cut_sessions, topic_sessions,
)


def _recipient(rows):
    return build_dataset(rows)


def test_split_is_deterministic_balanced_and_session_level():
    samples = _recipient(annotated_sessions(sessions=10, per_session=5))
    first = split_by_session(samples, holdout_ratio=0.3)
    second = split_by_session(samples, holdout_ratio=0.3)
    assert [s.sample_id for s in first.holdout] == [s.sample_id for s in second.holdout]
    train_sessions = {s.session_hash for s in first.train}
    holdout_sessions = {s.session_hash for s in first.holdout}
    assert train_sessions.isdisjoint(holdout_sessions)
    assert 1 <= len(holdout_sessions) <= 8
    assert len(first.train) + len(first.holdout) == len(samples)


def test_a_single_session_cannot_be_split():
    samples = _recipient(annotated_sessions(sessions=1, per_session=5))
    split = split_by_session(samples, holdout_ratio=0.3)
    assert split.holdout == ()
    report = evaluate_dataset(samples)
    assert report.verdict == VERDICT_INSUFFICIENT
    assert any("无法切出留出集" in reason for reason in report.reasons)


def test_no_samples_is_insufficient_not_a_crash():
    report = evaluate_dataset([])
    assert report.verdict == VERDICT_INSUFFICIENT
    assert report.reasons


def test_small_holdout_is_insufficient_rather_than_a_guess():
    samples = _recipient(annotated_sessions(sessions=4, per_session=3))
    report = evaluate_dataset(samples, config=LearningConfig(min_samples_for_evaluation=1000))
    assert report.verdict == VERDICT_INSUFFICIENT
    assert any("留出集样本不足" in reason for reason in report.reasons)


def test_a_learnable_batch_produces_an_accepted_candidate():
    samples = _recipient(annotated_sessions(sessions=16, per_session=18, biased=True))
    report = evaluate_dataset(samples)
    assert report.verdict == VERDICT_ACCEPTED, report.reasons
    assert report.candidate is not None
    assert report.candidate.status == "validated"
    recipient = report.tasks["recipient"]
    assert recipient.deltas["accuracy"]["delta"] >= 0.02
    assert recipient.fitted["calibrated_on"] == "train"
    assert recipient.baseline["support"] == recipient.holdout


def test_the_learned_scorer_is_reported_but_never_gated():
    samples = _recipient(annotated_sessions(sessions=16, per_session=18, biased=True))
    report = evaluate_dataset(samples)
    recipient = report.tasks["recipient"]
    assert recipient.learned_scorer is not None
    assert recipient.learned_scorer["requires_host_support"] == "环境层评分替换"
    # It must never leak into the exportable candidate's parameters.
    params = report.candidate.params if report.candidate else {}
    assert "cut" not in params
    assert "requires_host_support" not in recipient.deltas


def test_an_unlearnable_batch_is_rejected_with_a_reason():
    samples = _recipient(annotated_sessions(sessions=16, per_session=18, biased=False))
    report = evaluate_dataset(samples, tasks=("recipient",))
    assert report.verdict in {VERDICT_REJECTED, VERDICT_INSUFFICIENT}
    assert report.reasons
    if report.verdict == VERDICT_REJECTED:
        assert any("门槛" in reason or "回退" in reason for reason in report.reasons)


def test_every_evaluated_task_reports_a_support_matching_the_holdout():
    samples = _recipient(annotated_sessions(sessions=14, per_session=16))
    report = evaluate_dataset(samples)
    for name, task in report.tasks.items():
        # Binary tasks report a confusion-matrix support; the topic task reports
        # within-session pairs, which is its own notion of exposure.
        if "support" in task.baseline:
            assert task.holdout == task.baseline["support"], name
            assert task.candidate["support"] == task.baseline["support"], name
        else:
            assert task.baseline["pairs"] > 0
            assert task.candidate["pairs"] == task.baseline["pairs"]
        assert task.primary_metric


def test_the_report_states_that_it_replays_a_decision_function():
    samples = _recipient(annotated_sessions(sessions=12, per_session=12))
    report = evaluate_dataset(samples)
    assert any("不是 ChatDynamics 路由器的完整重跑" in note for note in report.notes)
    assert any("按会话切分" in note for note in report.notes)
    payload = report.as_dict()
    assert payload["split"]["holdout_sessions"] >= 1
    assert payload["baseline_params"]["strong_addressivity_threshold"] == BASE_POLICY["strong_addressivity_threshold"]


def test_topic_thresholds_are_only_moved_when_the_training_split_supports_it():
    samples = build_dataset(topic_sessions(sessions=12, per_session=10))
    report = evaluate_dataset(samples)
    topic = report.tasks.get("topic")
    assert topic is not None
    if report.candidate is not None and "topic_commit_threshold" in report.candidate.params:
        moved = report.candidate.params["topic_commit_threshold"]
        assert abs(moved - BASE_POLICY["topic_commit_threshold"]) <= \
            BASE_POLICY["topic_commit_threshold"] * 0.05 + 1e-9


def test_the_candidate_is_bounded_even_when_the_sweep_wants_a_big_move():
    # This batch wants the recipient cut far below the host default, so the cap
    # is what actually decides the exported value.
    samples = _recipient(strict_cut_sessions(sessions=20, per_session=16))
    config = LearningConfig(max_param_delta_ratio=0.05)
    report = evaluate_dataset(samples, config=config)
    candidate = report.candidate
    assert candidate is not None
    for name, value in candidate.params.items():
        base = BASE_POLICY[name]
        assert value == pytest.approx(base, abs=abs(base) * 0.05 + 1e-9), name
    assert candidate.params["strong_addressivity_threshold"] == pytest.approx(0.665, abs=1e-9)


def test_a_batch_that_only_barely_moves_is_rejected_with_the_shortfall():
    samples = _recipient(strict_cut_sessions(sessions=24, per_session=20))
    report = evaluate_dataset(samples, tasks=("recipient",))
    assert report.verdict == VERDICT_REJECTED
    assert any("门槛" in reason or "回退" in reason for reason in report.reasons)


def test_evaluation_works_when_only_features_are_stored():
    """store_raw_trace=False must still replay, from the feature map alone."""
    from astrbot_plugin_dynamics_learning.core.samples import build_dataset as build

    rows = annotated_sessions(sessions=14, per_session=16, biased=True)
    samples = build(rows, config=LearningConfig(store_raw_trace=False))
    assert samples
    for sample in samples:
        # No contract snapshot: only the summary and the candidate list remain.
        assert "routing_schema_version" not in sample.trace
        assert not sample.trace.get("participation")
        assert sample.features.get("base_score") is not None
    report = evaluate_dataset(samples)
    assert report.tasks, "the feature-only path must still evaluate"
    recipient = report.tasks["recipient"]
    assert recipient.baseline["support"] == recipient.holdout
    assert report.split["holdout_samples"] > 0


def test_evaluation_is_reproducible_apart_from_the_timestamp():
    samples = _recipient(annotated_sessions(sessions=12, per_session=12))
    first = evaluate_dataset(samples, now=1.0).as_dict()
    second = evaluate_dataset(samples, now=1.0).as_dict()
    assert first == second
    assert PolicyCandidate.from_dict(first["candidate"]) is not None
