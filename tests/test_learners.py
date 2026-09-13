"""v0.2 recipient learning and v0.3 topic learning."""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY, PARAM_SPECS
from astrbot_plugin_dynamics_learning.core.recipient_learner import learn as learn_recipient
from astrbot_plugin_dynamics_learning.core.recommendation import KIND_CONFIG_PARAM, KIND_DIAGNOSTIC
from astrbot_plugin_dynamics_learning.core.samples import TASK_TOPIC, build_dataset
from astrbot_plugin_dynamics_learning.core.topic_learner import learn as learn_topic, replay_label, replay_pairs, _rows

from .factories import ambient_record, annotated_sessions, make_record, make_trace, topic_record


def _recipient_samples(rows):
    return [s for s in build_dataset(rows) if s.task == "recipient"]


def _topic_row(session, record):
    """One replayed topic row, built through the real sample pipeline."""
    samples = [s for s in build_dataset([(session, record)]) if s.task == TASK_TOPIC]
    assert samples, "record produced no topic sample"
    return _rows(samples)[0]


def test_recipient_learner_reports_statistics_without_claiming_accuracy():
    report = learn_recipient(_recipient_samples(annotated_sessions(sessions=6, per_session=10)))
    assert report.samples > 0
    assert report.accuracy["note"].startswith("仅统计人工标注样本")
    assert report.confusion["support"] == report.accuracy["total"]
    assert report.positive_rate is not None
    assert any("不代表真实准确率" in note for note in report.notes)


def test_recipient_learner_separates_structural_turns_from_the_fitted_layer():
    rows = annotated_sessions(sessions=4, per_session=8)
    explicit = make_record(
        "explicit-1",
        trace=make_trace(evidence=[("bot_mention", "recipient", 1.0),
                                   ("ambient_baseline", "baseline", 0.2)],
                         bot_targeted=True),
        bot_targeted=True)
    samples = _recipient_samples(rows + [("umo:group:0", explicit)])
    report = learn_recipient(samples)
    assert report.explicit_samples == 1
    assert (report.ambient_samples + report.explicit_samples
            + report.early_return_samples) == report.samples
    assert report.ambient_samples == report.samples - 1 - report.early_return_samples
    assert any("结构化" in note for note in report.notes)


def test_recipient_learner_reports_insufficient_samples_instead_of_a_number():
    rows = [("s", ambient_record(f"m{index}", seed=index)) for index in range(5)]
    report = learn_recipient(_recipient_samples(rows), config=LearningConfig(min_samples_for_recommendation=100))
    assert report.recommendations == []
    assert any("低于建议阈值" in note for note in report.notes)
    assert report.evidence_lift is not None


def test_evidence_lift_is_bayesian_smoothed_and_skips_thin_codes():
    rows = [("s", ambient_record(f"m{index}", seed=index)) for index in range(120)]
    report = learn_recipient(_recipient_samples(rows))
    assert report.evidence_lift, "a 120-sample batch must expose at least one code"
    for row in report.evidence_lift:
        assert row.present >= 5
        # Never exactly 0 or 1, which is what the Beta prior buys.
        assert 0.0 < row.smoothed_rate < 1.0
        assert row.positive <= row.present
    # Every lift shares one base rate, so differences between lifts must equal
    # differences between the smoothed rates exactly.
    first = report.evidence_lift[0]
    for row in report.evidence_lift[1:]:
        assert row.lift - first.lift == pytest.approx(
            row.smoothed_rate - first.smoothed_rate, abs=1e-12)


def test_recipient_recommendation_targets_a_real_host_key_and_respects_the_cap():
    rows = [("s", ambient_record(f"m{index}", seed=index)) for index in range(200)]
    report = learn_recipient(_recipient_samples(rows))
    proposals = [row for row in report.recommendations if row.kind == KIND_CONFIG_PARAM]
    assert proposals, "expected a threshold proposal on a separable batch"
    proposal = proposals[0]
    assert proposal.param in PARAM_SPECS
    base = BASE_POLICY[proposal.param]
    assert abs(proposal.after - base) <= abs(base) * 0.05 + 1e-9


def test_recipient_learner_is_deterministic():
    samples = _recipient_samples(annotated_sessions(sessions=5, per_session=8))
    first = learn_recipient(samples).as_dict()
    second = learn_recipient(samples).as_dict()
    assert first == second


def test_empty_input_yields_an_explicit_empty_report():
    report = learn_recipient([])
    assert report.samples == 0
    assert report.recommendations == []
    assert report.notes
    assert report.as_dict()["task"] == "recipient"


# ---- topic ---------------------------------------------------------------


def test_topic_replay_only_removes_or_relaxes_recorded_assignments():
    committed_record = topic_record("m1", predicted="t1", expected="t1", confidence=0.7)
    committed = _topic_row("s", committed_record)
    assert replay_label(committed, 0.6) == "t1"
    assert replay_label(committed, 0.9) == ""

    unassigned_record = topic_record(
        "m2", predicted="UNKNOWN", expected="t1", confidence=0.3,
        candidates=[[0.55, "t1"], [0.4, "t2"]])
    unassigned = _topic_row("s", unassigned_record)
    assert replay_label(unassigned, 0.5) == "t1"
    assert replay_label(unassigned, 0.6) == ""
    assert replay_label(unassigned, 0.5, allow_relax=False) == ""


def test_topic_learner_flags_fragmentation_as_a_direction_not_a_parameter():
    rows = []
    for index in range(60):
        # Truth says these belong together; the host left them unassigned.
        rows.append((f"umo:g{index // 10}", topic_record(
            f"a{index}", predicted="UNKNOWN", expected="same", confidence=0.3)))
        rows.append((f"umo:g{index // 10}", topic_record(
            f"b{index}", predicted="UNKNOWN", expected="same", confidence=0.32)))
    report = learn_topic([s for s in build_dataset(rows) if s.task == TASK_TOPIC])
    assert report.pair_metrics["fragmentation"] > 0
    assert report.pair_metrics["wrong_merge"] == 0
    directions = [row for row in report.recommendations if row.kind == KIND_DIAGNOSTIC]
    assert any("过度拆分" in row.title for row in directions)
    assert not [row for row in report.recommendations if row.kind == KIND_CONFIG_PARAM], \
        "a relaxation the data cannot replay must not become a parameter proposal"
    assert any("topic_candidates" in str(row.evidence) for row in report.recommendations)


def test_topic_learner_proposes_tightening_when_merges_dominate():
    rows = []
    for index in range(40):
        session = f"umo:g{index // 10}"
        rows.append((session, topic_record(f"a{index}", predicted="t1", expected="t1",
                                           confidence=0.62)))
        rows.append((session, topic_record(f"b{index}", predicted="t1", expected="t2",
                                           confidence=0.61)))
    report = learn_topic([s for s in build_dataset(rows) if s.task == TASK_TOPIC])
    assert report.pair_metrics["wrong_merge"] > 0
    proposals = [row for row in report.recommendations if row.kind == KIND_CONFIG_PARAM]
    assert proposals and proposals[0].param == "topic_commit_threshold"
    assert proposals[0].after >= BASE_POLICY["topic_commit_threshold"] - 1e-9


def test_topic_learner_reports_balance_without_moving_a_parameter():
    rows = []
    for index in range(30):
        rows.append((f"umo:g{index // 6}", topic_record(f"a{index}", predicted="t1",
                                                        expected="t1", confidence=0.8)))
    report = learn_topic([s for s in build_dataset(rows) if s.task == TASK_TOPIC])
    assert report.pair_metrics["wrong_merge"] == 0
    assert report.pair_metrics["fragmentation"] == 0
    assert not [row for row in report.recommendations if row.kind == KIND_CONFIG_PARAM]


def test_topic_learner_handles_an_empty_batch():
    report = learn_topic([])
    assert report.samples == 0
    assert report.notes
    assert replay_pairs([], 0.5) == []
