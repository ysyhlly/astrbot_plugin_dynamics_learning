"""The iterative tuner: three trust tiers, small steps, and the stops.

Each branch is driven by a controlled score ladder whose per-session composition
is fixed, so the holdout accuracy moves by exactly `recovered / plan_size` and
the assertions can be arithmetic rather than approximate.
"""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.autotune import (
    DECISION_CANDIDATE, DECISION_INSUFFICIENT, DECISION_NEEDS_REVIEW, DECISION_NO_CHANGE,
    DECISION_PROMOTE, DECISION_REJECT, DECISION_ROLLBACK, DECISION_STRONG_PROMOTE,
    TuneRules, error_improved, run_all, run_tuning,
)
from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.metrics import ErrorRate
from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY
from astrbot_plugin_dynamics_learning.core.samples import build_dataset

from .factories import (
    PROMOTE_PER_SESSION, guard_ladder, ladder_sessions,
    promote_ladder, stall_ladder,
)

CONFIG = LearningConfig()


def _ladder(plan, sessions=5):
    return build_dataset(ladder_sessions(plan, sessions=sessions))


def test_the_rule_defaults_match_the_agreed_budget():
    rules = TuneRules()
    assert rules.step_delta_ratio == 0.05
    assert rules.cumulative_delta_ratio == 0.15
    assert rules.max_steps == 3
    # 0.95 ** 3 = -14.3%, so the step cap and the cumulative cap agree.
    assert (1 - 0.95 ** rules.max_steps) <= rules.cumulative_delta_ratio
    assert (1 - 0.95 ** (rules.max_steps + 1)) > rules.cumulative_delta_ratio
    assert rules.promote_improvement == 0.010
    assert rules.promote_error_relative == 0.10
    assert rules.strong_improvement == 0.020
    assert rules.safe_floor == -0.002
    assert rules.advance_improvement == 0.005
    assert rules.marginal_stall == 0.002


def test_a_single_small_step_does_not_promote_on_its_own():
    """The whole point: +0.8% is not enough to promote, but it is enough to go on."""
    run = run_tuning(_ladder(promote_ladder()), task="recipient", config=CONFIG)
    first = run.steps[0]
    assert first.safe is True
    assert first.advanced is True
    assert first.cumulative_delta == pytest.approx(2 / PROMOTE_PER_SESSION, abs=1e-4)
    assert first.decision == "continue"
    # Neither promote branch fired at step 1: the global move is under +1%, and
    # the target error did not fall 10% relative either.
    assert first.cumulative_delta < TuneRules().promote_improvement
    assert first.target_error_cumulative is not None
    assert first.target_error_cumulative > -TuneRules().promote_error_relative


def test_two_small_steps_promote_together():
    run = run_tuning(_ladder(promote_ladder()), task="recipient", config=CONFIG)
    assert run.decision == DECISION_PROMOTE, run.reasons
    assert run.adopted_steps == 2
    assert len(run.steps) == 2
    # Each step spends exactly the ±5% budget, and the total stays under ±15%.
    for step in run.steps:
        assert step.policy["strong_addressivity_threshold"] < 0.70
    # 0.70 x 0.95 x 0.95, to the four decimals a parameter is stored at.
    assert run.steps[0].policy["strong_addressivity_threshold"] == pytest.approx(0.665)
    assert run.steps[1].policy["strong_addressivity_threshold"] == pytest.approx(0.6318, abs=1e-4)
    cumulative = run.steps[-1].cumulative_delta
    assert cumulative >= TuneRules().promote_improvement
    assert cumulative < TuneRules().strong_improvement
    assert run.candidate is not None
    assert run.candidate.status == "validated"
    assert run.high_confidence is False


def test_each_step_is_judged_on_the_error_it_was_aimed_at():
    run = run_tuning(_ladder(promote_ladder()), task="recipient", config=CONFIG)
    for step in run.steps:
        # Lowering the addressee cut can only intend to stop missing the bot.
        assert step.target_error == "missed_bot"
        assert step.collateral["missed_bot"]["is_target"] is True
        assert step.collateral["missed_bot"]["improved"] is True
        assert step.collateral["false_bot"]["is_target"] is False


def test_a_two_percent_gain_is_flagged_high_confidence():
    # The ladder's first step earns +0.8%; setting the strong tier just under
    # that exercises the Strong promote branch without a second fixture.
    run = run_tuning(_ladder(promote_ladder()), task="recipient", config=CONFIG,
                     rules=TuneRules(strong_improvement=0.006, promote_improvement=0.002))
    assert run.decision == DECISION_STRONG_PROMOTE
    assert run.high_confidence is True
    assert run.adopted_steps == 1
    assert run.candidate is not None
    assert run.candidate.status == "validated"
    assert run.as_dict()["decision_label"] == "强采纳（高置信度）"


def test_a_plateau_stops_after_two_weak_steps_and_keeps_the_candidate():
    run = run_tuning(_ladder(stall_ladder()), task="recipient", config=CONFIG)
    assert run.decision == DECISION_CANDIDATE
    assert run.adopted_steps == 2
    assert "停止" in run.stop_reason
    for step in run.steps:
        assert step.safe is True
        assert step.step_delta is not None
        assert step.step_delta < TuneRules().marginal_stall
    assert run.candidate is not None
    assert run.candidate.status == "proposed"


def test_a_guard_break_rolls_the_whole_run_back():
    run = run_tuning(_ladder(guard_ladder()), task="recipient", config=CONFIG)
    assert run.decision == DECISION_ROLLBACK
    assert run.adopted_steps == 0
    assert run.final_policy == run.baseline
    assert run.steps[0].guard_failures
    # Accuracy barely moves; it is F1 that catches the precision collapse.
    assert any("f1" in failure for failure in run.steps[0].guard_failures)
    assert abs(run.steps[0].cumulative_delta) < TuneRules().guard_regression
    assert run.candidate is None


def test_a_step_that_does_not_improve_the_target_error_is_rejected():
    """No measurable gain, no promotion, and the target error has to move."""
    flat = [(0.95, True)] * 200 + [(0.30, True)] * 50  # the misses are out of reach
    run = run_tuning(_ladder(flat), task="recipient", config=CONFIG)
    assert run.decision == DECISION_REJECT
    assert run.steps[0].safe is False
    assert run.candidate is None


def test_hitting_the_cumulative_cap_asks_a_human():
    # A 4% cumulative budget is spent by the first step, while the run is still
    # improving — exactly the case that must not continue on its own.
    rules = TuneRules(cumulative_delta_ratio=0.04)
    run = run_tuning(_ladder(promote_ladder()), task="recipient", config=CONFIG, rules=rules)
    assert run.decision == DECISION_NEEDS_REVIEW
    assert run.steps[0].clamped == ["strong_addressivity_threshold"]
    assert "人工确认" in run.stop_reason
    assert any("人工确认" in reason for reason in run.steps[0].reasons)
    drift = run.steps[0].drift[0]
    assert abs(drift["delta_ratio"]) <= 0.04 + 1e-9


def test_the_cumulative_cap_is_never_exceeded_even_over_three_steps():
    rules = TuneRules(promote_improvement=0.99, promote_error_relative=0.99,
                      strong_improvement=0.99, advance_improvement=0.0,
                      advance_error_relative=0.99)
    run = run_tuning(_ladder(promote_ladder()), task="recipient", config=CONFIG, rules=rules)
    assert run.final_policy is not None
    for row in run.as_dict()["drift"]:
        assert abs(row["delta_ratio"]) <= rules.cumulative_delta_ratio + 1e-9
    assert len(run.steps) <= rules.max_steps


def test_a_task_without_its_own_parameter_is_not_tuned():
    run = run_tuning(_ladder(promote_ladder()), task="reply", config=CONFIG)
    assert run.decision == DECISION_NO_CHANGE
    assert run.steps == []


def test_too_few_samples_is_reported_not_guessed():
    single = build_dataset(ladder_sessions(promote_ladder(), sessions=1))
    tiny = run_tuning(single, task="recipient",
                      config=LearningConfig(min_samples_for_evaluation=10_000))
    assert tiny.decision == DECISION_INSUFFICIENT
    assert "留出集" in tiny.stop_reason
    assert tiny.candidate is None


def test_a_run_that_moves_a_parameter_without_moving_a_metric_earns_nothing():
    """Every turn is already right, so any threshold move is a no-op."""
    flat = [(0.95, True)] * 250
    run = run_tuning(_ladder(flat), task="recipient", config=CONFIG)
    assert run.decision == DECISION_NO_CHANGE
    assert run.final_policy is None
    assert run.candidate is None
    assert "没有任何指标变化" in run.stop_reason
    assert any("没有可保留的收益" in reason for reason in run.reasons)


def test_the_tuner_never_touches_the_other_tasks_parameters():
    run = run_tuning(_ladder(promote_ladder()), task="recipient", config=CONFIG)
    assert run.final_policy is not None
    assert run.final_policy["topic_commit_threshold"] == BASE_POLICY["topic_commit_threshold"]
    assert run.final_policy["topic_join_threshold"] == BASE_POLICY["topic_join_threshold"]


def test_every_run_reports_that_nothing_is_applied_automatically():
    run = run_tuning(_ladder(promote_ladder()), task="recipient", config=CONFIG)
    payload = run.as_dict()
    assert "不会自动修改" in payload["note"]
    assert payload["rules"]["step_delta_ratio"] == 0.05
    assert payload["decision_label"] == "采纳"


def test_run_all_gives_each_task_its_own_verdict_and_version():
    samples = _ladder(promote_ladder())
    runs = run_all(samples, config=CONFIG)
    assert [row.task for row in runs] == ["recipient", "topic"]
    versions = [row.candidate.version for row in runs if row.candidate]
    assert len(versions) == len(set(versions))


def test_error_improvement_handles_a_rate_that_is_already_zero():
    zero = ErrorRate("missed_bot", 0, 100)
    assert error_improved(zero, ErrorRate("missed_bot", 0, 100)) is True
    assert error_improved(zero, ErrorRate("missed_bot", 1, 100)) is False
    worse = ErrorRate("missed_bot", 10, 100)
    assert error_improved(worse, ErrorRate("missed_bot", 5, 100)) is True
    assert error_improved(worse, ErrorRate("missed_bot", 20, 100)) is False
