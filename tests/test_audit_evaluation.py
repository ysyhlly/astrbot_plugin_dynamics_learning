import pytest
from astrbot_plugin_dynamics_learning.core.evaluator import score_task, _units
from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY, PolicyCandidate, published_policy
from astrbot_plugin_dynamics_learning.core.logistic import LogisticModel
from astrbot_plugin_dynamics_learning.core.samples import build_dataset, TASK_RECIPIENT, TASK_REPLY_ADMISSION
from astrbot_plugin_dynamics_learning.core.trace import DecisionTrace
from .factories import make_trace, make_record

@pytest.mark.parametrize('task', [TASK_RECIPIENT, TASK_REPLY_ADMISSION])
def test_missing_additive_score_is_excluded_from_metrics_and_bootstrap(task):
    trace = make_trace(evidence=[('pending_hover', 'dialogue', .8)], bot_targeted=True,
                       participation_level='strong', participation_score=.8)
    trace['participation']['contribution_total'] = None
    rows = [s for s in build_dataset([('group', make_record('a', trace=trace, bot_targeted=True,
                                                          expected_reply=True))]) if s.task == task]
    result = score_task(rows, BASE_POLICY, task)
    assert result.support == 0
    assert result.unreplayable == 1
    assert _units(rows, task, BASE_POLICY, BASE_POLICY) == []

def test_legacy_duplicate_coefficients_preserve_score_when_aligned():
    model = LogisticModel(weights=(.3, .7, -.2), bias=.1, feature_names=('x', 'x', 'y'))
    assert model.aligned(('x', 'y')).score((.8, .2)) == pytest.approx(model.score((.8, .8, .2)))

def test_empty_trace_does_not_claim_prior_bot():
    assert DecisionTrace().prior_bot_proxy == -1

def test_missing_admission_does_not_become_silent_sample():
    rows = build_dataset([('group', make_record('a', trace=make_trace(), expected_reply=True))])
    assert not any(s.task == TASK_REPLY_ADMISSION for s in rows)

def test_forward_evaluation_is_not_shadow_observation():
    candidate = PolicyCandidate(version='v1', params={}, forward_result={'accepted': True})
    assert published_policy(candidate)['shadow_observed'] is False


def test_shadow_stage_alone_is_not_observation():
    candidate = PolicyCandidate(version="v1", params={}).with_status("shadow")
    payload = published_policy(candidate)
    assert payload["shadow_entered"] is True
    assert payload["shadow_observed"] is False


def test_score_without_context_does_not_invent_early_return():
    trace = make_trace(bot_targeted=True, participation_level="strong", participation_score=.9)
    trace["participation"]["contribution_total"] = .9
    rows = [s for s in build_dataset([("group", make_record("a", trace=trace, bot_targeted=True))])
            if s.task == TASK_RECIPIENT]
    result = score_task(rows, BASE_POLICY, TASK_RECIPIENT)
    assert result.support == 0
    assert result.unreplayable == 1
    assert _units(rows, TASK_RECIPIENT, BASE_POLICY, BASE_POLICY) == []


def test_unreplayable_rows_do_not_satisfy_evaluation_minimum():
    from astrbot_plugin_dynamics_learning.core.config import LearningConfig
    from astrbot_plugin_dynamics_learning.core.evaluator import (
        EvaluationReport, TaskEvaluation, _verdict, VERDICT_INSUFFICIENT,
    )
    report = EvaluationReport(tasks={TASK_RECIPIENT: TaskEvaluation(
        task=TASK_RECIPIENT, holdout=100, unreplayable=99)})
    _verdict(report, LearningConfig(min_samples_for_evaluation=10))
    assert report.verdict == VERDICT_INSUFFICIENT
    assert "recipient(1)" in report.reasons[0]
