"""Gate responsibilities follow changed parameters and the actual replay labels."""
from astrbot_plugin_dynamics_learning.core.evaluation_plan import evaluation_plan
from astrbot_plugin_dynamics_learning.core.evaluator import (
    EvaluationReport, TaskEvaluation, _verdict, promotion_check,
)
from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY
from astrbot_plugin_dynamics_learning.core.samples import (
    TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY_ADMISSION,
)


def row(task, delta, support=100):
    metric = "pair_accuracy" if task == TASK_TOPIC else "accuracy"
    return TaskEvaluation(task=task, holdout=support, primary_metric=metric,
                          deltas={metric: {"delta": delta}, "f1": {"delta": delta}})


def report_for(changes):
    plan = evaluation_plan({**BASE_POLICY, **changes}, BASE_POLICY)
    return EvaluationReport(plan=plan.as_dict())


def test_topic_change_ignores_unaffected_recipient_support_and_gain():
    report = report_for({"topic_commit_threshold": BASE_POLICY["topic_commit_threshold"] - .05})
    report.tasks = {TASK_TOPIC: row(TASK_TOPIC, .1), TASK_RECIPIENT: row(TASK_RECIPIENT, -.5, 1)}
    _verdict(report, LearningConfig())
    assert report.verdict == "accepted"


def test_recipient_change_requires_reply_admission_evidence():
    report = report_for({"strong_addressivity_threshold": BASE_POLICY["strong_addressivity_threshold"] + .05})
    report.tasks = {TASK_RECIPIENT: row(TASK_RECIPIENT, .1)}
    _verdict(report, LearningConfig())
    assert report.verdict == "insufficient"
    report.tasks[TASK_REPLY_ADMISSION] = row(TASK_REPLY_ADMISSION, -.5)
    _verdict(report, LearningConfig())
    assert report.verdict == "rejected"


def test_hover_has_no_replayable_label_effect():
    plan = evaluation_plan({**BASE_POLICY, "safe_hover_threshold": BASE_POLICY["safe_hover_threshold"] -.05}, BASE_POLICY)
    assert plan.target_tasks == ()
    assert plan.unsupported_params == ("safe_hover_threshold",)


def test_mixed_metric_versions_cannot_promote():
    session = EvaluationReport(verdict="accepted")
    forward = EvaluationReport(verdict="accepted", metric_schema_version=1)
    assert promotion_check(session, forward, config=LearningConfig()).verdict == "rejected"


def test_evidence_serialization_records_metric_version_and_plan():
    from astrbot_plugin_dynamics_learning.core.evaluator import holdout_record
    report = report_for({"topic_commit_threshold": BASE_POLICY["topic_commit_threshold"] -.05})
    assert report.as_dict()["metric_schema_version"] == 2
    assert holdout_record(report)["metric_schema_version"] == 2
    assert holdout_record(report)["evaluation_plan"]["target_tasks"] == [TASK_TOPIC]


def test_forward_split_does_not_treat_unknown_times_as_chronology():
    from types import SimpleNamespace
    from astrbot_plugin_dynamics_learning.core.evaluator import split_by_time
    unknown = [SimpleNamespace(timestamp=0, session_hash=str(i), msg_id=str(i), task="topic")
               for i in range(10)]
    assert not split_by_time(unknown, ratio=.5).holdout
    dated = SimpleNamespace(timestamp=10, session_hash="dated", msg_id="dated", task="topic")
    split = split_by_time([*unknown, dated], ratio=.5)
    assert split.holdout == (dated,)
    assert len(split.train) == 10


def test_dataset_identity_ignores_import_clock_but_preserves_annotation_time():
    from dataclasses import replace
    from astrbot_plugin_dynamics_learning.core.evaluator import dataset_fingerprint
    from astrbot_plugin_dynamics_learning.core.samples import build_dataset
    from .factories import annotated_sessions
    sample = build_dataset(annotated_sessions(sessions=1, per_session=2))[0]
    assert dataset_fingerprint([sample]) == dataset_fingerprint([replace(sample, ingested_at=12345)])
    assert dataset_fingerprint([sample]) != dataset_fingerprint([replace(sample, annotated_at=12345)])


def test_baseline_reproduction_compares_recorded_decision_not_human_label(monkeypatch):
    from types import SimpleNamespace
    from astrbot_plugin_dynamics_learning.core import evaluator
    monkeypatch.setattr(evaluator, "_unreplayable", lambda sample: False)
    monkeypatch.setattr(evaluator, "_decision", lambda *args: SimpleNamespace(recipient_label="bot"))
    sample = SimpleNamespace(task=TASK_RECIPIENT, predicted="other", expected="bot")
    result = evaluator.baseline_reproduction([sample], BASE_POLICY)
    assert result["tasks"][TASK_RECIPIENT] == {"checked": 1, "mismatched": 1, "excluded": 0}
    assert result["diagnostic_only"] is True


def test_joint_target_order_does_not_select_the_largest_holdout_gain():
    from astrbot_plugin_dynamics_learning.core.evaluator import primary_task
    report = report_for({"strong_addressivity_threshold": BASE_POLICY["strong_addressivity_threshold"] + .05,
                         "topic_commit_threshold": BASE_POLICY["topic_commit_threshold"] -.05})
    report.tasks = {TASK_TOPIC: row(TASK_TOPIC, .9), TASK_RECIPIENT: row(TASK_RECIPIENT, .01)}
    assert primary_task(report).task == TASK_RECIPIENT
