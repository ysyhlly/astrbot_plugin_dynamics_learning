"""Final validation is evidence for one fixed candidate and untouched data."""
from dataclasses import replace

import pytest

from astrbot_plugin_dynamics_learning.core import report as reporting
from astrbot_plugin_dynamics_learning.core import evaluation_pipeline as pipeline
from astrbot_plugin_dynamics_learning.core.autotune import TuneRun
from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.evaluator import (
    EvaluationReport, dataset_fingerprint, promotion_check,
)
from astrbot_plugin_dynamics_learning.core.policy import (
    BASE_POLICY, baseline_config_hash, candidate_from, validate_candidate_evidence,
    published_policy, published_payload, candidate_payload,
)
from astrbot_plugin_dynamics_learning.core.samples import build_dataset
from .factories import annotated_sessions


def test_content_fingerprint_changes_with_labels_features_and_trace():
    rows = build_dataset(annotated_sessions(sessions=3, per_session=3))
    digest = dataset_fingerprint(rows)
    assert digest == dataset_fingerprint(list(reversed(rows)))
    for changed in (replace(rows[0], expected="changed"),
                    replace(rows[0], features={**rows[0].features, "base_score": 0.123}),
                    replace(rows[0], trace={**rows[0].trace, "routing_schema_version": 99})):
        assert digest != dataset_fingerprint([changed, *rows[1:]])


@pytest.mark.parametrize("tuning", [False, True])
def test_final_validation_never_fits_on_its_evidence(monkeypatch, tuning):
    rows = build_dataset(annotated_sessions(sessions=12, per_session=8, start=1000))
    config = LearningConfig(min_samples_for_evaluation=1, bootstrap_iterations=20)
    candidate = candidate_from({"strong_addressivity_threshold": 0.66}, now=2000)
    seen = []
    validation_splits = []
    original_validate = pipeline.validate_frozen
    def validate(*args, **kwargs):
        validation_splits.append((kwargs["kind"], kwargs["split"]))
        return original_validate(*args, **kwargs)
    monkeypatch.setattr(pipeline, "validate_frozen", validate)
    def generate(samples, **kwargs):
        seen.append(tuple(samples))
        return EvaluationReport(candidate=candidate.with_fields(status="validated"), verdict="accepted")
    monkeypatch.setattr(reporting, "evaluate_dataset", generate)
    def tune(samples, **kwargs):
        seen.append(tuple(samples))
        return [TuneRun(task="recipient", baseline=dict(BASE_POLICY), decision="promote",
                        candidate=candidate, final_policy=dict(candidate.params))]
    monkeypatch.setattr(reporting, "run_tuning", tune)
    result = reporting.analyze(rows, config=config, with_tuning=tuning, now=2000,
                               host_version="1.5.2", baseline_source="host_effective")
    saved = result.tuning[0].candidate if tuning else result.evaluation.candidate
    assert saved is not None
    assert validate_candidate_evidence(saved) == []
    assert saved.target["baseline_verified"] is True
    assert saved.compatibility["trace_schema_versions"]
    assert saved.holdout_result["split"]["candidate_hash"] == baseline_config_hash(saved.params)
    assert saved.forward_result["split"]["candidate_hash"] == baseline_config_hash(saved.params)
    train = seen[0]
    assert all(tuple(batch) == train for batch in seen)
    assert dataset_fingerprint(train) == saved.holdout_result["split"]["train_fingerprint"]
    assert dataset_fingerprint(train) == saved.forward_result["split"]["train_fingerprint"]
    assert len(train) < len(rows)
    for kind, split in validation_splits:
        train_messages = {(row.session_hash, row.msg_id) for row in train}
        assert not train_messages.intersection((row.session_hash, row.msg_id) for row in split.holdout)
        if kind == "final_session":
            assert not {row.session_hash for row in train}.intersection(
                row.session_hash for row in split.holdout)
    # A parameter mutation cannot borrow the previous candidate's measurements.
    changed = saved.with_fields(params={**saved.params, "strong_addressivity_threshold": 0.9})
    assert "holdout.candidate_hash" in validate_candidate_evidence(changed)
    if result.promotion["verdict"] != "accepted" or not result.dataset_gate["ok"]:
        assert saved.status != "validated"


def test_promotion_rejects_different_frozen_candidates():
    session = EvaluationReport(verdict="accepted", split={"candidate_hash": "one"})
    future = EvaluationReport(verdict="accepted", split={"candidate_hash": "two"})
    assert promotion_check(session, future, config=LearningConfig()).verdict == "rejected"


def test_stale_candidates_are_not_offered():
    candidate = candidate_from({"strong_addressivity_threshold": 0.66}).with_fields(
        status="promoted", evidence={"stale": True})
    assert not published_payload([candidate])["policies"]
    assert not candidate_payload([candidate])["policies"]
    assert published_policy(candidate)["candidate_hash"] == baseline_config_hash(candidate.params)


def test_equal_sum_parameter_moves_are_not_discarded():
    from astrbot_plugin_dynamics_learning.core.autotune import _finalise
    rows = build_dataset(annotated_sessions(sessions=3, per_session=3))
    final = {**BASE_POLICY, "strong_addressivity_threshold": BASE_POLICY["strong_addressivity_threshold"] + 0.01,
             "topic_commit_threshold": BASE_POLICY["topic_commit_threshold"] - 0.01}
    run = TuneRun(task="recipient", baseline=dict(BASE_POLICY), final_policy=final)
    _finalise(run, rows, LearningConfig(), "recipient", BASE_POLICY, (), 2000)
    assert run.candidate is not None
    assert run.candidate.training_dataset["fingerprint"] == dataset_fingerprint(rows)
    assert run.candidate.compatibility["trace_schema_versions"]
    assert run.candidate.target["baseline_config_hash"] == baseline_config_hash(BASE_POLICY)


def test_old_metric_evidence_requires_revalidation():
    from .factories import evidenced_policy
    candidate = evidenced_policy()
    assert validate_candidate_evidence(candidate) == []
    legacy = candidate.with_fields(evidence={**candidate.evidence, "metric_schema_version": 1})
    assert "metric_schema_version" in validate_candidate_evidence(legacy)
    legacy_split = {**candidate.holdout_result["split"], "metric_schema_version": 1}
    mixed = candidate.with_fields(holdout_result={"split": legacy_split})
    assert "holdout.metric_schema_version" in validate_candidate_evidence(mixed)


def test_context_owns_snapshot_and_detects_generator_mutation():
    rows = build_dataset(annotated_sessions(sessions=12, per_session=8, start=1000))
    context = pipeline.EvaluationContext.reserve(
        rows, config=LearningConfig(), now=2000, host_version=None,
        baseline_source="default_reference", evaluation_enabled=True, dataset_gate_ok=True)
    digest = context.dataset_fingerprint
    rows[0].features["base_score"] = 42
    assert dataset_fingerprint(context.samples) == digest
    context.development[0].features["base_score"] = 99
    with pytest.raises(ValueError, match="snapshot changed"):
        context.finalize(candidate_from({"topic_commit_threshold": 0.62}))
