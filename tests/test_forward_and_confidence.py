"""Forward validation, confidence intervals, the dataset gate and the state machine.

Four separate answers to "should this policy exist", and each one is tested for
the case that makes it *not* decorative: a check that cannot run must report
insufficient rather than fine, an interval that spans zero must block, and a
corpus that is too small or too old must produce no policy record at all.
"""
from __future__ import annotations

import time

import pytest

from astrbot_plugin_dynamics_learning.core.bootstrap import (
    Unit, advantage_ratio, bootstrap_delta, human_interval,
)
from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.evaluator import (
    VERDICT_ACCEPTED, VERDICT_INSUFFICIENT, VERDICT_REJECTED, EvaluationReport,
    TaskEvaluation, dataset_record, evaluate_dataset, forward_evaluation, group_diagnostics,
    promotion_check, split_by_time, target_facts,
)
from astrbot_plugin_dynamics_learning.core.policy import (
    ACTION_STATUS, BASE_POLICY, LEARNING_VERSION, POLICY_CONTRACT_VERSION, STATUS_PROPOSED,
    STATUS_PROMOTED, STATUS_SHADOW, STATUS_SUPERSEDED, STATUS_VALIDATED, baseline_config_hash,
    can_transition, candidate_from, normalize_status, published_payload, published_policy,
)
from astrbot_plugin_dynamics_learning.core.quality import (
    GATE_BLOCK, GATE_OK, GATE_WARN, dataset_gate,
)
from astrbot_plugin_dynamics_learning.core.report import analyze
from astrbot_plugin_dynamics_learning.core.samples import (
    TASK_RECIPIENT, build_dataset,
)

from .factories import (
    annotated_sessions, delivered_record, make_record, make_trace, recipient_record,
    suppressed_record,
)

NOW = 2_000_000_000.0
SESSION = "umo:group:1"


def _binary_unit(key, *, base_tp, base_fn, cand_tp, cand_fn, tn=0):
    return Unit(
        key=key,
        baseline={"tp": base_tp, "fp": 0, "tn": tn, "fn": base_fn},
        candidate={"tp": cand_tp, "fp": 0, "tn": tn, "fn": cand_fn},
    )


# ---- the confidence interval -------------------------------------------

def test_a_bootstrap_is_deterministic_for_a_seed():
    units = [_binary_unit(f"s{index}", base_tp=5, base_fn=5, cand_tp=6, cand_fn=4)
             for index in range(20)]

    first = bootstrap_delta(units, metric="f1", iterations=200, seed=11)
    second = bootstrap_delta(units, metric="f1", iterations=200, seed=11)
    other = bootstrap_delta(units, metric="f1", iterations=200, seed=12)

    assert first == second
    assert first["lower"] <= first["delta"] <= first["upper"]
    assert first["seed"] != other["seed"]


def test_an_interval_that_spans_zero_is_reported_as_such():
    """Half the groups improve, half get worse: the delta is not distinguishable."""
    units = []
    for index in range(20):
        if index % 2:
            units.append(_binary_unit(f"up{index}", base_tp=4, base_fn=6, cand_tp=6, cand_fn=4))
        else:
            units.append(_binary_unit(f"down{index}", base_tp=6, base_fn=4, cand_tp=4, cand_fn=6))
    interval = bootstrap_delta(units, metric="f1", iterations=400, seed=3)

    assert interval["crosses_zero"] is True
    assert interval["lower"] < 0 < interval["upper"]
    assert "CI" in human_interval(interval)


def test_a_uniform_improvement_produces_an_interval_above_zero():
    units = [_binary_unit(f"s{index}", base_tp=5, base_fn=5, cand_tp=8, cand_fn=2)
             for index in range(30)]
    interval = bootstrap_delta(units, metric="f1", iterations=400, seed=5)

    assert interval["crosses_zero"] is False
    assert interval["lower"] > 0
    assert advantage_ratio([1.0, 2.0, -1.0]) == pytest.approx(0.6667)
    assert advantage_ratio([]) is None


def test_an_empty_corpus_gives_no_interval_rather_than_a_zero():
    empty = bootstrap_delta([], metric="f1")

    assert empty["delta"] is None
    assert empty["crosses_zero"] is None
    assert "重采样单元" in empty["reason"]
    assert human_interval(empty) == "区间不可计算"


def test_an_unknown_metric_is_refused_by_name():
    refused = bootstrap_delta([_binary_unit("s", base_tp=1, base_fn=1, cand_tp=2, cand_fn=0)],
                              metric="not_a_metric")

    assert refused["delta"] is None
    assert "未知指标" in refused["reason"]


# ---- the time-ordered split --------------------------------------------

def test_the_time_split_puts_the_newest_labels_in_the_forward_holdout():
    rows = annotated_sessions(sessions=8, per_session=10, start=NOW - 100_000)
    samples = build_dataset(rows)
    split = split_by_time(samples, ratio=0.25)

    assert split.train and split.holdout
    assert len(split.train) + len(split.holdout) == len(samples)
    assert max(row.timestamp for row in split.train) <= min(row.timestamp for row in split.holdout)


def test_the_time_split_is_deterministic_and_keeps_every_sample():
    rows = annotated_sessions(sessions=6, per_session=8, start=NOW - 50_000)
    samples = build_dataset(rows)

    first = split_by_time(samples, ratio=0.3)
    second = split_by_time(samples, ratio=0.3)

    assert [row.sample_id for row in first.holdout] == [row.sample_id for row in second.holdout]


def test_forward_evaluation_reports_a_time_split():
    rows = annotated_sessions(sessions=12, per_session=16, start=NOW - 200_000)
    samples = build_dataset(rows)
    report = forward_evaluation(samples, config=LearningConfig(), now=NOW)

    assert report.split["kind"] == "time"
    assert report.split["holdout_samples"] > 0


def test_a_session_evaluation_still_reports_a_session_split():
    rows = annotated_sessions(sessions=12, per_session=16, start=NOW - 200_000)
    report = evaluate_dataset(build_dataset(rows), now=NOW)

    assert report.split["kind"] == "session"


# ---- the promotion gate -------------------------------------------------

def _report(verdict, *, crosses_zero=None, reasons=None):
    """A report carrying one real TaskEvaluation, so the gate reads real fields."""
    interval = ({} if crosses_zero is None
                else {"crosses_zero": crosses_zero, "delta": 0.01, "lower": -0.01,
                      "upper": 0.03, "confidence": 0.95})
    row = TaskEvaluation(task=TASK_RECIPIENT, holdout=60, primary_metric="accuracy",
                         bootstrap=interval)
    return EvaluationReport(verdict=verdict, reasons=list(reasons or ["因为"]),
                            tasks={TASK_RECIPIENT: row})


def test_a_missing_forward_check_is_insufficient_not_accepted():
    check = promotion_check(_report(VERDICT_ACCEPTED), None, config=LearningConfig())

    assert check.verdict == VERDICT_INSUFFICIENT
    assert "前向验证" in check.reasons[0]


def test_the_forward_holdout_can_reject_a_policy_the_session_holdout_accepted():
    check = promotion_check(_report(VERDICT_ACCEPTED),
                            _report(VERDICT_REJECTED, reasons=["前向没复现"]),
                            config=LearningConfig())

    assert check.verdict == VERDICT_REJECTED
    assert "前向验证未通过" in check.reasons[0]


def test_an_interval_spanning_zero_blocks_promotion():
    check = promotion_check(_report(VERDICT_ACCEPTED, crosses_zero=True),
                            _report(VERDICT_ACCEPTED),
                            config=LearningConfig())

    assert check.verdict == VERDICT_REJECTED
    assert "置信区间跨 0" in check.reasons[0]
    assert check.interval_ok is False


def test_two_holdouts_and_a_positive_interval_accept():
    check = promotion_check(_report(VERDICT_ACCEPTED, crosses_zero=False),
                            _report(VERDICT_ACCEPTED),
                            config=LearningConfig())

    assert check.verdict == VERDICT_ACCEPTED
    assert check.forward_verdict == VERDICT_ACCEPTED
    assert "前向验证通过" in check.reasons[-1]


def test_switching_the_forward_requirement_off_is_stated_in_the_result():
    config = LearningConfig(require_forward_validation=False)
    check = promotion_check(_report(VERDICT_ACCEPTED, crosses_zero=False), None, config=config)

    assert check.verdict == VERDICT_ACCEPTED
    assert any("配置关闭" in reason for reason in check.reasons)


# ---- per-group diagnostics ---------------------------------------------

def test_group_diagnostics_gives_no_number_below_the_support_floor():
    units = [_binary_unit("small", base_tp=1, base_fn=1, cand_tp=2, cand_fn=0),
             _binary_unit("big", base_tp=10, base_fn=10, cand_tp=16, cand_fn=4)]
    result = group_diagnostics(units, task=TASK_RECIPIENT, min_support=12)

    rows = {row["group"]: row for row in result["groups"]}
    assert rows["small"]["eligible"] is False
    assert rows["small"]["delta"] is None
    assert rows["small"]["reason"] == "样本不足"
    assert rows["big"]["eligible"] is True
    assert result["improved"] == 1
    assert any("不产生本地策略" in note for note in result["notes"])


# ---- the dataset gate ---------------------------------------------------

def test_the_gate_blocks_a_corpus_that_is_too_small():
    samples = build_dataset([(SESSION, recipient_record(f"m{index}",
                                                        annotated_at=time.time()))
                             for index in range(3)])
    gate = dataset_gate(samples, config=LearningConfig(), now=time.time())

    assert gate["ok"] is False
    assert set(gate["blocked_by"]) == {"samples", "sessions"}
    assert any(row["status"] == GATE_BLOCK for row in gate["checks"])


def test_the_gate_blocks_stale_labels():
    samples = build_dataset([(f"umo:{index}", recipient_record(f"m{index}",
                                                              annotated_at=1_000.0))
                             for index in range(200)])
    gate = dataset_gate(samples, config=LearningConfig(), now=time.time())
    row = next(item for item in gate["checks"] if item["name"] == "label_age")

    assert gate["ok"] is False
    assert row["status"] == GATE_BLOCK
    assert "label_age" in gate["blocked_by"]


def test_the_gate_passes_a_fresh_corpus_but_still_qualifies_it():
    rows = annotated_sessions(sessions=10, per_session=16, start=time.time() - 7200)
    samples = build_dataset(rows)
    gate = dataset_gate(samples, config=LearningConfig(), now=time.time())

    assert gate["ok"] is True, gate["blocked_by"]
    coverage = next(item for item in gate["checks"] if item["name"] == "candidate_coverage")
    assert coverage["blocking"] is False, "少候选只限定话题方向，不该一票否决"
    assert coverage["status"] in (GATE_OK, GATE_WARN)


def test_a_blocked_corpus_produces_no_actionable_recommendation():
    samples = build_dataset([(SESSION, recipient_record(f"m{index}", annotated_at=1_000.0))
                             for index in range(3)])
    result = analyze(samples, now=time.time())

    assert result.dataset_gate["ok"] is False
    assert all(not row.actionable for row in result.recommendations)
    assert any("数据门槛未通过" in note for note in result.notes)


# ---- the state machine and the published file ---------------------------

def test_the_state_machine_refuses_to_skip_validation():
    assert can_transition(STATUS_VALIDATED, STATUS_PROMOTED) is True
    assert can_transition(STATUS_PROPOSED, STATUS_SHADOW) is False
    assert can_transition(STATUS_PROMOTED, STATUS_VALIDATED) is False
    assert can_transition(STATUS_SUPERSEDED, STATUS_PROPOSED) is False
    assert can_transition(STATUS_PROMOTED, STATUS_PROMOTED) is True, "重复点击不是错误"


def test_legacy_status_words_map_onto_the_new_ones():
    assert normalize_status("candidate") == STATUS_PROPOSED
    assert normalize_status("accepted") == STATUS_VALIDATED
    assert normalize_status("nonsense") == STATUS_PROPOSED


def test_only_promoted_policies_are_published():
    proposed = candidate_from({"topic_commit_threshold": 0.62})
    validated = proposed.with_status(STATUS_VALIDATED)
    promoted = proposed.with_status(STATUS_PROMOTED)
    payload = published_payload([proposed, validated, promoted], issued_at=1.0)

    assert [row["policy_id"] for row in payload["policies"]] == [promoted.version]
    assert payload["policy_contract_version"] == POLICY_CONTRACT_VERSION
    assert payload["policies"][0]["state"] == STATUS_PROMOTED


def test_a_published_policy_carries_every_parameter_and_the_shadow_flag():
    candidate = candidate_from({"strong_addressivity_threshold": 0.67})
    published = published_policy(candidate.with_status(STATUS_PROMOTED), issued_at=1.0)

    assert set(published["params"]) == set(published["baseline"])
    assert published["changed"] == ["strong_addressivity_threshold"]
    assert published["params"]["strong_addressivity_threshold"] == pytest.approx(0.67)
    assert published["shadow_observed"] is False
    assert "写入" in published_payload([])["note"]


def test_the_published_file_separates_the_three_versions():
    """One file, three numbers, and none of them can be read off another."""
    candidate = candidate_from({"strong_addressivity_threshold": 0.67}).with_fields(
        training_dataset={"fingerprint": "abc123", "samples": 120, "sessions": 6},
        compatibility={"trace_schema_version": 3, "trace_schema_versions": {"2": 20, "3": 100}},
        target={"chat_dynamics_version": "1.6.2", "baseline_config_hash": "deadbeef",
                "validated_host_versions": ["1.6.2"]},
    )
    published = published_policy(candidate.with_status(STATUS_PROMOTED), issued_at=1.0)

    assert published["policy_contract_version"] == POLICY_CONTRACT_VERSION
    assert published["source"] == {
        "trace_schema_version": 3,
        "trace_schema_versions": {"2": 20, "3": 100},
        "dataset_fingerprint": "abc123",
        "learning_version": LEARNING_VERSION,
    }
    assert published["target"] == {
        "chat_dynamics_version": "1.6.2",
        "baseline_config_hash": "deadbeef",
        "validated_host_versions": ["1.6.2"],
    }


def test_a_policy_with_no_observed_host_version_says_so():
    """An unknown version is not a match; it is the absence of a basis for active."""
    candidate = candidate_from({"topic_commit_threshold": 0.62}).with_status(STATUS_PROMOTED)
    published = published_policy(candidate, issued_at=1.0)

    assert published["target"]["chat_dynamics_version"] is None
    assert published["target"]["validated_host_versions"] == []


def test_target_facts_record_only_the_version_actually_seen():
    facts = target_facts(None, BASE_POLICY)
    named = target_facts("1.6.2", BASE_POLICY)

    assert facts["validated_host_versions"] == []
    assert facts["baseline_config_hash"] == named["baseline_config_hash"]
    assert named["validated_host_versions"] == ["1.6.2"]


def test_the_baseline_config_hash_is_canonical_and_rounding_stable():
    """The host has to reproduce this digest, so its shape is the contract."""
    first = baseline_config_hash(BASE_POLICY)
    again = baseline_config_hash(dict(BASE_POLICY))

    assert first == again and len(first) == 16
    # Rounding to four decimals is part of the canonical form, not noise.
    nudged = dict(BASE_POLICY)
    nudged["strong_addressivity_threshold"] = 0.70004
    assert baseline_config_hash(nudged) == first
    nudged["strong_addressivity_threshold"] = 0.7001
    assert baseline_config_hash(nudged) != first


def test_the_dataset_fingerprint_is_provenance_and_never_a_rejection_reason():
    """It must exist on the policy; pinning it is the consumer's choice."""
    candidate = candidate_from({"topic_commit_threshold": 0.62})
    assert "fingerprint" in dataset_record({"samples": 10, "sessions": 2, "tasks": {}})
    published = published_policy(candidate.with_status(STATUS_PROMOTED), issued_at=1.0)

    assert "dataset_fingerprint" in published["source"]
    assert published["source"]["dataset_fingerprint"] == "", "空指纹照样发布，不因此拒绝"


def test_the_console_actions_are_the_state_machine():
    assert ACTION_STATUS["accept"] == STATUS_PROMOTED
    assert ACTION_STATUS["validate"] == STATUS_VALIDATED
    assert ACTION_STATUS["shadow"] == "shadow"
    assert ACTION_STATUS["rollback"] == "rolled_back"


def test_a_policy_record_keeps_the_results_it_was_judged_on():
    rows = build_dataset([
        ("umo:a", delivered_record("m1", annotated_at=NOW - 100)),
        ("umo:b", suppressed_record("m2", annotated_at=NOW - 50)),
        ("umo:c", make_record("m3", trace=make_trace(participation_level="weak"),
                              predicted_topic="UNKNOWN", expected_topic="UNKNOWN",
                              expected_reply=False, annotated_at=NOW - 10)),
    ])
    report = evaluate_dataset(rows, config=LearningConfig(), now=NOW)

    assert report.outcome["support"] >= 1
    assert report.outcome["stages"].get("gate") == 1
    assert report.outcome["replayable"] is False


def test_candidate_offer_filters_unvalidated_and_retired_policies():
    from astrbot_plugin_dynamics_learning.core.policy import candidate_payload

    row = candidate_from({"strong_addressivity_threshold": 0.67})
    states = ["proposed", "validated", "shadow", "promoted", "superseded", "rolled_back"]
    payload = candidate_payload([row.with_status(state) for state in states], issued_at=1)
    assert {item["state"] for item in payload["policies"]} == {"validated", "shadow", "promoted"}
    assert payload["eligible_modes"] == ["shadow"]
