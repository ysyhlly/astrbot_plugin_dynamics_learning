from dataclasses import replace

import pytest

from core.config import LearningConfig
from core.evaluator import score_task, contract_compatibility
from core.policy import BASE_POLICY, PolicyCandidate, published_policy
from core.samples import LearningSample, TASK_REPLY_ADMISSION, samples_from_annotation
from tests.factories import make_record, make_trace


def admission(*, mode="legacy", outcome=None, explicit=None):
    trace = make_trace(mode=mode, participation_level="strong", bot_targeted=True,
                       evidence=[("bot_mention", "recipient", 1.0)])
    record = make_record("m", trace=trace, expected_reply=False)
    if outcome:
        record["outcome"] = outcome
    if explicit is not None:
        record["expected_rule_reply"] = explicit
    return next(s for s in samples_from_annotation(record, "s", config=LearningConfig(
        store_raw_trace=False)) if s.task == TASK_REPLY_ADMISSION)


@pytest.mark.parametrize("mode", ["persona", "unknown", "future_mode"])
def test_persona_and_unknown_final_labels_cannot_score_threshold(mode):
    row = admission(mode=mode)
    assert not row.rule_reply_supervision_eligible
    assert row.error_type == "unattributable"
    score = score_task([row], BASE_POLICY, TASK_REPLY_ADMISSION)
    assert score.support == 0 and score.unreplayable == 1
    restored = LearningSample.from_dict(row.as_dict())
    assert restored is not None and not restored.rule_reply_supervision_eligible
    assert restored.trace["decision_scope"]["decision_mode"] == mode


def test_explicit_rule_label_is_separate_from_final_reply_preference():
    row = admission(mode="persona", explicit=True)
    assert row.expected == "reply"
    assert row.rule_reply_supervision_eligible
    assert score_task([row], BASE_POLICY, TASK_REPLY_ADMISSION).support == 1


@pytest.mark.parametrize("value", ["suppressed", "generation_failed", "delivery_failed", "not_delivered"])
def test_downstream_non_delivery_is_not_rule_supervision(value):
    row = admission(outcome={"value": value, "delivered": False})
    assert not row.rule_reply_supervision_eligible


def test_old_persona_row_still_excluded_without_new_marker():
    row = admission(mode="persona")
    old = replace(row, trace={"contribution_total_recorded": True})
    assert not old.rule_reply_supervision_eligible
    assert score_task([old], BASE_POLICY, TASK_REPLY_ADMISSION).support == 0


def test_legacy_rule_contract_is_preserved():
    assert admission().rule_reply_supervision_eligible


def test_policy_does_not_claim_persona_validation():
    row = admission(mode="persona")
    compatibility = contract_compatibility([row])
    assert compatibility["reply_rule_excluded"] == 1
    assert compatibility["decision_modes"] == ["persona"]
    candidate = PolicyCandidate(version="p1", params=BASE_POLICY, compatibility=compatibility)
    payload = published_policy(candidate)
    assert payload["applicability"]["persona_behavior_validated"] is False
    assert payload["applicability"]["decision_stage"] == "rule"


def test_new_host_persona_context_overrides_legacy_routing_mode():
    trace = make_trace(mode="legacy", participation_level="strong")
    trace["review_context"] = {"decision_mode": "persona_model",
                               "persona_fingerprint": "session-fingerprint"}
    trace["decision_stages"] = {"rule": {"level": "strong", "score": 0.9},
                                "persona": {"action": "ignore", "prompt": "private"},
                                "gate": {"evaluated": False}}
    record = make_record("m", trace=trace, expected_reply=False)
    row = next(s for s in samples_from_annotation(record, "s")
               if s.task == TASK_REPLY_ADMISSION)
    restored = LearningSample.from_dict(row.as_dict())
    assert restored is not None and not restored.rule_reply_supervision_eligible
    assert restored.trace["decision_scope"]["decision_mode"] == "persona_model"
    assert restored.trace["decision_stages"]["persona"] == {"action": "ignore"}


def test_persona_outcome_is_not_gate_suppression():
    from core.outcome import parse_record_outcome
    outcome = parse_record_outcome({"outcome": {"value": "suppressed",
                                                "delivered": False, "stage": "persona"}})
    assert outcome.stage == "persona"
    assert not outcome.is_gate_suppression
    assert outcome.as_dict()["value_label"] == "角色选择沉默"
