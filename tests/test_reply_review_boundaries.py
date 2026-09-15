"""Regression evidence for identity, abstention and causal attribution."""
import json

import pytest

from astrbot_plugin_dynamics_learning.core.reply_review import (
    build_digest, compare, parse_reply_review, select_messages,
)
from .test_reply_review import SESSION, _record, _reply, _verdicts


@pytest.mark.parametrize("confidence", [0, -1, None, True, float("nan"), float("inf"), 10**500])
def test_invalid_confidence_cannot_become_a_training_or_missed_signal(confidence):
    chosen, _ = select_messages([(SESSION, _record("m", text="hello", delivered=False))])
    result = parse_reply_review(_verdicts(_reply("m", confidence=confidence)), chosen)
    assert result["rows"][0]["decided"] is False
    assert result["rows"][0]["model_should_reply"] is None
    assert result["counts"]["model_reply"] == result["counts"]["missed"] == 0


def test_no_text_abstains_even_if_model_claims_certainty():
    chosen, _ = select_messages([(SESSION, _record("m", delivered=False))])
    row = parse_reply_review(_verdicts(_reply("m", confidence=1)), chosen)["rows"][0]
    assert row["decided"] is False
    assert row["model_reason"] == "没有正文"


def test_duplicate_message_ids_in_different_sessions_keep_separate_facts():
    chosen, _ = select_messages([
        ("room-a", _record("m", text="first", delivered=True)),
        ("room-b", _record("m", text="second", delivered=False)),
    ])
    digest = build_digest(chosen)
    ids = [row["review_id"] for row in digest["messages"]]
    assert len(set(ids)) == 2
    payload = _verdicts(*[dict(review_id=identifier, should_reply=bool(index == 0), confidence=1)
                         for index, identifier in enumerate(ids)])
    result = parse_reply_review(payload, chosen)
    assert [row["text"] for row in result["rows"]] == ["first", "second"]
    assert [row["delivered"] for row in result["rows"]] == [True, False]
    assert [row["model_should_reply"] for row in result["rows"]] == [True, False]
    ambiguous = parse_reply_review(_verdicts(_reply("m")), chosen)
    assert ambiguous["counts"]["undecided"] == 2


def test_admission_silence_never_claims_a_reply_was_sent():
    row = compare({"host_level": "weak", "recorded": False}, False, True)
    assert row["verdict"] == "agreed_admission_silent"
    assert "发送未知" in row["verdict_label"]
    assert row["delivered"] is None


@pytest.mark.parametrize("recorded", [True, False])
def test_unknown_outcome_never_means_silence(recorded):
    assert compare({"recorded": recorded}, False, True)["verdict"] == "outcome_unknown"


def test_generation_in_progress_is_not_a_final_negative():
    record = _record("m", text="hi", delivered=False, expected_reply=True)
    record["decision_trace"]["outcome"] = {
        "final_outcome": "in_flight", "delivered": False, "stage": "generation"}
    chosen, stats = select_messages([(SESSION, record)])
    assert stats["mismatched"] == 0
    result = parse_reply_review(_verdicts(_reply("m")), chosen)
    assert result["rows"][0]["delivered"] is None
    assert result["rows"][0]["verdict"] == "outcome_unknown"
    assert result["counts"]["reply_preference"] == 0


@pytest.mark.parametrize("stage", ["persona", "gate", "generation", "delivery", "unknown"])
def test_non_delivery_does_not_prove_a_routing_error(stage):
    chosen, _ = select_messages([(SESSION, _record("m", text="hi", delivered=False, stage=stage))])
    result = parse_reply_review(_verdicts(_reply("m")), chosen)
    assert result["counts"]["missed"] == 0
    assert result["counts"]["reply_preference"] == 1
    assert result["rows"][0]["rule_error_confirmed"] is False


def test_persona_without_historical_principles_abstains_and_does_not_leak_answers():
    record = _record("m", text="you there?", delivered=False, expected_reply=True)
    record["decision_trace"].update(
        mode="persona_model", review_context={"decision_mode": "persona_model", "interaction_state": "observing"},
        decision_stages={"persona": {"action": "ignore"}, "gate": {"reason_code": "asleep_ambient"}})
    chosen, _ = select_messages([(SESSION, record)])
    digest = build_digest(chosen)
    blob = json.dumps(digest)
    assert "observing" in blob
    assert "ignore" not in blob and "asleep_ambient" not in blob
    assert "expected_reply" not in blob and SESSION not in blob
    result = parse_reply_review(_verdicts(_reply("m")), chosen)
    assert result["rows"][0]["decided"] is False
    assert result["rows"][0]["decision_stages"]["persona"]["action"] == "ignore"


@pytest.mark.parametrize("rows,patterns", [(1, 2), ({}, {}), ("bad", "bad")])
def test_malformed_collections_are_not_iterated_as_model_rows(rows, patterns):
    chosen, _ = select_messages([(SESSION, _record("m", text="hi"))])
    result = parse_reply_review(json.dumps(dict(summary="review", rows=rows, patterns=patterns)), chosen)
    assert result["counts"]["undecided"] == 1
    assert result["patterns"] == []
