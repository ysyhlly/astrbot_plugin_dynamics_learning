"""The attribution chain: one bucket per message, and the partition holds.

The tests are written around two promises the table makes. First, every message
lands in exactly one bucket, so the counts total the corpus. Second, the bucket a
message lands in is the *layer a reviewer should open first* — and when a message
fails two layers, the second one is still visible in also_failed rather than
being dropped by the single-count rule.
"""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core import buckets
from astrbot_plugin_dynamics_learning.core.attribution import (
    MIN_ATTRIBUTED_MESSAGES, attribute, attribution_report, build_chains, classify,
)
from astrbot_plugin_dynamics_learning.core.samples import build_dataset

from .factories import candidate, make_record, make_trace_v3, outcome_block

SESSION = "umo:group:1"
EVIDENCE = {"semantic": 0.74, "participant_overlap": 0.81}


def chain_record(
    msg_id,
    *,
    recipient_truth=True,
    host_targeted=True,
    expected_topic="UNKNOWN",
    predicted_topic="UNKNOWN",
    candidates=None,
    expected_reply=None,
    level=None,
    outcome=None,
    annotated_at=1000.0,
):
    """One message with every link of the chain controllable independently."""
    selected = predicted_topic if predicted_topic != "UNKNOWN" else ""
    trace = make_trace_v3(
        candidates=candidates, selected_topic=selected, outcome=outcome,
        bot_targeted=host_targeted,
        topic_id=selected, topic_confidence=0.7 if selected else 0.0,
        participation_level=level,
        participation_score=0.8 if level == "strong" else 0.2,
    )
    record = make_record(msg_id, trace=trace, predicted_topic=predicted_topic,
                         expected_topic=expected_topic, annotated_at=annotated_at)
    if recipient_truth is not None:
        record["bot_targeted"] = recipient_truth
    if expected_reply is not None:
        record["expected_reply"] = expected_reply
    return record


def bucket_of(record, msg_id="m1"):
    samples = build_dataset([(SESSION, record)])
    rows = {row.msg_id: row for row in attribute(samples)}
    return rows[msg_id]


def test_the_buckets_partition_the_corpus():
    rows = [
        chain_record("ok", recipient_truth=False, host_targeted=False),
        chain_record("recipient", recipient_truth=True, host_targeted=False),
        chain_record("miss", expected_topic="t1", predicted_topic="t2",
                     candidates=[candidate("t2", 0.7), candidate("t3", 0.4)]),
        chain_record("rank", expected_topic="t1", predicted_topic="t2",
                     candidates=[candidate("t2", 0.7), candidate("t1", 0.6)]),
    ]
    samples = build_dataset([(SESSION, record) for record in rows])
    attributed = attribute(samples)

    assert len(attributed) == len(rows)
    assert all(row.bucket in buckets.ORDER for row in attributed)
    report = attribution_report(samples)
    assert sum(report["counts"].values()) == report["messages"] == len(rows)


def test_a_recipient_error_is_attributed_to_the_understanding_layer():
    row = bucket_of(chain_record("m1", recipient_truth=True, host_targeted=False))

    assert row.bucket == buckets.RECIPIENT_ERROR
    assert row.is_model_error is True
    assert row.evidence["recipient"] == "other"
    assert row.evidence["recipient_expected"] == "bot"


def test_a_topic_that_never_reached_the_candidate_set_is_a_generation_miss():
    row = bucket_of(chain_record("m1", expected_topic="t1", predicted_topic="t2",
                                 candidates=[candidate("t2", 0.7), candidate("t3", 0.4)]))

    assert row.bucket == buckets.TOPIC_CANDIDATE_MISS
    assert "候选生成" in row.reason


def test_a_topic_that_was_a_candidate_but_lost_is_a_ranking_error():
    row = bucket_of(chain_record("m1", expected_topic="t1", predicted_topic="t2",
                                 candidates=[candidate("t2", 0.7), candidate("t1", 0.6)]))

    assert row.bucket == buckets.TOPIC_RANKING_ERROR
    assert "打分与排序" in row.reason


def test_a_wrong_topic_without_a_candidate_set_is_unattributable():
    row = bucket_of(chain_record("m1", expected_topic="t1", predicted_topic="t2"))

    assert row.bucket == buckets.UNATTRIBUTABLE
    assert "没有候选集" in row.reason


def test_an_admission_mistake_is_a_participation_error():
    row = bucket_of(chain_record("m1", expected_reply=True, level="weak",
                                 outcome=outcome_block("not_attempted",
                                                       delivered=False)))

    assert row.bucket == buckets.PARTICIPATION_ERROR
    assert row.is_model_error is True


def test_a_suppressed_send_is_a_gate_event_not_a_router_error():
    row = bucket_of(chain_record("m1", expected_reply=True, level="strong",
                                 outcome=outcome_block("suppressed", delivered=False,
                                                       reason="asleep_ambient")))

    assert row.bucket == buckets.GATE_SUPPRESSION
    assert row.is_model_error is False
    assert row.evidence["suppression_reason"] == "asleep_ambient"


@pytest.mark.parametrize("value,reason,expected", [
    ("generation_failed", "llm_timeout", buckets.GENERATION_FAILURE),
    ("delivery_failed", "send_failed", buckets.DELIVERY_FAILURE),
])
def test_each_execution_stage_gets_its_own_bucket(value, reason, expected):
    row = bucket_of(chain_record("m1", expected_reply=True, level="strong",
                                 outcome=outcome_block(value, delivered=False, reason=reason)))

    assert row.bucket == expected
    assert row.is_model_error is False


def test_a_non_delivery_with_no_stage_is_unattributable():
    """Delivered=false with no reason is a fact without a cause."""
    row = bucket_of(chain_record("m1", expected_reply=True, level="strong",
                                 outcome=outcome_block("not_delivered", delivered=False)))

    assert row.bucket == buckets.UNATTRIBUTABLE
    assert row.bucket not in buckets.SYSTEM_EVENTS


def test_gate_outcome_does_not_turn_final_reply_label_into_rule_supervision():
    """A final preference cannot diagnose a threshold error before a recorded gate."""
    row = bucket_of(chain_record("m1", expected_reply=True, level="weak",
                                 outcome=outcome_block("suppressed", delivered=False,
                                                       reason="asleep_ambient")))

    assert row.bucket == buckets.GATE_SUPPRESSION
    assert buckets.GATE_SUPPRESSION not in row.also_failed


def test_a_message_that_was_right_all_the_way_down_is_ok():
    row = bucket_of(chain_record("m1", recipient_truth=True, host_targeted=True,
                                 expected_topic="t1", predicted_topic="t1",
                                 candidates=[candidate("t1", 0.7, evidence=EVIDENCE)],
                                 expected_reply=True, level="strong",
                                 outcome=outcome_block("delivered", delivered=True)))

    assert row.bucket == buckets.OK
    assert row.reason == ""
    assert row.outcome_unavailable is False


def test_a_second_failure_is_recorded_rather_than_counted_twice():
    row = bucket_of(chain_record("m1", recipient_truth=True, host_targeted=False,
                                 expected_topic="t1", predicted_topic="t2",
                                 candidates=[candidate("t2", 0.7)]))

    assert row.bucket == buckets.RECIPIENT_ERROR, "the earlier link owns the count"
    assert row.also_failed == (buckets.TOPIC_CANDIDATE_MISS,)
    report = attribution_report(build_dataset([(SESSION, chain_record(
        "m1", recipient_truth=True, host_targeted=False, expected_topic="t1",
        predicted_topic="t2", candidates=[candidate("t2", 0.7)]))]))
    assert sum(report["counts"].values()) == 1, "one message, one bucket"
    assert report["also_failed"] == {buckets.TOPIC_CANDIDATE_MISS: 1}


def test_schema_2_admits_what_it_cannot_attribute():
    """No outcome recorded is a gap, not a negative."""
    record = chain_record("m1", expected_reply=True, level="strong", outcome=None)
    record["decision_trace"]["routing_schema_version"] = 2
    row = bucket_of(record)

    assert row.bucket == buckets.OK
    assert row.outcome_unavailable is True
    assert row.bucket not in buckets.SYSTEM_EVENTS


def test_a_schema_2_miss_is_still_attributed_to_the_router():
    record = chain_record("m1", expected_reply=True, level="weak", outcome=None)
    record["decision_trace"]["routing_schema_version"] = 2
    row = bucket_of(record)

    assert row.bucket == buckets.PARTICIPATION_ERROR
    assert row.outcome_unavailable is True


def test_a_label_that_only_covers_the_recipient_link_is_not_read_as_a_pass():
    record = chain_record("m1", recipient_truth=True, host_targeted=True,
                          expected_topic="UNKNOWN", expected_reply=None)
    row = bucket_of(record)

    assert row.present == ("recipient",)
    assert row.bucket == buckets.OK
    assert row.evidence["topic"] == "unrecorded"


def test_a_spurious_delivery_has_no_recorded_link_to_blame():
    row = bucket_of(chain_record("m1", expected_reply=False, level="weak",
                                 outcome=outcome_block("delivered", delivered=True)))

    assert row.bucket == buckets.UNATTRIBUTABLE
    assert "没有任何一环该为这次发送负责" in row.reason


def test_chains_never_merge_two_sessions_on_a_shared_message_id():
    samples = build_dataset([
        ("umo:group:1", chain_record("m1", recipient_truth=True, host_targeted=False)),
        ("umo:group:2", chain_record("m1", recipient_truth=False, host_targeted=False)),
    ])
    chains = build_chains(samples)

    assert len(chains) == 2
    assert len({chain.session_hash for chain in chains}) == 2
    assert all(len(classify(chain).present) == 1 for chain in chains)


def test_the_report_says_when_it_is_reading_counts_not_estimates():
    samples = build_dataset([(SESSION, chain_record("m1"))])
    report = attribution_report(samples)

    assert report["messages"] == 1
    assert report["messages"] < MIN_ATTRIBUTED_MESSAGES
    assert any("不是比例估计" in note for note in report["notes"])
    assert report["labels"][buckets.GATE_SUPPRESSION]
    assert report["actions"][buckets.TOPIC_CANDIDATE_MISS]


def test_an_empty_dataset_reports_itself_rather_than_zeroes():
    report = attribution_report([])

    assert report["messages"] == 0
    assert report["counts"][buckets.OK] == 0
    assert report["notes"] == ["还没有可归因的消息：先导入标注。"]


def test_a_suppression_note_explains_why_the_bucket_matters():
    samples = build_dataset([(SESSION, chain_record(
        "m1", expected_reply=True, level="strong",
        outcome=outcome_block("suppressed", delivered=False, reason="cool_command")))])
    report = attribution_report(samples)

    assert report["counts"][buckets.GATE_SUPPRESSION] == 1
    assert report["suppression_reasons"] == {"cool_command": 1}
    assert report["execution_stages"] == {"gate": 1}
    assert any("门禁压制" in note for note in report["notes"])
