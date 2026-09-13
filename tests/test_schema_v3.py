"""Schema 3: two readers, one normalisation layer, and what each one marks.

Schema 2 could describe a routing decision. Schema 3 also records the topic
candidate set with per-candidate evidence, the selection, and where the turn
finally ended up. These tests pin the *markers* as much as the values: a schema
2 row must read as outcome_unavailable and candidate_evidence_partial, because
the whole point of carrying a version is that a later reader can tell "the host
did not say" from "the host said no".
"""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.outcome import (
    STAGE_DELIVERY, STAGE_GATE, STAGE_UNKNOWN, VALUE_DELIVERED, VALUE_NOT_DELIVERED,
    parse_outcome, reason_stage,
)
from astrbot_plugin_dynamics_learning.core.samples import (
    SAMPLE_SCHEMA_VERSION, TASK_RECIPIENT, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME,
    TASK_TOPIC, build_dataset,
)
from astrbot_plugin_dynamics_learning.core.trace import (
    CANDIDATE_EVIDENCE_FULL, CANDIDATE_EVIDENCE_NONE, CANDIDATE_EVIDENCE_PARTIAL,
    SCHEMA_V2, SCHEMA_V3, parse_decision_trace, read_schema_v2, read_schema_v3,
)

from .factories import (
    candidate, delivered_record, make_record, make_trace, make_trace_v3, outcome_block,
    suppressed_record,
)

SESSION = "umo:group:1"
EVIDENCE = {"semantic": 0.74, "reply_edge": 0.0, "participant_overlap": 0.81}


# ---- the readers --------------------------------------------------------

def test_schema_2_marks_the_two_facts_it_cannot_answer():
    trace = parse_decision_trace(make_trace(evidence=[("ambient_baseline", "baseline", 0.2)]))

    assert trace.source_schema == SCHEMA_V2
    assert trace.trace_schema_version == SCHEMA_V2
    assert trace.degraded is False, "schema 2 is old, not malformed"
    assert trace.outcome_unavailable is True
    assert trace.outcome.recorded is False
    assert trace.candidate_evidence_partial is True, "no candidate field at all is still partial"


def test_schema_3_reads_candidates_selection_and_outcome():
    trace = parse_decision_trace(make_trace_v3(
        candidates=[candidate("t2", 0.72, rank=1, evidence=EVIDENCE),
                    candidate("t1", 0.68, rank=2, evidence=EVIDENCE)],
        selected_topic="t2",
        outcome=outcome_block("suppressed", delivered=False, reason="asleep_ambient"),
        topic_id="t2", topic_confidence=0.72))

    assert trace.source_schema == SCHEMA_V3
    assert trace.trace_schema_version == SCHEMA_V3
    assert [row.topic_id for row in trace.topic_candidates] == ["t2", "t1"]
    assert trace.topic_candidates[0].evidence == EVIDENCE
    assert trace.selected_topic == "t2"
    assert trace.candidate_evidence == CANDIDATE_EVIDENCE_FULL
    assert trace.candidate_evidence_partial is False
    assert trace.outcome.value == "suppressed"
    assert trace.outcome.is_gate_suppression is True
    assert trace.outcome.suppression_reason == "asleep_ambient"


def test_full_candidate_evidence_needs_schema_3_and_evidence_on_every_candidate():
    with_evidence = parse_decision_trace(make_trace_v3(
        candidates=[candidate("t1", 0.7, evidence=EVIDENCE)],
        outcome=outcome_block("delivered", delivered=True)))
    without_evidence = parse_decision_trace(make_trace_v3(
        candidates=[candidate("t1", 0.7)], outcome=outcome_block("delivered", delivered=True)))
    schema_2_with_evidence = make_trace_v3(
        candidates=[candidate("t1", 0.7, evidence=EVIDENCE)],
        outcome=outcome_block("delivered", delivered=True))
    schema_2_with_evidence["routing_schema_version"] = SCHEMA_V2

    assert with_evidence.candidate_evidence == CANDIDATE_EVIDENCE_FULL
    assert without_evidence.candidate_evidence == CANDIDATE_EVIDENCE_PARTIAL
    # The version is the host's own statement about what it wrote; evidence-ish
    # keys inside a record that declares schema 2 do not upgrade it.
    assert parse_decision_trace(schema_2_with_evidence).candidate_evidence == \
        CANDIDATE_EVIDENCE_PARTIAL


def test_no_candidate_field_at_all_is_none_not_partial():
    trace = parse_decision_trace(make_trace())

    assert trace.candidate_evidence == CANDIDATE_EVIDENCE_NONE
    assert trace.topic_candidates_recorded is False
    assert trace.candidate_evidence_partial is True, "none is more degraded than partial"


def test_an_unknown_version_falls_back_to_the_older_reader():
    raw = make_trace_v3(candidates=[candidate("t1", 0.7, evidence=EVIDENCE)],
                        outcome=outcome_block("delivered", delivered=True))
    raw["routing_schema_version"] = 9

    trace = parse_decision_trace(raw)

    assert trace.source_schema == 9
    assert trace.trace_schema_version == SCHEMA_V2
    assert trace.degraded is True
    # The self-describing blocks are still consumed — losing a fact the host
    # really wrote would be worse than the version mismatch it arrived under...
    assert trace.outcome.value == VALUE_DELIVERED
    # ...but an unreadable version can never promote the evidence level.
    assert trace.candidate_evidence == CANDIDATE_EVIDENCE_PARTIAL


def test_the_two_readers_are_separately_addressable():
    raw = make_trace_v3(candidates=[candidate("t1", 0.7)], outcome=outcome_block("delivered"))

    assert read_schema_v3(raw).trace_schema_version == SCHEMA_V3
    forced_v2 = read_schema_v2(raw)
    assert forced_v2.trace_schema_version == SCHEMA_V2
    assert forced_v2.degraded is True, "reading a v3 record as v2 is a degradation, and says so"


def test_round_trip_keeps_the_source_schema_and_the_schema_3_sections():
    raw = make_trace_v3(candidates=[candidate("t2", 0.72, evidence=EVIDENCE)],
                        selected_topic="t2",
                        outcome=outcome_block("suppressed", delivered=False,
                                              reason="conflict_silence"))
    first = parse_decision_trace(raw)
    second = parse_decision_trace(first.to_contract())

    assert second.trace_schema_version == SCHEMA_V3
    assert second.to_contract() == first.to_contract()
    assert second.outcome.value == "suppressed"
    assert [row.topic_id for row in second.topic_candidates] == ["t2"]


def test_a_schema_2_round_trip_does_not_invent_schema_3_sections():
    first = parse_decision_trace(make_trace(evidence=[("ambient_baseline", "baseline", 0.2)]))
    payload = first.to_contract()

    assert payload["trace_schema_version"] == SCHEMA_V2
    assert "outcome" not in payload
    assert "routing" not in payload


def test_candidates_written_beside_the_trace_are_adopted():
    """The schema 2 host splits one decision across routing and decision_trace."""
    from astrbot_plugin_dynamics_learning.core.trace import trace_from_sample_record

    record = make_record("m1", trace=make_trace(topic_id="t1", topic_confidence=0.7),
                         predicted_topic="t1", expected_topic="t1",
                         topic_candidates=[[0.7, "t1"], [0.4, "t-other"]])
    trace = trace_from_sample_record(record)

    assert trace.topic_candidates_recorded is True
    assert [row.topic_id for row in trace.topic_candidates] == ["t1", "t-other"]
    assert trace.candidate_evidence == CANDIDATE_EVIDENCE_PARTIAL


def test_an_outcome_written_beside_the_trace_is_read_too():
    from astrbot_plugin_dynamics_learning.core.trace import trace_from_sample_record

    record = make_record("m1", trace=make_trace_v3(outcome=None), predicted_topic="UNKNOWN",
                         expected_topic="UNKNOWN")
    record["outcome"] = outcome_block("delivered", delivered=True)

    assert trace_from_sample_record(record).outcome.value == VALUE_DELIVERED


# ---- the outcome vocabulary --------------------------------------------

def test_a_bare_not_delivered_is_not_filed_under_the_gate():
    guess = parse_outcome({"delivered": False})
    named = parse_outcome({"delivered": False, "suppression_reason": "asleep_ambient"})

    assert guess.value == VALUE_NOT_DELIVERED
    assert guess.stage == STAGE_UNKNOWN, "the host did not say where it stopped"
    assert named.stage == STAGE_GATE
    assert named.suppression_reason == "asleep_ambient"


def test_a_delivered_flag_names_the_value_and_a_reason_names_a_suppression():
    assert parse_outcome({"delivered": True}).value == VALUE_DELIVERED
    assert parse_outcome({"delivered": True}).stage == STAGE_DELIVERY
    assert parse_outcome({"suppression_reason": "cool_command"}).value == "suppressed"


def test_an_unlisted_reason_is_unknown_rather_than_a_gate():
    """A host that adds a reason must show up as unclassified, not as a gate."""
    assert reason_stage("asleep_ambient") == STAGE_GATE
    assert reason_stage("brand_new_reason_code") == STAGE_UNKNOWN
    parsed = parse_outcome({"final_outcome": "suppressed",
                            "suppression_reason": "brand_new_reason_code"})

    assert parsed.value == "suppressed"
    assert parsed.stage == STAGE_UNKNOWN
    assert parsed.is_gate_suppression is False


def test_the_first_source_that_recorded_an_outcome_wins():
    assert parse_outcome({"outcome": outcome_block("delivered", delivered=True)},
                         {"delivered": False}).value == VALUE_DELIVERED
    assert parse_outcome({}, {"delivered": False}).value == VALUE_NOT_DELIVERED


# ---- the two reply layers ----------------------------------------------

def test_reply_is_split_into_admission_and_outcome():
    samples = build_dataset([(SESSION, delivered_record("m1"))])
    by_task = {sample.task: sample for sample in samples}

    assert TASK_REPLY_ADMISSION in by_task
    assert TASK_REPLY_OUTCOME in by_task
    assert by_task[TASK_REPLY_ADMISSION].predicted == "reply"
    assert by_task[TASK_REPLY_OUTCOME].predicted == "reply"
    assert by_task[TASK_REPLY_OUTCOME].correct is True


def test_schema_2_data_produces_no_outcome_samples_at_all():
    samples = build_dataset([(SESSION, make_record(
        "m1", trace=make_trace(participation_level="strong", participation_score=0.8),
        predicted_topic="UNKNOWN", expected_topic="UNKNOWN", expected_reply=True))])

    assert [sample.task for sample in samples] == [TASK_REPLY_ADMISSION]


def test_a_suppressed_send_is_not_a_missed_reply():
    """The finding the two-layer split exists for."""
    samples = build_dataset([(SESSION, suppressed_record("m1"))])
    by_task = {sample.task: sample for sample in samples}

    admission = by_task[TASK_REPLY_ADMISSION]
    outcome = by_task[TASK_REPLY_OUTCOME]
    assert admission.correct is True, "the router admitted the turn; it was right"
    assert admission.error_type == "correct"
    assert outcome.correct is False
    assert outcome.error_type == "gate_suppression", "and the miss is named by its stage"
    assert outcome.outcome.suppression_reason == "asleep_ambient"


def test_an_unrecognised_suppression_stage_is_unattributable():
    samples = build_dataset([(SESSION, suppressed_record("m1", reason="brand_new_gate"))])
    outcome = next(sample for sample in samples if sample.task == TASK_REPLY_OUTCOME)

    assert outcome.error_type == "unattributable"


def test_the_outcome_travels_on_every_sample_of_the_message():
    record = suppressed_record("m1")
    record["expected_topic"] = "t1"
    record["predicted_topic"] = "t1"
    trace = record["decision_trace"]
    trace["topic"]["topic_id"] = "t1"
    record["bot_targeted"] = True
    trace["recipient"]["bot_targeted"] = True

    samples = build_dataset([(SESSION, record)])
    tasks = {sample.task for sample in samples}

    assert {TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME} <= tasks
    assert all(sample.outcome.recorded for sample in samples)
    assert all(sample.outcome.suppression_reason == "asleep_ambient" for sample in samples)


def test_the_outcome_survives_a_round_trip_without_the_raw_trace():
    config = LearningConfig(store_raw_trace=False)
    samples = build_dataset([(SESSION, suppressed_record("m1"))], config=config)
    outcome = next(sample for sample in samples if sample.task == TASK_REPLY_OUTCOME)

    payload = outcome.as_dict(include_trace=False)
    restored = type(outcome).from_dict(payload)

    assert restored is not None
    assert restored.outcome.recorded is True
    assert restored.outcome.suppression_reason == "asleep_ambient"


def test_a_legacy_reply_row_loads_as_the_admission_task():
    from astrbot_plugin_dynamics_learning.core.samples import LearningSample

    sample = LearningSample.from_dict({
        "task": "reply", "session_key": SESSION, "session_hash": "a" * 64,
        "msg_id": "m1", "predicted": "reply", "expected": "reply", "sample_id": "keepme",
    })

    assert sample is not None
    assert sample.task == TASK_REPLY_ADMISSION
    assert sample.sample_id == "keepme", "the stored id is kept, not re-derived"


def test_the_sample_schema_version_is_declared_as_three():
    assert SAMPLE_SCHEMA_VERSION == 3


@pytest.mark.parametrize("task", [TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY_ADMISSION,
                                  TASK_REPLY_OUTCOME])
def test_every_task_is_reachable_from_the_fixtures(task):
    record = delivered_record("m1")
    record["expected_topic"] = "t1"
    record["predicted_topic"] = "t1"
    record["decision_trace"]["topic"]["topic_id"] = "t1"
    record["bot_targeted"] = False
    record["decision_trace"]["recipient"]["bot_targeted"] = False

    assert task in {sample.task for sample in build_dataset([(SESSION, record)])}
