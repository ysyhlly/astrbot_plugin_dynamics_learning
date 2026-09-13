"""Sample construction and the supervised metrics built on top of it."""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.metrics import (
    binary_counts, binary_report, error_distribution, label_confusion, ratio,
    sample_accuracy, topic_pair_metrics,
)
from astrbot_plugin_dynamics_learning.core.samples import (
    TASK_RECIPIENT, TASK_REPLY, TASK_TOPIC, UNASSIGNED, build_dataset,
    samples_from_annotation, session_hash,
)

from .factories import make_record, make_trace


def test_one_record_can_produce_all_three_tasks():
    record = make_record(
        "m1",
        trace=make_trace(
            evidence=[("continuation_cue", "dialogue", 0.15), ("ambient_baseline", "baseline", 0.2)],
            bot_targeted=False, topic_id="t1", topic_confidence=0.62,
            participation_score=0.35, participation_level="hover"),
        predicted_topic="t1", expected_topic="t2",
        bot_targeted=True, expected_reply=True, error_type="topic_split",
        recipient_error_type="wrong_recipient")
    samples = samples_from_annotation(record, "umo:group:1")
    tasks = {sample.task for sample in samples}
    assert tasks == {TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY}
    recipient = next(sample for sample in samples if sample.task == TASK_RECIPIENT)
    assert recipient.predicted == "other" and recipient.expected == "bot"
    assert recipient.correct is False
    topic = next(sample for sample in samples if sample.task == TASK_TOPIC)
    assert topic.predicted == "t1" and topic.expected == "t2"
    reply = next(sample for sample in samples if sample.task == TASK_REPLY)
    assert reply.predicted == "silent" and reply.expected == "reply"


def test_missing_labels_never_become_samples():
    record = make_record("m1", trace=make_trace(topic_id="t1", topic_confidence=0.5))
    assert samples_from_annotation(record, "s") == []
    assert samples_from_annotation({}, "s") == []
    assert samples_from_annotation({"msg_id": "m"}, "s") == []


def test_unknown_topic_labels_are_excluded_but_new_is_a_singleton():
    unknown = make_record("m1", trace=make_trace(topic_id="t1", topic_confidence=0.5),
                          predicted_topic="t1", expected_topic="UNKNOWN")
    assert [s.task for s in samples_from_annotation(unknown, "s")] == []

    fresh = make_record("m2", trace=make_trace(topic_confidence=0.5),
                        predicted_topic="UNKNOWN", expected_topic="NEW")
    topic = next(s for s in samples_from_annotation(fresh, "s") if s.task == TASK_TOPIC)
    assert topic.expected == "NEW:m2"
    assert topic.predicted == UNASSIGNED


def test_unassigned_prediction_is_kept_for_pair_metrics():
    record = make_record("m3", trace=make_trace(topic_confidence=0.3),
                         predicted_topic="UNKNOWN", expected_topic="t7")
    topic = next(s for s in samples_from_annotation(record, "s") if s.task == TASK_TOPIC)
    assert topic.predicted == UNASSIGNED
    metrics = topic_pair_metrics([[("", "t7"), ("t7", "t7")]])
    assert metrics["fragmentation"] == 1 and metrics["true_positive"] == 0


def test_session_hash_matches_the_host_keying():
    import hashlib
    assert session_hash("umo:group:1") == hashlib.sha256(b"umo:group:1").hexdigest()


def test_build_dataset_keeps_the_latest_annotation_per_sample():
    first = make_record("m1", trace=make_trace(evidence=[("ambient_baseline", "baseline", 0.2)],
                                               bot_targeted=False),
                        bot_targeted=False, annotated_at=10.0)
    second = dict(first, bot_targeted=True, annotated_at=20.0)
    samples = build_dataset([("s", first), ("s", second)])
    recipient = [s for s in samples if s.task == TASK_RECIPIENT]
    assert len(recipient) == 1
    assert recipient[0].expected == "bot"
    assert recipient[0].annotated_at == 20.0


def test_topic_pair_metrics_are_renaming_invariant():
    # Session 1 fragments one truth into two predictions; session 2 agrees.
    pairs = [[("a", "x"), ("b", "x")], [("c", "z"), ("c", "z")]]
    renamed = [[("P", "Q"), ("R", "Q")], [("S", "T"), ("S", "T")]]
    assert topic_pair_metrics(pairs) == topic_pair_metrics(renamed)
    metrics = topic_pair_metrics(pairs)
    # One pair per session: comparisons never cross session boundaries.
    assert metrics["pairs"] == 2
    assert metrics["fragmentation"] == 1
    assert metrics["wrong_merge"] == 0
    assert metrics["pair_accuracy"] == pytest.approx(0.5, abs=1e-4)


def test_topic_pair_metrics_separate_wrong_merge_from_fragmentation():
    merged = topic_pair_metrics([[("t1", "a"), ("t1", "b")]])
    assert merged["wrong_merge"] == 1 and merged["fragmentation"] == 0
    split = topic_pair_metrics([[("t1", "a"), ("t2", "a")]])
    assert split["wrong_merge"] == 0 and split["fragmentation"] == 1
    assert topic_pair_metrics([[]])["pairs"] == 0
    assert topic_pair_metrics([[]])["precision"] is None


def test_binary_report_reports_undefined_rather_than_zero():
    report = binary_report(binary_counts([]))
    assert report["support"] == 0
    assert report["precision"] is None and report["recall"] is None and report["f1"] is None
    assert report["accuracy"] is None
    assert ratio(1, 0) is None and ratio(0, 5) == 0.0


def test_binary_report_f1_matches_its_definition():
    report = binary_report({"tp": 6, "fp": 2, "fn": 3, "tn": 9})
    assert report["precision"] == pytest.approx(0.75)
    assert report["recall"] == pytest.approx(6 / 9, abs=1e-4)
    assert report["f1"] == pytest.approx(2 * 0.75 * (6 / 9) / (0.75 + 6 / 9), abs=1e-4)


def test_error_types_name_the_direction_of_the_mistake():
    missed = make_record("m1", trace=make_trace(evidence=[("ambient_baseline", "baseline", 0.2)],
                                                bot_targeted=False),
                         bot_targeted=True, expected_reply=True)
    # The host admitted a reply (level strong) but the human says it should not
    # have: that is a premature reply, not a missed one.
    false_positive = make_record("m2", trace=make_trace(evidence=[("ambient_baseline", "baseline", 0.2)],
                                                        bot_targeted=True,
                                                        participation_level="strong"),
                                 bot_targeted=False, expected_reply=False)
    missed_samples = samples_from_annotation(missed, "s")
    fp_samples = samples_from_annotation(false_positive, "s")
    assert {s.error_type for s in missed_samples if s.task == TASK_RECIPIENT} == {"missed_bot"}
    assert {s.error_type for s in missed_samples if s.task == TASK_REPLY} == {"missed_reply"}
    assert {s.error_type for s in fp_samples if s.task == TASK_RECIPIENT} == {"false_bot"}
    assert {s.error_type for s in fp_samples if s.task == TASK_REPLY} == {"premature_reply"}


def test_an_explicit_human_error_type_wins_over_the_derived_one():
    record = make_record("m1", trace=make_trace(evidence=[("ambient_baseline", "baseline", 0.2)],
                                                bot_targeted=False),
                         bot_targeted=True, recipient_error_type="subject_confusion")
    recipient = next(s for s in samples_from_annotation(record, "s") if s.task == TASK_RECIPIENT)
    assert recipient.error_type == "subject_confusion"


def test_error_distribution_counts_only_incorrect_samples():
    records = [
        make_record("a", trace=make_trace(evidence=[("ambient_baseline", "baseline", 0.2)],
                                          bot_targeted=False),
                    bot_targeted=False, recipient_error_type="correct"),
        make_record("b", trace=make_trace(evidence=[("ambient_baseline", "baseline", 0.2)],
                                          bot_targeted=True),
                    bot_targeted=False, recipient_error_type="missed_bot"),
    ]
    samples = [s for s in build_dataset([("s", r) for r in records]) if s.task == TASK_RECIPIENT]
    assert sample_accuracy(samples)["accuracy"] == 0.5
    assert error_distribution(samples) == {"missed_bot": 1}
    assert label_confusion((s.predicted, s.expected) for s in samples)


def test_samples_survive_a_serialisation_round_trip():
    from astrbot_plugin_dynamics_learning.core.samples import LearningSample
    record = make_record("m1", trace=make_trace(topic_id="t1", topic_confidence=0.5),
                         predicted_topic="t1", expected_topic="t1", bot_targeted=True)
    sample = samples_from_annotation(record, "s")[0]
    restored = LearningSample.from_dict(sample.as_dict(include_trace=True))
    assert restored is not None
    assert restored.sample_id == sample.sample_id
    assert restored.features == sample.features
    assert LearningSample.from_dict({"task": "nope"}) is None
    assert LearningSample.from_dict(None) is None
