"""Topic candidate parsing, recall, selection and error attribution."""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.candidates import (
    ATTRIBUTION_CANDIDATE_MISS, ATTRIBUTION_CORRECT, ATTRIBUTION_NEW_TOPIC,
    ATTRIBUTION_NOT_RECORDED, ATTRIBUTION_RANKING_ERROR, KIND_LABELLED, KIND_NEW_TOPIC,
    CandidateObservation, attribution, candidate_ids, candidate_recall, parse_candidates,
    parse_topic_candidates, records_to_payload, selection_accuracy, summarise,
)
from astrbot_plugin_dynamics_learning.core.samples import (
    TASK_TOPIC, LearningSample, build_dataset,
)

from .factories import make_record, make_trace, topic_record


def _obs(expected, selected, raw, *, kind=KIND_LABELLED):
    parsed = parse_candidates(raw)
    return CandidateObservation(expected=expected, selected=selected,
                                candidates=parsed.items, recorded=parsed.recorded,
                                kind=kind, dropped=parsed.dropped)


def test_legacy_pairs_and_structured_records_parse_to_the_same_ranking():
    legacy = parse_topic_candidates([[0.68, "A"], [0.72, "B"]])
    structured = parse_topic_candidates([
        {"topic_id": "A", "final_score": 0.68, "rank": 2,
         "evidence": {"semantic": 0.74, "reply_edge": 0.0}},
        {"topic_id": "B", "final_score": 0.72, "rank": 1,
         "evidence": {"semantic": 0.66, "reply_edge": 1.0, "recency": 0.9}},
    ])
    assert candidate_ids(legacy) == ("B", "A")
    assert candidate_ids(structured) == ("B", "A")
    # The structured form keeps per-candidate evidence; the legacy form cannot.
    assert structured[0].evidence["reply_edge"] == 1.0
    assert structured[0].rank == 1
    assert legacy[0].evidence == {}


def test_rank_is_authoritative_over_score_order():
    records = parse_topic_candidates([
        {"topic_id": "low", "final_score": 0.10, "rank": 1},
        {"topic_id": "high", "final_score": 0.90, "rank": 2},
    ])
    assert candidate_ids(records) == ("low", "high")


def test_parsing_is_total_over_garbage():
    assert parse_topic_candidates(None) == ()
    assert parse_topic_candidates("nope") == ()
    assert parse_topic_candidates([]) == ()
    # Unusable entries are dropped, usable ones in the same list survive.
    records = parse_topic_candidates([
        {"topic_id": "", "final_score": 0.5},
        {"topic_id": "ok", "final_score": "x"},
        {"topic_id": "good", "final_score": 0.4},
        [0.9, "pair"],
        [None, "bad"],
        ["0.3", "stringly"],
    ])
    assert candidate_ids(records) == ("pair", "good", "ok")
    # A candidate the host recorded without a score still counts for recall —
    # the topic *was* a candidate — but is flagged so a threshold replay can
    # leave it alone instead of reading the missing score as zero.
    unscored = parse_topic_candidates([{"topic_id": "ok"}])[0]
    assert unscored.score_known is False
    assert "final_score" not in unscored.as_dict()


def test_an_unscored_candidate_is_never_chosen_by_a_relaxed_threshold():
    from astrbot_plugin_dynamics_learning.core.topic_learner import TopicPairRow, replay_label

    row = TopicPairRow(predicted="", expected="A", confidence=0.3, ambiguous=False,
                       candidates=parse_topic_candidates([{"topic_id": "A"}]),
                       session_hash="s")
    # Relaxing the cut cannot pick A, because nothing recorded how it scored.
    assert replay_label(row, 0.0) == ""


def test_candidate_records_round_trip_through_the_sample_store():
    record = topic_record("m1", predicted="B", expected="A", confidence=0.7,
                          candidates=[{"topic_id": "A", "final_score": 0.68, "rank": 2,
                                       "evidence": {"semantic": 0.74}},
                                      {"topic_id": "B", "final_score": 0.72, "rank": 1}])
    sample = next(s for s in build_dataset([("s", record)]) if s.task == TASK_TOPIC)
    assert candidate_ids(sample.topic_candidates) == ("B", "A")
    assert sample.topic_candidates[1].evidence == {"semantic": 0.74}
    assert sample.selected_topic == "B"
    assert records_to_payload(sample.topic_candidates)[0]["topic_id"] == "B"


def test_selected_topic_falls_back_to_the_predicted_label():
    record = topic_record("m1", predicted="B", expected="A", confidence=0.7)
    sample = next(s for s in build_dataset([("s", record)]) if s.task == TASK_TOPIC)
    assert sample.selected_topic == "B"


def test_attribution_separates_generation_from_ranking():
    observations = [
        # correct
        _obs("A", "A", [[0.9, "A"]]),
        # A was a candidate but B won -> ranking / scoring
        _obs("A", "B", [[0.9, "B"], [0.7, "A"]]),
        # A never made the candidate set -> generation
        _obs("A", "B", [[0.9, "B"], [0.4, "C"]]),
        # nothing recorded -> unattributable, never blamed on generation
        _obs("A", "B", None),
    ]
    result = attribution(observations)
    assert result["counts"][ATTRIBUTION_CORRECT] == 1
    assert result["counts"][ATTRIBUTION_RANKING_ERROR] == 1
    assert result["counts"][ATTRIBUTION_CANDIDATE_MISS] == 1
    assert result["counts"][ATTRIBUTION_NOT_RECORDED] == 1
    assert result["rates"][ATTRIBUTION_CANDIDATE_MISS] == pytest.approx(0.25)


def test_recall_and_selection_are_conditioned_separately():
    observations = [
        _obs("A", "A", [[0.9, "A"]]),          # hit@1 and selected
        _obs("A", "B", [[0.9, "B"], [0.7, "A"]]),  # hit@2, not selected
        _obs("A", "B", [[0.9, "B"]]),          # miss
    ]
    recall = candidate_recall(observations)
    assert recall["recorded"] == 3
    assert recall["coverage"] == 1.0
    assert recall["hits_at_1"] == 1
    assert recall["hits_at_3"] == 2
    assert recall["recall_at_3"] == pytest.approx(2 / 3, abs=1e-4)
    # Selection accuracy only counts the two turns where A was available.
    selection = selection_accuracy(observations)
    assert selection["eligible"] == 2
    assert selection["correct"] == 1
    assert selection["accuracy"] == pytest.approx(0.5)


def test_undefined_ratios_stay_undefined():
    summary = summarise([])
    assert summary["candidate_recall"]["recall_at_3"] is None
    assert summary["selection_accuracy"]["accuracy"] is None
    assert summary["attribution"]["total"] == 0


def test_new_topic_singletons_are_counted_but_never_scored():
    from astrbot_plugin_dynamics_learning.core.topic_learner import candidate_observations

    fresh = make_record("m1", trace=make_trace(topic_confidence=0.5),
                        predicted_topic="UNKNOWN", expected_topic="NEW")
    known = topic_record("m2", predicted="A", expected="A", confidence=0.7,
                         candidates=[[0.9, "A"]])
    samples = build_dataset([("s", fresh), ("s", known)])
    observations = candidate_observations([s for s in samples if s.task == TASK_TOPIC])
    # A brand-new topic cannot have been a candidate, so blaming generation for
    # it would be blaming the host for being right. It is still *counted*, so the
    # attribution table totals the corpus instead of quietly shrinking it.
    assert [row.kind for row in observations] == [KIND_NEW_TOPIC, KIND_LABELLED]
    summary = summarise(observations)
    assert summary["attribution"]["counts"][ATTRIBUTION_NEW_TOPIC] == 1
    assert summary["candidate_recall"]["total"] == 1
    assert summary["candidate_recall"]["excluded_new_topic"] == 1


def test_an_empty_recorded_list_is_a_generation_miss_not_a_missing_field():
    """The distinction the whole attribution rests on.

    "The host looked and proposed nothing" and "the host recorded nothing" are
    different facts. Merging them turns every unattributable error into a
    generation failure the data never showed.
    """
    recorded_empty = _obs("A", "B", [])
    missing = _obs("A", "B", None)
    assert recorded_empty.recorded is True
    assert recorded_empty.attribution == ATTRIBUTION_CANDIDATE_MISS
    assert missing.recorded is False
    assert missing.attribution == ATTRIBUTION_NOT_RECORDED


def test_an_empty_recorded_list_counts_against_recall_and_a_missing_one_does_not():
    observations = [_obs("A", "B", []), _obs("A", "B", None), _obs("A", "A", [[0.9, "A"]])]
    recall = candidate_recall(observations)
    assert recall["recorded"] == 2
    assert recall["not_recorded"] == 1
    assert recall["coverage"] == pytest.approx(2 / 3)
    assert recall["missed"] == 1
    assert recall["recall_at_3"] == pytest.approx(0.5)
    assert attribution(observations)["counts"][ATTRIBUTION_NOT_RECORDED] == 1


def test_the_hosts_silence_survives_ingestion():
    """A routing snapshot without the key, versus one holding an empty list."""
    silent = topic_record("m1", predicted="B", expected="A", confidence=0.7)
    empty = topic_record("m2", predicted="B", expected="A", confidence=0.7, candidates=[])
    samples = {s.msg_id: s for s in build_dataset([("s", silent), ("s", empty)])
               if s.task == TASK_TOPIC}
    assert samples["m1"].topic_candidates_recorded is False
    assert samples["m2"].topic_candidates_recorded is True
    assert samples["m2"].topic_candidates == ()


def test_rows_written_before_the_flag_keep_the_conservative_reading():
    """No flag: an empty payload cannot be told apart, so it is not blamed."""
    def sample(trace):
        return LearningSample(sample_id="x", session_key="s", session_hash="h", msg_id="m1",
                              timestamp=0.0, task=TASK_TOPIC, predicted="B", expected="A",
                              confidence=0.7, source="test", trace=trace)

    assert sample({"topic_candidates": []}).topic_candidates_recorded is False
    assert sample({"topic_candidates": [],
                   "topic_candidates_recorded": True}).topic_candidates_recorded is True


def test_attribution_totals_every_labeled_topic_sample():
    from astrbot_plugin_dynamics_learning.core.topic_learner import candidate_observations

    fresh = make_record("m1", trace=make_trace(topic_confidence=0.5),
                        predicted_topic="UNKNOWN", expected_topic="NEW")
    known = topic_record("m2", predicted="B", expected="A", confidence=0.7,
                         candidates=[[0.9, "B"], [0.7, "A"]])
    samples = [s for s in build_dataset([("s", fresh), ("s", known)]) if s.task == TASK_TOPIC]
    summary = summarise(candidate_observations(samples))
    assert summary["attribution"]["total"] == len(samples) == 2
    assert sum(summary["attribution"]["counts"].values()) == 2
    assert summary["attribution"]["counts"][ATTRIBUTION_RANKING_ERROR] == 1


def test_unparsed_candidate_entries_are_counted_not_hidden():
    parsed = parse_candidates([[0.5, "ok"], [0.5], "junk", {"nope": 1}, [0.5, ""]])
    assert parsed.ids == ("ok",)
    assert parsed.dropped == 4
    summary = summarise([_obs("A", "B", [[0.5, "ok"], [0.5], {"nope": 1}])])
    assert summary["dropped_entries"] == 2
    assert any("无法解析" in note for note in summary["notes"])


def test_truncated_candidate_lists_are_flagged():
    summary = summarise([_obs("A", "A", [[0.9, "A"]]), _obs("A", "A", [[0.9, "A"], [0.5, "B"]])])
    assert summary["candidate_lengths"] == {"1": 1, "2": 1}
    assert any("Recall@5" in note for note in summary["notes"])


def test_an_undefined_selection_accuracy_does_not_break_the_candidate_notes():
    """Candidates existed but none was ever the right one: accuracy is undefined.

    Formatting that None as a percentage used to raise, so the whole topic
    report died on a batch whose only sin was never retrieving the right topic.
    """
    from astrbot_plugin_dynamics_learning.core.topic_learner import learn

    record = topic_record("m1", predicted="B", expected="A", confidence=0.7,
                          candidates=[[0.9, "B"]])
    samples = [s for s in build_dataset([("s", record)]) if s.task == TASK_TOPIC]
    report = learn(samples)
    assert report.candidate_metrics["candidate_recall"]["recall_at_3"] == 0.0
    assert report.candidate_metrics["selection_accuracy"]["accuracy"] is None
    assert any("无法计算" in note for note in report.notes)
