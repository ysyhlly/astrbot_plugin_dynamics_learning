"""Scope review profiles: the review layer, the LOO baseline, the diagnosis gates.

The numbers these tests pin are the ones a reader is most likely to misread, so
they are asserted against hand-computed values rather than against whatever the
implementation happens to return.
"""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core import scope_profile
from astrbot_plugin_dynamics_learning.core.policy import (
    ERROR_FALSE_BOT, ERROR_FRAGMENTATION, ERROR_MISSED_BOT,
)
from astrbot_plugin_dynamics_learning.core.samples import build_dataset
from astrbot_plugin_dynamics_learning.core.scope import session_hash
from astrbot_plugin_dynamics_learning.core.scope_profile import (
    CONFIDENCE_INSUFFICIENT, CONFIDENCE_LOW, CONFIDENCE_MODERATE, CONFIDENCE_STABLE,
    DIAGNOSIS_CANDIDATE_GENERATION, DIAGNOSIS_INSUFFICIENT, DIAGNOSIS_RANKING,
    STABLE_MIN_DAYS, compare_scope, confidence_for, diagnose_candidates, profile_rows,
    scope_payload, scopes_payload,
)

from .factories import recipient_record, scope_batch, topic_record

DAY = 86_400.0


def _recipient(msg_id: str, *, error: bool, kind: str = ERROR_MISSED_BOT,
               annotated_at: float = 1_000_000.0):
    return recipient_record(msg_id, error=error, kind=kind, annotated_at=annotated_at)


def _scope(scope: str, *, errors: int, correct: int, kind: str = ERROR_MISSED_BOT,
           annotated_at: float = 1_000_000.0):
    rows = [(scope, _recipient(f"{scope}-e{index}", error=True, kind=kind,
                               annotated_at=annotated_at)) for index in range(errors)]
    rows += [(scope, _recipient(f"{scope}-c{index}", error=False,
                                annotated_at=annotated_at)) for index in range(correct)]
    return rows


def _row(comparison, kind: str):
    return next(row for row in comparison.rates if row.kind == kind)


# ---- the baseline -------------------------------------------------------

def test_delta_uses_a_leave_one_out_baseline():
    """A: 8/10 missed, B and C: 1/10 each.

    The naive global would be 10/30 = 33.3%; the LOO baseline is 2/20 = 10%.
    Comparing A against a global that contains A pulls the difference toward
    zero, which is exactly the wrong direction for a tool whose job is to point
    at where one conversation differs.
    """
    rows = _scope("scope-a", errors=8, correct=2)
    rows += _scope("scope-b", errors=1, correct=9)
    rows += _scope("scope-c", errors=1, correct=9)
    samples = build_dataset(rows)

    comparison = compare_scope(samples, session_hash("scope-a"))
    assert comparison is not None
    row = _row(comparison, ERROR_MISSED_BOT)

    assert (row.count, row.support) == (8, 10)
    assert row.raw_rate == pytest.approx(0.8)
    assert (row.loo_count, row.loo_support) == (2, 20)
    assert row.loo_rate == pytest.approx(0.1)
    assert row.loo_rate != pytest.approx(10 / 30), "the baseline must exclude this scope"
    # smoothed = (8 + 20 * 0.1) / (10 + 20) = 1/3
    assert row.smoothed_rate == pytest.approx(1 / 3, abs=1e-4)
    assert row.delta_pp == pytest.approx(23.3, abs=0.1)
    # The raw rate is still reported, so the movement is explainable.
    assert row.as_dict()["raw_rate"] == pytest.approx(0.8)
    assert row.as_dict()["smoothed_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_a_tiny_scope_is_shrunk_and_refuses_to_compare():
    """Three reviewed samples are not a trait, smoothed or not."""
    rows = _scope("scope-a", errors=2, correct=1)
    rows += _scope("scope-b", errors=20, correct=80)
    samples = build_dataset(rows)

    comparison = compare_scope(samples, session_hash("scope-a"))
    assert comparison is not None
    row = _row(comparison, ERROR_MISSED_BOT)

    assert row.raw_rate == pytest.approx(2 / 3)
    assert row.comparable is False
    assert "只有 3 条" in row.reason
    assert row.delta is None
    assert comparison.profile.confidence == CONFIDENCE_INSUFFICIENT
    assert "3 条被检查样本" in comparison.profile.confidence_reason


def test_smoothing_moves_a_small_scope_toward_the_baseline():
    rows = _scope("scope-a", errors=1, correct=4)          # 5/5, support hits the floor
    rows += _scope("scope-b", errors=10, correct=90)       # baseline 10%
    samples = build_dataset(rows)

    row = _row(compare_scope(samples, session_hash("scope-a")), ERROR_MISSED_BOT)

    assert row.raw_rate == pytest.approx(0.2)
    assert row.loo_rate == pytest.approx(0.1)
    # (1 + 20*0.1) / (5 + 20) = 0.12 — pulled down from 20% toward 10%.
    assert row.smoothed_rate == pytest.approx(0.12, abs=1e-6)
    assert 0.1 < row.smoothed_rate < row.raw_rate


def test_a_kind_with_no_exposure_is_undefined_not_zero():
    """Two different facts: "0 of 10 were false_bot" and "there was nothing to check".

    The first is a measurement and stays a real 0.0; the second has no
    denominator, and reporting it as 0% would invent a perfect score out of an
    absence of evidence.
    """
    rows = _scope("scope-a", errors=6, correct=4)
    rows += _scope("scope-b", errors=6, correct=4)
    samples = build_dataset(rows)
    comparison = compare_scope(samples, session_hash("scope-a"))

    exposed = _row(comparison, ERROR_FALSE_BOT)
    assert (exposed.count, exposed.support) == (0, 10)
    assert exposed.raw_rate == 0.0

    # No reply supervision anywhere, so the reply kinds have no denominator.
    unexposed = _row(comparison, "missed_reply")
    assert unexposed.support == 0
    assert unexposed.raw_rate is None
    assert unexposed.comparable is False
    assert unexposed.reason == "本作用域没有该类样本"


# ---- topic kinds and their own denominator ------------------------------

def test_topic_kinds_count_pairs_not_samples():
    """A pair count divided by a sample count would be a different number."""
    rows = [
        ("scope-a", topic_record("m1", predicted="t1", expected="t1", confidence=0.6)),
        ("scope-a", topic_record("m2", predicted="t2", expected="t1", confidence=0.6)),
        ("scope-a", topic_record("m3", predicted="t3", expected="t1", confidence=0.6)),
        ("scope-a", topic_record("m4", predicted="t4", expected="t1", confidence=0.6)),
    ]
    samples = build_dataset(rows)

    comparison = compare_scope(samples, session_hash("scope-a"))
    assert comparison is not None
    row = _row(comparison, ERROR_FRAGMENTATION)

    assert row.support == 6, "four labelled messages in one session make six pairs"
    assert row.count == 6
    assert row.raw_rate == pytest.approx(1.0)
    assert comparison.profile.topic["samples"] == 4
    assert comparison.profile.topic["pairs"] == 6


# ---- confidence tiers ---------------------------------------------------

def _bulk(scope: str, count: int, *, sessions: int = 1, days: int = 1):
    rows = []
    for index in range(count):
        session = scope if sessions == 1 else f"{scope}-s{index % sessions}"
        day = float(index % days)
        rows.append((session, _recipient(f"{scope}-{index}", error=False,
                                         annotated_at=1_000_000.0 + day * DAY)))
    return rows


def test_confidence_tiers_are_reachable():
    assert confidence_for(labelled_samples=5, labelled_sessions=1, days=1)[0] == \
        CONFIDENCE_INSUFFICIENT
    assert confidence_for(labelled_samples=50, labelled_sessions=1, days=3)[0] == CONFIDENCE_LOW
    assert confidence_for(labelled_samples=120, labelled_sessions=1, days=1)[0] == \
        CONFIDENCE_MODERATE


def test_a_hundred_samples_from_one_sitting_are_not_stable():
    samples = build_dataset(_bulk("scope-a", 120, days=1))

    profile = compare_scope(samples, session_hash("scope-a")).profile

    assert profile.labelled_samples == 120
    assert profile.days == 1
    assert profile.confidence == CONFIDENCE_MODERATE
    assert "标注日" in profile.confidence_reason


def test_spreading_the_review_over_days_reaches_stable():
    samples = build_dataset(_bulk("scope-a", 120, days=STABLE_MIN_DAYS))

    profile = compare_scope(samples, session_hash("scope-a")).profile

    assert profile.days == STABLE_MIN_DAYS
    assert profile.confidence == CONFIDENCE_STABLE


def test_the_session_gate_scales_with_the_contract(monkeypatch):
    """One scope is one session today, so three sessions cannot be demanded yet."""
    assert scope_profile.required_sessions() == 1

    monkeypatch.setattr(scope_profile, "SCOPE_SPANS_SESSIONS", True)
    assert scope_profile.required_sessions() == scope_profile.STABLE_MIN_SESSIONS

    samples = build_dataset(_bulk("scope-a", 120, days=STABLE_MIN_DAYS))
    profile = compare_scope(samples, session_hash("scope-a")).profile
    assert profile.labelled_sessions == 1
    assert profile.confidence == CONFIDENCE_MODERATE, "the gate tightens on its own"


# ---- diagnosis gates ----------------------------------------------------

def _topic_rows(*, hits: int, misses: int, selected_wrong: int = 0):
    """`(session, record)` pairs with a known candidate chain.

    A hit is a message whose expected topic was in the candidate list; a miss is
    one where the correct topic never entered it. Which of the two dominates is
    the whole question the diagnosis has to answer.
    """
    rows = []
    for index in range(hits):
        expected = f"T{index}"
        wrong = index < selected_wrong
        predicted = "X" if wrong else expected
        rows.append(("scope-a", topic_record(f"h{index}", predicted=predicted, expected=expected,
                                             confidence=0.6,
                                             candidates=[[0.9, expected], [0.4, "Y"]])))
    for index in range(misses):
        expected = f"M{index}"
        rows.append(("scope-a", topic_record(f"m{index}", predicted="Z", expected=expected,
                                             confidence=0.6, candidates=[[0.9, "Z"]])))
    return rows


def test_diagnosis_refuses_without_candidate_evidence():
    rows = _topic_rows(hits=6, misses=1)
    rows += [("scope-a", topic_record("n1", predicted="t1", expected="t1", confidence=0.6)),
             ("scope-a", topic_record("n2", predicted="t2", expected="t2", confidence=0.6)),
             ("scope-a", topic_record("n3", predicted="t3", expected="t3", confidence=0.6)),
             ("scope-a", topic_record("n4", predicted="t4", expected="t4", confidence=0.6)),
             ("scope-a", topic_record("n5", predicted="t5", expected="t5", confidence=0.6)),
             ("scope-a", topic_record("n6", predicted="t6", expected="t6", confidence=0.6)),
             ("scope-a", topic_record("n7", predicted="t7", expected="t7", confidence=0.6))]
    samples = build_dataset(rows)

    comparison = compare_scope(samples, session_hash("scope-a"))
    assert comparison is not None
    metrics = comparison.profile.candidate_metrics
    diagnosis = diagnose_candidates(metrics)

    assert diagnosis.code == DIAGNOSIS_INSUFFICIENT
    assert "覆盖率" in diagnosis.detail
    assert diagnosis.recommended_target is None


def test_diagnosis_names_candidate_generation():
    """Recall is low while everything that was offered was picked correctly."""
    samples = build_dataset(_topic_rows(hits=21, misses=6))

    comparison = compare_scope(samples, session_hash("scope-a"))
    assert comparison is not None
    metrics = comparison.profile.candidate_metrics
    assert metrics["candidate_recall"]["recall_at_3"] < 0.8
    assert metrics["selection_accuracy"]["accuracy"] == 1.0

    diagnosis = comparison.diagnosis

    assert diagnosis.code == DIAGNOSIS_CANDIDATE_GENERATION
    assert diagnosis.recommended_target == "topic_candidate_generation"
    assert "候选生成" in diagnosis.detail
    assert "暂不建议继续调整 topic_commit_threshold" in diagnosis.detail


def test_diagnosis_names_ranking_when_recall_is_high():
    """Everything was offered; a fifth of the picks were still wrong."""
    samples = build_dataset(_topic_rows(hits=24, misses=0, selected_wrong=4))

    comparison = compare_scope(samples, session_hash("scope-a"))
    assert comparison is not None
    metrics = comparison.profile.candidate_metrics
    assert metrics["candidate_recall"]["recall_at_3"] >= 0.8
    assert metrics["selection_accuracy"]["accuracy"] < 0.9

    diagnosis = comparison.diagnosis

    assert diagnosis.code == DIAGNOSIS_RANKING
    assert diagnosis.recommended_target == "topic_ranking_or_scoring"
    assert "topic_margin_threshold" in diagnosis.detail


def test_the_diagnosis_is_never_a_bare_verdict():
    samples = build_dataset(_topic_rows(hits=21, misses=6))
    comparison = compare_scope(samples, session_hash("scope-a"))

    assert comparison is not None
    assert any("候选链诊断" in line for line in comparison.diagnostics)
    assert comparison.diagnosis.label in "".join(comparison.diagnostics)


# ---- payloads -----------------------------------------------------------

def test_dominant_errors_need_a_delta_over_the_threshold():
    rows = _scope("scope-a", errors=9, correct=1)
    rows += _scope("scope-b", errors=9, correct=1)
    samples = build_dataset(rows)

    comparison = compare_scope(samples, session_hash("scope-a"))

    assert comparison is not None
    row = _row(comparison, ERROR_MISSED_BOT)
    assert row.loo_rate == pytest.approx(0.9)
    assert comparison.dominant_errors == [], "a scope that matches the baseline has none"


def test_scopes_payload_lists_the_worst_scope_first():
    rows = _scope("scope-quiet", errors=1, correct=9)
    rows += _scope("scope-loud", errors=18, correct=2)
    samples = build_dataset(rows)

    payload = scopes_payload(samples)

    assert payload["total"] == 2
    assert payload["scope_level"] == "session"
    assert [row["scope_label"] for row in payload["rows"]] == \
        [session_hash("scope-loud")[:12], session_hash("scope-quiet")[:12]]
    loud = payload["rows"][0]
    assert loud["samples"] == 20
    assert loud["dominant_errors"] == [ERROR_MISSED_BOT]
    assert loud["dominant_labels"] == ["漏识别 Bot"]
    assert "被检查过" in payload["notes"][0]


def test_scope_payload_is_the_review_layer_not_a_population_estimate():
    rows = _scope("scope-a", errors=8, correct=2)
    rows += _scope("scope-b", errors=1, correct=9)
    samples = build_dataset(rows)

    payload = scope_payload(samples, session_hash("scope-a"))

    assert payload is not None
    profile = payload["profile"]
    assert profile["labelled_samples"] == 10
    assert profile["recipient"]["missed_bot"] == 8
    assert profile["confidence"] == CONFIDENCE_INSUFFICIENT
    assert "被检查过" in profile["note"]
    assert payload["global"]["baseline"] == "leave_one_out"
    # A (10 reviewed) vs B (10 reviewed): the baseline is B, not A plus B.
    assert payload["global"]["rates"][ERROR_MISSED_BOT]["support"] == 10
    assert payload["global"]["rates"][ERROR_MISSED_BOT]["count"] == 1
    assert payload["totals"]["scopes"] == 2
    assert payload["totals"]["other_scope_samples"] == 10
    assert payload["diagnosis"]["code"] in {DIAGNOSIS_INSUFFICIENT, DIAGNOSIS_CANDIDATE_GENERATION,
                                            DIAGNOSIS_RANKING, "no_clear_bottleneck"}


def test_the_fixture_batch_produces_a_clear_ranking():
    """The whole path on a batch built for it: three scopes, one worse, three days."""
    samples = build_dataset(scope_batch(scopes=3, per_scope=30, days=3, topics=6))

    payload = scopes_payload(samples)

    assert payload["total"] == 3
    assert [row["samples"] for row in payload["rows"]] == [36, 36, 36]
    assert payload["rows"][0]["scope_label"] == session_hash("umo:scope:0")[:12]
    assert ERROR_MISSED_BOT in payload["rows"][0]["dominant_errors"]
    assert payload["rows"][0]["diagnosis"] in {
        DIAGNOSIS_INSUFFICIENT, DIAGNOSIS_CANDIDATE_GENERATION, DIAGNOSIS_RANKING,
        "no_clear_bottleneck",
    }


def test_an_unknown_scope_has_no_payload():
    samples = build_dataset(_scope("scope-a", errors=1, correct=1))

    assert scope_payload(samples, session_hash("nope")) is None
    assert profile_rows(samples)[0]["scope_hash"] == session_hash("scope-a")


# ---- plugin surface -----------------------------------------------------

@pytest.mark.asyncio
async def test_plugin_scope_endpoints_round_trip(plugin):
    rows = _scope("scope-a", errors=8, correct=2) + _scope("scope-b", errors=1, correct=9)
    await plugin.ingest(source="export", payload={"sessions": [
        {"session_key": scope, "records": [record for key, record in rows if key == scope]}
        for scope in ("scope-a", "scope-b")
    ]})

    listing = await plugin.scopes_payload()
    assert listing["total"] == 2

    scope_hash = listing["rows"][0]["scope_hash"]
    payload = await plugin.scope_payload(scope_hash)
    assert payload["profile"]["scope_hash"] == scope_hash
    assert payload["diagnosis"]["code"]

    with pytest.raises(ValueError):
        await plugin.scope_payload(session_hash("nope"))
    with pytest.raises(ValueError):
        await plugin.scope_payload(scope_hash[:12])

    filtered = await plugin.samples_payload(scope=scope_hash)
    assert filtered["total"] == listing["rows"][0]["samples"]


@pytest.mark.asyncio
async def test_scope_endpoint_reports_a_missing_scope_as_404(plugin):
    plugin.web.register()
    handlers = {route: handler for route, handler, _methods, _desc in plugin.context.routes}

    response = await handlers["/astrbot_plugin_dynamics_learning/scope"]()

    assert response["status"] == "error"
    assert response["status_code"] == 400
