"""Contract health: the two planes, the accounting identity, the matrix.

The tests here exist because the health report is itself a claim about data
quality, so it has to be the least trustworthy-looking thing in the plugin: every
number either totals the corpus or says why it does not.
"""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core import ingest as ingest_module
from astrbot_plugin_dynamics_learning.core.ingest import parse_export, parse_preferences
from astrbot_plugin_dynamics_learning.core.quality import (
    CAPABILITY_RECIPIENT_REPLAY, CAPABILITY_SCOPE_IDENTITY, CAPABILITY_TOPIC_ATTRIBUTION,
    CAPABILITY_TOPIC_THRESHOLD_REPLAY, STATUS_INSUFFICIENT, STATUS_OK, STATUS_UNSUPPORTED,
    STATUS_WARNING, capabilities, contract_findings, quality_report, recipient_replay,
    scope_identity, topic_attribution, topic_threshold_replay,
)
from astrbot_plugin_dynamics_learning.core.samples import TASK_RECIPIENT, build_dataset, session_hash

from .factories import export_payload, make_record, make_trace, topic_record

SESSION = "umo:group:1"


def _preference(key, value):
    return {"key": key, "value": {"val": value}}


def _runtime(sessions):
    return _preference("panel_runtime_v1", {"version": 1, "sessions": sessions})


def _annotations(session, records):
    return _preference("topic_annotations_v1_" + session_hash(session), records)


def _ambient(msg_id: str, *, evidence=(), targeted: bool = True, level: str = "strong",
             total: float | None = 0.8, expected_reply: bool = True, annotated_at: float = 1.0):
    trace = make_trace(evidence=list(evidence), bot_targeted=targeted,
                       participation_score=total, participation_level=level,
                       recipient_confidence=0.5)
    if total is None:
        trace["participation"]["contribution_total"] = None
    else:
        trace["participation"]["contribution_total"] = total
    return make_record(msg_id, trace=trace, predicted_topic="t1", expected_topic="t1",
                       bot_targeted=targeted, expected_reply=expected_reply,
                       annotated_at=annotated_at)


# ---- contract plane: accounting ----------------------------------------

def test_raw_counters_account_for_every_row():
    """seen == kept + malformed + unknown_session, or the report is lying."""
    rows = [
        _runtime([{"session_key": SESSION, "umo": SESSION, "group_id": "1"}]),
        _annotations(SESSION, [_ambient("m1"), {"msg_id": ""}, "junk"]),
        _preference("topic_annotations_v1_" + "a" * 64, [_ambient("m2"), _ambient("m3")]),
        _preference("topic_annotations_v1_" + "b" * 64, "not-a-list"),
    ]

    result = parse_preferences(rows)
    stats = result.contract

    assert stats.annotation_keys == 3
    assert stats.unreadable_keys == 1
    assert stats.annotations_seen == 5
    assert stats.annotations_kept == 1
    assert stats.malformed == 2
    assert stats.unknown_session == 2
    assert stats.balanced is True
    assert stats.annotations_seen == (stats.annotations_kept + stats.malformed
                                      + stats.unknown_session)
    assert result.records == 1
    # The historical diagnostic keeps its meaning (unreadable keys + rejects).
    assert result.diagnostics["malformed"] == 3
    assert result.diagnostics["balanced"] is True


def test_oversized_records_are_counted_apart_from_other_rejects(monkeypatch):
    monkeypatch.setattr(ingest_module, "_MAX_KEY_BYTES", 40)
    rows = [
        _runtime([{"session_key": SESSION, "umo": SESSION}]),
        _annotations(SESSION, [{"msg_id": "m1"}, {"msg_id": "m2", "pad": "x" * 80}]),
    ]

    stats = parse_preferences(rows).contract

    assert stats.malformed == 1
    assert stats.oversized == 1
    assert stats.annotations_kept == 1
    assert stats.balanced is True


def test_truncated_records_are_reported_not_silently_dropped(monkeypatch):
    monkeypatch.setattr(ingest_module, "MAX_RECORDS_PER_SESSION", 2)
    rows = [
        _runtime([{"session_key": SESSION, "umo": SESSION}]),
        _annotations(SESSION, [_ambient("m1"), _ambient("m2"), _ambient("m3")]),
    ]

    result = parse_preferences(rows)

    assert result.contract.truncated == 1
    assert result.contract.annotations_seen == 2
    assert result.contract.balanced is True


# ---- contract plane: the facts normalisation erases ---------------------

def test_schema_versions_are_counted_before_normalisation_erases_them():
    """A stored sample always reads back as schema 2; the raw record does not."""
    legacy_trace = make_trace(topic_id="t1", topic_confidence=0.5,
                              evidence=[("ambient_baseline", "baseline", 0.2)])
    legacy_trace["routing_schema_version"] = 1
    legacy = make_record("m1", trace=legacy_trace, predicted_topic="t1", expected_topic="t1",
                         bot_targeted=True)
    legacy["annotation_schema_version"] = 1
    current = _ambient("m2", evidence=[("continuation_cue", "dialogue", 0.15)])
    no_trace = {"msg_id": "m3", "predicted_topic": "t1", "expected_topic": "t1"}

    result = parse_preferences([
        _runtime([{"session_key": SESSION, "umo": SESSION}]),
        _annotations(SESSION, [legacy, current, no_trace]),
    ])
    stats = result.contract

    assert stats.annotation_schema_versions == {"1": 1, "2": 1, "missing": 1}
    assert stats.routing_schema_versions == {"1": 1, "2": 1}
    assert stats.decision_trace_present == 2
    assert stats.decision_trace_absent == 1

    samples = build_dataset(result.annotations, session_meta=result.sessions)
    assert samples
    # Every stored trace claims schema 2 — which is exactly why the raw plane,
    # and not the sample plane, has to answer this question.
    assert {sample.trace["routing_schema_version"] for sample in samples} == {2}


def test_a_missing_additive_total_is_counted_before_it_becomes_zero():
    record = _ambient("m1", evidence=[("continuation_cue", "dialogue", 0.15)], total=None)

    result = parse_preferences([
        _runtime([{"session_key": SESSION, "umo": SESSION}]),
        _annotations(SESSION, [record]),
    ])

    assert result.contract.contribution_total_absent == 1
    assert result.contract.contribution_total_present == 0

    samples = build_dataset(result.annotations, session_meta=result.sessions)
    recipient = next(sample for sample in samples if sample.task == TASK_RECIPIENT)
    # The value is erased into a decisive-looking zero...
    assert recipient.features["base_score"] == 0.0
    # ...so the flag, not the value, is what the replay health is read from.
    assert recipient.contribution_total_recorded is False


def test_contribution_total_present_is_distinguished_from_zero():
    record = _ambient("m1", evidence=[("continuation_cue", "dialogue", 0.15)], total=0.0)

    result = parse_preferences([
        _runtime([{"session_key": SESSION, "umo": SESSION}]),
        _annotations(SESSION, [record]),
    ])

    assert result.contract.contribution_total_present == 1
    samples = build_dataset(result.annotations, session_meta=result.sessions)
    recipient = next(sample for sample in samples if sample.task == TASK_RECIPIENT)
    assert recipient.features["base_score"] == 0.0
    assert recipient.contribution_total_recorded is True


def test_topic_candidate_buckets_separate_missing_from_empty():
    rows = [
        _runtime([{"session_key": SESSION, "umo": SESSION}]),
        _annotations(SESSION, [
            topic_record("m1", predicted="t1", expected="t2", confidence=0.5),
            topic_record("m2", predicted="t1", expected="t2", confidence=0.5, candidates=[]),
            topic_record("m3", predicted="t1", expected="t2", confidence=0.5,
                         candidates=[[0.6, "t1"], [0.4, "t2"]]),
        ]),
    ]

    stats = parse_preferences(rows).contract

    assert stats.topic_candidates_missing == 1
    assert stats.topic_candidates_empty == 1
    assert stats.topic_candidates_nonempty == 1


def test_export_import_counts_the_same_way():
    payload = export_payload([(SESSION, _ambient("m1")), (SESSION, {"msg_id": ""})])

    result = parse_export(payload)

    assert result.contract.source == "export"
    assert result.contract.balanced is True
    assert result.contract.annotations_seen == 2
    assert result.contract.annotations_kept == 1
    assert result.contract.malformed == 1


def test_sessions_and_identity_are_counted_from_the_runtime_snapshot():
    other = "platform-b:GroupMessage:room"
    result = parse_preferences([
        _runtime([{"session_key": SESSION, "umo": SESSION, "group_id": "1"},
                  {"session_key": other, "umo": other, "group_id": "room"}]),
        _annotations(SESSION, [_ambient("m1")]),
        _annotations(other, [_ambient("m2")]),
    ])

    assert result.contract.sessions == 2
    assert result.contract.scope_sources == {"session_umo_equal": 2}
    assert result.contract.distinct_group_id_sessions == 2


# ---- sample plane: capabilities ----------------------------------------

def _recipient_batch(*, total: int, eligible: int, explicit: int = 0,
                     no_prior_bot: int = 0, no_score: int = 0):
    """A batch with an exact composition, in samples (one record yields a few)."""
    records = []
    index = 0
    for _ in range(eligible):
        index += 1
        records.append(_ambient(f"e{index}", evidence=[("continuation_cue", "dialogue", 0.15)]))
    for _ in range(explicit):
        index += 1
        records.append(_ambient(f"x{index}", evidence=[("bot_mention", "recipient", 0.5)]))
    for _ in range(no_prior_bot):
        index += 1
        records.append(_ambient(f"p{index}", evidence=[("ambient_baseline", "baseline", 0.2)]))
    for _ in range(no_score):
        index += 1
        records.append(_ambient(f"s{index}", evidence=[("continuation_cue", "dialogue", 0.15)],
                                total=None))
    assert len(records) == total
    return build_dataset([(SESSION, record) for record in records])


def test_recipient_replay_separates_every_reason_a_sample_cannot_move():
    samples = _recipient_batch(total=4, eligible=1, explicit=1, no_prior_bot=1, no_score=1)

    health = recipient_replay(samples, min_samples=1)

    assert health.detail["total"] == 4
    assert health.detail["eligible"] == 1
    assert health.detail["explicit"] == 1
    assert health.detail["no_prior_bot"] == 1
    assert health.detail["no_score"] == 1
    assert health.status == STATUS_WARNING
    assert any("结构化直判" in reason for reason in health.reasons)
    assert any("contribution_total" in reason for reason in health.reasons)


def test_status_distinguishes_unavailable_from_undersized():
    assert recipient_replay([], min_samples=1).status == STATUS_UNSUPPORTED
    assert recipient_replay([], min_samples=1).total == 0

    small = _recipient_batch(total=3, eligible=3)
    assert recipient_replay(small, min_samples=20).status == STATUS_INSUFFICIENT
    assert recipient_replay(small, min_samples=3).status == STATUS_OK

    # Coverage below the bar is a warning with the reasons attached, never a zero
    # masquerading as a measurement.
    mixed = _recipient_batch(total=10, eligible=5, explicit=5)
    assert recipient_replay(mixed, min_samples=1).status == STATUS_WARNING


def test_topic_attribution_matches_the_learners_own_denominator():
    samples = build_dataset([
        (SESSION, topic_record("m1", predicted="t1", expected="t2", confidence=0.5,
                               candidates=[[0.6, "t1"], [0.4, "t2"]])),
        (SESSION, topic_record("m2", predicted="t1", expected="t1", confidence=0.6)),
        (SESSION, topic_record("m3", predicted="UNKNOWN", expected="t3", confidence=0.2)),
    ])

    health = topic_attribution(samples, min_samples=1)

    assert health.total == 3
    assert health.eligible == 1
    assert health.detail["not_recorded"] == 2
    assert health.status == STATUS_WARNING
    assert any("没有记录候选集" in reason for reason in health.reasons)


def test_new_topic_samples_are_excluded_from_the_attribution_denominator():
    samples = build_dataset([
        (SESSION, topic_record("m1", predicted="UNKNOWN", expected="NEW", confidence=0.2)),
    ])

    health = topic_attribution(samples, min_samples=1)

    assert health.total == 0
    assert health.status == STATUS_UNSUPPORTED


def test_topic_threshold_replay_follows_the_replay_predicate():
    samples = build_dataset([
        (SESSION, topic_record("m1", predicted="t1", expected="t2", confidence=0.5,
                               candidates=[[0.6, "t1"], [0.4, "t2"]])),
        (SESSION, topic_record("m2", predicted="t1", expected="t1", confidence=0.6)),
        (SESSION, topic_record("m3", predicted="UNKNOWN", expected="t3", confidence=0.2)),
    ])

    health = topic_threshold_replay(samples, min_samples=1)

    # m1 and m2 are committed at some confidence; m3 has neither an assignment
    # nor a scored candidate, so no threshold can move it.
    assert health.eligible == 2
    assert health.total == 3
    assert health.detail["frozen"] == 1


def test_scope_identity_reports_the_level_instead_of_claiming_a_group():
    records = [(SESSION, _ambient("m1"))]
    confirmed = build_dataset(records, session_meta={SESSION: {"session_key": SESSION, "umo": SESSION}})
    fallback = build_dataset(records)

    with_meta = scope_identity(confirmed, min_samples=1)
    without_meta = scope_identity(fallback, min_samples=1)

    assert with_meta.status == STATUS_OK
    assert with_meta.eligible == with_meta.total
    assert with_meta.detail["host_confirmed"] == with_meta.total
    assert without_meta.status == STATUS_OK, "a fallback scope is still a usable scope"
    assert without_meta.detail["host_confirmed"] == 0
    assert any("会话" in reason for reason in without_meta.reasons)


def test_capabilities_cover_every_registered_entry():
    found = capabilities(_recipient_batch(total=2, eligible=2), min_samples=1)

    assert set(found) == {
        CAPABILITY_RECIPIENT_REPLAY, CAPABILITY_TOPIC_ATTRIBUTION,
        CAPABILITY_TOPIC_THRESHOLD_REPLAY, "reply_admission_replay", CAPABILITY_SCOPE_IDENTITY,
        "final_reply_outcome",
    }
    assert all(row.as_dict()["status_label"] for row in found.values())


# ---- assembly -----------------------------------------------------------

def test_the_final_send_outcome_is_unsupported_by_contract():
    """A capability the host cannot answer is a row, not an error and not silence."""
    from astrbot_plugin_dynamics_learning.core.quality import final_reply_outcome

    samples = _recipient_batch(total=3, eligible=3)
    rows = [sample for sample in samples if sample.task == "reply"]

    health = final_reply_outcome(samples, min_samples=1)

    assert rows, "the fixture is meant to carry reply supervision"
    assert health.status == STATUS_UNSUPPORTED
    assert health.eligible == 0
    assert health.total == len(rows)
    assert any("should_reply" in reason for reason in health.reasons)


def test_quality_report_without_an_import_says_so():
    report = quality_report(build_dataset([(SESSION, _ambient("m1"))]), now=10.0)

    assert report["generated_at"] == 10.0
    assert report["contract"] is None
    assert report["contract_findings"] == [
        "还没有导入记录：契约面（本体到底写了什么字段）暂无数据，先执行一次导入。"]
    assert report["dataset"]["scope_level"] == "session"
    assert report["dataset"]["timestamp_semantics"] == "annotated_at"


def test_quality_report_carries_the_contract_snapshot_and_its_age():
    rows = [
        _runtime([{"session_key": SESSION, "umo": SESSION}]),
        _annotations(SESSION, [_ambient("m1", evidence=[("ambient_baseline", "baseline", 0.2)])]),
    ]
    result = parse_preferences(rows)
    samples = build_dataset(result.annotations, session_meta=result.sessions)

    report = quality_report(samples, contract=result.contract.as_dict(), ingest_at=99.0, now=10.0)

    assert report["contract"]["annotations_kept"] == 1
    assert report["contract_at"] == 99.0
    assert report["contract"]["balanced"] is True
    assert any("标注 schema" in finding for finding in report["contract_findings"])


def test_contract_findings_name_the_field_that_is_missing():
    contract = {
        "balanced": True,
        "annotation_schema_versions": {"1": 3, "2": 7},
        "decision_trace_absent": 3,
        "topic_candidates": {"missing": 4, "empty": 1, "nonempty": 5},
        "contribution_total": {"present": 7, "absent": 0, "no_trace": 3},
        "truncated": 2,
    }

    findings = contract_findings(contract)

    assert any("1×3" in finding for finding in findings)
    assert any("decision_trace" in finding for finding in findings)
    assert any("4/10" in finding for finding in findings)
    assert any("截断" in finding for finding in findings)


def test_contract_findings_shout_when_the_counters_do_not_balance():
    findings = contract_findings({"balanced": False})

    assert any("不守恒" in finding for finding in findings)


# ---- plugin surface -----------------------------------------------------

@pytest.mark.asyncio
async def test_plugin_quality_payload_merges_both_planes(plugin):
    from .factories import annotated_sessions

    await plugin.ingest(source="export", payload=export_payload(
        annotated_sessions(sessions=3, per_session=4)))

    payload = await plugin.quality_payload()

    assert payload["dataset"]["samples"] > 0
    assert payload["contract"]["source"] == "export"
    assert payload["contract"]["balanced"] is True
    assert payload["contract_at"] is not None
    assert "topic_attribution" in payload["capabilities"]


@pytest.mark.asyncio
async def test_plugin_quality_is_empty_before_any_import(plugin):
    payload = await plugin.quality_payload()

    assert payload["dataset"]["samples"] == 0
    assert payload["contract"] is None
    assert payload["capabilities"][CAPABILITY_TOPIC_ATTRIBUTION]["status"] == STATUS_UNSUPPORTED


@pytest.mark.asyncio
async def test_session_filter_refuses_a_display_label(plugin):
    from astrbot_plugin_dynamics_learning.main import resolve_session_digest

    digest = resolve_session_digest(SESSION)
    assert digest == session_hash(SESSION)

    with pytest.raises(ValueError):
        resolve_session_digest(digest[:12])
    with pytest.raises(ValueError):
        await plugin.samples_payload(session=digest[:12])


@pytest.mark.asyncio
async def test_quality_endpoint_is_registered_and_answers(plugin):
    await plugin.ingest(source="export", payload=export_payload([(SESSION, _ambient("m1"))]))
    plugin.web.register()
    handlers = {route: handler for route, handler, _methods, _desc in plugin.context.routes}

    response = await handlers["/astrbot_plugin_dynamics_learning/quality"]()

    assert response["status"] == "ok"
    assert response["data"]["capabilities"]


@pytest.mark.asyncio
async def test_samples_endpoint_maps_a_bad_filter_to_400(plugin):
    async def broken(**_kwargs):
        raise ValueError("session 需要完整 64 位 scope_hash")

    plugin.samples_payload = broken  # type: ignore[method-assign]
    response = await plugin.web.samples()

    assert response["status"] == "error"
    assert response["status_code"] == 400
