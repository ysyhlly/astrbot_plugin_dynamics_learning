"""Cross-repository contract test: the host writes it, the reader must read it.

Every other test on both sides uses a fixture. This one does not: it calls the
**real** ChatDynamics `build_routing_trace` and `outcome_recorder`, turns the
result into the annotation record `TopicAnnotations.save` writes, and runs it
through the real Dynamics Learning ingest -> samples -> attribution path.

That matters because the two repositories version their protocols independently
and neither one's fixtures can catch a rename that only happened on one side.
When the host package is not importable the test is skipped rather than faked —
a cross-repo check that quietly falls back to a fixture is the failure mode it
exists to prevent.
"""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core import buckets
from astrbot_plugin_dynamics_learning.core.attribution import attribute
from astrbot_plugin_dynamics_learning.core.ingest import parse_export
from astrbot_plugin_dynamics_learning.core.samples import (
    TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME, build_dataset,
)
from astrbot_plugin_dynamics_learning.core.trace import (
    CANDIDATE_EVIDENCE_FULL, SCHEMA_V3, parse_decision_trace, trace_from_sample_record,
)

host = pytest.importorskip(
    "astrbot_plugin_chat_dynamics.core.routing_trace",
    reason="ChatDynamics 不在导入路径上；跨仓库契约检查不能退化成 fixture")
recorder = pytest.importorskip("astrbot_plugin_chat_dynamics.core.outcome_recorder")

SESSION = "umo:group:cross-repo"


def _host_routing(**overrides):
    """A routing mapping with the recipient facts a real turn always has.

    Without them the trace says "the bot was not the addressee", which the
    attribution chain correctly reads as a recipient error — and every later
    layer of the test would then be measuring that instead of what it meant to.
    """
    routing = {"topic_id": "t1", "topic_confidence": 0.72,
               "addressee_ids": ["bot"], "bot_is_addressee": True,
               "addressee_confidence": 0.9}
    routing.update(overrides)
    return routing


def _host_trace(**kwargs):
    kwargs.setdefault("routing", _host_routing())
    return host.build_routing_trace(**kwargs)


def _annotation_record(trace, *, expected_topic="t1", predicted_topic="t1",
                       bot_targeted=True, expected_reply=True):
    """The record shape `TopicAnnotations.save` writes, reduced to the fields
    the learning layer reads. Built here from a **real** host trace."""
    return {
        "annotation_schema_version": 2,
        "msg_id": "m1",
        "predicted_topic": predicted_topic,
        "expected_topic": expected_topic,
        "error_type": "correct",
        "annotated_at": 1_000.0,
        "routing": {"topic_confidence": trace["topic"]["confidence"],
                    "topic_ambiguous": trace["topic"]["ambiguous"],
                    "topic_status": "formed"},
        "bot_targeted": bot_targeted,
        "expected_reply": expected_reply,
        "decision_trace": trace,
    }


def test_the_host_trace_is_read_as_schema_three():
    trace = _host_trace(
        routing=_host_routing(topic_candidates=[[0.72, "t1"], [0.4, "t0"]],
                              topic_candidate_evidence={"t1": {"semantic": 0.81},
                                                        "t0": {"semantic": 0.32}}),
        participation={"score": 0.8, "level": "strong", "should_reply": None,
                       "evidence": [], "family_contributions": {}, "contribution_total": 0.8})

    parsed = trace_from_sample_record(_annotation_record(trace))

    assert parsed.source_schema == SCHEMA_V3
    assert parsed.trace_schema_version == SCHEMA_V3
    assert parsed.candidate_evidence == CANDIDATE_EVIDENCE_FULL
    assert [row.topic_id for row in parsed.topic_candidates] == ["t1", "t0"]
    assert parsed.topic_candidates[0].evidence == {"semantic": 0.81}
    assert parsed.selected_topic == "t1"


def test_a_suppression_recorded_by_the_host_is_not_a_router_error():
    """The whole point of schema 3, exercised through both codebases."""
    node = type("Node", (), {"metadata": {}})()
    recorder.mark_suppressed(node, "asleep_ambient")
    trace = _host_trace(
        participation={"score": 0.8, "level": "strong", "should_reply": None},
        outcome=node.metadata["outcome"])

    samples = build_dataset([(SESSION, _annotation_record(trace))])
    by_task = {sample.task: sample for sample in samples}

    assert by_task[TASK_REPLY_ADMISSION].correct is True, "准入判定是对的"
    outcome = by_task[TASK_REPLY_OUTCOME]
    assert outcome.correct is False
    assert outcome.error_type == buckets.GATE_SUPPRESSION

    row = next(item for item in attribute(samples))
    assert row.bucket == buckets.GATE_SUPPRESSION
    assert row.evidence["suppression_reason"] == "asleep_ambient"
    assert row.is_model_error is False


def test_a_delivered_turn_records_an_outcome_sample_that_is_correct():
    node = type("Node", (), {"metadata": {}})()
    recorder.mark_delivered(node)
    trace = _host_trace(
        participation={"score": 0.8, "level": "strong", "should_reply": None},
        outcome=node.metadata["outcome"])

    samples = build_dataset([(SESSION, _annotation_record(trace))])
    outcome = next(s for s in samples if s.task == TASK_REPLY_OUTCOME)

    assert outcome.correct is True
    assert outcome.outcome.is_delivered is True
    assert attribute(samples)[0].bucket == buckets.OK


def test_an_explicit_host_stage_survives_an_unknown_reason():
    """An explicit host stage is evidence even for a newly introduced reason."""
    trace = _host_trace(
        participation={"score": 0.8, "level": "strong", "should_reply": None},
        outcome={"final_outcome": "suppressed", "delivered": False,
                 "suppression_reason": "some_future_reason", "stage": "gate"})

    samples = build_dataset([(SESSION, _annotation_record(trace))])
    row = attribute(samples)[0]

    assert row.bucket == buckets.GATE_SUPPRESSION


def test_a_not_attempted_turn_is_a_participation_error_when_reply_was_expected():
    node = type("Node", (), {"metadata": {}})()
    recorder.mark_not_attempted(node)
    trace = _host_trace(
        participation={"score": 0.2, "level": "weak", "should_reply": None},
        outcome=node.metadata["outcome"])

    samples = build_dataset([(SESSION, _annotation_record(trace))])
    row = attribute(samples)[0]

    assert row.bucket == buckets.PARTICIPATION_ERROR
    assert row.is_model_error is True


def _host_effective(config=None):
    """The host's effective values for the six shared parameters.

    Mirrors DynamicsLearningPlugin._learning_policy_effective_config: read the
    configured fields, then apply the documented topic_commit_threshold
    derivation (max(0.58, join + 0.10) below the legacy threshold).
    """
    from astrbot_plugin_chat_dynamics.core.config import parse_runtime_config
    from astrbot_plugin_chat_dynamics.core.thread_router import TOPIC_JOIN_THRESHOLD

    from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY

    host, _warnings = parse_runtime_config(config or {})
    values = {name: float(getattr(host, name)) for name in BASE_POLICY}
    if not values["topic_commit_threshold"]:
        join = values["topic_join_threshold"]
        values["topic_commit_threshold"] = (max(TOPIC_JOIN_THRESHOLD, join + 0.10)
                                            if join < TOPIC_JOIN_THRESHOLD else join)
    return values


def test_the_two_sides_agree_on_the_default_baseline():
    """The digest is only meaningful if both sides hash the same values.

    Dynamics Learning's BASE_POLICY is its own copy of this plugin's defaults,
    and topic_commit_threshold is *derived* on the host rather than stored. If
    either drifts, every published baseline_config_hash stops matching and
    active becomes unreachable — silently, because a mismatch looks exactly
    like "the operator changed the configuration".
    """
    from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY

    values = _host_effective()

    for name, value in BASE_POLICY.items():
        assert values[name] == pytest.approx(value), (
            f"{name}: Learning {value} vs host {values[name]}")


def test_the_host_derives_the_topic_commit_threshold_the_documented_way():
    assert _host_effective({"topic_join_threshold": 0.52})["topic_commit_threshold"] \
        == pytest.approx(0.62)
    assert _host_effective({"topic_join_threshold": 0.70})["topic_commit_threshold"] \
        == pytest.approx(0.70)
    assert _host_effective({"topic_commit_threshold": 0.66})["topic_commit_threshold"] \
        == pytest.approx(0.66)


def test_a_published_policy_resolves_to_the_same_override_on_the_host():
    """The consumer's `would-override` must equal what Learning published."""
    consumer_module = pytest.importorskip("astrbot_plugin_chat_dynamics.core.learning_policy")
    from astrbot_plugin_dynamics_learning.core.policy import (
        STATUS_PROMOTED, candidate_from, published_payload,
    )

    effective = _host_effective()
    version = "v1.6.2"
    candidate = candidate_from({"strong_addressivity_threshold": 0.67}).with_fields(
        training_dataset={"fingerprint": "9f2c1a", "samples": 120, "sessions": 6},
        compatibility={"trace_schema_version": 3, "trace_schema_versions": {"3": 120}},
        target={"chat_dynamics_version": version,
                "baseline_config_hash": consumer_module.baseline_config_hash(effective),
                "validated_host_versions": [version]},
    ).with_status(STATUS_PROMOTED)

    payload = published_payload([candidate])
    decision = consumer_module.resolve(payload, mode="active", host_version=version,
                                       effective_config=effective)

    assert decision.status == consumer_module.STATUS_ACTIVE
    assert decision.applied is True
    # The producer publishes the whole resolved set, so the consumer's
    # would-override must equal it key for key — not just the parameter that
    # moved. A consumer that invented the unchanged ones would be applying a
    # configuration nobody published.
    assert decision.overrides == dict(candidate.params)
    assert decision.overrides["strong_addressivity_threshold"] == pytest.approx(0.67)


def test_a_shadow_mode_consumer_marks_the_mismatch_and_applies_nothing():
    consumer_module = pytest.importorskip("astrbot_plugin_chat_dynamics.core.learning_policy")
    from astrbot_plugin_dynamics_learning.core.policy import (
        STATUS_PROMOTED, candidate_from, published_payload,
    )

    effective = _host_effective()
    candidate = candidate_from({"strong_addressivity_threshold": 0.67}).with_fields(
        target={"chat_dynamics_version": "v1.6.1",
                "baseline_config_hash": consumer_module.baseline_config_hash(effective),
                "validated_host_versions": ["v1.6.1"]},
    ).with_status(STATUS_PROMOTED)

    decision = consumer_module.resolve(published_payload([candidate]), mode="shadow",
                                       host_version="v1.6.2", effective_config=effective)

    assert decision.status == consumer_module.STATUS_SHADOW
    assert decision.applied is False, "shadow 绝不实际应用"
    assert decision.version_mismatch is True


def _host_built_sessions(sessions=16, per_session=18, start=None):
    """Annotation records whose traces were built by the **real** host module.

    The fixtures already produce the routing and participation inputs; this
    routes them through ChatDynamics' own allowlisted builder, so what the
    learner reads is a schema 3 trace produced by the same code path a deployed
    host uses — not a fixture shaped like one.
    """
    import time as _time
    from astrbot_plugin_dynamics_learning.tests.factories import annotated_sessions

    rows = annotated_sessions(sessions=sessions, per_session=per_session, biased=True,
                              start=(start if start is not None else _time.time() - 7200))
    rebuilt = []
    for session, record in rows:
        fixture = record.get("decision_trace") or {}
        trace = host.build_routing_trace(
            routing={"topic_id": fixture.get("topic", {}).get("topic_id", ""),
                     "topic_confidence": fixture.get("topic", {}).get("confidence", 0.0),
                     "addressee_ids": (["bot"] if fixture.get("recipient", {}).get("bot_targeted")
                                       else []),
                     "bot_is_addressee": fixture.get("recipient", {}).get("bot_targeted", False),
                     "addressee_confidence": fixture.get("recipient", {}).get("confidence", 0.0)},
            identity=fixture.get("identity"),
            participation=fixture.get("participation"),
            state=fixture.get("state"),
        )
        record = dict(record)
        record["decision_trace"] = trace
        rebuilt.append((session, record))
    return rebuilt


def test_the_whole_loop_from_a_host_trace_to_a_resolved_policy():
    """③ 跨仓库闭环：本体写轨迹 → 学习层导入训练验证 → 发布 → 本体读取并确认一致。

    Every stage uses the real module from its own repository. The one thing this
    test exists to catch is a rename or a shape change on one side that the
    other side's fixtures would happily keep passing.
    """
    consumer_module = pytest.importorskip("astrbot_plugin_chat_dynamics.core.learning_policy")

    from astrbot_plugin_dynamics_learning.core.config import LearningConfig
    from astrbot_plugin_dynamics_learning.core.policy import (
        BASE_POLICY, STATUS_PROMOTED, STATUS_VALIDATED, published_payload,
    )
    from astrbot_plugin_dynamics_learning.core.report import analyze
    from astrbot_plugin_dynamics_learning.core.samples import build_dataset

    # 1. ChatDynamics 生成真实 trace
    rows = _host_built_sessions()
    samples = build_dataset(rows)
    assert samples
    assert {sample.trace["trace_schema_version"] for sample in samples} == {3}

    # 2. 学习层导入、训练、验证
    host_version = _host_effective_version()
    result = analyze(samples, config=LearningConfig(), host_version=host_version,
                     now=10_000.0)
    assert result.dataset_gate["ok"] is True, result.dataset_gate.get("blocked_by")

    candidate = None
    if result.evaluation is not None and result.evaluation.candidate is not None:
        candidate = result.evaluation.candidate
    candidate = candidate or next((run.candidate for run in result.tuning
                                   if run.candidate is not None), None)
    assert candidate is not None, "这批数据应当产出一个候选"
    assert candidate.compatibility["trace_schema_version"] == 3
    assert candidate.training_dataset["fingerprint"]
    assert candidate.target["baseline_config_hash"]

    # 3. 采纳并发布（validated -> promoted 是允许的箭头，且会记下 shadow 与否）
    promoted = candidate.with_status(STATUS_VALIDATED).with_status(STATUS_PROMOTED)
    payload = published_payload([promoted], issued_at=10_000.0)

    # 4. 本体读取，并确认 would-override 与学习层发布的完全一致
    effective = _host_effective()
    decision = consumer_module.resolve(
        payload, mode="active",
        host_version=_host_effective_version(),
        effective_config=effective)

    assert decision.applied is True, decision.reasons
    assert decision.overrides == dict(promoted.params)
    assert consumer_module.baseline_config_hash(BASE_POLICY) == \
        consumer_module.baseline_config_hash({name: BASE_POLICY[name] for name in BASE_POLICY})
    applied = consumer_module.resolve(
        payload, mode="active", host_version="v0.0.0", effective_config=effective)
    assert applied.applied is False, "版本不匹配时绝不应用"
    assert applied.version_mismatch is True


def _host_effective_version():
    runtime_persistence = pytest.importorskip(
        "astrbot_plugin_chat_dynamics.core.runtime_persistence")
    return runtime_persistence.host_version()


def test_a_host_trace_that_never_recorded_an_outcome_still_trains():
    """The schema 2 path must keep working; only the outcome layer is empty."""
    from astrbot_plugin_dynamics_learning.core.samples import (
        TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME, build_dataset,
    )

    record = _annotation_record(_host_trace(participation={
        "score": 0.8, "level": "strong", "should_reply": None}))
    record["expected_topic"] = "t1"
    record["decision_trace"]["topic"]["topic_id"] = "t1"
    record["predicted_topic"] = "t1"
    record["decision_trace"]["trace_schema_version"] = 2

    samples = build_dataset([(SESSION, record)])
    tasks = {sample.task for sample in samples}

    assert TASK_REPLY_ADMISSION in tasks
    assert TASK_REPLY_OUTCOME not in tasks, "schema 2 不产生结果层样本"
    assert all(sample.outcome_unavailable for sample in samples)


def test_a_host_shadow_decision_reads_as_one_disagreement_end_to_end():
    """⑤ 闭环的影子分支：本体算判定 → 冻结进轨迹 → 学习层读出分歧。

    The comparison is computed by the host at decision time and read back by
    the learning layer. Both sides have to agree on what "changed" means, or
    the subset the whole evaluation is built on would be a different set on
    each side of the contract.
    """
    consumer_module = pytest.importorskip("astrbot_plugin_chat_dynamics.core.learning_policy")
    from astrbot_plugin_dynamics_learning.core.policy import (
        STATUS_PROMOTED, candidate_from, published_payload,
    )
    from astrbot_plugin_dynamics_learning.core.samples import build_dataset
    from astrbot_plugin_dynamics_learning.core.shadow import disagreement_table, shadow_rows

    effective = _host_effective()
    version = _host_effective_version()
    candidate = candidate_from({"strong_addressivity_threshold": 0.67}).with_fields(
        target={"chat_dynamics_version": version,
                "baseline_config_hash": consumer_module.baseline_config_hash(effective),
                "validated_host_versions": [version]},
    ).with_status(STATUS_PROMOTED)

    consumer = consumer_module.LearningPolicyConsumer(mode="shadow", host_version=version)
    consumer.decision = consumer_module.resolve(
        published_payload([candidate]), mode="shadow", host_version=version,
        effective_config=effective)
    assert consumer.decision.status == consumer_module.STATUS_SHADOW

    # A turn the baseline cut off at 0.70 and the policy would admit at 0.67.
    shadow = consumer.shadow_decision(
        score=0.68, level="hover", evidence_codes=["ambient_baseline"],
        has_prior_bot=True, baseline_threshold=0.70)
    assert shadow["changed"] is True

    trace = host.build_routing_trace(
        routing=_host_routing(),
        participation={"score": 0.68, "level": "hover", "should_reply": None},
        shadow=shadow)
    samples = build_dataset([(SESSION, _annotation_record(trace))])

    rows = shadow_rows(samples)
    assert len(rows) == 1
    assert rows[0].policy_id == candidate.version
    assert rows[0].changed is True
    assert rows[0].baseline_reply is False and rows[0].shadow_reply is True

    table = disagreement_table(rows)
    assert table["changed"] == 1
    assert table["shadow_only"] == 1, "人工标了该回复，策略判对了、基线判错了"
    assert table["net_gain"] == 1
    assert table["balanced"] is True


def test_a_structural_turn_records_no_disagreement():
    """主机的结构化短路与学习层的判定必须是同一件事。"""
    consumer_module = pytest.importorskip("astrbot_plugin_chat_dynamics.core.learning_policy")
    from astrbot_plugin_dynamics_learning.core.policy import (
        STATUS_PROMOTED, candidate_from, published_payload,
    )
    from astrbot_plugin_dynamics_learning.core.samples import build_dataset
    from astrbot_plugin_dynamics_learning.core.shadow import disagreement_table, shadow_rows

    effective = _host_effective()
    version = _host_effective_version()
    candidate = candidate_from({"strong_addressivity_threshold": 0.67}).with_fields(
        target={"chat_dynamics_version": version,
                "baseline_config_hash": consumer_module.baseline_config_hash(effective),
                "validated_host_versions": [version]},
    ).with_status(STATUS_PROMOTED)
    consumer = consumer_module.LearningPolicyConsumer(mode="shadow", host_version=version)
    consumer.decision = consumer_module.resolve(
        published_payload([candidate]), mode="shadow", host_version=version,
        effective_config=effective)

    shadow = consumer.shadow_decision(
        score=0.9, level="strong", evidence_codes=["bot_mention", "ambient_baseline"],
        has_prior_bot=True, baseline_threshold=0.70)
    assert shadow["reason"] == "structural" and shadow["changed"] is False

    trace = host.build_routing_trace(
        routing=_host_routing(),
        participation={"score": 0.9, "level": "strong", "should_reply": None},
        shadow=shadow)
    samples = build_dataset([(SESSION, _annotation_record(trace))])
    table = disagreement_table(shadow_rows(samples))

    assert table["changed"] == 0
    assert table["both_correct"] == 1
    assert table["balanced"] is True


def test_the_host_runtime_snapshot_reports_a_version_the_reader_can_record():
    """`plugin_version` is what makes `target.chat_dynamics_version` possible."""
    runtime_persistence = pytest.importorskip(
        "astrbot_plugin_chat_dynamics.core.runtime_persistence")

    assert runtime_persistence.host_version().startswith("v1.")


def test_an_export_round_trip_keeps_the_schema_three_facts():
    """The path a real deployment takes: host records -> shared preferences ->
    export -> ingest -> samples, with nothing re-derived in between."""
    node = type("Node", (), {"metadata": {}})()
    recorder.mark_suppressed(node, "cool_command")
    trace = _host_trace(
        routing=_host_routing(topic_candidates=[[0.72, "t1"]]),
        participation={"score": 0.8, "level": "strong", "should_reply": None},
        outcome=node.metadata["outcome"])
    payload = {"sessions": [{"session_key": SESSION,
                             "records": [_annotation_record(trace)]}]}

    result = parse_export(payload)
    samples = build_dataset(result.annotations, session_meta=result.sessions)

    assert result.diagnostics["trace_schema_versions"] == {"3": 1}
    parsed = parse_decision_trace(samples[0].trace)
    assert parsed.source_schema == SCHEMA_V3
    assert parsed.outcome.suppression_reason == "cool_command"
