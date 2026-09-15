"""Synthetic ChatDynamics annotation records with controlled signal.

The fixtures build exactly the shape `TopicAnnotations.save` writes, so the
ingest path is exercised end to end rather than against a convenient fiction.
Signal is deliberately imperfect: the recorded scorer must be wrong often
enough that a learner has something to improve on, and right often enough that
the baseline is not degenerate.
"""
from __future__ import annotations

import random
from typing import Any, Iterable, Mapping, Sequence

# Kept for the ladders that need a score ladder rather than an evidence draw.
SCORE_LADDER_NOTE = "ladder batches only carry recipient/reply supervision"

ANNOTATION_SCHEMA_VERSION = 2
ROUTING_SCHEMA_VERSION = 2


def make_trace(
    *,
    evidence: Sequence[tuple] = (),
    bot_targeted: bool = False,
    topic_id: str = "",
    topic_confidence: float = 0.0,
    topic_ambiguous: bool = False,
    participation_score: float | None = None,
    participation_level: str | None = None,
    recipient_confidence: float = 0.0,
    recipient_ambiguous: bool = True,
    state: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
    mode: str = "legacy",
) -> dict[str, Any]:
    facts = [{"code": item[0], "family": item[1], "strength": round(float(item[2]), 4),
              "source": item[3] if len(item) > 3 else "policy"} for item in evidence]
    families: dict[str, float] = {}
    for item in facts:
        families[item["family"]] = round(families.get(item["family"], 0.0) + item["strength"], 4)
    return {
        "routing_schema_version": ROUTING_SCHEMA_VERSION,
        "trace_version": 1,
        "calibrated": False,
        "parent": {"message_id": "", "confidence": 0.0, "ambiguous": True, "candidates": []},
        "topic": {"topic_id": topic_id, "confidence": round(topic_confidence, 4),
                  "threshold": 0.58, "margin_threshold": 0.06, "ambiguous": topic_ambiguous},
        "recipient": {"ids": ["bot"] if bot_targeted else [], "bot_targeted": bot_targeted,
                      "confidence": round(recipient_confidence, 4), "threshold": 0.7,
                      "ambiguous": recipient_ambiguous},
        "identity": identity or {"bot_reference": "none", "mention": False,
                                 "vocative": False, "subject": False},
        "participation": {
            "score": participation_score,
            "level": participation_level,
            "should_reply": None,
            "evidence": facts,
            "family_contributions": families,
            "contribution_total": round(sum(item["strength"] for item in facts), 4),
        },
        "state": {"pending_hover": False, "active_interlocutor": None, "intervening_users": 0,
                  "waiting_for_answer": False, "last_bot_was_question": False,
                  "last_bot_message_id": None, **(state or {})},
        "mode": mode,
        "weights_version": "routing-weights-v1",
    }


def make_record(
    msg_id: str,
    *,
    trace: dict[str, Any],
    predicted_topic: str = "UNKNOWN",
    expected_topic: str | None = None,
    error_type: str = "unknown",
    bot_targeted: bool | None = None,
    expected_reply: bool | None = None,
    recipient_error_type: str | None = None,
    annotated_at: float = 1000.0,
    topic_candidates: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "msg_id": msg_id,
        "predicted_topic": predicted_topic,
        "expected_topic": expected_topic if expected_topic is not None else predicted_topic,
        "error_type": error_type,
        "annotated_at": annotated_at,
        "routing": {"topic_confidence": trace["topic"]["confidence"],
                    "topic_ambiguous": trace["topic"]["ambiguous"],
                    "topic_status": "formed" if trace["topic"]["topic_id"] else "unformed"},
        "decision_trace": trace,
    }
    if topic_candidates is not None:
        # Structured rows must pass through unchanged; only the legacy pair form
        # needs normalising to a list.
        record["routing"]["topic_candidates"] = [
            list(row) if isinstance(row, (list, tuple)) else row for row in topic_candidates]
    if bot_targeted is not None:
        record["bot_targeted"] = bool(bot_targeted)
        record["recipient_correct"] = bool(bot_targeted) == bool(trace["recipient"]["bot_targeted"])
        record["recipient_ids"] = ["bot"] if bot_targeted else ["u9"]
    if expected_reply is not None:
        record["expected_reply"] = bool(expected_reply)
    if recipient_error_type is not None:
        record["recipient_error_type"] = recipient_error_type
    return record


AMBIENT_POOL = (
    ("temporal_gap", "temporal", 0.25),
    ("continuation_cue", "dialogue", 0.15),
    ("lexical_overlap", "topic", 0.15),
    ("active_interlocutor", "dialogue", 0.12),
    ("explicit_thread", "dialogue", 0.08),
    ("pending_hover", "dialogue", 0.20),
    ("intervening_messages", "dialogue", -0.15),
    ("human_quote", "recipient", -0.15),
    ("platform_wake", "platform", 0.15),
)


def ambient_record(
    msg_id: str,
    *,
    seed: int,
    biased: bool = True,
    annotated_at: float = 1000.0,
    session_topic: str = "t1",
) -> dict[str, Any]:
    """One ambient (scored) turn whose label depends on the evidence drawn.

    With `biased=True` the truth is a deterministic function of the drawn
    evidence plus noise, which is exactly the situation a re-weighted scorer can
    improve on and a memorised one cannot.
    """
    rng = random.Random(seed)
    evidence: list[tuple[str, str, float]] = [("ambient_baseline", "baseline", 0.20)]
    for code, family, strength in AMBIENT_POOL:
        if rng.random() < 0.45:
            evidence.append((code, family, strength))
    score = sum(item[2] for item in evidence)
    recorded_targeted = max(0.0, min(1.0, score)) >= 0.70
    if biased:
        # Truth follows the dialogue-family evidence more than the fixed
        # 0.70 cut does, with a little label noise.
        dialogue = sum(strength for _code, family, strength in evidence if family == "dialogue")
        truth = dialogue >= 0.30
        if rng.random() < 0.08:
            truth = not truth
    else:
        truth = recorded_targeted
    level = "strong" if recorded_targeted else ("hover" if score >= 0.40 else "weak")
    trace = make_trace(evidence=evidence, bot_targeted=recorded_targeted,
                       topic_id=session_topic, topic_confidence=round(rng.uniform(0.4, 0.9), 4),
                       participation_score=round(max(0.0, min(1.0, score)), 4),
                       participation_level=level,
                       recipient_confidence=0.5,
                       state={"intervening_users": rng.choice([0, 1, 3])})
    return make_record(msg_id, trace=trace, predicted_topic=session_topic,
                       expected_topic=session_topic,
                       bot_targeted=truth,
                       expected_reply=truth,
                       error_type="correct" if truth == recorded_targeted else "wrong_assignment",
                       recipient_error_type="correct" if truth == recorded_targeted else "wrong_recipient",
                       annotated_at=annotated_at)


def topic_record(
    msg_id: str,
    *,
    predicted: str,
    expected: str,
    confidence: float,
    ambiguous: bool = False,
    candidates: Sequence[Sequence[Any]] | None = None,
    annotated_at: float = 1000.0,
) -> dict[str, Any]:
    trace = make_trace(topic_id=predicted if predicted != "UNKNOWN" else "",
                       topic_confidence=confidence, topic_ambiguous=ambiguous,
                       evidence=[("ambient_baseline", "baseline", 0.20)])
    return make_record(msg_id, trace=trace, predicted_topic=predicted, expected_topic=expected,
                       error_type="correct" if predicted == expected else "topic_split",
                       annotated_at=annotated_at, topic_candidates=candidates)


# A pool tuned so the additive score lands mostly just below the host's 0.70
# cut, which is what makes a bounded threshold move worth anything.
STRICT_POOL = (
    ("continuation_cue", "dialogue", 0.15, 0.60),
    ("lexical_overlap", "topic", 0.15, 0.60),
    ("active_interlocutor", "dialogue", 0.12, 0.60),
    ("explicit_thread", "dialogue", 0.08, 0.60),
    ("platform_wake", "platform", 0.15, 0.50),
    ("temporal_gap", "temporal", 0.25, 0.40),
    ("pending_hover", "dialogue", 0.20, 0.30),
    ("intervening_messages", "dialogue", -0.15, 0.25),
    ("human_quote", "recipient", -0.15, 0.20),
)


def strict_cut_record(msg_id: str, *, seed: int, annotated_at: float = 1000.0,
                      session_topic: str = "t1") -> dict[str, Any]:
    """A turn where the host cut is genuinely too strict.

    The truth is "the bot is the addressee once the additive score reaches
    0.62", while the host commits at 0.70. Every sample in the 0.62–0.70 band is
    a false negative, and a bounded downward move of the cut can recover some of
    them — the situation the evaluation gate exists to detect.
    """
    rng = random.Random(seed)
    evidence: list[tuple[str, str, float]] = [("ambient_baseline", "baseline", 0.20)]
    for code, family, strength, probability in STRICT_POOL:
        if rng.random() < probability:
            evidence.append((code, family, strength))
    score = sum(item[2] for item in evidence)
    clamped = max(0.0, min(1.0, score))
    recorded_targeted = clamped >= 0.70
    truth = clamped >= 0.62
    if rng.random() < 0.03:
        truth = not truth
    level = "strong" if recorded_targeted else ("hover" if clamped >= 0.40 else "weak")
    trace = make_trace(evidence=evidence, bot_targeted=recorded_targeted,
                       topic_id=session_topic, topic_confidence=round(clamped, 4),
                       participation_score=round(clamped, 4), participation_level=level,
                       recipient_confidence=0.5)
    return make_record(msg_id, trace=trace, predicted_topic=session_topic,
                       expected_topic=session_topic, bot_targeted=truth, expected_reply=truth,
                       error_type="correct" if truth == recorded_targeted else "wrong_assignment",
                       recipient_error_type="correct" if truth == recorded_targeted else "missed_bot",
                       annotated_at=annotated_at)


def strict_cut_sessions(*, sessions: int = 24, per_session: int = 20,
                        start: float = 1_000_000.0) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for index in range(sessions):
        session = f"umo:group:{index}"
        for position in range(per_session):
            rows.append((session, strict_cut_record(
                f"s{index}-{position}", seed=index * 5000 + position * 7 + 3,
                annotated_at=start + index * 60 + position, session_topic=f"t{index}")))
    return rows


def ladder_record(msg_id: str, *, score: float, truth: bool,
                  annotated_at: float = 1000.0) -> dict[str, Any]:
    """One ambient turn at an **exact** additive score.

    `temporal_gap` carries the remainder so `contribution_total` lands on
    `score` to the last digit. That is what makes a threshold ladder
    reproducible: the test can predict the accuracy after a -5% cut by hand.
    """
    evidence = [("ambient_baseline", "baseline", 0.20)]
    remainder = round(float(score) - 0.20, 6)
    if remainder > 0:
        evidence.append(("temporal_gap", "temporal", remainder))
    recorded = float(score) >= 0.70
    level = "strong" if recorded else ("hover" if score >= 0.40 else "weak")
    trace = make_trace(evidence=evidence, bot_targeted=recorded,
                       participation_score=round(max(0.0, min(1.0, float(score))), 6),
                       participation_level=level, recipient_confidence=0.5)
    return make_record(msg_id, trace=trace, predicted_topic="UNKNOWN",
                       expected_topic="UNKNOWN", bot_targeted=truth, expected_reply=truth,
                       error_type="correct" if truth == recorded else "wrong_assignment",
                       recipient_error_type="correct" if truth == recorded else "missed_bot",
                       annotated_at=annotated_at)


def ladder_sessions(plan: Sequence[tuple[float, bool]], *, sessions: int = 5,
                    start: float = 1_000_000.0) -> list[tuple[str, dict[str, Any]]]:
    """Apply one `(score, truth)` plan to **every** session identically.

    The split is by session, so the holdout's composition is only predictable if
    every session has the same composition. That makes the arithmetic exact:
    with a per-session plan of `p` entries and `b` of them recovered by a step,
    the holdout accuracy moves by exactly `b/p`, whatever subset of sessions the
    split happens to pick.

    Only the recipient and reply tasks are produced (the topic label is
    `UNKNOWN`), so a tuning run over this batch has exactly one target error and
    its numbers can be checked by hand.
    """
    rows: list[tuple[str, dict[str, Any]]] = []
    for index in range(sessions):
        for position, (score, truth) in enumerate(plan):
            rows.append((f"umo:ladder:{index}", ladder_record(
                f"ld{index}-{position}", score=score, truth=truth,
                annotated_at=start + index * 60 + position)))
    return rows


# Per-step cuts the state machine actually reaches from a 0.70 baseline:
# step 1 -> 0.665, step 2 -> 0.63175, step 3 -> 0.60016, capped at 0.595.
PROMOTE_PER_SESSION = 250
STALL_PER_SESSION = 600
GUARD_PER_SESSION = 400


def promote_ladder(*, high: int = 186, quiet: int = 60, near: int = 2,
                   beyond: int = 2) -> list[tuple[float, bool]]:
    """The two-step promote case, sized to match a hand calculation.

    The 0.70 cut misses every truly-addressed turn at 0.55. Two of them sit at
    0.68, which the first -5% step recovers, and two at 0.65, which the second
    recovers. With a per-session plan of 250 that is +0.8% per step and +1.6%
    together, so the run continues exactly once and then promotes — never on the
    first step, and never through the target-error shortcut, because 64 turns
    per session are missed either way and the relative move stays under 10%.
    """
    plan: list[tuple[float, bool]] = [(0.95, True)] * high
    plan.extend([(0.55, True)] * quiet)
    plan.extend([(0.68, True)] * near)
    plan.extend([(0.65, True)] * beyond)
    return plan


def stall_ladder(*, high: int = 298, quiet: int = 300, near: int = 1,
                 beyond: int = 1) -> list[tuple[float, bool]]:
    """A batch where every step helps, but far too little to justify another.

    A plan of 600 puts each recovery at +0.17%: below the +0.2% stall line and
    below the +0.5% advance line, while the 1.6% relative error move stays under
    the 5% shortcut. Two such steps in a row must stop the run.
    """
    plan: list[tuple[float, bool]] = [(0.95, True)] * high
    plan.extend([(0.55, True)] * quiet)
    plan.extend([(0.68, True)] * near)
    plan.extend([(0.65, True)] * beyond)
    return plan


def guard_ladder(*, quiet: int = 390, hit: int = 8, near: int = 1,
                 beyond: int = 1) -> list[tuple[float, bool]]:
    """A batch where accuracy barely moves but precision collapses.

    Lowering the cut recovers the truly-addressed turn at 0.65 and, at the same
    time, starts firing on the false one at 0.68. Accuracy moves by about
    -0.25% while F1 falls about 5%, so the guard has to catch what the safe
    floor would only barely notice.
    """
    plan: list[tuple[float, bool]] = [(0.60, False)] * quiet
    plan.extend([(0.90, True)] * hit)
    plan.extend([(0.65, True)] * near)
    plan.extend([(0.68, False)] * beyond)
    return plan


def annotated_sessions(
    *,
    sessions: int = 12,
    per_session: int = 14,
    biased: bool = True,
    start: float = 1_000_000.0,
) -> list[tuple[str, dict[str, Any]]]:
    """A batch big enough for a holdout split that is not trivially small."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for index in range(sessions):
        session = f"umo:group:{index}"
        for position in range(per_session):
            seed = index * 1000 + position
            rows.append((session, ambient_record(
                f"m{index}-{position}", seed=seed, biased=biased,
                annotated_at=start + index * 60 + position,
                session_topic=f"t{index}",
            )))
    return rows


def topic_sessions(*, sessions: int = 10, per_session: int = 10) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    rng = random.Random(7)
    for index in range(sessions):
        session = f"umo:group:{index}"
        for position in range(per_session):
            # Two real topics per session, sometimes split, sometimes merged.
            expected = f"g{index}-{position // 5}"
            if rng.random() < 0.25:
                predicted = f"g{index}-{position // 5 + 1}"
            else:
                predicted = expected
            confidence = round(rng.uniform(0.35, 0.9), 4)
            rows.append((session, topic_record(
                f"t{index}-{position}", predicted=predicted, expected=expected,
                confidence=confidence, ambiguous=False,
                candidates=[[round(confidence - 0.05, 4), expected],
                            [round(confidence - 0.2, 4), f"g{index}-other"]]
                if confidence > 0.36 else None,
                annotated_at=1_000_000.0 + index * 60 + position,
            )))
    return rows


def recipient_record(
    msg_id: str,
    *,
    error: bool = False,
    kind: str = "missed_bot",
    annotated_at: float = 1000.0,
) -> dict[str, Any]:
    """One record that yields exactly one recipient sample and nothing else.

    The topic label is UNKNOWN and there is no reply supervision, so a batch of
    these has an exact per-scope composition — which is what a leave-one-out
    baseline and a smoothing prior have to be checked against by hand.
    """
    trace = make_trace(bot_targeted=False, recipient_confidence=0.5)
    return make_record(msg_id, trace=trace, predicted_topic="UNKNOWN", expected_topic="UNKNOWN",
                       bot_targeted=error, recipient_error_type=kind if error else "correct",
                       annotated_at=annotated_at)


def scope_batch(
    *,
    scopes: int = 3,
    per_scope: int = 14,
    wrong_scope: int = 0,
    wrong_ratio: float = 0.6,
    other_ratio: float = 0.1,
    days: int = 3,
    topics: int = 0,
    start: float = 1_000_000.0,
) -> list[tuple[str, dict[str, Any]]]:
    """Several scopes, one deliberately worse, spread over several annotation days.

    Three things the other fixtures do not produce and the scope view needs:
    more than one scope, a difference large enough to survive a leave-one-out
    baseline, and annotation timestamps on different days — without those the
    "stable" tier is unreachable no matter how much is reviewed.
    """
    rows: list[tuple[str, dict[str, Any]]] = []
    span = max(1, days)
    for scope_index in range(scopes):
        scope = f"umo:scope:{scope_index}"
        wrong = int(round(per_scope * (wrong_ratio if scope_index == wrong_scope else other_ratio)))
        for position in range(per_scope):
            rows.append((scope, recipient_record(
                f"s{scope_index}-m{position}", error=position < wrong,
                annotated_at=start + (position % span) * 86_400.0 + scope_index * 60 + position)))
        for position in range(topics):
            expected = f"t{scope_index}-{position // 4}"
            rows.append((scope, topic_record(
                f"s{scope_index}-t{position}",
                predicted=expected if position % 4 else f"t{scope_index}-other",
                expected=expected, confidence=0.62,
                candidates=[[0.62, expected], [0.4, "t-other"]],
                annotated_at=start + (position % span) * 86_400.0 + scope_index * 60 + position)))
    return rows


def export_payload(rows: Iterable[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for session, record in rows:
        grouped.setdefault(session, []).append(record)
    return {"sessions": [{"session_key": key, "records": value}
                         for key, value in grouped.items()]}


# ---- schema 3 fixtures ---------------------------------------------------

ROUTING_SCHEMA_VERSION_V3 = 3


def candidate(topic_id, score, *, rank=None, evidence=None):
    """One structured candidate in the schema 3 shape."""
    row = {"topic_id": topic_id, "final_score": round(float(score), 6)}
    if rank is not None:
        row["rank"] = rank
    if evidence is not None:
        row["evidence"] = {key: round(float(value), 6) for key, value in evidence.items()}
    return row


def outcome_block(value=None, *, delivered=None, reason="", stage=""):
    """A schema 3 outcome block, carrying only the fields the caller wrote."""
    block = {}
    if value is not None:
        block["final_outcome"] = value
    if delivered is not None:
        block["delivered"] = delivered
    if reason:
        block["suppression_reason"] = reason
    if stage:
        block["stage"] = stage
    return block


def shadow_block(*, policy_id="policy_v3", baseline_reply=False, shadow_reply=True,
                 baseline_threshold=0.70, shadow_threshold=0.67, score=0.68,
                 reason="ambient", recorded_at=1000.0):
    """One shadow comparison, in the shape the host records."""
    return {
        "policy_id": policy_id,
        "baseline_threshold": baseline_threshold,
        "shadow_threshold": shadow_threshold,
        "baseline_reply": baseline_reply,
        "shadow_reply": shadow_reply,
        "changed": bool(baseline_reply) != bool(shadow_reply),
        "score": score,
        "baseline_margin": round(score - baseline_threshold, 6),
        "shadow_margin": round(score - shadow_threshold, 6),
        "reason": reason,
        "recorded_at": recorded_at,
    }


def shadow_record(msg_id, *, expected_reply, shadow, annotated_at=1000.0,
                  level=None, selected_topic="", outcome=None):
    """A message carrying a reply label and a recorded shadow comparison."""
    trace = make_trace_v3(
        outcome=outcome, selected_topic=selected_topic,
        participation_level=(level or ("strong" if shadow["baseline_reply"] else "weak")),
        participation_score=shadow["score"],
    )
    trace["shadow"] = dict(shadow)
    return make_record(msg_id, trace=trace, predicted_topic="UNKNOWN", expected_topic="UNKNOWN",
                       expected_reply=expected_reply, annotated_at=annotated_at)


def make_trace_v3(*, candidates=None, selected_topic="", outcome=None, shadow=None, **kwargs):
    """A schema 3 trace: every schema 2 field, plus routing and outcome."""
    trace = make_trace(**kwargs)
    trace["routing_schema_version"] = ROUTING_SCHEMA_VERSION_V3
    routing = {"selected_topic": selected_topic or trace["topic"]["topic_id"]}
    if candidates is not None:
        routing["topic_candidates"] = [
            dict(row) if isinstance(row, Mapping) else row for row in candidates]
    trace["routing"] = routing
    if outcome is not None:
        trace["outcome"] = outcome
    if shadow is not None:
        trace["shadow"] = dict(shadow)
    return trace


def delivered_record(msg_id, *, expected_reply=True, annotated_at=1000.0, **kwargs):
    """A message that was admitted and then actually sent."""
    trace = make_trace_v3(
        outcome=outcome_block("delivered", delivered=True, stage="delivery"),
        participation_level="strong" if expected_reply else "weak",
        participation_score=0.8 if expected_reply else 0.2,
        **kwargs)
    return make_record(msg_id, trace=trace, predicted_topic="UNKNOWN", expected_topic="UNKNOWN",
                       expected_reply=expected_reply, annotated_at=annotated_at)


def suppressed_record(msg_id, *, reason="asleep_ambient", expected_reply=True,
                      annotated_at=1000.0, **kwargs):
    """Admitted, then stopped by the gate — the case schema 2 read as a miss."""
    trace = make_trace_v3(
        outcome=outcome_block("suppressed", delivered=False, reason=reason, stage="gate"),
        participation_level="strong",
        participation_score=0.8,
        **kwargs)
    return make_record(msg_id, trace=trace, predicted_topic="UNKNOWN", expected_topic="UNKNOWN",
                       expected_reply=expected_reply, annotated_at=annotated_at)


def evidenced_policy(params=None):
    """Consistent offline evidence fixture for storage/state tests, not evaluator tests."""
    from dataclasses import replace
    from astrbot_plugin_dynamics_learning.core.policy import candidate_from, baseline_config_hash
    candidate = candidate_from(params or {"strong_addressivity_threshold": 0.67})
    identity = {"metric_schema_version": 2,
                "candidate_hash": baseline_config_hash(candidate.params),
                "baseline_hash": baseline_config_hash(candidate.baseline),
                "dataset_fingerprint": "fixture-dataset"}
    return replace(candidate, training_dataset={"fingerprint": "fixture-dataset"},
                   holdout_result={"split": identity}, forward_result={"split": identity},
                   target={"baseline_config_hash": identity["baseline_hash"]},
                   evidence={"metric_schema_version": 2, "final_validation": {"verdict": "accepted"}, "dataset_gate_ok": True})
