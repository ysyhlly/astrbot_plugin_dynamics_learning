"""Read-only normalisation of a ChatDynamics routing decision trace.

The host stores `decision_trace` snapshots built by
`astrbot_plugin_chat_dynamics.core.routing_trace.build_routing_trace`
(schema 2). That snapshot is already a field-allowlisted copy, so this module
never has to strip message text: it only has to be defensive about shape.

Contract version 2 mirrors:
  routing_schema_version, participation.{evidence, family_contributions,
  contribution_total}, recipient, topic, identity, state, mode, weights_version.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

TRACE_CONTRACT_VERSION = 2

# Mirrors astrbot_plugin_chat_dynamics.core.participation_policy.EVIDENCE_CODES.
EVIDENCE_CODES = frozenset({
    "canonical_recipient", "bot_mention", "other_mention", "vocative", "bot_reply",
    "routed_bot", "routed_other", "bot_subject", "ambient_baseline", "human_quote",
    "platform_wake", "temporal_gap", "intervening_messages", "continuation_cue",
    "lexical_overlap", "embedding_without_tokens", "recipient_confidence",
    "active_interlocutor", "explicit_thread", "pending_hover",
})
EVIDENCE_FAMILIES = frozenset({"recipient", "baseline", "platform", "temporal", "dialogue", "topic"})

# Codes that short-circuit the host policy: the decision is structural, not scored.
EXPLICIT_CODES = frozenset({
    "canonical_recipient", "bot_mention", "other_mention", "vocative", "bot_reply",
    "routed_bot", "routed_other", "bot_subject",
})
# Codes the host scores additively in its ambient tier.
AMBIENT_CODES = frozenset(EVIDENCE_CODES - EXPLICIT_CODES)

LEVELS = ("strong", "hover", "weak")
_UNKNOWN_FAMILY = "unknown"


def _finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def _optional_finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _text(value: Any, limit: int = 160) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _identifier_list(value: Any, limit: int = 32) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[str] = []
    for item in value[:limit]:
        if isinstance(item, str) and item:
            result.append(item[:160])
    return tuple(dict.fromkeys(result))


def _flag(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _optional_flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


@dataclass(frozen=True)
class EvidenceFact:
    code: str
    family: str
    strength: float
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "family": self.family,
                "strength": self.strength, "source": self.source}


@dataclass(frozen=True)
class DecisionTrace:
    """A frozen, JSON-safe view of one routing decision."""

    contract_version: int = TRACE_CONTRACT_VERSION
    trace_version: int = 1
    mode: str = "legacy"
    weights_version: str = "default"

    topic_id: str = ""
    topic_confidence: float = 0.0
    topic_threshold: float | None = None
    topic_margin_threshold: float | None = None
    topic_ambiguous: bool = False

    recipient_ids: tuple[str, ...] = ()
    bot_targeted: bool = False
    recipient_confidence: float = 0.0
    recipient_threshold: float | None = None
    recipient_ambiguous: bool = True

    parent_message_id: str = ""
    parent_confidence: float = 0.0
    parent_ambiguous: bool = True

    identity: Mapping[str, Any] = field(default_factory=dict)
    state: Mapping[str, Any] = field(default_factory=dict)

    participation_score: float | None = None
    participation_level: str | None = None
    should_reply: bool | None = None

    evidence: tuple[EvidenceFact, ...] = ()
    family_contributions: Mapping[str, float] = field(default_factory=dict)
    contribution_total: float = 0.0

    ledger_entries: int = 0
    degraded: bool = False

    # ---- derived views -------------------------------------------------

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(item.code for item in self.evidence)

    def strengths(self) -> dict[str, float]:
        return {item.code: item.strength for item in self.evidence}

    @property
    def explicit_code(self) -> str | None:
        """The host's short-circuit code, if the trace recorded one."""
        for item in self.evidence:
            if item.code in EXPLICIT_CODES:
                return item.code
        return None

    @property
    def is_explicit(self) -> bool:
        return self.explicit_code is not None

    @property
    def prior_bot_proxy(self) -> int:
        """1 when the host produced a scored (non short-circuit, non early-return) turn.

        The host's `ParticipationPolicy.evaluate` returns early with only
        `ambient_baseline` (plus an optional `human_quote` penalty) when the
        session has no prior bot message. That exact evidence signature is the
        only observable proxy for `has_prior_bot`, so it is named a proxy.
        """
        codes = self.codes
        if not codes:
            return 1
        return 0 if codes <= {"ambient_baseline", "human_quote"} else 1

    def to_contract(self) -> dict[str, Any]:
        """The nested shape the host writes, so storage round-trips losslessly.

        Emitting the host's schema-2 layout rather than a private one means a
        stored sample can be fed straight back through
        :func:`parse_decision_trace`, and a future host schema change has exactly
        one place to adapt.
        """
        return {
            "routing_schema_version": self.contract_version,
            "trace_version": self.trace_version,
            "mode": self.mode,
            "weights_version": self.weights_version,
            "parent": {
                "message_id": self.parent_message_id,
                "confidence": self.parent_confidence,
                "ambiguous": self.parent_ambiguous,
            },
            "topic": {
                "topic_id": self.topic_id,
                "confidence": self.topic_confidence,
                "threshold": self.topic_threshold,
                "margin_threshold": self.topic_margin_threshold,
                "ambiguous": self.topic_ambiguous,
            },
            "recipient": {
                "ids": list(self.recipient_ids),
                "bot_targeted": self.bot_targeted,
                "confidence": self.recipient_confidence,
                "threshold": self.recipient_threshold,
                "ambiguous": self.recipient_ambiguous,
            },
            "identity": dict(self.identity),
            "participation": {
                "score": self.participation_score,
                "level": self.participation_level,
                "should_reply": self.should_reply,
                "evidence": [item.as_dict() for item in self.evidence],
                "family_contributions": dict(self.family_contributions),
                "contribution_total": self.contribution_total,
            },
            "state": dict(self.state),
            "ledger_entries": self.ledger_entries,
            "learning_degraded": self.degraded,
        }


def _parse_evidence(value: Any) -> tuple[tuple[EvidenceFact, ...], bool]:
    if not isinstance(value, (list, tuple)):
        return (), bool(value)
    facts: list[EvidenceFact] = []
    degraded = False
    seen: set[tuple[str, str]] = set()
    for item in value[:64]:
        if not isinstance(item, Mapping):
            degraded = True
            continue
        code = item.get("code")
        if not isinstance(code, str) or not code:
            degraded = True
            continue
        family = item.get("family")
        if not isinstance(family, str) or family not in EVIDENCE_FAMILIES:
            # A host upgrade may add a family; keep the fact and flag it rather
            # than silently dropping a real signal.
            family = _UNKNOWN_FAMILY
            degraded = True
        key = (code, family)
        if key in seen:
            continue
        seen.add(key)
        facts.append(EvidenceFact(code=code[:64], family=family,
                                  strength=_finite(item.get("strength")),
                                  source=_text(item.get("source"), 64)))
    return tuple(facts), degraded


def _section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Narrow a nested mapping without repeating the isinstance dance."""
    value = raw.get(key)
    return value if isinstance(value, Mapping) else {}


def _parse_families(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        number = _optional_finite(raw)
        if number is not None:
            result[key[:32]] = number
    return result


def parse_decision_trace(raw: Any) -> DecisionTrace:
    """Normalise a stored trace. Never raises; malformed input yields `degraded`."""
    if not isinstance(raw, Mapping):
        return DecisionTrace(degraded=True)
    degraded = False

    schema = raw.get("routing_schema_version")
    if schema is not None and schema != TRACE_CONTRACT_VERSION:
        degraded = True
    if raw.get("learning_degraded") is True:
        degraded = True

    recipient = _section(raw, "recipient")
    topic = _section(raw, "topic")
    parent = _section(raw, "parent")
    identity = _section(raw, "identity")
    state = _section(raw, "state")
    participation = _section(raw, "participation")
    evidence, evidence_degraded = _parse_evidence(participation.get("evidence"))
    degraded = degraded or evidence_degraded
    families = _parse_families(participation.get("family_contributions"))

    level = participation.get("level")
    if level is not None and level not in LEVELS:
        level = None
        degraded = True

    ledger = _section(raw, "ledger")
    ledger_entries = ledger.get("entries")
    ledger_count = len(ledger_entries) if isinstance(ledger_entries, list) else 0

    intervening = state.get("intervening_users")
    clean_state = {
        "pending_hover": _optional_flag(state.get("pending_hover")),
        "active_interlocutor": _text(state.get("active_interlocutor")) or None,
        "intervening_users": intervening
        if isinstance(intervening, int) and not isinstance(intervening, bool) else None,
        "waiting_for_answer": _optional_flag(state.get("waiting_for_answer")),
        "last_bot_was_question": _optional_flag(state.get("last_bot_was_question")),
        "last_bot_message_id": _text(state.get("last_bot_message_id")) or None,
    }

    clean_identity = {
        "bot_reference": _text(identity.get("bot_reference"), 32) or "none",
        "mention": _flag(identity.get("mention")),
        "vocative": _flag(identity.get("vocative")),
        "subject": _flag(identity.get("subject")),
    }

    trace_version = raw.get("trace_version")
    return DecisionTrace(
        contract_version=TRACE_CONTRACT_VERSION,
        trace_version=trace_version if isinstance(trace_version, int) and not isinstance(trace_version, bool) else 1,
        mode=_text(raw.get("mode"), 32) or "legacy",
        weights_version=_text(raw.get("weights_version"), 64) or "default",
        topic_id=_text(topic.get("topic_id")),
        topic_confidence=_finite(topic.get("confidence")),
        topic_threshold=_optional_finite(topic.get("threshold")),
        topic_margin_threshold=_optional_finite(topic.get("margin_threshold")),
        topic_ambiguous=_flag(topic.get("ambiguous")),
        recipient_ids=_identifier_list(recipient.get("ids")),
        bot_targeted=_flag(recipient.get("bot_targeted")),
        recipient_confidence=_finite(recipient.get("confidence")),
        recipient_threshold=_optional_finite(recipient.get("threshold")),
        recipient_ambiguous=(lambda value: value if isinstance(value, bool) else True)(recipient.get("ambiguous")),
        parent_message_id=_text(parent.get("message_id")),
        parent_confidence=_finite(parent.get("confidence")),
        parent_ambiguous=(lambda value: value if isinstance(value, bool) else True)(parent.get("ambiguous")),
        identity=clean_identity,
        state=clean_state,
        participation_score=_optional_finite(participation.get("score")),
        participation_level=level,
        should_reply=_optional_bool(participation.get("should_reply")),
        evidence=evidence,
        family_contributions=families,
        contribution_total=_finite(participation.get("contribution_total")),
        ledger_entries=ledger_count,
        degraded=degraded,
    )


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def known_topic_label(value: Any) -> bool:
    """A label counts only when it names something; `UNKNOWN`/blank never score."""
    return isinstance(value, str) and bool(value.strip()) and value.strip().upper() != "UNKNOWN"


def trace_from_sample_record(record: Mapping[str, Any]) -> DecisionTrace:
    """Convenience: pull the trace out of a stored annotation record."""
    if not isinstance(record, Mapping):
        return DecisionTrace(degraded=True)
    trace = record.get("decision_trace")
    if trace is None:
        # Very old records stored the raw routing mapping instead.
        return parse_legacy_routing(record.get("routing"))
    return parse_decision_trace(trace)


def parse_legacy_routing(raw: Any) -> DecisionTrace:
    """Map a pre-schema-2 `routing` mapping onto the same shape.

    Old records have no participation evidence at all, so the resulting trace is
    marked `degraded`; callers must not treat its empty evidence list as
    evidence of absence.
    """
    if not isinstance(raw, Mapping):
        return DecisionTrace(degraded=True)
    return DecisionTrace(
        degraded=True,
        topic_id=_text(raw.get("topic_id")),
        topic_confidence=_finite(raw.get("topic_confidence")),
        topic_ambiguous=_flag(raw.get("topic_ambiguous")) or _flag(raw.get("ambiguous")),
        recipient_confidence=_finite(raw.get("addressee_confidence")),
        recipient_threshold=_optional_finite(raw.get("recipient_threshold")),
        recipient_ids=_identifier_list(raw.get("addressee_ids")),
        bot_targeted=_flag(raw.get("bot_is_addressee")),
    )


def sequence_of_traces(values: Sequence[Any]) -> list[DecisionTrace]:
    return [parse_decision_trace(value) for value in values]
