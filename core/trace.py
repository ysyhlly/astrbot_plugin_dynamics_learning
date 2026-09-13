"""Read-only normalisation of a ChatDynamics routing decision trace.

The host stores `decision_trace` snapshots built by
`astrbot_plugin_chat_dynamics.core.routing_trace.build_routing_trace`. Those
snapshots are already field-allowlisted copies, so this module never has to
strip message text: it only has to be defensive about shape **and about
version**, because the host is upgraded independently of this plugin.

Two readers and one normalisation layer, because "read whatever is there" and
"know what the record could not say" are different jobs:

```text
read_schema_v2   routing_schema_version == 2
read_schema_v3   routing_schema_version == 3
normalize_trace  dispatch on the recorded version, then fill the gaps
```

Both readers produce the same `DecisionTrace`. What they cannot produce is
recorded the same way:

| fact | schema 2 | schema 3 |
| --- | --- | --- |
| participation evidence | yes | yes |
| topic candidates | list of `[score, id]` | structured, with per-candidate evidence |
| selected topic | `routing.selected_topic` (optional) | required |
| final outcome | **never** | `outcome.{final_outcome, delivered, suppression_reason}` |

So a schema 2 trace is marked, not silently filled in:

* `outcome_unavailable` — nothing was recorded about whether anything was sent;
* `candidate_evidence_partial` — the candidate set may be there, the evidence
  behind each candidate is not.

The marker is derived from the **source schema as well as the payload**: a row
that says schema 2 can never be promoted to `full` candidate evidence no matter
what keys an older writer happened to leave behind, because the version is the
host's own statement about what it wrote.

`trace_schema_version` is therefore the schema the trace is *expressed in* — the
source schema after normalisation — and not a constant. Re-emitting schema 2 for
a schema 2 row is what lets a stored sample round-trip byte-compatibly through
:func:`parse_decision_trace` and :meth:`DecisionTrace.to_contract`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .candidates import CandidateRecord, parse_candidates
from .outcome import EMPTY as OUTCOME_EMPTY, FinalOutcome, parse_outcome

# ---- the two protocol versions, named so they can never be confused -----
#
#   trace_schema_version    ChatDynamics -> decision trace -> Dynamics Learning
#                           (the host's own `decision_trace.routing_schema_version`)
#   policy_contract_version Dynamics Learning -> /published -> ChatDynamics
#                           (see `core/policy.POLICY_CONTRACT_VERSION`)
#
# They move independently, and this module only owns the first one. A change to
# this plugin's readers, API or UI must never move the trace schema: that number
# describes what *ChatDynamics* writes, and only ChatDynamics can change it.
#
# The host trace layout the reader understands.
SCHEMA_V2 = 2
SCHEMA_V3 = 3
SUPPORTED_SCHEMAS = (SCHEMA_V2, SCHEMA_V3)
# What the host writes today. The reader accepts both, so a host that has not
# been upgraded yet is read, counted and marked — never guessed at.
LATEST_TRACE_SCHEMA = SCHEMA_V3
LEGACY_SCHEMA = SCHEMA_V2

# The key the shadow block travels under, inside the trace.
SHADOW_KEY = "shadow"

# How much of the per-candidate evidence the host actually recorded.
CANDIDATE_EVIDENCE_FULL = "full"
CANDIDATE_EVIDENCE_PARTIAL = "partial"
CANDIDATE_EVIDENCE_NONE = "none"

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
class ShadowDecision:
    """What a shadow policy *would* have decided, recorded beside what was done.

    Phase one of a shadow A/B run: the runtime keeps the baseline behaviour and
    the policy's admission decision is computed next to it, so the two can be
    compared against a human label later. Nothing here changes behaviour — it is
    the other half of a comparison whose first half is `participation.level`.

    `changed` is the field the whole evaluation is built on. When ninety-five
    percent of decisions are identical, an overall accuracy delta averages the
    policy's effect away; the only place its effect exists is the subset where
    the two decisions differ, and this flag is what selects it.
    """

    recorded: bool = False
    policy_id: str = ""
    baseline_threshold: float | None = None
    shadow_threshold: float | None = None
    baseline_reply: bool = False
    shadow_reply: bool = False
    changed: bool = False
    score: float | None = None
    baseline_margin: float | None = None
    shadow_margin: float | None = None
    # Why the two agreed, when they did: `structural` (the evidence decided it,
    # no threshold involved), `early_return` (no prior bot message), or
    # `ambient` (the score was compared). An empty string for a disagreement.
    reason: str = ""
    recorded_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "recorded": self.recorded,
            "policy_id": self.policy_id,
            "baseline_threshold": self.baseline_threshold,
            "shadow_threshold": self.shadow_threshold,
            "baseline_reply": self.baseline_reply,
            "shadow_reply": self.shadow_reply,
            "changed": self.changed,
            "score": self.score,
            "baseline_margin": self.baseline_margin,
            "shadow_margin": self.shadow_margin,
            "reason": self.reason,
            "recorded_at": self.recorded_at,
        }


SHADOW_EMPTY = ShadowDecision()


def parse_shadow(raw: Any) -> ShadowDecision:
    """Read a shadow block. Never raises; anything unreadable stays unrecorded."""
    if not isinstance(raw, Mapping):
        return SHADOW_EMPTY
    policy_id = _text(raw.get("policy_id"), 64)
    if not policy_id:
        # A block with no policy id cannot be attributed to a policy, and an
        # unattributable row in a comparison table is worse than a missing one.
        return SHADOW_EMPTY
    baseline = raw.get("baseline_reply")
    shadow = raw.get("shadow_reply")
    if not isinstance(baseline, bool) or not isinstance(shadow, bool):
        return SHADOW_EMPTY
    return ShadowDecision(
        recorded=True,
        policy_id=policy_id,
        baseline_threshold=_optional_finite(raw.get("baseline_threshold")),
        shadow_threshold=_optional_finite(raw.get("shadow_threshold")),
        baseline_reply=baseline,
        shadow_reply=shadow,
        changed=bool(raw.get("changed")) or baseline != shadow,
        score=_optional_finite(raw.get("score")),
        baseline_margin=_optional_finite(raw.get("baseline_margin")),
        shadow_margin=_optional_finite(raw.get("shadow_margin")),
        reason=_text(raw.get("reason"), 32),
        recorded_at=_finite(raw.get("recorded_at")),
    )


@dataclass(frozen=True)
class DecisionTrace:
    """A frozen, JSON-safe view of one routing decision."""

    # The schema this trace is expressed in — the host's `routing_schema_version`
    # after normalisation. A trace built in-process (not read from a host record)
    # has no schema 3 facts, so it defaults to the older schema rather than
    # claiming an outcome it was never told.
    trace_schema_version: int = LEGACY_SCHEMA
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

    # Schema 3: the candidate set with the host's own per-candidate evidence,
    # and which topic it finally selected.
    topic_candidates: tuple[CandidateRecord, ...] = ()
    topic_candidates_recorded: bool = False
    selected_topic: str = ""
    # Schema 3: where the turn finally ended up (sent / suppressed / failed).
    # Stays `EMPTY` — recorded=False — for every schema 2 row, which is a
    # statement about the record, not about the message.
    outcome: FinalOutcome = OUTCOME_EMPTY
    # Schema 3: what a shadow policy would have decided, when one was being
    # observed. `recorded=False` on every row from a run without a shadow
    # policy, which is not a disagreement of zero — it is the absence of a
    # comparison.
    shadow: ShadowDecision = SHADOW_EMPTY
    # The version the host wrote (0 when absent or unreadable). Distinct from
    # `trace_schema_version`, which is the schema this trace is expressed in
    # after normalisation: a v1 row is *expressed* as v2 while `source_schema`
    # keeps saying 1, or 0.
    source_schema: int = 0

    ledger_entries: int = 0
    degraded: bool = False

    # ---- derived views -------------------------------------------------

    @property
    def candidate_evidence(self) -> str:
        """How complete the per-candidate evidence is: full / partial / none.

        `full` requires both a schema 3 source and an evidence map on every
        parsed candidate. A schema 2 row cannot reach it even if a writer left
        evidence-shaped keys behind: the version is the host's own statement
        about what it recorded, and it outranks a guess made from key names.
        """
        if not self.topic_candidates_recorded:
            return CANDIDATE_EVIDENCE_NONE
        if self.source_schema < SCHEMA_V3 or self.trace_schema_version < SCHEMA_V3:
            return CANDIDATE_EVIDENCE_PARTIAL
        if self.topic_candidates and all(row.evidence for row in self.topic_candidates):
            return CANDIDATE_EVIDENCE_FULL
        return CANDIDATE_EVIDENCE_PARTIAL

    @property
    def candidate_evidence_partial(self) -> bool:
        """True whenever the candidate evidence is not fully available.

        Named as the plan names it. A row with `none` is *more* degraded than
        one with `partial`, so this flag is true for both — the three-valued
        :attr:`candidate_evidence` is what tells them apart.
        """
        return self.candidate_evidence != CANDIDATE_EVIDENCE_FULL

    @property
    def outcome_unavailable(self) -> bool:
        """No final outcome was recorded: schema 2's signature, not a negative."""
        return not self.outcome.recorded

    @property
    def shadow_recorded(self) -> bool:
        return self.shadow.recorded

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

        The layout is the host's, not a private one, so a stored sample can be
        fed straight back through :func:`parse_decision_trace` — and the schema
        written back is the schema that was read, so a schema 2 row stays a
        schema 2 row instead of silently acquiring fields the host never wrote.

        The schema 3 sections are emitted **only when they carry a fact**:
        an empty `routing`/`outcome` block would be indistinguishable, after a
        round trip, from a host that looked and recorded nothing.
        """
        payload: dict[str, Any] = {
            "trace_schema_version": self.trace_schema_version,
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
        if self.topic_candidates_recorded or self.selected_topic:
            routing: dict[str, Any] = {"selected_topic": self.selected_topic}
            if self.topic_candidates_recorded:
                routing["topic_candidates"] = [row.as_dict() for row in self.topic_candidates]
            payload["routing"] = routing
        if self.outcome.recorded:
            payload["outcome"] = {
                "final_outcome": self.outcome.value,
                "delivered": self.outcome.delivered,
                "suppression_reason": self.outcome.suppression_reason,
                "stage": self.outcome.stage,
            }
        if self.shadow.recorded:
            block = self.shadow.as_dict()
            block.pop("recorded", None)
            payload["shadow"] = block
        if self.source_schema:
            payload["source_schema"] = self.source_schema
        return payload


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


# The key the schema number travels under, newest first.
#
# `routing_schema_version` was the original name and it said the wrong thing:
# the number describes the whole *trace* (recipient, topic, participation and —
# from schema 3 — the outcome), not the routing section alone. ChatDynamics
# writes `trace_schema_version` from v1.6.2; the old key is still read so every
# annotation stored before that keeps its meaning instead of reading as
# "unrecorded".
SCHEMA_KEYS = ("trace_schema_version", "routing_schema_version")


def declared_schema_value(raw: Any) -> Any:
    """The raw schema value a mapping declares, under either key."""
    if not isinstance(raw, Mapping):
        return None
    for key in SCHEMA_KEYS:
        if key in raw:
            return raw[key]
    return None


def _read_schema(raw: Mapping[str, Any]) -> int:
    """The version the host wrote, or 0 when it did not say.

    A string version is **not** coerced: `"2"` is a different record from `2`,
    and guessing which one a writer meant is how a version check stops meaning
    anything. It is reported as unreadable instead.
    """
    value = declared_schema_value(raw)
    # `bool` is an `int` subclass, and JSON `true` is not version 1. The
    # positive form is also what narrows `Any` to `int` for the type checker.
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _read_candidates(raw: Mapping[str, Any]) -> tuple[tuple[CandidateRecord, ...], bool]:
    """The candidate set, plus whether the field was there at all.

    `topic_candidates` is schema 3's key; `candidates` is the older alias the
    host writes into `record["routing"]`. An absent key returns
    `(recorded=False)` even though the payload is empty, because "the host
    looked and proposed nothing" is a candidate-generation miss while "the host
    recorded nothing" is not evidence of anything (see `core/candidates.py`).
    """
    routing = _section(raw, "routing")
    if "topic_candidates" in routing:
        parsed = parse_candidates(routing.get("topic_candidates"))
    elif "candidates" in routing:
        parsed = parse_candidates(routing.get("candidates"))
    else:
        return (), False
    return parsed.items, parsed.recorded


def _read_outcome(raw: Mapping[str, Any]) -> FinalOutcome:
    return parse_outcome(raw, _section(raw, "routing"))


def _read_shadow(raw: Mapping[str, Any]) -> ShadowDecision:
    return parse_shadow(raw.get(SHADOW_KEY))


def read_schema_v2(raw: Any) -> DecisionTrace:
    """Read a schema 2 trace: everything except the final outcome.

    Schema 2 is not "worse data" — it is data with a **documented hole**. The
    participation evidence, the scores and the topic decision are all there and
    are all replayable; what it cannot answer is whether anything was sent. So
    the reader fills in what is present and leaves `outcome` empty rather than
    defaulting it, which is what makes `outcome_unavailable` a fact instead of
    an assumption.
    """
    return _read_common(raw, source_schema=SCHEMA_V2, trace_schema_version=SCHEMA_V2)


def read_schema_v3(raw: Any) -> DecisionTrace:
    """Read a schema 3 trace: schema 2 plus candidates, selection and outcome."""
    return _read_common(raw, source_schema=SCHEMA_V3, trace_schema_version=SCHEMA_V3)


def normalize_trace(raw: Any) -> DecisionTrace:
    """Dispatch on the recorded version and normalise to a `DecisionTrace`.

    Version handling, stated once:

    ```text
    2        -> schema 2 field set
    3        -> schema 3 field set
    absent   -> schema 2 field set, degraded  (a pre-versioning writer)
    anything -> schema 2 field set, degraded  (an unknown future, or a typo)
    ```

    Both readers consume the self-describing `routing` and `outcome` blocks
    wherever they appear, because their keys name themselves: an `outcome` block
    is a fact the host wrote, and refusing to read it because the version moved
    would lose it. What the version gates is the **evidence level**, not the
    field set: `full` candidate evidence requires a record that *declares*
    schema 3, so key names can never promote an older record to a completeness it
    never claimed.

    An unrecognised version therefore degrades loudly and keeps reading. The
    degradation is the finding — a host that ships schema 4 should see this
    plugin report "unreadable version 4" rather than silently produce schema 2
    numbers that look fine.
    """
    if not isinstance(raw, Mapping):
        return DecisionTrace(degraded=True)
    schema = _read_schema(raw)
    if schema == SCHEMA_V3:
        return read_schema_v3(raw)
    return read_schema_v2(raw)


def parse_decision_trace(raw: Any) -> DecisionTrace:
    """Normalise a stored trace. Never raises; malformed input yields `degraded`."""
    return normalize_trace(raw)


def _read_common(raw: Any, *, source_schema: int, trace_schema_version: int) -> DecisionTrace:
    if not isinstance(raw, Mapping):
        return DecisionTrace(degraded=True)
    recorded = _read_schema(raw)
    # An absent version is *legacy*, not malformed: pre-versioning writers left
    # no statement either way, and `source_schema=0` already carries that fact.
    # Only a version that contradicts the reader we were asked to use is a
    # degradation.
    degraded = recorded not in (0, source_schema)
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
    candidates, candidates_recorded = _read_candidates(raw)
    return DecisionTrace(
        trace_schema_version=trace_schema_version,
        source_schema=recorded,
        topic_candidates=candidates,
        topic_candidates_recorded=candidates_recorded,
        selected_topic=_text(_section(raw, "routing").get("selected_topic")),
        outcome=_read_outcome(raw),
        shadow=_read_shadow(raw),
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
    """Read one stored annotation record as a single decision.

    The host splits the same decision across two places: `decision_trace` holds
    the evidence and the participation score, `routing` holds the candidate set
    and the selection. Reading only the nested snapshot would lose the candidate
    set on every schema 2 row, which is precisely the half the attribution chain
    needs, so the record is read as one fact.

    A field the trace already carries is **never** overwritten. The frozen
    snapshot is the more specific record of what the router saw; the enclosing
    record is the fallback for a host that writes a fact one level up.
    """
    if not isinstance(record, Mapping):
        return DecisionTrace(degraded=True)
    raw_trace = record.get("decision_trace")
    if raw_trace is None:
        # Very old records stored the raw routing mapping instead.
        return parse_legacy_routing(record.get("routing"))
    trace = parse_decision_trace(raw_trace)
    routing = _section(record, "routing")
    changes: dict[str, Any] = {}
    if not trace.topic_candidates_recorded:
        candidates, recorded = _read_candidates(record)
        if recorded:
            changes["topic_candidates"] = candidates
            changes["topic_candidates_recorded"] = True
    if not trace.selected_topic:
        selected = _text(routing.get("selected_topic"))
        if selected:
            changes["selected_topic"] = selected
    if not trace.outcome.recorded:
        # The outcome is written after the snapshot was frozen, so it is
        # normally one level up — the trace is checked first only so that a host
        # which *does* freeze it wins over a stale copy beside it.
        fallback = parse_outcome(routing, record)
        if fallback.recorded:
            changes["outcome"] = fallback
    return replace(trace, **changes) if changes else trace


def parse_legacy_routing(raw: Any) -> DecisionTrace:
    """Map a pre-schema-2 `routing` mapping onto the same shape.

    Old records have no participation evidence at all, so the resulting trace is
    marked `degraded`; callers must not treat its empty evidence list as
    evidence of absence.
    """
    if not isinstance(raw, Mapping):
        return DecisionTrace(degraded=True)
    candidates, candidates_recorded = _read_candidates({"routing": raw})
    return DecisionTrace(
        degraded=True,
        source_schema=0,
        topic_candidates=candidates,
        topic_candidates_recorded=candidates_recorded,
        selected_topic=_text(raw.get("selected_topic")),
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
