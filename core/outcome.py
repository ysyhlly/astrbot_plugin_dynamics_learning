"""The host's *final* outcome, and the suppression that stopped it.

Schema 2 could say "the router admitted a reply". It could not say whether
anything was ever sent. That one missing fact makes three unrelated failures
look identical to a learner:

    admitted, but 作息/降温/媒体静音 压掉了    -> the gate said no, not a model error
    admitted, but 生成超时/空回复              -> generation failed
    admitted, 生成成功, but 平台发送失败       -> delivery failed

Fed into a reply-F1 as one "missed_reply", all three read as "the router should
have replied". Two of them are not router decisions at all, and moving
`strong_addressivity_threshold` cannot fix any of them. So the outcome is read
as its own fact, with four honest states and no guessing:

* a **recorded** outcome names what actually happened;
* an **absent** outcome is `recorded=False`, which is not `delivered=False` —
  schema 2 rows are *unavailable*, not negative.

Two vocabularies, deliberately separate:

* `VALUE_*` — what happened, closed set. Unknown strings collapse to
  `unknown` because a learner cannot act on a value it does not understand.
* suppression **reasons** — an *open* vocabulary mirrored from the host's
  `GateResult.reason_code`. Keeping them verbatim means a host that adds a
  reason is still readable; `reason_stage` maps the known ones onto the stage
  that produced them and reports the rest as `unknown` instead of quietly
  filing them under "gate".

One rule the rest of the plugin depends on: a suppression is only ever a
**gate** suppression when the host explicitly names that stage or a known reason says so. "The bot did not reply" is not
evidence about *why*.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# ---- what happened -----------------------------------------------------

VALUE_DELIVERED = "delivered"
VALUE_SUPPRESSED = "suppressed"
VALUE_GENERATION_FAILED = "generation_failed"
VALUE_DELIVERY_FAILED = "delivery_failed"
VALUE_NOT_ATTEMPTED = "not_attempted"
# Delivered=false with no reason recorded. Naming the gap beats choosing
# between "suppressed" and "failed" on the reader side: the host did not say,
# and a bucket that means "we were told, but not why" is not the same as one
# that means "a gate stopped it".
VALUE_NOT_DELIVERED = "not_delivered"
VALUE_UNKNOWN = "unknown"

VALUES = (
    VALUE_DELIVERED, VALUE_SUPPRESSED, VALUE_GENERATION_FAILED, VALUE_DELIVERY_FAILED,
    VALUE_NOT_ATTEMPTED, VALUE_NOT_DELIVERED, VALUE_UNKNOWN,
)
VALUE_LABEL = {
    VALUE_DELIVERED: "已发送",
    VALUE_SUPPRESSED: "被门禁压掉",
    VALUE_GENERATION_FAILED: "生成失败",
    VALUE_DELIVERY_FAILED: "发送失败",
    VALUE_NOT_ATTEMPTED: "未进入回复流程",
    VALUE_NOT_DELIVERED: "未发送（未记录原因）",
    VALUE_UNKNOWN: "未知",
}

# ---- which stage the outcome belongs to --------------------------------

STAGE_ADMISSION = "admission"
STAGE_GATE = "gate"
STAGE_GENERATION = "generation"
STAGE_DELIVERY = "delivery"
STAGE_UNKNOWN = "unknown"

STAGES = (STAGE_ADMISSION, STAGE_GATE, STAGE_GENERATION, STAGE_DELIVERY, STAGE_UNKNOWN)
STAGE_LABEL = {
    STAGE_ADMISSION: "参与准入",
    STAGE_GATE: "门禁",
    STAGE_GENERATION: "生成",
    STAGE_DELIVERY: "发送",
    STAGE_UNKNOWN: "未知",
}

# The stage each value implies. `not_delivered` is *not* mapped to a stage: the
# host said "not sent" without saying where it stopped, so any stage here would
# be this plugin guessing wearing the host name.
VALUE_STAGE = {
    VALUE_DELIVERED: STAGE_DELIVERY,
    VALUE_SUPPRESSED: STAGE_GATE,
    VALUE_GENERATION_FAILED: STAGE_GENERATION,
    VALUE_DELIVERY_FAILED: STAGE_DELIVERY,
    VALUE_NOT_ATTEMPTED: STAGE_ADMISSION,
    VALUE_NOT_DELIVERED: STAGE_UNKNOWN,
    VALUE_UNKNOWN: STAGE_UNKNOWN,
}

# ---- the host reason vocabulary (open) ---------------------------------

# Mirrors the `reason_code` strings ChatDynamics' decision gate and arbiter
# actually produce. A code that is not in here is **not** filed under `gate`:
# it stays `unknown` and is counted, so adding a reason in the host shows up as
# an unclassified code rather than as a silent gate suppression.
GATE_REASONS = frozenset({
    # gate / arbiter silences
    "arbiter_silence", "deep_cooling", "safe_hover", "energy_asymmetry",
    "private_topic", "filter_gate", "wts_low", "media_others_field",
    "voice_low_info", "image_privacy_skip", "media_listen", "media_meme_listen",
    "deciding_no_banter", "proactive_quota", "newcomer_caution", "gap_wait",
    "wind_later_silence", "conflict_silence", "cool_command", "occasion_silence",
    "presence_ghost", "ambient_budget", "insomnia_cap", "brief_wake_cooldown",
    # rhythm
    "asleep_plain_gn", "asleep_wake_ok", "asleep_ambient", "wind_goodnight_ok",
    "wind_hot_delay",
})
GENERATION_REASONS = frozenset({
    "generation_failed", "generation_timeout", "empty_reply", "llm_error",
    "llm_timeout", "no_content",
})
DELIVERY_REASONS = frozenset({
    "delivery_failed", "send_failed", "platform_error", "upload_failed",
})
# Reasons that mean the flow was never entered. They are admission outcomes,
# not gate suppressions: the gate never ran.
ADMISSION_REASONS = frozenset({
    "not_attempted", "admission_declined", "no_admission", "level_not_strong",
})

_REASON_STAGE = {
    **{code: STAGE_GATE for code in GATE_REASONS},
    **{code: STAGE_GENERATION for code in GENERATION_REASONS},
    **{code: STAGE_DELIVERY for code in DELIVERY_REASONS},
    **{code: STAGE_ADMISSION for code in ADMISSION_REASONS},
}

# Which values count as "the host told us something went wrong".
FAILURE_VALUES = frozenset({
    VALUE_SUPPRESSED, VALUE_GENERATION_FAILED, VALUE_DELIVERY_FAILED,
    VALUE_NOT_ATTEMPTED, VALUE_NOT_DELIVERED,
})

# ---- where the fact was read from --------------------------------------

SOURCE_TRACE = "trace"
SOURCE_RECORD = "record"
SOURCE_NONE = "none"
SOURCE_LABEL = {SOURCE_TRACE: "decision_trace", SOURCE_RECORD: "标注记录", SOURCE_NONE: "未记录"}
NESTED_KEY = "outcome"
_FLAT_KEYS = ("final_outcome", "outcome_value", "delivered", "suppression_reason")


def reason_stage(reason: str) -> str:
    """Which stage produced a suppression reason, or `unknown` if unlisted."""
    return _REASON_STAGE.get(reason, STAGE_UNKNOWN)


def _text(value: Any, limit: int = 96) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


@dataclass(frozen=True)
class FinalOutcome:
    """One message final outcome. `recorded=False` means *not told*, not *no*."""

    recorded: bool = False
    value: str = VALUE_UNKNOWN
    delivered: bool | None = None
    suppression_reason: str = ""
    stage: str = STAGE_UNKNOWN
    source: str = SOURCE_NONE

    # ---- derived -------------------------------------------------------

    @property
    def outcome_unavailable(self) -> bool:
        """The plan schema-2 marker, spelled the way the report prints it."""
        return not self.recorded

    @property
    def is_delivered(self) -> bool:
        return self.value == VALUE_DELIVERED

    @property
    def is_failure(self) -> bool:
        """Was a non-delivery recorded? `recorded=False` is never a failure."""
        if not self.recorded:
            return False
        if self.delivered is False:
            return True
        return self.value in FAILURE_VALUES

    @property
    def is_gate_suppression(self) -> bool:
        return self.is_failure and self.stage == STAGE_GATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "recorded": self.recorded,
            "value": self.value,
            "value_label": VALUE_LABEL.get(self.value, self.value),
            "delivered": self.delivered,
            "suppression_reason": self.suppression_reason,
            "stage": self.stage,
            "stage_label": STAGE_LABEL.get(self.stage, self.stage),
            "source": self.source,
            "outcome_unavailable": self.outcome_unavailable,
        }


EMPTY = FinalOutcome()


def _normalise_value(raw: Any) -> str:
    if not isinstance(raw, str):
        return VALUE_UNKNOWN
    text = raw.strip().lower()
    return text if text in VALUES else VALUE_UNKNOWN


def _from_block(block: Any, source: str) -> FinalOutcome | None:
    """Read one candidate location. `None` means it recorded nothing at all."""
    if isinstance(block, str):
        value = _normalise_value(block)
        if value == VALUE_UNKNOWN:
            return None
        return _assemble(value=value, delivered=None, reason="", stage="", source=source)
    if not isinstance(block, Mapping):
        return None

    raw_value = block.get("final_outcome")
    if raw_value is None:
        raw_value = block.get("value")
    if raw_value is None:
        raw_value = block.get("outcome")
    delivered = _optional_bool(block.get("delivered"))
    reason = _text(block.get("suppression_reason") or block.get("reason"))
    stage = _text(block.get("stage"), 32)

    if raw_value is None and delivered is None and not reason and not stage:
        return None
    return _assemble(value=_normalise_value(raw_value), delivered=delivered,
                     reason=reason, stage=stage, source=source)


def _assemble(*, value: str, delivered: bool | None, reason: str,
              stage: str, source: str) -> FinalOutcome:
    """Fill the value from whatever the host did record, and never further.

    Derivation is only allowed in one direction: a recorded `delivered` flag with
    no value names `delivered` / `not_delivered`, and a reason with no value
    names `suppressed` — because a reason is exactly what a suppression is. A
    bare `delivered=false` with no reason stays `not_delivered`: choosing
    between the generation and delivery stages there would be this plugin guess.
    """
    if value == VALUE_UNKNOWN:
        if delivered is True:
            value = VALUE_DELIVERED
        elif delivered is False:
            value = VALUE_SUPPRESSED if reason else VALUE_NOT_DELIVERED
        elif reason:
            value = VALUE_SUPPRESSED
    resolved_stage = stage if stage in STAGES else ""
    if not resolved_stage:
        resolved_stage = reason_stage(reason) if reason else ""
    if not resolved_stage:
        resolved_stage = VALUE_STAGE.get(value, STAGE_UNKNOWN)
    if stage not in STAGES and resolved_stage == STAGE_GATE and reason and reason_stage(reason) != STAGE_GATE:
        # The host said "suppressed" but named a reason this plugin does not
        # recognise as a gate code. That is a stage nobody can vouch for.
        resolved_stage = STAGE_UNKNOWN
    if delivered is None and value in (VALUE_DELIVERED, VALUE_NOT_ATTEMPTED):
        delivered = value == VALUE_DELIVERED
    if delivered is None and value in FAILURE_VALUES:
        delivered = False
    return FinalOutcome(recorded=True, value=value, delivered=delivered,
                        suppression_reason=reason, stage=resolved_stage, source=source)


def from_mapping(raw: Any, *, source: str = SOURCE_TRACE) -> FinalOutcome:
    """Read the outcome out of one mapping, nested block first, then flat keys.

    The nested form is schema 3 (`{"outcome": {...}}`); the flat form is
    accepted so the host can start writing one field at a time instead of
    landing the whole block at once.
    """
    if not isinstance(raw, Mapping):
        return EMPTY
    nested = _from_block(raw.get(NESTED_KEY), source)
    if nested is not None:
        return nested
    flat = {key: raw.get(key) for key in _FLAT_KEYS if key in raw}
    return _from_block(flat, source) or EMPTY


def parse_outcome(*sources: Any) -> FinalOutcome:
    """First source that recorded an outcome wins.

    Called as `parse_outcome(trace_raw, record_raw)`: the frozen decision trace is
    authoritative when it carries the fact, and the enclosing annotation record
    is the fallback for a host that writes it there instead. Order is explicit
    rather than merged, so two disagreeing sources cannot produce a third answer
    that neither of them stated.
    """
    for index, source in enumerate(sources):
        found = from_mapping(source, source=SOURCE_TRACE if index == 0 else SOURCE_RECORD)
        if found.recorded:
            return found
    return EMPTY


def parse_record_outcome(record: Mapping[str, Any]) -> FinalOutcome:
    """Convenience for a stored annotation record or a stored sample payload."""
    if not isinstance(record, Mapping):
        return EMPTY
    trace = record.get("decision_trace")
    first = trace if isinstance(trace, Mapping) else {}
    return parse_outcome(first, record)


__all__ = [
    "ADMISSION_REASONS", "DELIVERY_REASONS", "EMPTY", "FAILURE_VALUES", "GATE_REASONS",
    "GENERATION_REASONS", "NESTED_KEY", "SOURCE_LABEL", "SOURCE_NONE", "SOURCE_RECORD",
    "SOURCE_TRACE", "STAGES", "STAGE_ADMISSION", "STAGE_DELIVERY", "STAGE_GATE",
    "STAGE_GENERATION", "STAGE_LABEL", "STAGE_UNKNOWN", "VALUE_DELIVERED",
    "VALUE_DELIVERY_FAILED", "VALUE_GENERATION_FAILED", "VALUE_LABEL", "VALUE_NOT_ATTEMPTED",
    "VALUE_NOT_DELIVERED", "VALUE_STAGE", "VALUE_SUPPRESSED", "VALUE_UNKNOWN", "VALUES",
    "FinalOutcome", "from_mapping", "parse_outcome", "parse_record_outcome", "reason_stage",
]
