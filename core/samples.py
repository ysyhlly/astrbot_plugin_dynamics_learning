"""The unified `LearningSample` and the conversion from human labels.

One sample answers exactly one supervised question about one message. The point
of the format is that it stores **features, not just outcomes**: a wrong label
is only useful later if the evidence that produced it was preserved.

Three tasks are produced from a single ChatDynamics annotation record:

* `recipient`  - was the bot the addressee?          (features: ambient evidence)
* `topic`      - which topic does the message belong to? (features: topic scores)
* `reply`      - did the host admit a reply?           (features: same score, other cut)

The predicted value is always something the host actually recorded. It is never
re-derived from a different source, so a sample can be replayed later.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import candidates as candidates_module
from .candidates import (
    CandidateRecord, parse_candidates, parse_topic_candidates, records_to_payload,
)
from .config import LearningConfig
from .features import FEATURE_SCHEMA_VERSION, build_features, feature_summary
from .trace import known_topic_label, trace_from_sample_record

SAMPLE_SCHEMA_VERSION = 1

TASK_RECIPIENT = "recipient"
TASK_TOPIC = "topic"
TASK_REPLY = "reply"
TASKS = (TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY)

SOURCE_MANUAL_REPLAY = "manual_replay"
SOURCE_RUNTIME_OUTCOME = "runtime_outcome"

BOT = "bot"
OTHER = "other"
REPLY = "reply"
SILENT = "silent"
NEW_TOPIC = "NEW"
UNASSIGNED = ""  # the host recorded no committed topic for this message
MAX_TOPIC_CANDIDATES = candidates_module.MAX_CANDIDATES


def _topic_candidates(record: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Ranked candidates, plus whether the host recorded the field at all.

    Both facts matter and only one of them survives the payload. An **empty
    list** means the host looked and proposed nothing, which is a genuine
    candidate-generation miss; a **missing field** means it recorded nothing, so
    whether the right topic was offered simply is not knowable. Returning the
    payload alone would merge the two and blame generation for both.

    Accepts both the legacy `[[score, topic_id], ...]` pairs and the structured
    `[{topic_id, final_score, evidence, rank}, ...]` form.
    """
    routing = record.get("routing")
    if not isinstance(routing, Mapping):
        return [], False
    raw = routing.get("topic_candidates")
    if not isinstance(raw, list):
        raw = routing.get("candidates")
    parsed = parse_candidates(raw)
    return records_to_payload(parsed.items), parsed.recorded


def _selected_topic(record: Mapping[str, Any]) -> str:
    routing = record.get("routing")
    if isinstance(routing, Mapping):
        selected = routing.get("selected_topic")
        if isinstance(selected, str) and selected:
            return selected[:160]
    predicted = _topic_label(record.get("predicted_topic"), str(record.get("msg_id") or ""))
    return predicted or UNASSIGNED


def _finite(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def session_hash(session_key: str) -> str:
    """Same hashing the host uses for its per-session annotation key."""
    return hashlib.sha256(str(session_key).encode("utf-8")).hexdigest()


def sample_identifier(session_key: str, msg_id: str, task: str) -> str:
    digest = hashlib.sha256(f"{session_key}\x1f{msg_id}\x1f{task}".encode("utf-8")).hexdigest()
    return digest[:32]


@dataclass(frozen=True)
class LearningSample:
    sample_id: str
    session_key: str
    session_hash: str
    msg_id: str
    timestamp: float
    task: str
    predicted: str
    expected: str
    confidence: float
    source: str
    error_type: str = "unknown"
    annotated_at: float = 0.0
    features: Mapping[str, float] = field(default_factory=dict)
    trace: Mapping[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        return self.predicted == self.expected

    @property
    def topic_candidates(self) -> tuple[CandidateRecord, ...]:
        raw = self.trace.get("topic_candidates") if isinstance(self.trace, Mapping) else None
        return parse_topic_candidates(raw)

    @property
    def topic_candidates_recorded(self) -> bool:
        """Whether the host recorded a candidate field — not whether it was non-empty.

        Rows written before this flag existed stored the payload alone, so an
        empty list there is ambiguous. They keep the earlier reading, "not
        recorded", because claiming a generation miss that the record does not
        support is worse than admitting the record cannot say.
        """
        if not isinstance(self.trace, Mapping):
            return False
        flagged = self.trace.get("topic_candidates_recorded")
        if isinstance(flagged, bool):
            return flagged
        return bool(self.topic_candidates)

    @property
    def selected_topic(self) -> str:
        value = self.trace.get("selected_topic") if isinstance(self.trace, Mapping) else None
        return value if isinstance(value, str) else self.predicted

    def as_dict(self, *, include_trace: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "session_key": self.session_key,
            "session_hash": self.session_hash,
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
            "task": self.task,
            "predicted": self.predicted,
            "expected": self.expected,
            "confidence": self.confidence,
            "source": self.source,
            "error_type": self.error_type,
            "annotated_at": self.annotated_at,
            "features": {key: float(value) for key, value in self.features.items()},
        }
        if include_trace:
            payload["trace"] = dict(self.trace)
        return payload

    @classmethod
    def from_dict(cls, raw: Any) -> "LearningSample | None":
        if not isinstance(raw, Mapping):
            return None
        task = raw.get("task")
        if task not in TASKS:
            return None
        session_key = raw.get("session_key")
        msg_id = raw.get("msg_id")
        if not isinstance(session_key, str) or not isinstance(msg_id, str) or not session_key or not msg_id:
            return None
        predicted, expected = raw.get("predicted"), raw.get("expected")
        if not isinstance(predicted, str) or not isinstance(expected, str):
            return None
        features = raw.get("features")
        clean_features: dict[str, float] = {}
        if isinstance(features, Mapping):
            for key, value in list(features.items())[:512]:
                if isinstance(key, str):
                    clean_features[key[:64]] = _finite(value)
        trace = raw.get("trace")
        return cls(
            sample_id=str(raw.get("sample_id") or sample_identifier(session_key, msg_id, task))[:64],
            session_key=session_key[:256],
            session_hash=str(raw.get("session_hash") or session_hash(session_key))[:64],
            msg_id=msg_id[:256],
            timestamp=_finite(raw.get("timestamp")),
            task=task,
            predicted=predicted[:256],
            expected=expected[:256],
            confidence=_finite(raw.get("confidence")),
            source=str(raw.get("source") or SOURCE_MANUAL_REPLAY)[:64],
            error_type=str(raw.get("error_type") or "unknown")[:64],
            annotated_at=_finite(raw.get("annotated_at")),
            features=clean_features,
            trace=dict(trace) if isinstance(trace, Mapping) else {},
        )


def _topic_label(value: Any, msg_id: str) -> str | None:
    """`NEW` is a per-message singleton; blank and UNKNOWN never score."""
    if not isinstance(value, str) or not known_topic_label(value):
        return None
    text = value.strip()
    if text.upper() == NEW_TOPIC:
        return f"{NEW_TOPIC}:{msg_id}"
    return text


def _error_label(explicit: Any, predicted: str, expected: str, *,
                 missed: str, false_positive: str, wrong: str) -> str:
    """Name the mistake the way the review page talks about it.

    A human-supplied type wins when present. Otherwise the type is derived from
    the direction of the error, so the console's distribution reads as
    "missed_bot / false_bot / missed_reply / premature_reply" instead of a wall
    of "unknown".
    """
    if isinstance(explicit, str) and explicit and explicit != "unknown":
        return explicit[:64]
    if predicted == expected:
        return "correct"
    if expected in (BOT, REPLY):
        return missed
    if predicted in (BOT, REPLY):
        return false_positive
    return wrong


def samples_from_annotation(
    record: Mapping[str, Any],
    session_key: str,
    *,
    config: LearningConfig | None = None,
    now: float | None = None,
) -> list[LearningSample]:
    """Convert one stored host annotation record into zero or more samples.

    A record with no usable supervision yields no samples rather than a sample
    with an invented label. Missing labels are never treated as negatives.
    """
    config = config or LearningConfig()
    if not isinstance(record, Mapping):
        return []
    msg_id = record.get("msg_id")
    if not isinstance(msg_id, str) or not msg_id:
        return []
    trace = trace_from_sample_record(record)
    annotated_at = _finite(record.get("annotated_at"), now if now is not None else time.time())
    digest = session_hash(session_key)
    features = build_features(trace)
    trace_payload = trace.to_contract() if config.store_raw_trace else {}
    summary = feature_summary(trace)
    short_id = msg_id[:256]
    produced: list[LearningSample] = []

    def make(task: str, predicted: str, expected: str, confidence: float,
             error_type: str, extra: Mapping[str, Any] | None = None) -> LearningSample:
        payload: dict[str, Any] = dict(trace_payload)
        payload["evidence_summary"] = summary
        payload.update(extra or {})
        return LearningSample(
            sample_id=sample_identifier(session_key, msg_id, task),
            session_key=session_key,
            session_hash=digest,
            msg_id=short_id,
            timestamp=annotated_at,
            task=task,
            predicted=predicted,
            expected=expected,
            confidence=confidence,
            source=SOURCE_MANUAL_REPLAY,
            error_type=error_type,
            annotated_at=annotated_at,
            features=dict(features),
            trace=payload,
        )

    # --- recipient ------------------------------------------------------
    bot_targeted = record.get("bot_targeted")
    if isinstance(bot_targeted, bool):
        predicted_recipient = BOT if trace.bot_targeted else OTHER
        expected_recipient = BOT if bot_targeted else OTHER
        produced.append(make(
            TASK_RECIPIENT, predicted_recipient, expected_recipient,
            trace.recipient_confidence,
            _error_label(record.get("recipient_error_type"),
                         predicted_recipient, expected_recipient,
                         missed="missed_bot", false_positive="false_bot",
                         wrong="wrong_recipient"),
        ))

    # --- topic ----------------------------------------------------------
    # A message the host left unassigned is still a scorable sample: the host's
    # own routing metrics count "labelled together but predicted apart
    # (including unassigned predictions)" as fragmentation. The unassigned
    # marker is the empty string, which never equals another label.
    expected_topic = _topic_label(record.get("expected_topic"), msg_id)
    if expected_topic is not None:
        predicted_topic = _topic_label(record.get("predicted_topic"), msg_id) or UNASSIGNED
        candidates_payload, candidates_recorded = _topic_candidates(record)
        produced.append(make(
            TASK_TOPIC, predicted_topic, expected_topic, trace.topic_confidence,
            str(record.get("error_type") or "unknown")[:64],
            {"topic_candidates": candidates_payload,
             "topic_candidates_recorded": candidates_recorded,
             "selected_topic": _selected_topic(record)},
        ))

    # --- reply admission ------------------------------------------------
    expected_reply = record.get("expected_reply")
    if isinstance(expected_reply, bool):
        # The host never recorded a final send decision (should_reply stays
        # null), so the honest replay target is its routing admission cut.
        predicted_reply = REPLY if trace.participation_level == "strong" else SILENT
        expected_reply_label = REPLY if expected_reply else SILENT
        produced.append(make(
            TASK_REPLY, predicted_reply, expected_reply_label,
            float(trace.participation_score or 0.0),
            _error_label(None, predicted_reply, expected_reply_label,
                         missed="missed_reply", false_positive="premature_reply",
                         wrong="wrong_reply"),
        ))
    return produced


def build_dataset(
    annotations: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    config: LearningConfig | None = None,
) -> list[LearningSample]:
    """Convert `(session_key, record)` pairs, de-duplicating by sample id."""
    config = config or LearningConfig()
    latest: dict[str, LearningSample] = {}
    for session_key, record in annotations:
        for sample in samples_from_annotation(record, session_key, config=config):
            previous = latest.get(sample.sample_id)
            if previous is None or sample.annotated_at >= previous.annotated_at:
                latest[sample.sample_id] = sample
    return sorted(latest.values(), key=lambda item: (item.session_hash, item.msg_id, item.task))


def by_task(samples: Sequence[LearningSample], task: str) -> list[LearningSample]:
    return [sample for sample in samples if sample.task == task]


def sessions_of(samples: Sequence[LearningSample]) -> dict[str, list[LearningSample]]:
    grouped: dict[str, list[LearningSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.session_hash, []).append(sample)
    return grouped


__all__ = [
    "BOT", "CandidateRecord", "MAX_TOPIC_CANDIDATES", "NEW_TOPIC", "OTHER", "REPLY",
    "SAMPLE_SCHEMA_VERSION",
    "SILENT", "SOURCE_MANUAL_REPLAY", "SOURCE_RUNTIME_OUTCOME", "TASKS", "TASK_RECIPIENT",
    "TASK_REPLY", "TASK_TOPIC", "UNASSIGNED", "LearningSample", "build_dataset", "by_task",
    "sample_identifier", "samples_from_annotation", "session_hash", "sessions_of",
    "FEATURE_SCHEMA_VERSION",
]
