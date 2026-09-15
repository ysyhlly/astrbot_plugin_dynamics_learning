"""Read-only ingestion of ChatDynamics' learning contract.

ChatDynamics already stores everything needed, under its own shared
preferences scope (`plugin` / `{author}/{name}`):

| Key | Written by | Used for |
| --- | --- | --- |
| `panel_runtime_v1` | `core/runtime_persistence.export_runtime_state` | session -> hash mapping, identity facts, bounded runtime context |
| `topic_annotations_v1_<sha256(session)>` | `core/topic_annotations.TopicAnnotations.save` | human labels + frozen `decision_trace` |

The annotation key is a hash, so the session name is recovered by hashing every
`session_key` in the runtime snapshot. Records whose session cannot be
recovered are reported as diagnostics and skipped: a label without a session
cannot be scored, because topic metrics are within-session only.

Every record is also counted **before** it is cleaned or converted, into a
`RawContractStats`. Those counters are the only place where "the host did not
write this field" is still distinguishable from "it was written as zero": the
sample layer normalises traces into schema 2 and re-emits them, which erases
both facts. The counters are therefore taken here, where the records are still
raw, and they are made to satisfy an accounting identity so the health report
cannot quietly lose rows.

This module never writes to the host plugin.

Two protocol versions live on either side of this module and are deliberately
*not* the same number (see `core/trace.py`):

    trace_schema_version       what the host writes   (routing_schema_version)
    policy_contract_version    what this plugin publishes (core/policy.py)

`READER_VERSION` is neither: it is this plugin's own reader revision.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .quality import RawContractStats
from .samples import session_hash

ANNOTATION_KEY_PREFIX = "topic_annotations_v1_"
RUNTIME_KEY = "panel_runtime_v1"
RUNTIME_VERSION = 1
# ---- the two protocol versions, named so they can never be confused -----
#
#   trace_schema_version    ChatDynamics -> decision trace -> Dynamics Learning
#   policy_contract_version Dynamics Learning -> /published -> ChatDynamics
#
# `READER_VERSION` is neither of them. It is this plugin's own reader/API
# revision, and it answers exactly one question: "was this snapshot taken by a
# reader that could already see field X, or by one that had never heard of it?"
# Conflating that with the host's trace schema is what made the old name
# (`contract_version`) dangerous — one word meant both, so a reader upgrade
# looked like a host contract change.
#
# Reset to 1 at v0.9.0, when the two were separated. The history lives in
# CHANGELOG.md, not in the number.
READER_VERSION = 1
MAX_RECORDS_PER_SESSION = 2_000

_ANNOTATION_KEY = re.compile(r"^topic_annotations_v1_[0-9a-f]{64}$")
_MAX_KEY_BYTES = 4_000_000

REASON_OK = ""
REASON_BAD_SHAPE = "bad_shape"
REASON_MISSING_ID = "missing_msg_id"
REASON_NOT_SERIALISABLE = "not_serialisable"
REASON_TOO_LARGE = "too_large"


@dataclass
class IngestResult:
    confirmed_sessions: set[str] = field(default_factory=set)
    partial_sessions: set[str] = field(default_factory=set)
    annotations: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    contract: RawContractStats = field(default_factory=RawContractStats)

    @property
    def records(self) -> int:
        return len(self.annotations)

    def session_rows(self) -> list[dict[str, Any]]:
        return list(self.sessions.values())


def _preference_pairs(rows: Any) -> list[tuple[str, Any]]:
    """Accept AstrBot `Preference` objects, dicts, or plain pairs."""
    pairs: list[tuple[str, Any]] = []
    if not isinstance(rows, Iterable):
        return pairs
    for row in rows:
        key = getattr(row, "key", None)
        value = getattr(row, "value", None)
        if key is None and isinstance(row, Mapping):
            key = row.get("key")
            value = row.get("value")
        if not isinstance(key, str):
            continue
        if isinstance(value, Mapping) and "val" in value:
            value = value.get("val")
        pairs.append((key, value))
    return pairs


# Keys the host may use to report its own software version in the runtime
# snapshot. Read from several names because the field does not exist yet: this
# is the *forward* half of the version contract, and a host that starts writing
# any of them is immediately usable.
HOST_VERSION_KEYS = ("plugin_version", "host_version", "chat_dynamics_version",
                     "version_name")


def _runtime_host_version(payload: Any) -> str:
    """The host's self-reported version, or "" when it does not report one.

    The result is never inferred. Learning cannot know which ChatDynamics built
    a trace it was handed, so an absent field stays absent — and a policy
    published without a host version says so, rather than implying a match that
    was never established.
    """
    if not isinstance(payload, Mapping):
        return ""
    for key in HOST_VERSION_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:64]
    meta = payload.get("meta")
    if isinstance(meta, Mapping):
        for key in HOST_VERSION_KEYS:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:64]
    return ""


def _runtime_sessions(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        return []
    result: list[dict[str, Any]] = []
    for item in sessions:
        if not isinstance(item, Mapping):
            continue
        key = item.get("session_key")
        if isinstance(key, str) and key:
            result.append(dict(item))
    return result


def clean_record(raw: Any) -> tuple[dict[str, Any] | None, str]:
    """Return the usable record (or None) and why it was rejected."""
    if not isinstance(raw, Mapping):
        return None, REASON_BAD_SHAPE
    msg_id = raw.get("msg_id")
    if not isinstance(msg_id, str) or not msg_id:
        return None, REASON_MISSING_ID
    try:
        encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None, REASON_NOT_SERIALISABLE
    if len(encoded) > _MAX_KEY_BYTES:
        return None, REASON_TOO_LARGE
    return dict(raw), REASON_OK


def _clean_record(raw: Any) -> dict[str, Any] | None:
    record, _reason = clean_record(raw)
    return record


def parse_preferences(rows: Any) -> IngestResult:
    """Split host preferences into runtime sessions and labelled records."""
    result = IngestResult()
    stats = RawContractStats(source="shared_preferences")
    pairs = _preference_pairs(rows)
    raw_annotations: list[tuple[str, Any]] = []
    runtime_payload: Any = None
    for key, value in pairs:
        if key == RUNTIME_KEY:
            runtime_payload = value
        elif _ANNOTATION_KEY.match(key):
            raw_annotations.append((key[len(ANNOTATION_KEY_PREFIX):], value))

    sessions = _runtime_sessions(runtime_payload)
    by_hash: dict[str, dict[str, Any]] = {}
    for item in sessions:
        by_hash[session_hash(item["session_key"])] = item
    result.sessions = {item["session_key"]: item for item in sessions}

    stats.annotation_keys = len(raw_annotations)
    unknown_sessions = 0
    kept = 0
    used: list[str] = []
    for digest, value in raw_annotations:
        if not isinstance(value, list):
            # The key exists but carries nothing readable: counted, not parsed.
            stats.unreadable_keys += 1
            continue
        stats.truncated += max(0, len(value) - MAX_RECORDS_PER_SESSION)
        session = by_hash.get(digest)
        if session is None:
            unknown_sessions += 1
        session_key = session["session_key"] if session is not None else ""
        if session_key and len(value) <= MAX_RECORDS_PER_SESSION and all(clean_record(row)[0] is not None for row in value):
            result.confirmed_sessions.add(session_key)
        for row in value[:MAX_RECORDS_PER_SESSION]:
            # Counted before anything can drop it, so the accounting identity
            # covers every row the host actually stored.
            stats.annotations_seen += 1
            if not session_key:
                stats.unknown_session += 1
                continue
            record, reason = clean_record(row)
            if record is None:
                stats.malformed += 1
                if reason == REASON_TOO_LARGE:
                    stats.oversized += 1
                continue
            stats.observe_record(row)
            result.annotations.append((session_key, record))
            used.append(session_key)
            kept += 1

    stats.annotations_kept = kept
    stats.observe_sessions(result.sessions, used)
    result.contract = stats
    result.diagnostics = {
        "reader_version": READER_VERSION,
        "trace_schema_versions": dict(stats.routing_schema_versions),
        "host_version": _runtime_host_version(runtime_payload),
        "runtime_present": runtime_payload is not None,
        "runtime_version": runtime_payload.get("version") if isinstance(runtime_payload, Mapping) else None,
        "runtime_version_supported": isinstance(runtime_payload, Mapping) and runtime_payload.get("version") == RUNTIME_VERSION,
        "runtime_sessions": len(sessions),
        "annotation_keys": stats.annotation_keys,
        "records": kept,
        "unknown_sessions": unknown_sessions,
        # Kept in its historical shape (unreadable keys + rejected records) so the
        # console copy does not change meaning; the precise split lives in the
        # contract block.
        "malformed": stats.malformed + stats.unreadable_keys,
        "balanced": stats.balanced,
        "source": "shared_preferences",
    }
    return result


def parse_export(payload: Any) -> IngestResult:
    """Offline import used by tests and by manual exports.

    Accepted shapes:
      * `{"sessions": [{"session_key": ..., "records": [...]}]}`
      * `[{"session_key": ..., "records": [...]}]`
      * a bare `[record, ...]` list, which requires `session_key` in each record
    """
    result = IngestResult()
    stats = RawContractStats(source="export")
    if isinstance(payload, Mapping) and payload.get("schema") == "dynamics_learning_export_v1":
        result.diagnostics = {"source": "export", "available": False, "error": "analysis export is not a restorable annotation backup", "records": 0}
        return result
    if isinstance(payload, Mapping) and "schema" in payload:
        result.diagnostics = {"source": "export", "available": False,
                              "error": "unsupported export schema", "records": 0}
        return result
    rows: Sequence[Any]
    if isinstance(payload, Mapping):
        candidate = payload.get("sessions")
        rows = candidate if isinstance(candidate, list) else [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        result.contract = stats
        result.diagnostics = {"source": "export", "available": False, "records": 0, "malformed": 1,
                              "balanced": stats.balanced,
                              "error": "unsupported export shape"}
        return result

    used: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            stats.annotation_keys += 1
            stats.annotations_seen += 1
            stats.malformed += 1
            continue
        session_key = row.get("session_key") or row.get("umo")
        records = row.get("records")
        if isinstance(session_key, str) and session_key and isinstance(records, list):
            if len(records) <= MAX_RECORDS_PER_SESSION and all(clean_record(record)[0] is not None for record in records):
                result.confirmed_sessions.add(session_key)
            result.sessions.setdefault(session_key, {"session_key": session_key})
            stats.annotation_keys += 1
            stats.truncated += max(0, len(records) - MAX_RECORDS_PER_SESSION)
            for record in records[:MAX_RECORDS_PER_SESSION]:
                stats.annotations_seen += 1
                clean, reason = clean_record(record)
                if clean is None:
                    stats.malformed += 1
                    if reason == REASON_TOO_LARGE:
                        stats.oversized += 1
                    continue
                stats.observe_record(record)
                result.annotations.append((session_key, clean))
                used.append(session_key)
                stats.annotations_kept += 1
            continue
        stats.annotation_keys += 1
        stats.annotations_seen += 1
        clean, reason = clean_record(row)
        if clean is None or not isinstance(session_key, str) or not session_key:
            stats.malformed += 1
            if reason == REASON_TOO_LARGE:
                stats.oversized += 1
            continue
        result.partial_sessions.add(session_key)
        stats.observe_record(row)
        result.annotations.append((session_key, clean))
        used.append(session_key)
        stats.annotations_kept += 1

    for session_key, _ in result.annotations:
        result.sessions.setdefault(session_key, {"session_key": session_key})
    stats.observe_sessions(result.sessions, used)
    result.contract = stats
    result.diagnostics = {
        "reader_version": READER_VERSION,
        "trace_schema_versions": dict(stats.routing_schema_versions),
        "source": "export",
        "runtime_sessions": len(result.sessions),
        "records": len(result.annotations),
        "unknown_sessions": 0,
        "malformed": stats.malformed,
        "balanced": stats.balanced,
    }
    if not result.annotations and not result.confirmed_sessions:
        result.diagnostics.update(available=False, error="export contains no usable records or confirmed sessions")
    return result


async def collect_from_host(source_plugin_id: str, *, sp_module: Any = None) -> IngestResult:
    """Read the host contract through AstrBot's shared preferences.

    Returns an empty result with a diagnostic when AstrBot or the host plugin
    is unavailable. This never raises on a missing host: a companion plugin
    must not break the bot it observes.
    """
    module = sp_module
    if module is None:
        try:
            # The host SDK is an optional runtime dependency, so it is resolved
            # by name: a plain import would type-check only in whichever of the
            # two environments happens to be installed, and the ignore code that
            # fits one (import-untyped) is wrong in the other (import-not-found).
            module = getattr(importlib.import_module("astrbot.core"), "sp", None)
        except Exception:
            module = None
    if module is None or not hasattr(module, "range_get_async"):
        return IngestResult(
            contract=RawContractStats(source="shared_preferences"),
            diagnostics={"source": "shared_preferences", "available": False,
                         "error": "astrbot shared preferences unavailable", "records": 0})
    try:
        rows = await module.range_get_async("plugin", source_plugin_id, None)
    except Exception as exc:
        return IngestResult(
            contract=RawContractStats(source="shared_preferences"),
            diagnostics={"source": "shared_preferences", "available": False,
                         "error": type(exc).__name__, "records": 0})
    result = parse_preferences(rows)
    result.diagnostics["available"] = True
    result.diagnostics["source_plugin_id"] = source_plugin_id
    return result


def merge_results(results: Iterable[IngestResult]) -> IngestResult:
    merged = IngestResult()
    for result in results:
        merged.confirmed_sessions.update(result.confirmed_sessions)
        merged.partial_sessions.update(result.partial_sessions)
        merged.annotations.extend(result.annotations)
        merged.sessions.update(result.sessions)
        merged.contract.merge(result.contract)
    merged.diagnostics = {"merged": True, "records": merged.records,
                          "runtime_sessions": len(merged.sessions),
                          "balanced": merged.contract.balanced}
    return merged


def digest_of(session_key: str) -> str:
    return hashlib.sha256(str(session_key).encode("utf-8")).hexdigest()


__all__ = [
    "ANNOTATION_KEY_PREFIX", "HOST_VERSION_KEYS", "READER_VERSION", "IngestResult",
    "REASON_BAD_SHAPE",
    "REASON_MISSING_ID", "REASON_NOT_SERIALISABLE", "REASON_OK", "REASON_TOO_LARGE", "RUNTIME_KEY",
    "clean_record", "collect_from_host", "digest_of", "merge_results", "parse_export",
    "parse_preferences",
]
