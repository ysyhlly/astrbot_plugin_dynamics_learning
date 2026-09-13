"""Read-only ingestion of ChatDynamics' learning contract.

ChatDynamics already stores everything needed, under its own shared
preferences scope (`plugin` / `{author}/{name}`):

| Key | Written by | Used for |
| --- | --- | --- |
| `panel_runtime_v1` | `core/runtime_persistence.export_runtime_state` | session -> hash mapping, bounded runtime context |
| `topic_annotations_v1_<sha256(session)>` | `core/topic_annotations.TopicAnnotations.save` | human labels + frozen `decision_trace` |

The annotation key is a hash, so the session name is recovered by hashing every
`session_key` in the runtime snapshot. Records whose session cannot be
recovered are reported as diagnostics and skipped: a label without a session
cannot be scored, because topic metrics are within-session only.

This module never writes to the host plugin.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .samples import session_hash

ANNOTATION_KEY_PREFIX = "topic_annotations_v1_"
RUNTIME_KEY = "panel_runtime_v1"
RUNTIME_VERSION = 1
CONTRACT_VERSION = 2
MAX_RECORDS_PER_SESSION = 2_000

_ANNOTATION_KEY = re.compile(r"^topic_annotations_v1_[0-9a-f]{64}$")
_MAX_KEY_BYTES = 4_000_000


@dataclass
class IngestResult:
    annotations: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

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


def _runtime_sessions(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping) or payload.get("version") != RUNTIME_VERSION:
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


def _clean_record(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    msg_id = raw.get("msg_id")
    if not isinstance(msg_id, str) or not msg_id:
        return None
    try:
        encoded = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return None
    if len(encoded) > _MAX_KEY_BYTES:
        return None
    return dict(raw)


def parse_preferences(rows: Any) -> IngestResult:
    """Split host preferences into runtime sessions and labelled records."""
    result = IngestResult()
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

    unknown_sessions = 0
    malformed = 0
    kept = 0
    for digest, value in raw_annotations:
        if not isinstance(value, list):
            malformed += 1
            continue
        session = by_hash.get(digest)
        if session is None:
            unknown_sessions += 1
            continue
        session_key = session["session_key"]
        for row in value[:MAX_RECORDS_PER_SESSION]:
            record = _clean_record(row)
            if record is None:
                malformed += 1
                continue
            result.annotations.append((session_key, record))
            kept += 1

    result.diagnostics = {
        "contract_version": CONTRACT_VERSION,
        "runtime_present": runtime_payload is not None,
        "runtime_sessions": len(sessions),
        "annotation_keys": len(raw_annotations),
        "records": kept,
        "unknown_sessions": unknown_sessions,
        "malformed": malformed,
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
    rows: Sequence[Any]
    if isinstance(payload, Mapping):
        candidate = payload.get("sessions")
        rows = candidate if isinstance(candidate, list) else [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        result.diagnostics = {"source": "export", "records": 0, "malformed": 1,
                              "error": "unsupported export shape"}
        return result

    malformed = 0
    for row in rows:
        if not isinstance(row, Mapping):
            malformed += 1
            continue
        session_key = row.get("session_key") or row.get("umo")
        records = row.get("records")
        if isinstance(session_key, str) and session_key and isinstance(records, list):
            for record in records[:MAX_RECORDS_PER_SESSION]:
                clean = _clean_record(record)
                if clean is None:
                    malformed += 1
                    continue
                result.annotations.append((session_key, clean))
            continue
        clean = _clean_record(row)
        if clean is None or not isinstance(session_key, str) or not session_key:
            malformed += 1
            continue
        result.annotations.append((session_key, clean))

    result.sessions = {}
    for session_key, _ in result.annotations:
        result.sessions.setdefault(session_key, {"session_key": session_key})
    result.diagnostics = {
        "contract_version": CONTRACT_VERSION,
        "source": "export",
        "runtime_sessions": len(result.sessions),
        "records": len(result.annotations),
        "unknown_sessions": 0,
        "malformed": malformed,
    }
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
        return IngestResult(diagnostics={"source": "shared_preferences", "available": False,
                                         "error": "astrbot shared preferences unavailable",
                                         "records": 0})
    try:
        rows = await module.range_get_async("plugin", source_plugin_id, None)
    except Exception as exc:
        return IngestResult(diagnostics={"source": "shared_preferences", "available": False,
                                         "error": type(exc).__name__, "records": 0})
    result = parse_preferences(rows)
    result.diagnostics["available"] = True
    result.diagnostics["source_plugin_id"] = source_plugin_id
    return result


def merge_results(results: Iterable[IngestResult]) -> IngestResult:
    merged = IngestResult()
    for result in results:
        merged.annotations.extend(result.annotations)
        merged.sessions.update(result.sessions)
    merged.diagnostics = {"merged": True, "records": merged.records,
                          "runtime_sessions": len(merged.sessions)}
    return merged


def digest_of(session_key: str) -> str:
    return hashlib.sha256(str(session_key).encode("utf-8")).hexdigest()


__all__ = [
    "ANNOTATION_KEY_PREFIX", "CONTRACT_VERSION", "IngestResult", "RUNTIME_KEY",
    "collect_from_host", "digest_of", "merge_results", "parse_export", "parse_preferences",
]
