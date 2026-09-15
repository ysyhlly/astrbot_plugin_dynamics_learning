"""Operational counts from retained telemetry, never labelled training samples."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Mapping, TypeGuard


def _number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _counts(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    changed = sum(row["changed"] for row in rows)
    return {
        "comparisons": len(rows),
        "disagreements": changed,
        "disagreement_rate": changed / len(rows) if rows else None,
    }


def evaluate_shadow_coverage(payload: Any, *, now: float | None = None) -> dict[str, Any]:
    """Validate, deduplicate and bucket a schema-1 host snapshot.

    The denominator is explicitly the retained window, not lifetime traffic.
    Duplicate IDs with conflicting content are excluded entirely.
    """
    current = time.time() if now is None else now
    result: dict[str, Any] = {
        **_counts([]),
        "available": False,
        "invalid": 0,
        "duplicates": 0,
        "expired": 0,
        "buckets": [],
        "by_reason": [],
        "by_session": [],
        "by_hour": [],
        "window": {
            "from": None,
            "to": None,
            "retention_seconds": None,
            "max_records": None,
            "denominator": "retained_unique_valid_observations",
        },
    }
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        return result
    retention, maximum = payload.get("retention_seconds"), payload.get("max_records")
    observations = payload.get("observations")
    if (
        not _number(retention)
        or retention <= 0
        or type(maximum) is not int
        or maximum <= 0
        or not isinstance(observations, list)
    ):
        return result
    result["available"] = True
    result["updated_at"] = payload.get("updated_at") if _number(payload.get("updated_at")) else None
    result["window"].update(retention_seconds=retention, max_records=maximum)
    unique: dict[str, Mapping[str, Any]] = {}
    conflicts: set[str] = set()
    for row in observations:
        if not isinstance(row, Mapping) or not all(
            isinstance(row.get(k), str) and 0 < len(row[k]) <= 256
            for k in ("observation_id", "policy_id", "host_version", "session_hash")
        ):
            result["invalid"] += 1
            continue
        if any(key in row and (not isinstance(row[key], str) or len(row[key]) > 256)
               for key in ("experiment_id", "candidate_hash", "baseline_hash")):
            result["invalid"] += 1
            continue
        if (
            row.get("reason") not in ("structural", "early_return", "ambient")
            or not all(type(row.get(k)) is bool for k in ("baseline_reply", "shadow_reply", "changed"))
            or not all(_number(row.get(k)) for k in ("recorded_at", "score", "baseline_threshold", "shadow_threshold"))
            or row["changed"] != (row["baseline_reply"] != row["shadow_reply"])
            or row["recorded_at"] > current
            or row["recorded_at"] < 0
        ):
            result["invalid"] += 1
            continue
        if row["recorded_at"] < current - retention:
            result["expired"] += 1
            continue
        identity = row["observation_id"]
        if identity in unique:
            result["duplicates"] += 1
            if row != unique[identity]:
                conflicts.add(identity)
            continue
        unique[identity] = row
    result["invalid"] += len(conflicts)
    rows = sorted((r for key, r in unique.items() if key not in conflicts), key=lambda r: r["recorded_at"])[-maximum:]
    result.update(_counts(rows))
    if rows:
        result["window"].update({"from": rows[0]["recorded_at"], "to": rows[-1]["recorded_at"]})
    for field, output in (
        ("reason", "by_reason"),
        ("session_hash", "by_session"),
        ("hour", "by_hour"),
        ("bucket", "buckets"),
    ):
        groups: dict[Any, list[Mapping[str, Any]]] = {}
        for row in rows:
            key = (
                tuple(row.get(name, "") for name in ("policy_id", "host_version", "experiment_id", "candidate_hash", "baseline_hash"))
                if field == "bucket"
                else datetime.fromtimestamp(row["recorded_at"], timezone.utc).strftime("%Y-%m-%dT%H:00Z")
                if field == "hour"
                else row[field]
            )
            groups.setdefault(key, []).append(row)
        result[output] = [
            (dict(zip(("policy_id", "host_version", "experiment_id", "candidate_hash", "baseline_hash"), key)) if field == "bucket" else {"key": key}) | _counts(group)
            for key, group in sorted(groups.items())
        ]
    return result
