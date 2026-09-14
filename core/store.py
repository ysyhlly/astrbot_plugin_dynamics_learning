"""Bounded, sharded persistence for samples, policies and run state.

Samples are sharded one shared-preferences key per session, mirroring the host's
own annotation keying. A single giant blob would be re-serialised on every write
and could exceed what the preferences table should hold, so the store keeps a
small index plus one bounded list per session and prunes whole sessions when the
global cap is exceeded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .config import LearningConfig
from .policy import PolicyCandidate, can_transition, transition_error
from .samples import LearningSample, session_hash

SAMPLE_INDEX_KEY = "learning_index_v1"
SAMPLE_KEY_PREFIX = "learning_samples_v1_"
POLICY_KEY = "learning_policies_v1"
# The publish contract, materialised as one key so the consumer can read a
# single agreed artefact instead of re-deriving the shape from the policy
# records. Two implementations of the same contract is two chances to disagree,
# and the disagreement would only show up as a policy that silently never
# applies.
PUBLISHED_KEY = "learning_published_v1"
CANDIDATE_KEY = "learning_candidate_v1"
STATE_KEY = "learning_state_v1"
# The model-written contract review, kept beside the data it describes: the
# fingerprint inside it is what decides whether it still applies.
REVIEW_KEY = "learning_contract_review_v1"
STORE_SCHEMA_VERSION = 1
MAX_SESSIONS_TRACKED = 2_000


class KeyValueBackend(Protocol):
    async def get_kv_data(self, key: str, default: Any) -> Any: ...
    async def put_kv_data(self, key: str, value: Any) -> None: ...
    async def delete_kv_data(self, key: str) -> None: ...


class MemoryBackend:
    """Dictionary-backed backend for tests and offline scripting."""

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = dict(initial or {})
        self.writes = 0
        self.deletes = 0

    async def get_kv_data(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    async def put_kv_data(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.writes += 1

    async def delete_kv_data(self, key: str) -> None:
        self.data.pop(key, None)
        self.deletes += 1


def _as_int(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def _as_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


@dataclass
class StoreStats:
    sessions: int = 0
    samples: int = 0
    tasks: dict[str, int] = field(default_factory=dict)
    pruned_sessions: int = 0
    pruned_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"sessions": self.sessions, "samples": self.samples,
                "tasks": dict(self.tasks), "pruned_sessions": self.pruned_sessions,
                "pruned_samples": self.pruned_samples}


class LearningStore:
    def __init__(self, backend: KeyValueBackend) -> None:
        self.backend = backend

    # ---- index ---------------------------------------------------------

    async def _load_index(self) -> dict[str, Any]:
        raw = await self.backend.get_kv_data(SAMPLE_INDEX_KEY, {})
        if not isinstance(raw, Mapping):
            return {"store_schema_version": STORE_SCHEMA_VERSION, "sessions": {}, "total": 0}
        sessions = raw.get("sessions")
        clean: dict[str, dict[str, Any]] = {}
        if isinstance(sessions, Mapping):
            for digest, row in list(sessions.items())[:MAX_SESSIONS_TRACKED]:
                if not isinstance(digest, str) or not isinstance(row, Mapping):
                    continue
                clean[digest] = {
                    "session_key": str(row.get("session_key") or "")[:256],
                    "count": max(0, _as_int(row.get("count"))),
                    "updated_at": _as_float(row.get("updated_at")),
                }
        return {"store_schema_version": STORE_SCHEMA_VERSION, "sessions": clean,
                "total": sum(row["count"] for row in clean.values())}

    async def _save_index(self, index: Mapping[str, Any]) -> None:
        sessions = index.get("sessions") if isinstance(index, Mapping) else {}
        sessions = sessions if isinstance(sessions, Mapping) else {}
        await self.backend.put_kv_data(SAMPLE_INDEX_KEY, {
            "store_schema_version": STORE_SCHEMA_VERSION,
            "sessions": dict(sessions),
            "total": sum(_as_int(row.get("count")) for row in sessions.values()
                         if isinstance(row, Mapping)),
        })

    # ---- samples -------------------------------------------------------

    async def load_samples(self, *, config: LearningConfig | None = None) -> list[LearningSample]:
        config = config or LearningConfig()
        index = await self._load_index()
        samples: list[LearningSample] = []
        for digest in list(index["sessions"])[:MAX_SESSIONS_TRACKED]:
            rows = await self.backend.get_kv_data(SAMPLE_KEY_PREFIX + digest, [])
            if not isinstance(rows, list):
                continue
            for row in rows[: config.max_samples]:
                sample = LearningSample.from_dict(row)
                if sample is not None:
                    samples.append(sample)
        return sorted(samples, key=lambda item: (item.session_hash, item.msg_id, item.task))

    async def index_rows(self) -> list[dict[str, Any]]:
        index = await self._load_index()
        return [{"session_hash": digest, **row} for digest, row in index["sessions"].items()]

    async def replace_session(
        self,
        session_key: str,
        samples: Sequence[LearningSample],
        *,
        config: LearningConfig | None = None,
        now: float | None = None,
    ) -> StoreStats:
        """Replace every sample belonging to one session, then enforce the cap."""
        config = config or LearningConfig()
        digest = session_hash(session_key)
        rows = [sample.as_dict(include_trace=config.store_raw_trace) for sample in samples]
        if rows:
            await self.backend.put_kv_data(SAMPLE_KEY_PREFIX + digest, rows)
        else:
            await self.backend.delete_kv_data(SAMPLE_KEY_PREFIX + digest)
        index = await self._load_index()
        if rows:
            index["sessions"][digest] = {"session_key": session_key[:256], "count": len(rows),
                                         "updated_at": now if now is not None else time.time()}
        else:
            index["sessions"].pop(digest, None)
        stats = await self._prune(index, config, now=now)
        return stats

    async def _prune(self, index: dict[str, Any], config: LearningConfig,
                     *, now: float | None = None) -> StoreStats:
        index["total"] = sum(_as_int(row.get("count")) for row in index["sessions"].values())
        pruned_sessions = 0
        pruned_samples = 0
        if index["total"] > config.max_samples:
            order = sorted(index["sessions"].items(), key=lambda item: _as_float(item[1].get("updated_at")))
            for digest, row in order:
                if index["total"] <= config.max_samples:
                    break
                await self.backend.delete_kv_data(SAMPLE_KEY_PREFIX + digest)
                index["total"] -= _as_int(row.get("count"))
                pruned_samples += _as_int(row.get("count"))
                pruned_sessions += 1
                index["sessions"].pop(digest, None)
        await self._save_index(index)
        # Task mix is recomputed from the loaded samples; the index only tracks
        # per-session counts so it stays small.
        return StoreStats(sessions=len(index["sessions"]), samples=max(0, index["total"]),
                          pruned_sessions=pruned_sessions, pruned_samples=pruned_samples)

    async def clear_samples(self) -> int:
        index = await self._load_index()
        removed = 0
        for digest, row in index["sessions"].items():
            await self.backend.delete_kv_data(SAMPLE_KEY_PREFIX + digest)
            removed += _as_int(row.get("count"))
        await self.backend.put_kv_data(SAMPLE_INDEX_KEY, {
            "store_schema_version": STORE_SCHEMA_VERSION, "sessions": {}, "total": 0})
        return removed

    # ---- policies ------------------------------------------------------

    async def load_policies(self) -> list[PolicyCandidate]:
        raw = await self.backend.get_kv_data(POLICY_KEY, {})
        rows = raw.get("policies") if isinstance(raw, Mapping) else None
        rows = rows if isinstance(rows, list) else []
        result: list[PolicyCandidate] = []
        for row in rows[-500:]:
            candidate = PolicyCandidate.from_dict(row)
            if candidate is not None:
                result.append(candidate)
        return result

    async def save_policies(self, policies: Sequence[PolicyCandidate]) -> None:
        await self.backend.put_kv_data(POLICY_KEY, {
            "store_schema_version": STORE_SCHEMA_VERSION,
            "policies": [candidate.as_dict() for candidate in list(policies)[-500:]],
        })

    async def append_policy(self, candidate: PolicyCandidate) -> list[PolicyCandidate]:
        policies = await self.load_policies()
        policies = [row for row in policies if row.version != candidate.version] + [candidate]
        await self.save_policies(policies)
        return policies

    async def update_policy_status(self, version: str, status: str, *,
                                   now: float | None = None,
                                   reason: str = "") -> PolicyCandidate | None:
        """Move one policy through the state machine, or refuse and say why.

        The store is where the transition is checked, because it is the only
        place that knows the *current* state. A record's own `with_status` is a
        setter; an illegal arrow — promoting something that was never validated —
        is refused here with the two states named, rather than stored and
        discovered later by whoever reads the published file.
        """
        policies = await self.load_policies()
        updated: PolicyCandidate | None = None
        result: list[PolicyCandidate] = []
        for row in policies:
            if row.version == version:
                if not can_transition(row.status, status):
                    raise ValueError(transition_error(row.status, status))
                updated = row.with_status(status, now=now, reason=reason)
                result.append(updated)
            else:
                result.append(row)
        if updated is not None:
            await self.save_policies(result)
        return updated

    # ---- the publish contract -------------------------------------------

    async def save_candidate(self, payload: Mapping[str, Any]) -> None:
        await self.backend.put_kv_data(CANDIDATE_KEY, dict(payload))

    async def load_candidate(self) -> dict[str, Any]:
        raw = await self.backend.get_kv_data(CANDIDATE_KEY, {})
        return dict(raw) if isinstance(raw, dict) else {}

    async def clear_candidate(self) -> None:
        await self.backend.delete_kv_data(CANDIDATE_KEY)

    async def save_published(self, payload: Mapping[str, Any]) -> None:
        await self.backend.put_kv_data(PUBLISHED_KEY, dict(payload))

    async def load_published(self) -> dict[str, Any]:
        raw = await self.backend.get_kv_data(PUBLISHED_KEY, {})
        return dict(raw) if isinstance(raw, Mapping) else {}

    async def clear_published(self) -> None:
        await self.backend.delete_kv_data(PUBLISHED_KEY)

    # ---- model-written contract review ---------------------------------

    async def save_review(self, payload: Mapping[str, Any]) -> None:
        await self.backend.put_kv_data(REVIEW_KEY, dict(payload))

    async def load_review(self) -> dict[str, Any]:
        raw = await self.backend.get_kv_data(REVIEW_KEY, {})
        return dict(raw) if isinstance(raw, Mapping) else {}

    async def clear_review(self) -> None:
        await self.backend.delete_kv_data(REVIEW_KEY)

    # ---- run state -----------------------------------------------------

    async def load_state(self) -> dict[str, Any]:
        raw = await self.backend.get_kv_data(STATE_KEY, {})
        if not isinstance(raw, Mapping):
            return {}
        clean: dict[str, Any] = {}
        for key, value in list(raw.items())[:64]:
            if isinstance(key, str) and isinstance(value, (str, int, float, bool, list, dict, type(None))):
                clean[key[:64]] = value
        return clean

    async def save_state(self, state: Mapping[str, Any]) -> None:
        clean: dict[str, Any] = {}
        for key, value in list(state.items())[:64]:
            if isinstance(key, str) and isinstance(value, (str, int, float, bool, list, dict, type(None))):
                clean[key[:64]] = value
        clean["store_schema_version"] = STORE_SCHEMA_VERSION
        await self.backend.put_kv_data(STATE_KEY, clean)

    async def patch_state(self, **changes: Any) -> dict[str, Any]:
        state = await self.load_state()
        state.update(changes)
        await self.save_state(state)
        return state


def index_summary(index_rows: Iterable[Mapping[str, Any]], samples: Sequence[LearningSample]) -> dict[str, Any]:
    tasks: dict[str, int] = {}
    for sample in samples:
        tasks[sample.task] = tasks.get(sample.task, 0) + 1
    return {"sessions": len(list(index_rows)), "samples": len(samples), "tasks": tasks}


__all__ = [
    "CANDIDATE_KEY", "KeyValueBackend", "LearningStore", "MemoryBackend", "POLICY_KEY", "PUBLISHED_KEY",
    "REVIEW_KEY", "SAMPLE_INDEX_KEY", "SAMPLE_KEY_PREFIX", "STATE_KEY", "STORE_SCHEMA_VERSION",
    "StoreStats",
    "index_summary",
]
