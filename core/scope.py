"""Learning scope: which conversation a sample is evidence about.

A learning sample is only comparable with another sample from the *same*
conversation, so the scope is an identity, not a display detail. Getting it
wrong fails silently in both directions: two scopes that should be one split a
conversation's evidence in half, and two that should be separate quietly pool
unrelated behaviour into a single rate.

**What the host contract can prove today.** ChatDynamics creates every runtime
from an event as

    session_key = unified_msg_origin or group_id
    umo         = unified_msg_origin or session_key
    group_id    = get_group_id()

and its own restore path refuses a snapshot whose `umo` differs from the
session key. So in normal operation `umo == session_key`, while `group_id` is
the platform's raw group identifier — it is *not* platform-qualified (two
adapters can emit the same value), and it is written back as the session key
itself when an event carries no group id. So the contract exposes **no field
that proves two sessions are the same conversation**, and this module does not
pretend otherwise:

    session scope   the only scope this contract can prove stable
    group_id        diagnostic metadata, never an aggregation key

`resolve_scope` therefore returns the session identity for every input. A
future contract that defines a real cross-session conversation identity —
stating its behaviour across sessions, adapters and bot accounts — changes this
one function; nothing else in the pipeline has to know that it happened.

The invariant PR1 pins with tests:

    scope_hash == session_hash(session_key)     # byte for byte

Re-hashing the same key under a new prefix (`session:<key>`) would split every
stored sample from its own history on upgrade — the exact failure this module
exists to prevent.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

SCOPE_SOURCE_SESSION = "session"
SCOPE_SOURCE_SESSION_UMO_EQUAL = "session_umo_equal"
SCOPE_SOURCE_SESSION_UMO_MISMATCH = "session_umo_mismatch"

# Display only. Any comparison, storage key or API identity uses the full digest.
SCOPE_LABEL_CHARS = 12

# Whether one scope can cover several host sessions. False under the current
# contract, where `scope_hash` *is* `session_hash`, and it has one visible
# consequence: a stability rule of the form "3+ sessions per scope" would be
# unreachable, so the profile layer scales that requirement to this fact instead
# of carrying a gate nobody can pass. The day requirement keeps the "not one
# long evening of labelling" intent in the meantime.
SCOPE_SPANS_SESSIONS = False


def session_hash(session_key: str) -> str:
    """Same hashing the host uses for its per-session annotation key."""
    return hashlib.sha256(str(session_key).encode("utf-8")).hexdigest()


def scope_label(scope_hash: str, *, chars: int = SCOPE_LABEL_CHARS) -> str:
    """A shortened scope hash, for a label a human reads and nothing else.

    Separate from the identity on purpose: a truncated digest in a lookup key is
    a collision waiting to happen, and `profiles[scope_hash.slice(0, 12)]` in a
    page is hard to notice and harder to debug.
    """
    return str(scope_hash)[:max(1, int(chars))]


def _group_hint(runtime_meta: Mapping[str, Any], session_key: str) -> str:
    """A stable digest of the host's `group_id`, or empty when it says nothing.

    Diagnostic provenance only: it answers "did the host hand us a group id that
    is not just the session key", which is what makes the difference between a
    real platform group id and the contract's own fallback visible.

    It is deliberately **not** an aggregation key. Two adapters can share one
    raw group id, so grouping by it would pool two unrelated conversations; and
    because the host sometimes writes the session key here instead, a sample
    would look groupable or not depending on which code path created the
    runtime. Empty means "no distinct group id was recorded", which is honest
    about not knowing, rather than a hash of a value that carries no grouping.
    """
    group_id = runtime_meta.get("group_id")
    if not isinstance(group_id, str) or not group_id or group_id == session_key:
        return ""
    return hashlib.sha256(f"group\x1f{group_id}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LearningScope:
    """The conversation one sample belongs to, plus how that was decided."""

    scope_hash: str
    scope_source: str = SCOPE_SOURCE_SESSION
    # Diagnostic only. Never used as an aggregation key; see `_group_hint`.
    group_hint_hash: str = ""

    @property
    def label(self) -> str:
        return scope_label(self.scope_hash)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_hash": self.scope_hash,
            "scope_label": self.label,
            "scope_source": self.scope_source,
            "group_hint_hash": self.group_hint_hash,
        }


def resolve_scope(session_key: str, runtime_meta: Mapping[str, Any] | None = None) -> LearningScope:
    """Resolve one session's learning scope.

    The runtime snapshot is *not* allowed to change the identity: it only
    records which host facts were available, so a later contract can tell apart
    "the host confirmed the session scope" from "we fell back to it".
    """
    key = str(session_key)
    runtime = runtime_meta if isinstance(runtime_meta, Mapping) else {}
    umo = runtime.get("umo")
    if not isinstance(umo, str) or not umo:
        source = SCOPE_SOURCE_SESSION
    elif umo == key:
        source = SCOPE_SOURCE_SESSION_UMO_EQUAL
    else:
        # The host's own restore path treats this as a contract violation, so it
        # is reported rather than silently accepted — but it still never decides
        # the identity.
        source = SCOPE_SOURCE_SESSION_UMO_MISMATCH
    return LearningScope(scope_hash=session_hash(key), scope_source=source,
                         group_hint_hash=_group_hint(runtime, key))


__all__ = [
    "SCOPE_LABEL_CHARS", "SCOPE_SOURCE_SESSION", "SCOPE_SOURCE_SESSION_UMO_EQUAL",
    "SCOPE_SOURCE_SESSION_UMO_MISMATCH", "LearningScope", "resolve_scope", "scope_label",
    "session_hash",
]
