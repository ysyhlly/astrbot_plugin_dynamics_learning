"""Stable evidence for deferring automatic reviews and revising pending proposals."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .contract import MODEL_INPUT_LIMIT, candidate, digest, encode, policy_allowed

DEFER_FINGERPRINT = "review_defer_fingerprint"
REVIEW_EVIDENCE_VERSION = 1
_ADJUSTMENT_TARGET = {
    "scene_adjustment": "scene_pattern",
    "expression_adjustment": "expression_style",
}
_FEATURE_FLAGS = {
    "affection": ("affection_enabled", True),
    "jargon": ("jargon_enabled", True),
    "scene_pattern": ("scene_learning_enabled", False),
    "scene_adjustment": ("scene_learning_enabled", False),
    "expression_style": ("expression_style_enabled", False),
    "expression_adjustment": ("expression_style_enabled", False),
}
_PROPOSITION_FIELDS = (
    "id",
    "entry_id",
    "scope",
    "persona",
    "member",
    "category",
    "key",
    "value",
    "reason",
    "mode",
    "source",
    "base_revision",
    "baseline",
    "host_version",
    "days",
    "expression_policy",
    "profile_policy",
    "scene_policy",
)
_MATERIAL_FIELDS = (
    "id",
    "category",
    "value",
    "proposal_id",
    "revision",
    "baseline",
    "host_version",
)


@dataclass(frozen=True)
class ReviewContext:
    provider_id: str
    model: str
    provider_instance: object | None = field(repr=False)
    fingerprint: str
    target_material: dict | None
    automatic_allowed: bool
    evidence_authorized: bool


def _evidence_record(row):
    if not isinstance(row, dict):
        return row
    return {key: value for key, value in row.items() if key != "received_at"}


def _scoped_evidence(proposal):
    evidence = proposal.get("evidence", [])
    scoped = bool(evidence) and all(
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and isinstance(row.get("message_id"), str)
        and row.get("scope") == proposal["scope"]
        and row.get("persona") == proposal["persona"]
        for row in evidence
    )
    if not scoped:
        return False
    member = proposal["member"]
    if proposal["category"] == "expression_example":
        value = proposal["value"]
        by_id = {row.get("id"): row.get("member") for row in evidence}
        return (
            isinstance(value, dict)
            and len(evidence) == 2
            and value.get("context_id") in by_id
            and by_id.get(value.get("expression_id")) == member
            and value.get("context_id") != value.get("expression_id")
        )
    return not member or all(row.get("member") == member for row in evidence)


def provider_model(provider):
    get_model = getattr(provider, "get_model", None)
    model = get_model() if callable(get_model) else getattr(provider, "model_name", "")
    config = getattr(provider, "provider_config", None) or {}
    return str(model or config.get("model") or "")


def _route(engine):
    provider_id = engine.provider_for("review_provider")
    getter = getattr(getattr(engine, "_host_context", None), "get_provider_by_id", None)
    provider = getter(provider_id) if provider_id and callable(getter) else None
    config = getattr(provider, "provider_config", None) or {}
    route = {
        key: str(config[key])
        for key in ("type", "api_base", "base_url", "provider_source_id")
        if config.get(key)
    }
    ready = not callable(getter) or (
        callable(getattr(provider, "text_chat", None))
        and not getattr(provider, "is_native_decision_provider", False)
    )
    return (
        provider_id,
        provider if ready else None,
        provider_model(provider),
        route,
        ready,
    )


def _material(row, now):
    if row is None:
        return None
    return {
        **{key: row.get(key) for key in _MATERIAL_FIELDS},
        "active": bool(row.get("active")),
        "expired": row.get("category") != "affection" and row.get("expires", 0) <= now,
    }


def review_context(engine, proposal, instruction):
    """Bind an authorized proposal to the current review rules, material and route."""
    scope, persona, category = (
        proposal[key] for key in ("scope", "persona", "category")
    )
    now = time.time()
    provider_id, provider, model, route, ready = _route(engine)
    current, target = engine.store.review_material(
        proposal, _ADJUSTMENT_TARGET.get(category)
    )
    target_material = _material(target, now)
    active_target = (
        target_material and target_material["active"] and not target_material["expired"]
    )
    valid_target = category not in _ADJUSTMENT_TARGET or (
        active_target and target["revision"] == proposal["value"]["target_revision"]
    )
    target_payload = (
        {key: target[key] for key in ("category", "value", "proposal_id", "revision")}
        if category in _ADJUSTMENT_TARGET and valid_target
        else None
    )
    flag, default = _FEATURE_FLAGS.get(category, ("enabled", True))
    allowed = engine.allowed(scope)
    feature_enabled = bool(engine.config.get(flag, default))
    live_persona = engine.current_personas.get(scope, persona)
    mode = engine.mode(category)
    policy = policy_allowed(proposal)
    expired = proposal["created"] + proposal["days"] * 86400 <= now
    base_valid = (current["revision"] if current else 0) == proposal["base_revision"]
    if category == "participation":
        base_valid = base_valid and engine.latest.get((scope, persona)) == (
            proposal["baseline"],
            proposal["host_version"],
        )
    scoped = _scoped_evidence(
        proposal
    ) and not engine.store.source_withdrawals.has_withdrawn(scope, proposal["evidence"])
    fingerprint = digest(
        [
            REVIEW_EVIDENCE_VERSION,
            instruction,
            provider_id,
            model,
            route,
            ready,
            [allowed, feature_enabled, live_persona, mode, policy, expired, base_valid],
            {key: proposal.get(key) for key in _PROPOSITION_FIELDS},
            [_evidence_record(row) for row in proposal.get("evidence", [])],
            _material(proposal.get("before"), now),
            _material(current, now),
            target_material,
        ]
    )
    return ReviewContext(
        provider_id,
        model,
        provider,
        fingerprint,
        target_payload,
        bool(
            allowed
            and feature_enabled
            and policy
            and scoped
            and not expired
            and valid_target
            and base_valid
            and mode in {"llm", "llm_then_human"}
            and proposal["mode"] != "auto"
            and live_persona == persona
        ),
        scoped,
    )


def review_is_sleeping(proposal, context):
    return (
        proposal["state"] == "pending"
        and context.automatic_allowed
        and proposal.get(DEFER_FINGERPRINT) == context.fingerprint
    )


def _corrects(previous, current, source):
    if (
        previous["member"] != current["member"]
        or previous["turn_id"] != current["turn_id"]
    ):
        return False
    if source.startswith("human:"):
        return previous.get("kind") == current.get("kind") == "operator_correction"
    old_context, context = previous.get("context", {}), current.get("context", {})
    if source == "system:scene-feedback" and previous["id"] == current["id"]:
        return (
            context.get("scene_feedback", {}).get("verified") is True
            and old_context.get("scene_feedback") != context["scene_feedback"]
        )
    return (
        source == "llm:analysis"
        and current.get("kind") == "outcome"
        and (
            previous.get("kind") != "outcome"
            or previous.get("outcome") != current.get("outcome")
            or previous.get("reply") != current.get("reply")
        )
        and current.get("received_at", 0) > previous.get("received_at", 0)
    )


def pending_evidence_update(old, new, before, source, previous_sources):
    """Keep bounded, validated evidence while preserving the pending proposition's identity."""
    if (
        old["created"] + old["days"] * 86400 <= time.time()
        or not _scoped_evidence(new)
        or source.startswith("human:") != old["source"].startswith("human:")
    ):
        return None

    existing = {row["message_id"]: row for row in old["evidence"]}
    incoming = {}
    changed = False
    for row in new["evidence"]:
        identity = row["message_id"]
        previous = incoming.get(identity, existing.get(identity))
        if previous is None:
            if identity not in previous_sources:
                incoming[identity] = row
                changed = True
            elif previous_sources[identity] == row["id"]:
                incoming[identity] = row
        elif _evidence_record(previous) == _evidence_record(row):
            incoming[identity] = previous
        elif _corrects(previous, row, source):
            incoming[identity] = row
            changed = True
        elif previous["id"] == row["id"]:
            incoming[identity] = previous
    if not changed:
        return None

    category, value = new["category"], new["value"]
    raw = {key: new[key] for key in ("category", "key", "member", "reason")}
    raw["value"] = (
        {k: value[k] for k in ("experience_delta", "tone")}
        if category == "affection"
        else value
    )
    raw["days"] = old["days"]  # Evidence updates do not renew a proposal's lifetime.
    limit = (
        100 if category in {"typing_habit", "scene_pattern", "expression_style"} else 30
    )
    added = list(incoming.values())
    remainder = [row for identity, row in existing.items() if identity not in incoming]
    keep_old = limit - len(added)
    merged = (remainder[-keep_old:] if keep_old > 0 else []) + added
    for evidence in (merged, added):
        # When retaining old records would exceed the existing single-review input budget, use the new snapshot.
        if (
            evidence is merged
            and len(encode({"proposal": {**old, "evidence": evidence}}))
            > MODEL_INPUT_LIMIT
        ):
            continue
        raw["evidence_ids"] = [row["id"] for row in evidence]
        try:
            normalized = candidate(
                raw, evidence, before if before and before.get("active") else None
            )
        except (ValueError, TypeError, KeyError):
            continue
        if normalized["value"] == value and normalized["key"] == old["key"]:
            return normalized
    return None
