"""Deterministic feature extraction from a normalised decision trace.

Feature vectors are dense, ordered and float-only, so a fitted model can be
stored and replayed without the original trace. The order is frozen by
`FEATURE_NAMES`; changing it invalidates stored models, which is why
`FEATURE_SCHEMA_VERSION` is persisted alongside every model.

Nothing here reads message text: every input is a decision-trace field.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from .config import LearningConfig
from .trace import AMBIENT_CODES, EVIDENCE_FAMILIES, DecisionTrace

FEATURE_SCHEMA_VERSION = 1

_AMBIENT_SORTED = tuple(sorted(AMBIENT_CODES))
_STATE_FEATURES = (
    "st_pending_hover", "st_active_interlocutor", "st_waiting_for_answer",
    "st_last_bot_was_question", "st_intervening_zero", "st_intervening_few",
    "st_intervening_many",
)
_ROUTER_FEATURES = (
    "rc_recipient_confidence", "rc_recipient_ambiguous", "rc_recipient_threshold",
    "rc_topic_confidence", "rc_topic_ambiguous",
)
_IDENTITY_FEATURES = ("id_mention", "id_vocative", "id_subject", "id_any_reference")
_CONTEXT_FEATURES = ("ctx_prior_bot", "ctx_explicit", "ctx_mode_persona")

def _unique(names: tuple[str, ...]) -> tuple[str, ...]:
    """Drop a repeated name, keeping its first position.

    `st_pending_hover` and `st_active_interlocutor` are produced twice on
    purpose-built lists: once as the strength of an ambient evidence code and once
    as a decoded state flag. A duplicate here is not cosmetic — the vector would
    carry the same input twice, the coefficient would split across two slots that
    can never be told apart, and `LogisticModel.aligned` maps names to weights
    through a dict, so re-aligning a stored model would silently keep only the
    last of the two weights and change what the model scores.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


# The exact-name list a stored model is validated against.
FEATURE_NAMES: tuple[str, ...] = _unique((
    *(f"ev_{code}" for code in _AMBIENT_SORTED),
    *(f"st_{code}" for code in _AMBIENT_SORTED),
    *_STATE_FEATURES,
    *_ROUTER_FEATURES,
    *_IDENTITY_FEATURES,
    *_CONTEXT_FEATURES,
    *(f"fam_{family}" for family in sorted(EVIDENCE_FAMILIES)),
))
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def _flag(value: Any) -> float:
    return 1.0 if value is True else 0.0


def build_features(trace: DecisionTrace) -> dict[str, float]:
    """Return the named feature map. Missing values are 0.0, never NaN."""
    strengths = trace.strengths()
    result: dict[str, float] = {}
    for code in _AMBIENT_SORTED:
        present = code in strengths
        result[f"ev_{code}"] = 1.0 if present else 0.0
        result[f"st_{code}"] = float(strengths.get(code, 0.0))

    state = trace.state if isinstance(trace.state, Mapping) else {}
    intervening = state.get("intervening_users")
    count = intervening if isinstance(intervening, int) else 0
    result["st_pending_hover"] = _flag(state.get("pending_hover"))
    result["st_active_interlocutor"] = 1.0 if state.get("active_interlocutor") else 0.0
    result["st_waiting_for_answer"] = _flag(state.get("waiting_for_answer"))
    result["st_last_bot_was_question"] = _flag(state.get("last_bot_was_question"))
    result["st_intervening_zero"] = 1.0 if count == 0 else 0.0
    result["st_intervening_few"] = 1.0 if 0 < count <= 2 else 0.0
    result["st_intervening_many"] = 1.0 if count > 2 else 0.0

    result["rc_recipient_confidence"] = float(trace.recipient_confidence)
    result["rc_recipient_ambiguous"] = 1.0 if trace.recipient_ambiguous else 0.0
    result["rc_recipient_threshold"] = float(trace.recipient_threshold or 0.0)
    result["rc_topic_confidence"] = float(trace.topic_confidence)
    result["rc_topic_ambiguous"] = 1.0 if trace.topic_ambiguous else 0.0

    identity = trace.identity if isinstance(trace.identity, Mapping) else {}
    result["id_mention"] = _flag(identity.get("mention"))
    result["id_vocative"] = _flag(identity.get("vocative"))
    result["id_subject"] = _flag(identity.get("subject"))
    reference = identity.get("bot_reference")
    result["id_any_reference"] = 0.0 if not reference or reference == "none" else 1.0

    result["ctx_prior_bot"] = float(trace.prior_bot_proxy)
    result["ctx_explicit"] = 1.0 if trace.is_explicit else 0.0
    result["ctx_mode_persona"] = 1.0 if trace.mode == "persona" else 0.0

    for family in EVIDENCE_FAMILIES:
        result[f"fam_{family}"] = float(trace.family_contributions.get(family, 0.0))

    # Diagnostic-only keys: NOT part of FEATURE_NAMES, so they never enter a
    # model vector.
    #
    # base_score preserves the host's pre-clamp additive score, which is what a
    # threshold sweep has to replay.
    result["base_score"] = max(0.0, min(1.0, float(trace.contribution_total)))
    # ctx_bot_targeted caches the *structural* outcome of an explicit turn. It
    # is a deterministic function of the evidence codes, not a label: the host
    # never scores an explicit turn, so replaying one at any threshold has to
    # return the same answer. Storing it lets a sample persisted without a raw
    # trace still replay explicit turns faithfully.
    result["ctx_bot_targeted"] = 1.0 if trace.bot_targeted else 0.0

    for name in FEATURE_NAMES:
        value = result.get(name, 0.0)
        result[name] = value if math.isfinite(value) else 0.0
    return result


def vector(features: Mapping[str, float]) -> list[float]:
    """Project a feature map into the frozen order."""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def feature_summary(trace: DecisionTrace) -> dict[str, Any]:
    """Compact, human-readable evidence view for the console (no user identifiers)."""
    return {
        "codes": sorted(trace.codes),
        "explicit_code": trace.explicit_code,
        "prior_bot_proxy": trace.prior_bot_proxy,
        "contribution_total": trace.contribution_total,
        "family_contributions": {key: round(float(value), 4)
                                 for key, value in sorted(trace.family_contributions.items())},
        "level": trace.participation_level,
        "degraded": trace.degraded,
        "mode": trace.mode,
        "weights_version": trace.weights_version,
    }


def build_features_for_config(trace: DecisionTrace, config: LearningConfig) -> dict[str, float]:
    """Config-aware alias kept so a future schema revision can gate features."""
    del config
    return build_features(trace)


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_INDEX",
    "FEATURE_SCHEMA_VERSION",
    "build_features",
    "build_features_for_config",
    "feature_summary",
    "vector",
]
