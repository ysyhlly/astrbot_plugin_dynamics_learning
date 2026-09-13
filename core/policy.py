"""Policy parameters, bounded deltas, and the parameterised replay decision.

Every parameter here is a **real ChatDynamics configuration key** with the real
default taken from its `_conf_schema.json`. A recommendation that cannot be
expressed as one of these keys is reported as engineering diagnostics instead
of being dressed up as a config change.

This module deliberately contains **no write path**: nothing here can change
the host's configuration. `PolicyCandidate` objects are records, not actions.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .trace import DecisionTrace

POLICY_SCHEMA_VERSION = 1

# Host configuration keys this plugin is allowed to reason about, with the
# host defaults and the host's own slider ranges.
PARAM_SPECS: dict[str, dict[str, Any]] = {
    "strong_addressivity_threshold": {
        "default": 0.70, "min": 0.50, "max": 0.90, "step": 0.02,
        "label": "强指代判定阈值",
        "hint": "定向度评分达到该值判定为明确指向机器人；同时决定 legacy 准入是否回复。",
    },
    "safe_hover_threshold": {
        "default": 0.40, "min": 0.20, "max": 0.60, "step": 0.02,
        "label": "安全悬停判定阈值",
        "hint": "低于强指代且高于该值时只静默记入图谱。",
    },
    "topic_commit_threshold": {
        "default": 0.58, "min": 0.30, "max": 0.95, "step": 0.02,
        "label": "话题确定归属阈值",
        "hint": "匹配得分达到该值且满足领先间隔才确定归入已有话题。默认由 topic_join_threshold 映射而来。",
    },
    "topic_join_threshold": {
        "default": 0.48, "min": 0.30, "max": 0.85, "step": 0.02,
        "label": "话题阈值兼容设置",
        "hint": "旧兼容键：小于 0.58 时确定归属阈值为 max(0.58, 本值 + 0.10)。",
    },
    "topic_margin_threshold": {
        "default": 0.06, "min": 0.0, "max": 0.50, "step": 0.01,
        "label": "话题确定归属领先间隔",
        "hint": "最佳候选相对第二候选的最低得分差。",
    },
    "parent_accept_threshold": {
        "default": 0.72, "min": 0.50, "max": 0.95, "step": 0.02,
        "label": "推断回复边接受阈值",
        "hint": "多因子打分达到该值且领先时才建立推断回复边。",
    },
}
PARAM_NAMES = tuple(PARAM_SPECS)

BASE_POLICY: dict[str, float] = {name: float(spec["default"]) for name, spec in PARAM_SPECS.items()}

# Error kinds, named the way the review page talks about them. Kept as literals
# so this module stays free of any dependency on the sample layer.
ERROR_MISSED_BOT = "missed_bot"
ERROR_FALSE_BOT = "false_bot"
ERROR_FRAGMENTATION = "fragmentation"
ERROR_WRONG_MERGE = "wrong_merge"
ERROR_MISSED_REPLY = "missed_reply"
ERROR_PREMATURE_REPLY = "premature_reply"

# What each adjustment is *for*, keyed by parameter and the direction of the
# move. An adjustment that cannot name the error it is aimed at can only be
# judged on the global number — which is how a genuinely good change gets
# thrown away for being "only" +0.6%.
TARGET_ERROR_BY_DIRECTION: dict[tuple[str, int], str] = {
    ("strong_addressivity_threshold", -1): ERROR_MISSED_BOT,
    ("strong_addressivity_threshold", +1): ERROR_FALSE_BOT,
    ("safe_hover_threshold", -1): ERROR_MISSED_BOT,
    ("safe_hover_threshold", +1): ERROR_FALSE_BOT,
    ("topic_commit_threshold", -1): ERROR_FRAGMENTATION,
    ("topic_commit_threshold", +1): ERROR_WRONG_MERGE,
    ("topic_join_threshold", -1): ERROR_FRAGMENTATION,
    ("topic_join_threshold", +1): ERROR_WRONG_MERGE,
}

# Evidence codes whose structural outcome is "the bot is the addressee".
TARGETED_EXPLICIT_CODES = frozenset({
    "canonical_recipient", "bot_mention", "vocative", "bot_reply", "routed_bot",
})

STATUS_CANDIDATE = "candidate"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
STATUS_ROLLED_BACK = "rolled_back"
STATUSES = (STATUS_CANDIDATE, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_ROLLED_BACK)

SOURCE_RECIPIENT_SWEEP = "recipient_threshold_sweep"
SOURCE_TOPIC_SWEEP = "topic_threshold_sweep"
SOURCE_MANUAL = "manual"


def clamp_param(name: str, value: Any) -> float:
    spec = PARAM_SPECS.get(name)
    if spec is None:
        raise KeyError(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(spec["default"])
    number = float(value)
    if not math.isfinite(number):
        return float(spec["default"])
    return max(float(spec["min"]), min(float(spec["max"]), number))


def normalize_policy(raw: Any) -> dict[str, float]:
    """Fill missing keys with host defaults and clamp everything into range."""
    policy = dict(BASE_POLICY)
    if isinstance(raw, Mapping):
        for name in PARAM_NAMES:
            if name in raw:
                policy[name] = clamp_param(name, raw[name])
    policy["safe_hover_threshold"] = min(policy["safe_hover_threshold"],
                                         round(policy["strong_addressivity_threshold"] - 0.05, 4))
    policy["safe_hover_threshold"] = clamp_param("safe_hover_threshold", policy["safe_hover_threshold"])
    return policy


def bounded_target(name: str, base: float, target: float, max_delta_ratio: float) -> float:
    """Move `base` toward `target` by at most `max_delta_ratio` of the base value."""
    base = clamp_param(name, base)
    target = clamp_param(name, target)
    if max_delta_ratio <= 0:
        return base
    budget = abs(base) * max_delta_ratio
    if budget == 0:
        return base
    delta = max(-budget, min(budget, target - base))
    return clamp_param(name, round(base + delta, 4))


def policy_deltas(base: Mapping[str, float], candidate: Mapping[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in PARAM_NAMES:
        before = float(base.get(name, BASE_POLICY[name]))
        after = float(candidate.get(name, BASE_POLICY[name]))
        if abs(after - before) < 1e-9:
            continue
        spec = PARAM_SPECS[name]
        rows.append({
            "param": name,
            "label": spec["label"],
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(after - before, 4),
            "delta_ratio": round((after - before) / before, 4) if before else None,
        })
    return rows


@dataclass(frozen=True)
class PolicyCandidate:
    version: str
    params: Mapping[str, float]
    baseline: Mapping[str, float] = field(default_factory=lambda: dict(BASE_POLICY))
    source: str = SOURCE_MANUAL
    rationale: str = ""
    created_at: float = 0.0
    status: str = STATUS_CANDIDATE
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "version": self.version,
            "params": {key: round(float(value), 4) for key, value in self.params.items()},
            "baseline": {key: round(float(value), 4) for key, value in self.baseline.items()},
            "deltas": policy_deltas(self.baseline, self.params),
            "source": self.source,
            "rationale": self.rationale,
            "created_at": self.created_at,
            "status": self.status,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "PolicyCandidate | None":
        if not isinstance(raw, Mapping):
            return None
        version = raw.get("version")
        if not isinstance(version, str) or not version:
            return None
        status = raw.get("status")
        evidence = raw.get("evidence")
        return cls(
            version=version[:64],
            params=normalize_policy(raw.get("params")),
            baseline=normalize_policy(raw.get("baseline")),
            source=str(raw.get("source") or SOURCE_MANUAL)[:64],
            rationale=str(raw.get("rationale") or "")[:1000],
            created_at=float(raw.get("created_at") or 0.0),
            status=status if status in STATUSES else STATUS_CANDIDATE,
            evidence=dict(evidence) if isinstance(evidence, Mapping) else {},
        )

    def with_status(self, status: str) -> "PolicyCandidate":
        return PolicyCandidate(
            version=self.version, params=self.params, baseline=self.baseline,
            source=self.source, rationale=self.rationale, created_at=self.created_at,
            status=status if status in STATUSES else self.status, evidence=self.evidence,
        )


def next_version(existing: Iterable[str]) -> str:
    highest = 0
    for name in existing:
        if not isinstance(name, str) or not name.startswith("policy_v"):
            continue
        suffix = name[len("policy_v"):]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"policy_v{highest + 1}"


def candidate_from(
    changes: Mapping[str, float],
    *,
    baseline: Mapping[str, float] | None = None,
    source: str = SOURCE_MANUAL,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
    existing_versions: Iterable[str] = (),
    now: float | None = None,
) -> PolicyCandidate:
    base = normalize_policy(baseline)
    params = dict(base)
    for name, value in changes.items():
        if name in PARAM_SPECS:
            params[name] = clamp_param(name, value)
    params = normalize_policy(params)
    return PolicyCandidate(
        version=next_version(existing_versions), params=params, baseline=base,
        source=source, rationale=rationale[:1000],
        created_at=now if now is not None else time.time(),
        evidence=dict(evidence or {}),
    )


@dataclass(frozen=True)
class ReplayDecision:
    """What the parameterised decision function produces for one trace."""

    targeted: bool
    level: str
    score: float
    topic_committed: bool

    @property
    def recipient_label(self) -> str:
        return "bot" if self.targeted else "other"

    @property
    def reply_label(self) -> str:
        return "reply" if self.level == "strong" else "silent"


def ambient_score(trace: DecisionTrace) -> float:
    """Reproduce the host's ambient additive score from the recorded ledger.

    `contribution_total` is the sum of the recorded evidence contributions, which
    is exactly the host's pre-clamp score. Re-clamping here reproduces
    `ParticipationPolicy.evaluate` without re-deriving any weight.
    """
    return max(0.0, min(1.0, float(trace.contribution_total)))


def decide(
    trace: DecisionTrace,
    policy: Mapping[str, float],
    *,
    model_score: float | None = None,
) -> ReplayDecision:
    """Replay one trace under a policy.

    `model_score`, when given, replaces the additive score for ambient turns
    with a fitted probability in [0, 1]. Structural (explicit) turns and the
    no-prior-bot early return are threshold-independent, exactly as in the host.
    """
    policy = normalize_policy(policy)
    strong = float(policy["strong_addressivity_threshold"])
    hover = float(policy["safe_hover_threshold"])
    commit = float(policy["topic_commit_threshold"])

    topic_committed = bool(
        trace.topic_id and not trace.topic_ambiguous and trace.topic_confidence >= commit
    )

    if trace.is_explicit:
        targeted = bool(trace.bot_targeted)
        return ReplayDecision(targeted, "strong" if targeted else "weak",
                              float(trace.participation_score or 0.0), topic_committed)

    if trace.prior_bot_proxy == 0:
        # Host early return: no prior bot message, level stays weak.
        score = min(1.0, ambient_score(trace))
        return ReplayDecision(False, "weak", score, topic_committed)

    score = ambient_score(trace) if model_score is None else max(0.0, min(1.0, float(model_score)))
    level = "strong" if score >= strong else "hover" if score >= hover else "weak"
    return ReplayDecision(level == "strong", level, score, topic_committed)


def clamp_cumulative(original: Mapping[str, float], proposed: Mapping[str, float],
                     max_ratio: float) -> tuple[dict[str, float], list[str]]:
    """Bound the total drift from the original baseline.

    One step is capped at ±5%, but three of them compound to -14.3%. The
    cumulative cap is what actually stops an iteration from walking a parameter
    somewhere nobody agreed to, so it is enforced separately from the per-step
    budget and every clamped parameter is named.
    """
    base = normalize_policy(original)
    result = normalize_policy(proposed)
    clamped: list[str] = []
    if max_ratio <= 0:
        return base, list(PARAM_NAMES)
    for name in PARAM_NAMES:
        budget = abs(base[name]) * max_ratio
        low, high = base[name] - budget, base[name] + budget
        if result[name] < low - 1e-9:
            result[name] = clamp_param(name, low)
            clamped.append(name)
        elif result[name] > high + 1e-9:
            result[name] = clamp_param(name, high)
            clamped.append(name)
    return result, clamped


def drift_from(original: Mapping[str, float],
               policy: Mapping[str, float]) -> list[dict[str, Any]]:
    """Per-parameter cumulative drift, for reporting and for the ±15% stop."""
    base, current = normalize_policy(original), normalize_policy(policy)
    rows: list[dict[str, Any]] = []
    for name in PARAM_NAMES:
        before, after = base[name], current[name]
        if abs(after - before) < 1e-9:
            continue
        rows.append({"param": name, "label": PARAM_SPECS[name]["label"],
                     "baseline": round(before, 4), "value": round(after, 4),
                     "delta": round(after - before, 4),
                     "delta_ratio": round((after - before) / before, 4) if before else None})
    return rows


def target_error_for(changes: Mapping[str, float],
                     baseline: Mapping[str, float]) -> str | None:
    """Name the error kind the dominant change in `changes` is aimed at."""
    base = normalize_policy(baseline)
    dominant: tuple[float, str, int] | None = None
    for name, value in changes.items():
        if name not in PARAM_SPECS:
            continue
        delta = clamp_param(name, value) - base[name]
        if abs(delta) < 1e-9:
            continue
        magnitude = abs(delta) / abs(base[name]) if base[name] else abs(delta)
        direction = 1 if delta > 0 else -1
        if dominant is None or magnitude > dominant[0]:
            dominant = (magnitude, name, direction)
    if dominant is None:
        return None
    return TARGET_ERROR_BY_DIRECTION.get((dominant[1], dominant[2]))


def policy_summary(policy: Mapping[str, float]) -> list[dict[str, Any]]:
    normalized = normalize_policy(policy)
    return [{"param": name, "label": PARAM_SPECS[name]["label"], "value": normalized[name],
             "default": BASE_POLICY[name], "min": PARAM_SPECS[name]["min"],
             "max": PARAM_SPECS[name]["max"]}
            for name in PARAM_NAMES]


def sweep_values(name: str, *, steps: int = 21) -> list[float]:
    spec = PARAM_SPECS[name]
    low, high = float(spec["min"]), float(spec["max"])
    if steps <= 1:
        return [low]
    return [round(low + (high - low) * index / (steps - 1), 4) for index in range(steps)]


def sequences_equal(left: Sequence[float], right: Sequence[float], tol: float = 1e-6) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tol for a, b in zip(left, right))


__all__ = [
    "BASE_POLICY", "ERROR_FALSE_BOT", "ERROR_FRAGMENTATION", "ERROR_MISSED_BOT",
    "ERROR_MISSED_REPLY", "ERROR_PREMATURE_REPLY", "ERROR_WRONG_MERGE", "PARAM_NAMES",
    "PARAM_SPECS", "POLICY_SCHEMA_VERSION", "PolicyCandidate", "ReplayDecision", "SOURCE_MANUAL",
    "SOURCE_RECIPIENT_SWEEP", "SOURCE_TOPIC_SWEEP", "STATUS_ACCEPTED", "STATUS_CANDIDATE",
    "STATUS_REJECTED", "STATUS_ROLLED_BACK", "STATUSES", "TARGETED_EXPLICIT_CODES",
    "TARGET_ERROR_BY_DIRECTION", "ambient_score", "bounded_target", "candidate_from",
    "clamp_cumulative", "clamp_param", "decide", "drift_from", "next_version",
    "normalize_policy", "policy_deltas", "policy_summary", "sequences_equal", "sweep_values",
    "target_error_for",
]
