"""Bounded runtime configuration with explicit, documented defaults.

Every learned change is expressed as a delta against a real ChatDynamics
configuration key, and every delta is capped by `max_param_delta_ratio`.
Nothing in this module reads or writes the host plugin's configuration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

# The source plugin registers its shared preferences under `{author}/{name}`,
# matching astrbot.core.star.star.StarMetadata.plugin_id.
DEFAULT_SOURCE_PLUGIN_ID = "ysyhlly/astrbot_plugin_chat_dynamics"

# Hard ceilings. Configuration may lower these but never raise them.
HARD_MAX_SAMPLES = 200_000
HARD_MAX_DELTA_RATIO = 0.20
HARD_MAX_ITERATIONS = 5_000
MIN_RECOMMENDATION_SAMPLES = 20
MIN_EVALUATION_SAMPLES = 20


def _as_bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _as_int(value: Any, default: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    return max(low, min(high, int(value)))


def _as_float(value: Any, default: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    return max(low, min(high, number))


def _as_text(value: Any, default: str, limit: int = 256) -> str:
    if not isinstance(value, str):
        return default
    cleaned = value.strip()
    return cleaned[:limit] if cleaned else default


@dataclass(frozen=True)
class LearningConfig:
    """Resolved configuration. Immutable so a learner run sees one consistent view."""

    enabled: bool = True
    source_plugin_id: str = DEFAULT_SOURCE_PLUGIN_ID

    # Sampling and storage.
    max_samples: int = 5_000
    store_raw_trace: bool = True
    keep_message_text: bool = False

    # Recommendation gate.
    min_samples_for_recommendation: int = 100
    max_param_delta_ratio: float = 0.05

    # Evaluation gate.
    min_samples_for_evaluation: int = 40
    evaluation_min_improvement: float = 0.02
    evaluation_max_regression: float = 0.01
    holdout_ratio: float = 0.30

    # Time-ordered (forward) validation. A session holdout answers "does this
    # work on conversations it has not seen"; this one answers "does it still
    # work later", and a policy has to pass both.
    forward_holdout_ratio: float = 0.25
    require_forward_validation: bool = True

    # Confidence interval on the metric delta. A point estimate that a
    # resampling cannot separate from zero is not a small improvement; it is an
    # unmeasured one.
    bootstrap_iterations: int = 600
    bootstrap_seed: int = 7
    bootstrap_alpha: float = 0.05
    require_ci_positive: bool = True

    # Shadow A/B: what a policy has to clear before "active" is offered. These
    # are the plan's numbers, and every one is a knob — a threshold nobody can
    # move is a threshold nobody can justify changing.
    shadow_min_samples: int = 500
    shadow_min_disagreements: int = 100
    shadow_max_regression: float = 0.01
    shadow_target_relative: float = 0.10
    shadow_target_absolute: float = 0.01
    shadow_relative_min_error: float = 0.05
    shadow_ci_floor: float = -0.002
    shadow_min_sessions: int = 3
    shadow_min_active_hours: int = 4
    shadow_subgroup_max_regression: float = 0.05
    shadow_subgroup_min_support: int = 20

    # Group (session) diagnostics: the support a group needs before its delta is
    # printed at all. Below it the group is named and given no number.
    group_min_support: int = 12

    # The dataset gate: what the corpus must look like before any policy is
    # offered at all. Failing it produces diagnostics, not a recommendation.
    gate_min_samples: int = 60
    gate_min_sessions: int = 4
    gate_min_positive_rate: float = 0.05
    gate_max_degraded_ratio: float = 0.5
    gate_max_label_age_days: int = 120

    # Deterministic logistic fitting.
    learning_rate: float = 0.35
    l2_strength: float = 0.02
    max_iterations: int = 600

    # Automatic analysis is opt-in; default is manual from the console.
    auto_analyze: bool = False
    auto_analyze_interval_minutes: int = 360

    # Contract review: the panel reads the matrix through a model. On by default
    # because that is what the panel is for; the deterministic table is still
    # computed, still returned, and is what the page falls back to whenever the
    # model is off, unreachable, or answers with something unreadable.
    review_enabled: bool = True
    review_provider_id: str = ""
    review_timeout_seconds: int = 45

    # Per-message reply post-mortem. Off by default because it is the one path
    # that sends message text off the machine: the contract review above sends
    # counts only. Text is read from the host for a single call and is never
    # written to this plugin's store.
    reply_review_enabled: bool = False
    reply_review_provider_id: str = ""
    reply_review_timeout_seconds: int = 60
    reply_review_max_messages: int = 12

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_plugin_id": self.source_plugin_id,
            "max_samples": self.max_samples,
            "store_raw_trace": self.store_raw_trace,
            "keep_message_text": self.keep_message_text,
            "min_samples_for_recommendation": self.min_samples_for_recommendation,
            "max_param_delta_ratio": self.max_param_delta_ratio,
            "min_samples_for_evaluation": self.min_samples_for_evaluation,
            "evaluation_min_improvement": self.evaluation_min_improvement,
            "evaluation_max_regression": self.evaluation_max_regression,
            "holdout_ratio": self.holdout_ratio,
            "forward_holdout_ratio": self.forward_holdout_ratio,
            "require_forward_validation": self.require_forward_validation,
            "bootstrap_iterations": self.bootstrap_iterations,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_alpha": self.bootstrap_alpha,
            "require_ci_positive": self.require_ci_positive,
            "group_min_support": self.group_min_support,
            "shadow_min_samples": self.shadow_min_samples,
            "shadow_min_disagreements": self.shadow_min_disagreements,
            "shadow_max_regression": self.shadow_max_regression,
            "shadow_target_relative": self.shadow_target_relative,
            "shadow_target_absolute": self.shadow_target_absolute,
            "shadow_relative_min_error": self.shadow_relative_min_error,
            "shadow_ci_floor": self.shadow_ci_floor,
            "shadow_min_sessions": self.shadow_min_sessions,
            "shadow_min_active_hours": self.shadow_min_active_hours,
            "gate_min_samples": self.gate_min_samples,
            "gate_min_sessions": self.gate_min_sessions,
            "gate_min_positive_rate": self.gate_min_positive_rate,
            "gate_max_degraded_ratio": self.gate_max_degraded_ratio,
            "gate_max_label_age_days": self.gate_max_label_age_days,
            "learning_rate": self.learning_rate,
            "l2_strength": self.l2_strength,
            "max_iterations": self.max_iterations,
            "auto_analyze": self.auto_analyze,
            "auto_analyze_interval_minutes": self.auto_analyze_interval_minutes,
            "review_enabled": self.review_enabled,
            "review_provider_id": self.review_provider_id,
            "review_timeout_seconds": self.review_timeout_seconds,
            "reply_review_enabled": self.reply_review_enabled,
            "reply_review_provider_id": self.reply_review_provider_id,
            "reply_review_timeout_seconds": self.reply_review_timeout_seconds,
            "reply_review_max_messages": self.reply_review_max_messages,
        }

    def with_overrides(self, **changes: Any) -> "LearningConfig":
        return replace(self, **changes)


def parse_learning_config(raw: Any) -> LearningConfig:
    """Accept a host config mapping, a plain mapping or nothing at all."""
    if isinstance(raw, LearningConfig):
        return raw
    if raw is None:
        return LearningConfig()
    if not isinstance(raw, Mapping):
        getter = getattr(raw, "get", None)
        if not callable(getter):
            return LearningConfig()
        raw = {key: getter(key) for key in _KNOWN_KEYS}
    defaults = LearningConfig()

    def pick(name: str) -> Any:
        # Host configs wrap values in {"value": ...} in some AstrBot versions.
        value = raw.get(name)
        if isinstance(value, Mapping) and "value" in value:
            return value.get("value")
        return value

    return LearningConfig(
        enabled=_as_bool(pick("learning_enabled"), defaults.enabled),
        source_plugin_id=_as_text(pick("source_plugin_id"), defaults.source_plugin_id),
        max_samples=_as_int(pick("learning_max_samples"), defaults.max_samples, 50, HARD_MAX_SAMPLES),
        store_raw_trace=_as_bool(pick("learning_store_raw_trace"), defaults.store_raw_trace),
        keep_message_text=_as_bool(pick("learning_keep_message_text"), defaults.keep_message_text),
        min_samples_for_recommendation=_as_int(
            pick("learning_min_samples"), defaults.min_samples_for_recommendation,
            MIN_RECOMMENDATION_SAMPLES, HARD_MAX_SAMPLES),
        max_param_delta_ratio=_as_float(
            pick("learning_max_param_delta_ratio"), defaults.max_param_delta_ratio,
            0.0, HARD_MAX_DELTA_RATIO),
        min_samples_for_evaluation=_as_int(
            pick("learning_min_evaluation_samples"), defaults.min_samples_for_evaluation,
            MIN_EVALUATION_SAMPLES, HARD_MAX_SAMPLES),
        evaluation_min_improvement=_as_float(
            pick("learning_min_improvement"), defaults.evaluation_min_improvement, 0.0, 1.0),
        evaluation_max_regression=_as_float(
            pick("learning_max_regression"), defaults.evaluation_max_regression, 0.0, 1.0),
        holdout_ratio=_as_float(pick("learning_holdout_ratio"), defaults.holdout_ratio, 0.10, 0.60),
        forward_holdout_ratio=_as_float(pick("learning_forward_holdout_ratio"),
                                        defaults.forward_holdout_ratio, 0.05, 0.60),
        require_forward_validation=_as_bool(pick("learning_require_forward"),
                                            defaults.require_forward_validation),
        bootstrap_iterations=_as_int(pick("learning_bootstrap_iterations"),
                                     defaults.bootstrap_iterations, 0, 20_000),
        bootstrap_seed=_as_int(pick("learning_bootstrap_seed"), defaults.bootstrap_seed, 0, 1_000_000),
        bootstrap_alpha=_as_float(pick("learning_bootstrap_alpha"), defaults.bootstrap_alpha,
                                  0.001, 0.5),
        require_ci_positive=_as_bool(pick("learning_require_ci"), defaults.require_ci_positive),
        group_min_support=_as_int(pick("learning_group_min_support"), defaults.group_min_support,
                                  3, 10_000),
        shadow_min_samples=_as_int(pick("learning_shadow_min_samples"),
                                   defaults.shadow_min_samples, 0, HARD_MAX_SAMPLES),
        shadow_min_disagreements=_as_int(pick("learning_shadow_min_disagreements"),
                                         defaults.shadow_min_disagreements, 0, HARD_MAX_SAMPLES),
        shadow_max_regression=_as_float(pick("learning_shadow_max_regression"),
                                        defaults.shadow_max_regression, 0.0, 1.0),
        shadow_target_relative=_as_float(pick("learning_shadow_target_relative"),
                                         defaults.shadow_target_relative, 0.0, 1.0),
        shadow_target_absolute=_as_float(pick("learning_shadow_target_absolute"),
                                         defaults.shadow_target_absolute, 0.0, 1.0),
        shadow_relative_min_error=_as_float(pick("learning_shadow_relative_min_error"),
                                            defaults.shadow_relative_min_error, 0.0, 1.0),
        shadow_ci_floor=_as_float(pick("learning_shadow_ci_floor"), defaults.shadow_ci_floor,
                                  -1.0, 1.0),
        shadow_min_sessions=_as_int(pick("learning_shadow_min_sessions"),
                                    defaults.shadow_min_sessions, 1, 1_000),
        shadow_min_active_hours=_as_int(pick("learning_shadow_min_active_hours"),
                                        defaults.shadow_min_active_hours, 1, 24),
        shadow_subgroup_max_regression=_as_float(pick("learning_shadow_subgroup_regression"),
                                                 defaults.shadow_subgroup_max_regression,
                                                 0.0, 1.0),
        shadow_subgroup_min_support=_as_int(pick("learning_shadow_subgroup_support"),
                                            defaults.shadow_subgroup_min_support, 1, 10_000),
        gate_min_samples=_as_int(pick("learning_gate_min_samples"), defaults.gate_min_samples,
                                 MIN_EVALUATION_SAMPLES, HARD_MAX_SAMPLES),
        gate_min_sessions=_as_int(pick("learning_gate_min_sessions"), defaults.gate_min_sessions,
                                  1, 1_000),
        gate_min_positive_rate=_as_float(pick("learning_gate_min_positive_rate"),
                                         defaults.gate_min_positive_rate, 0.0, 0.5),
        gate_max_degraded_ratio=_as_float(pick("learning_gate_max_degraded_ratio"),
                                          defaults.gate_max_degraded_ratio, 0.0, 1.0),
        gate_max_label_age_days=_as_int(pick("learning_gate_max_label_age_days"),
                                        defaults.gate_max_label_age_days, 0, 3_650),
        learning_rate=_as_float(pick("learning_rate"), defaults.learning_rate, 0.001, 2.0),
        l2_strength=_as_float(pick("learning_l2"), defaults.l2_strength, 0.0, 1.0),
        max_iterations=_as_int(pick("learning_iterations"), defaults.max_iterations, 20, HARD_MAX_ITERATIONS),
        auto_analyze=_as_bool(pick("learning_auto_analyze"), defaults.auto_analyze),
        auto_analyze_interval_minutes=_as_int(
            pick("learning_auto_analyze_minutes"), defaults.auto_analyze_interval_minutes, 15, 10_080),
        review_enabled=_as_bool(pick("learning_review_enabled"), defaults.review_enabled),
        review_provider_id=_as_text(pick("learning_review_provider"), defaults.review_provider_id, 128),
        review_timeout_seconds=_as_int(
            pick("learning_review_timeout"), defaults.review_timeout_seconds, 5, 300),
        reply_review_enabled=_as_bool(pick("learning_reply_review_enabled"),
                                      defaults.reply_review_enabled),
        reply_review_provider_id=_as_text(pick("learning_reply_review_provider"),
                                          defaults.reply_review_provider_id, 128),
        reply_review_timeout_seconds=_as_int(
            pick("learning_reply_review_timeout"), defaults.reply_review_timeout_seconds, 5, 300),
        reply_review_max_messages=_as_int(
            pick("learning_reply_review_messages"), defaults.reply_review_max_messages, 1, 40),
    )


_KNOWN_KEYS = (
    "learning_enabled", "source_plugin_id", "learning_max_samples", "learning_store_raw_trace",
    "learning_keep_message_text", "learning_min_samples", "learning_max_param_delta_ratio",
    "learning_min_evaluation_samples", "learning_min_improvement", "learning_max_regression",
    "learning_holdout_ratio", "learning_forward_holdout_ratio", "learning_require_forward",
    "learning_bootstrap_iterations", "learning_bootstrap_seed", "learning_bootstrap_alpha",
    "learning_require_ci", "learning_group_min_support",
    "learning_shadow_min_samples", "learning_shadow_min_disagreements",
    "learning_shadow_max_regression", "learning_shadow_target_relative",
    "learning_shadow_target_absolute", "learning_shadow_relative_min_error",
    "learning_shadow_ci_floor",
    "learning_shadow_min_sessions", "learning_shadow_min_active_hours",
    "learning_shadow_subgroup_regression", "learning_shadow_subgroup_support",
    "learning_gate_min_samples", "learning_gate_min_sessions",
    "learning_gate_min_positive_rate", "learning_gate_max_degraded_ratio",
    "learning_gate_max_label_age_days",
    "learning_rate", "learning_l2", "learning_iterations",
    "learning_auto_analyze", "learning_auto_analyze_minutes",
    "learning_review_enabled", "learning_review_provider", "learning_review_timeout",
    "learning_reply_review_enabled", "learning_reply_review_provider",
    "learning_reply_review_timeout", "learning_reply_review_messages",
)
