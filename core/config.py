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

    # Deterministic logistic fitting.
    learning_rate: float = 0.35
    l2_strength: float = 0.02
    max_iterations: int = 600

    # Automatic analysis is opt-in; default is manual from the console.
    auto_analyze: bool = False
    auto_analyze_interval_minutes: int = 360

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
            "learning_rate": self.learning_rate,
            "l2_strength": self.l2_strength,
            "max_iterations": self.max_iterations,
            "auto_analyze": self.auto_analyze,
            "auto_analyze_interval_minutes": self.auto_analyze_interval_minutes,
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
        learning_rate=_as_float(pick("learning_rate"), defaults.learning_rate, 0.001, 2.0),
        l2_strength=_as_float(pick("learning_l2"), defaults.l2_strength, 0.0, 1.0),
        max_iterations=_as_int(pick("learning_iterations"), defaults.max_iterations, 20, HARD_MAX_ITERATIONS),
        auto_analyze=_as_bool(pick("learning_auto_analyze"), defaults.auto_analyze),
        auto_analyze_interval_minutes=_as_int(
            pick("learning_auto_analyze_minutes"), defaults.auto_analyze_interval_minutes, 15, 10_080),
    )


_KNOWN_KEYS = (
    "learning_enabled", "source_plugin_id", "learning_max_samples", "learning_store_raw_trace",
    "learning_keep_message_text", "learning_min_samples", "learning_max_param_delta_ratio",
    "learning_min_evaluation_samples", "learning_min_improvement", "learning_max_regression",
    "learning_holdout_ratio", "learning_rate", "learning_l2", "learning_iterations",
    "learning_auto_analyze", "learning_auto_analyze_minutes",
)
