"""Paired bootstrap confidence intervals for a metric *delta*.

A single "+0.8%" is not a result, it is a point estimate with its uncertainty
deleted. The plan asks for the interval, and the reason is not statistical
decoration: with 40 holdout samples, half of all "+0.8%" readings are noise, and
a rule that cannot tell them apart will promote a coin flip.

The resampling unit is the **session**, never the sample. Two messages from one
conversation are not independent draws — they share a topic, a recipient history
and a mood — so resampling samples would produce an interval that is too narrow
by exactly the amount the corpus is clustered. The pairs are *paired*: each
resample scores the baseline and the candidate on the same drawn sessions, which
is what makes the delta's interval meaningful rather than the difference of two
independent intervals.

Everything here is deterministic given a seed. A validation result that changed
between two runs of the same data would make every recorded policy
irreproducible, which is the one property the whole store depends on.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics import METRIC_SCHEMA_VERSION, f1_score, pair_accuracy

DEFAULT_ITERATIONS = 600
DEFAULT_SEED = 7
DEFAULT_ALPHA = 0.05

Metric = Callable[[Mapping[str, float]], "float | None"]


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


# The metrics a bootstrap can be run on, expressed over **pooled counts** rather
# than over samples. Pooling is what makes resampling possible: a metric that
# needed the raw rows could not be recomputed from a multiset of sessions.
def _binary_accuracy(counts: Mapping[str, float]) -> float | None:
    total = sum(counts.get(key, 0.0) for key in ("tp", "fp", "tn", "fn"))
    return _ratio(counts.get("tp", 0.0) + counts.get("tn", 0.0), total)


def _binary_f1(counts: Mapping[str, float]) -> float | None:
    return f1_score(counts.get("tp", 0.0), counts.get("fp", 0.0), counts.get("fn", 0.0))


def _pair_accuracy(counts: Mapping[str, float]) -> float | None:
    return pair_accuracy(counts)


METRICS: dict[str, Metric] = {
    "accuracy": _binary_accuracy,
    "f1": _binary_f1,
    "pair_accuracy": _pair_accuracy,
}


def _metric(name: str) -> Metric | None:
    return METRICS.get(name)


@dataclass(frozen=True)
class Unit:
    """One resampling unit: its counts under the baseline and the candidate.

    `key` is the session hash. It is carried for diagnostics only — the
    bootstrap itself never looks at it, because a resample that treated two
    draws of the same session as different sessions would be the bug this class
    exists to prevent.
    """

    key: str
    baseline: Mapping[str, float] = field(default_factory=dict)
    candidate: Mapping[str, float] = field(default_factory=dict)


def _add(target: dict[str, float], counts: Mapping[str, float]) -> None:
    for key, value in counts.items():
        target[key] = target.get(key, 0.0) + float(value)


def _pool(units: Iterable[Unit], attribute: str) -> dict[str, float]:
    total: dict[str, float] = {}
    for unit in units:
        _add(total, getattr(unit, attribute))
    return total


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Nearest-rank quantile: no interpolation, so no invented value.

    With 600 resamples and alpha=0.05 this is the 15th and 585th order
    statistics. Interpolating would produce a number that no resample ever
    produced, which is exactly the kind of precision this module exists to
    refuse.
    """
    if not ordered:
        return 0.0
    index = int(math.floor(fraction * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


def bootstrap_delta(
    units: Sequence[Unit],
    *,
    metric: str,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Paired session-level bootstrap of a metric delta.

    Returns `None`s rather than zeroes when the metric is undefined on the
    corpus: "no eligible pair" and "a delta of exactly nothing" are different
    findings, and the gate that reads this must not treat them the same.
    """
    function = _metric(metric)
    result: dict[str, Any] = {
        "metric": metric,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "units": len(units),
        "iterations": 0,
        "requested_iterations": max(1, int(iterations)),
        "seed": int(seed),
        "alpha": float(alpha),
        "delta": None,
        "lower": None,
        "upper": None,
        "crosses_zero": None,
        "confidence": None,
    }
    if function is None or not units:
        result["reason"] = "没有可用的重采样单元" if not units else f"未知指标 {metric}"
        return result

    baseline = function(_pool(units, "baseline"))
    candidate = function(_pool(units, "candidate"))
    if baseline is None or candidate is None:
        result["reason"] = "指标在留出集上未定义，无法给出区间"
        return result
    result["delta"] = round(candidate - baseline, 6)
    result["baseline"] = round(baseline, 6)
    result["candidate"] = round(candidate, 6)

    generator = random.Random(int(seed))
    size = len(units)
    deltas: list[float] = []
    for _ in range(result["requested_iterations"]):
        base_counts: dict[str, float] = {}
        candidate_counts: dict[str, float] = {}
        for _ in range(size):
            unit = units[generator.randrange(size)]
            _add(base_counts, unit.baseline)
            _add(candidate_counts, unit.candidate)
        base_value = function(base_counts)
        candidate_value = function(candidate_counts)
        if base_value is None or candidate_value is None:
            # A resample whose metric is undefined carries no information about
            # the delta, so it is dropped and *counted* rather than read as 0.
            continue
        deltas.append(candidate_value - base_value)
    result["iterations"] = len(deltas)
    result["dropped"] = result["requested_iterations"] - len(deltas)
    if not deltas:
        result["reason"] = "所有重采样都无法计算该指标"
        return result
    deltas.sort()
    lower = _quantile(deltas, float(alpha) / 2)
    upper = _quantile(deltas, 1 - float(alpha) / 2)
    result["lower"] = round(lower, 6)
    result["upper"] = round(upper, 6)
    result["crosses_zero"] = bool(lower <= 0.0 <= upper)
    result["confidence"] = round(1.0 - float(alpha), 4)
    return result


def advantage_ratio(deltas: Sequence[float]) -> float | None:
    """Fraction of resamples in which the candidate beat the baseline.

    Reported beside the interval because it answers a question the interval
    cannot: "how often" versus "how much". A change that is positive in 96% of
    resamples but tiny is a different bet from one that is +2% in 55% of them.
    """
    if not deltas:
        return None
    return round(sum(1 for value in deltas if value > 0) / len(deltas), 4)


def human_interval(interval: Mapping[str, Any], *, digits: int = 4) -> str:
    """`+0.8% 95% CI [-0.2%, +1.9%]` — the sentence the plan asks for."""
    if not isinstance(interval, Mapping) or interval.get("delta") is None:
        return "区间不可计算"
    confidence = int(round(float(interval.get("confidence") or 0.95) * 100))
    delta = float(interval["delta"])
    lower, upper = interval.get("lower"), interval.get("upper")
    if lower is None or upper is None:
        return f"{delta:+.{digits}f}（区间不可用）"
    return (f"{delta:+.{digits}f} {confidence}% CI "
            f"[{float(lower):+.{digits}f}, {float(upper):+.{digits}f}]")


__all__ = [
    "DEFAULT_ALPHA", "DEFAULT_ITERATIONS", "DEFAULT_SEED", "METRICS", "Metric", "Unit",
    "advantage_ratio", "bootstrap_delta", "human_interval",
]
