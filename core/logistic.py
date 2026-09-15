"""Dependency-free, deterministic logistic regression and threshold sweeps.

The plan calls for "规则统计 + logistic regression + Bayesian prior + 阈值优化"
before any neural work, so this module deliberately has no third-party
dependency and no randomness: full-batch gradient descent from a zero
initialisation over a fixed number of iterations always produces the same
weights for the same input. That is what makes a stored policy replayable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .metrics import binary_counts, binary_report, rounded

MODEL_SCHEMA_VERSION = 1
_EPSILON = 1e-9


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    factor = math.exp(max(value, -60.0))
    return factor / (1.0 + factor)


@dataclass(frozen=True)
class LogisticModel:
    weights: tuple[float, ...]
    bias: float
    feature_names: tuple[str, ...]
    schema_version: int = MODEL_SCHEMA_VERSION
    converged: bool = True
    iterations: int = 0
    training_size: int = 0
    training_positive: int = 0
    loss: float | None = None

    def score(self, vector: Sequence[float]) -> float:
        total = self.bias
        for weight, value in zip(self.weights, vector):
            total += weight * value
        return _sigmoid(total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "weights": [round(value, 6) for value in self.weights],
            "bias": round(self.bias, 6),
            "converged": self.converged,
            "iterations": self.iterations,
            "training_size": self.training_size,
            "training_positive": self.training_positive,
            "loss": rounded(self.loss),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "LogisticModel | None":
        if not isinstance(raw, Mapping):
            return None
        names = raw.get("feature_names")
        weights = raw.get("weights")
        if not isinstance(names, list) or not isinstance(weights, list) or len(names) != len(weights):
            return None
        if not all(isinstance(name, str) for name in names):
            return None
        clean_weights: list[float] = []
        for value in weights:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return None
            clean_weights.append(float(value))
        bias = raw.get("bias")
        return cls(
            weights=tuple(clean_weights),
            bias=float(bias) if isinstance(bias, (int, float)) and not isinstance(bias, bool) else 0.0,
            feature_names=tuple(names),
            schema_version=int(raw.get("model_schema_version") or MODEL_SCHEMA_VERSION),
            converged=bool(raw.get("converged", True)),
            iterations=int(raw.get("iterations") or 0),
            training_size=int(raw.get("training_size") or 0),
            training_positive=int(raw.get("training_positive") or 0),
            loss=float(raw["loss"]) if isinstance(raw.get("loss"), (int, float)) else None,
        )

    def aligned(self, names: Sequence[str]) -> "LogisticModel":
        """Re-order onto `names`, zero-filling anything the model never saw."""
        lookup: dict[str, float] = {}
        for name, weight in zip(self.feature_names, self.weights):
            lookup[name] = lookup.get(name, 0.0) + float(weight)
        return LogisticModel(
            weights=tuple(float(lookup.get(name, 0.0)) for name in names),
            bias=self.bias, feature_names=tuple(names), schema_version=self.schema_version,
            converged=self.converged, iterations=self.iterations,
            training_size=self.training_size, training_positive=self.training_positive,
            loss=self.loss,
        )


def _log_loss(weights: Sequence[float], bias: float, rows: Sequence[Sequence[float]],
              labels: Sequence[float]) -> float:
    total = 0.0
    for row, label in zip(rows, labels):
        score = bias + sum(weight * value for weight, value in zip(weights, row))
        probability = min(1.0 - _EPSILON, max(_EPSILON, _sigmoid(score)))
        total += -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
    return total / len(rows) if rows else 0.0


def fit(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[bool],
    *,
    feature_names: Sequence[str] = (),
    l2_strength: float = 0.02,
    learning_rate: float = 0.35,
    iterations: int = 600,
) -> LogisticModel:
    """Full-batch L2 logistic regression with a deterministic stopping rule."""
    names = tuple(feature_names) or tuple(f"f{index}" for index in range(len(vectors[0]) if vectors else 0))
    width = len(names)
    rows = [[float(value) for value in row[:width]] + [0.0] * max(0, width - len(row)) for row in vectors]
    targets = [1.0 if label else 0.0 for label in labels]
    positives = int(sum(targets))
    if not rows or width == 0:
        return LogisticModel(tuple([0.0] * width), 0.0, names,
                             converged=False, training_size=len(rows), training_positive=positives)
    if positives == 0 or positives == len(targets):
        # A single class carries no gradient signal; fall back to the base rate
        # so the model is still a usable, honest constant predictor.
        base = min(1.0 - _EPSILON, max(_EPSILON, positives / len(targets)))
        bias = math.log(base / (1.0 - base))
        return LogisticModel(tuple([0.0] * width), bias, names,
                             converged=True, iterations=0, training_size=len(rows),
                             training_positive=positives, loss=_log_loss([0.0] * width, bias, rows, targets))

    weights = [0.0] * width
    bias = 0.0
    rate = max(1e-4, float(learning_rate))
    penalty = max(0.0, float(l2_strength))
    size = float(len(rows))
    previous = _log_loss(weights, bias, rows, targets)
    converged = False
    performed = 0
    for step in range(max(1, int(iterations))):
        performed = step + 1
        grad_w = [0.0] * width
        grad_b = 0.0
        for row, target in zip(rows, targets):
            score = bias + sum(weight * value for weight, value in zip(weights, row))
            error = _sigmoid(score) - target
            for index, value in enumerate(row):
                if value:
                    grad_w[index] += error * value
            grad_b += error
        for index in range(width):
            weights[index] -= rate * (grad_w[index] / size + penalty * weights[index])
        bias -= rate * (grad_b / size)
        loss = _log_loss(weights, bias, rows, targets)
        if abs(previous - loss) <= 1e-9:
            previous = loss
            converged = True
            break
        previous = loss
    return LogisticModel(tuple(round(value, 8) for value in weights), round(bias, 8), names,
                         converged=converged, iterations=performed, training_size=len(rows),
                         training_positive=positives, loss=previous)


def sweep_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    metric: str = "f1",
    min_support: int = 1,
    steps: int = 41,
) -> dict[str, Any]:
    """Choose the decision cut that maximises `metric` on the given pairs.

    Ties are broken toward the threshold closest to 0.5, then toward the lower
    value, so the sweep is deterministic. A cut that admits nothing scores 0.0
    rather than `None` so it cannot accidentally win.
    """
    pairs = [(float(score), bool(label)) for score, label in zip(scores, labels)]
    if not pairs:
        return {"threshold": None, "metric": metric, "value": None, "support": 0,
                "evaluated": 0, "curve": []}
    candidates = {0.0, 1.0}
    for score, _ in pairs:
        candidates.add(round(min(1.0, max(0.0, score)), 4))
        candidates.add(round(min(1.0, max(0.0, score)) + 0.0005, 4))
    ordered = sorted(candidates)
    if steps > 1 and len(ordered) > steps:
        stride = (len(ordered) - 1) / (steps - 1)
        ordered = sorted({ordered[min(len(ordered) - 1, int(round(index * stride)))]
                          for index in range(steps)} | {ordered[0], ordered[-1]})

    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for threshold in ordered:
        report = binary_report(binary_counts((score >= threshold, label) for score, label in pairs))
        if report["support"] < min_support:
            continue
        value = report.get(metric)
        value = 0.0 if value is None else float(value)
        row = {"threshold": threshold, "value": round(value, 4), metric: round(value, 4),
               "precision": report["precision"], "recall": report["recall"],
               "accuracy": report["accuracy"], "positive_rate": report["positive_rate"]}
        curve.append(row)
        key = (round(value, 6), -abs(threshold - 0.5), -threshold)
        if best is None or key > best["_key"]:
            best = {**row, "_key": key}
    if best is None:
        return {"threshold": None, "metric": metric, "value": None, "support": len(pairs),
                "evaluated": 0, "curve": []}
    best.pop("_key", None)
    baseline = binary_report(binary_counts((score >= 0.5, label) for score, label in pairs))
    return {
        "threshold": best["threshold"], "metric": metric, "value": best["value"],
        "support": len(pairs), "evaluated": len(curve), "curve": curve[:64],
        "at_half": {"f1": baseline["f1"], "accuracy": baseline["accuracy"],
                    "precision": baseline["precision"], "recall": baseline["recall"]},
    }


__all__ = ["MODEL_SCHEMA_VERSION", "LogisticModel", "fit", "sweep_threshold"]
