"""The vocabulary the console and the evaluator share.

A `Recommendation` is either an actionable change to a real ChatDynamics
configuration key (`kind="config_param"`) or an engineering diagnostic that
cannot currently be expressed as configuration (`kind="diagnostic"`).

Keeping the two apart is the whole point: reporting "quote_weight should rise"
as if it were a setting would be a lie, because the host does not expose
per-evidence weights. Those observations stay diagnostics until the host does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

KIND_CONFIG_PARAM = "config_param"
KIND_DIAGNOSTIC = "diagnostic"

CONFIDENCE_INSUFFICIENT = "insufficient"
CONFIDENCE_LOW = "low"
CONFIDENCE_MODERATE = "moderate"
CONFIDENCE_ORDER = (CONFIDENCE_INSUFFICIENT, CONFIDENCE_LOW, CONFIDENCE_MODERATE)

# Below this many labelled samples a recommendation is reported but never
# treated as actionable.
CONFIDENCE_LOW_SAMPLES = 40
CONFIDENCE_MODERATE_SAMPLES = 150


def confidence_for(samples: int, *, min_samples: int) -> str:
    if samples < min_samples:
        return CONFIDENCE_INSUFFICIENT
    if samples < max(min_samples, CONFIDENCE_MODERATE_SAMPLES):
        return CONFIDENCE_LOW if samples >= CONFIDENCE_LOW_SAMPLES else CONFIDENCE_INSUFFICIENT
    return CONFIDENCE_MODERATE


@dataclass(frozen=True)
class Recommendation:
    kind: str
    title: str
    detail: str
    rationale: str = ""
    param: str | None = None
    label: str = ""
    before: float | None = None
    after: float | None = None
    confidence: str = CONFIDENCE_INSUFFICIENT
    samples: int = 0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def actionable(self) -> bool:
        return (self.kind == KIND_CONFIG_PARAM and self.param is not None
                and self.before is not None and self.after is not None
                and self.confidence != CONFIDENCE_INSUFFICIENT)

    @property
    def delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return round(self.after - self.before, 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "title": self.title, "detail": self.detail,
            "rationale": self.rationale, "param": self.param, "label": self.label,
            "before": self.before, "after": self.after, "delta": self.delta,
            "confidence": self.confidence, "samples": self.samples,
            "actionable": self.actionable, "evidence": dict(self.evidence),
        }


def config_recommendation(
    *,
    param: str,
    label: str,
    before: float,
    after: float,
    title: str,
    rationale: str,
    samples: int,
    min_samples: int,
    evidence: Mapping[str, Any] | None = None,
) -> Recommendation:
    detail = f"{before:.4f} -> {after:.4f}"
    return Recommendation(
        kind=KIND_CONFIG_PARAM, title=title, detail=detail, rationale=rationale,
        param=param, label=label, before=round(float(before), 4), after=round(float(after), 4),
        confidence=confidence_for(samples, min_samples=min_samples), samples=samples,
        evidence=dict(evidence or {}),
    )


def diagnostic(
    *,
    title: str,
    detail: str,
    rationale: str = "",
    samples: int = 0,
    confidence: str = CONFIDENCE_INSUFFICIENT,
    evidence: Mapping[str, Any] | None = None,
) -> Recommendation:
    return Recommendation(kind=KIND_DIAGNOSTIC, title=title, detail=detail,
                          rationale=rationale, samples=samples, confidence=confidence,
                          evidence=dict(evidence or {}))


__all__ = [
    "CONFIDENCE_INSUFFICIENT", "CONFIDENCE_LOW", "CONFIDENCE_LOW_SAMPLES",
    "CONFIDENCE_MODERATE", "CONFIDENCE_MODERATE_SAMPLES", "CONFIDENCE_ORDER",
    "KIND_CONFIG_PARAM", "KIND_DIAGNOSTIC", "Recommendation", "confidence_for",
    "config_recommendation", "diagnostic",
]
