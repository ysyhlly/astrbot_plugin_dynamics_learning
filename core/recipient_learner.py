"""v0.2 Recipient Learning: "who is this actually addressed to?"

The learner answers three questions over the labelled recipient samples:

1. How often is the recorded addressee right? (statistics, never an estimate)
2. Which ambient evidence codes actually predict "the bot is the addressee"?
   (Bayesian-smoothed lift, plus an in-sample fitted model)
3. Would a different cut on the recorded score have decided better?
   (in-sample threshold sweep -> a *proposed* candidate)

Nothing produced here is treated as validated. Only
`core.evaluator` may mark a candidate as accepted, and it does so on a held-out
set of sessions that the sweep never saw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import LearningConfig
from .features import FEATURE_NAMES, vector
from .logistic import LogisticModel, fit, sweep_threshold
from .metrics import (
    SAMPLE_NOTE, binary_counts, binary_report, error_distribution, ratio, rounded,
    sample_accuracy,
)
from .policy import (
    BASE_POLICY, PARAM_SPECS, bounded_target,
)
from .recommendation import Recommendation, config_recommendation, diagnostic
from .samples import BOT, LearningSample, TASK_RECIPIENT
from .trace import AMBIENT_CODES

SMOOTHING_ALPHA = 1.0
SMOOTHING_BETA = 1.0
MIN_CODE_SUPPORT = 5


@dataclass(frozen=True)
class EvidenceLift:
    code: str
    present: int
    positive: int
    smoothed_rate: float
    lift: float

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "present": self.present, "positive": self.positive,
                "smoothed_rate": round(self.smoothed_rate, 4), "lift": round(self.lift, 4)}


@dataclass
class RecipientLearning:
    samples: int = 0
    sessions: int = 0
    ambient_samples: int = 0
    explicit_samples: int = 0
    early_return_samples: int = 0
    degraded_samples: int = 0
    accuracy: dict[str, Any] = field(default_factory=dict)
    confusion: dict[str, Any] = field(default_factory=dict)
    error_types: dict[str, int] = field(default_factory=dict)
    positive_rate: float | None = None
    evidence_lift: list[EvidenceLift] = field(default_factory=list)
    model: LogisticModel | None = None
    threshold_sweep: dict[str, Any] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": TASK_RECIPIENT,
            "samples": self.samples, "sessions": self.sessions,
            "ambient_samples": self.ambient_samples, "explicit_samples": self.explicit_samples,
            "early_return_samples": self.early_return_samples,
            "degraded_samples": self.degraded_samples,
            "accuracy": self.accuracy, "confusion": self.confusion,
            "error_types": self.error_types, "positive_rate": self.positive_rate,
            "evidence_lift": [row.as_dict() for row in self.evidence_lift],
            "model": self.model.as_dict() if self.model else None,
            "threshold_sweep": {key: value for key, value in self.threshold_sweep.items()
                                if key != "curve"},
            "recommendations": [row.as_dict() for row in self.recommendations],
            "notes": list(self.notes),
        }


def _is_explicit(sample: LearningSample) -> bool:
    return sample.features.get("ctx_explicit", 0.0) > 0.0


def _is_ambient(sample: LearningSample) -> bool:
    """A scored ambient turn: neither structural nor the host's early return."""
    return not _is_explicit(sample) and sample.features.get("ctx_prior_bot", 1.0) >= 1.0


def evidence_lift(samples: Sequence[LearningSample], base_rate: float) -> list[EvidenceLift]:
    rows: list[EvidenceLift] = []
    for code in sorted(AMBIENT_CODES):
        key = f"ev_{code}"
        subset = [sample for sample in samples if sample.features.get(key, 0.0) > 0.0]
        present = len(subset)
        if present < MIN_CODE_SUPPORT:
            continue
        positive = sum(1 for sample in subset if sample.expected == BOT)
        smoothed = (positive + SMOOTHING_ALPHA) / (present + SMOOTHING_ALPHA + SMOOTHING_BETA)
        rows.append(EvidenceLift(code=code, present=present, positive=positive,
                                 smoothed_rate=smoothed, lift=smoothed - base_rate))
    rows.sort(key=lambda row: (-abs(row.lift), row.code))
    return rows


def learn(samples: Sequence[LearningSample], *, config: LearningConfig | None = None) -> RecipientLearning:
    config = config or LearningConfig()
    rows = [sample for sample in samples if sample.task == TASK_RECIPIENT]
    report = RecipientLearning(
        samples=len(rows),
        sessions=len({sample.session_hash for sample in rows}),
        degraded_samples=sum(1 for sample in rows if sample.features.get("ctx_explicit", 0.0) and
                            sample.trace.get("evidence_summary", {}).get("degraded")),
    )
    if not rows:
        report.notes.append("没有收件人标注样本；先在本体回放页标注 bot_targeted。")
        return report

    report.accuracy = sample_accuracy(rows)
    report.confusion = binary_report(binary_counts(
        (sample.predicted == BOT, sample.expected == BOT) for sample in rows))
    report.error_types = error_distribution(rows)
    report.positive_rate = rounded(ratio(sum(1 for sample in rows if sample.expected == BOT), len(rows)))
    report.notes.append(SAMPLE_NOTE)

    explicit = [sample for sample in rows if _is_explicit(sample)]
    report.explicit_samples = len(explicit)
    ambient = [sample for sample in rows if _is_ambient(sample)]
    report.ambient_samples = len(ambient)
    report.early_return_samples = len(rows) - len(explicit) - len(ambient)
    if report.explicit_samples:
        report.notes.append(
            f"{report.explicit_samples} 条为结构化（显式 @/引用/唤醒）判定，阈值无关，"
            "不参与环境层拟合。")
    if report.early_return_samples:
        report.notes.append(
            f"{report.early_return_samples} 条为无前序 Bot 消息的提前返回，同样与阈值无关。")
    if len(ambient) < config.min_samples_for_recommendation:
        report.notes.append(
            f"环境层样本 {len(ambient)} 条，低于建议阈值 {config.min_samples_for_recommendation} 条，"
            "本轮只出统计与诊断，不出参数建议。")
        report.evidence_lift = evidence_lift(ambient, _base_rate(ambient))
        return report

    base_rate = _base_rate(ambient)
    report.evidence_lift = evidence_lift(ambient, base_rate)

    labels = [sample.expected == BOT for sample in ambient]
    vectors = [vector(sample.features) for sample in ambient]
    report.model = fit(vectors, labels, feature_names=FEATURE_NAMES,
                       l2_strength=config.l2_strength, learning_rate=config.learning_rate,
                       iterations=config.max_iterations)

    recorded = [float(sample.features.get("base_score", 0.0)) for sample in ambient]
    report.threshold_sweep = sweep_threshold(recorded, labels, metric="f1")

    report.recommendations.extend(_threshold_recommendations(
        report=report, ambient=ambient, config=config, base_rate=base_rate))
    report.recommendations.extend(_diagnostics(report, base_rate))
    return report


def _base_rate(samples: Sequence[LearningSample]) -> float:
    if not samples:
        return 0.0
    return sum(1 for sample in samples if sample.expected == BOT) / len(samples)


def _threshold_recommendations(
    *,
    report: RecipientLearning,
    ambient: Sequence[LearningSample],
    config: LearningConfig,
    base_rate: float,
) -> list[Recommendation]:
    sweep = report.threshold_sweep
    threshold = sweep.get("threshold")
    if threshold is None:
        return []
    base = BASE_POLICY["strong_addressivity_threshold"]
    target = bounded_target("strong_addressivity_threshold", base, float(threshold),
                            config.max_param_delta_ratio)
    recorded = binary_report(binary_counts(
        (sample.predicted == BOT, sample.expected == BOT) for sample in ambient))
    current = base
    verdict = binary_report(binary_counts(
        (float(sample.features.get("base_score", 0.0)) >= current, sample.expected == BOT)
        for sample in ambient))
    rationale = (
        f"在 {len(ambient)} 条环境层标注上扫描强指代阈值："
        f"最优 {float(threshold):.3f}（F1 {float(sweep.get('value') or 0):.3f}），"
        f"当前 {current:.2f} 对应 F1 {float(verdict.get('f1') or 0):.3f}。"
        f"样本正例率 {base_rate:.3f}。建议值已按 ±{config.max_param_delta_ratio:.0%} 上限截断，"
        "并且尚未通过留出集评测。")
    return [config_recommendation(
        param="strong_addressivity_threshold",
        label=PARAM_SPECS["strong_addressivity_threshold"]["label"],
        before=base, after=target,
        title="调整强指代判定阈值",
        rationale=rationale,
        samples=len(ambient), min_samples=config.min_samples_for_recommendation,
        evidence={"sweep": {key: value for key, value in sweep.items() if key != "curve"},
                  "recorded_accuracy": recorded["accuracy"],
                  "in_sample_f1": verdict["f1"]},
    )]


def _diagnostics(report: RecipientLearning, base_rate: float) -> list[Recommendation]:
    low = [row for row in report.evidence_lift if row.lift <= -0.05]
    high = [row for row in report.evidence_lift if row.lift >= 0.05]
    notes: list[Recommendation] = []
    if high:
        top = "、".join(f"{row.code}(+{row.lift:.2f}, n={row.present})" for row in high[:4])
        notes.append(diagnostic(
            title="有效证据族",
            detail=top,
            rationale="这些证据出现时，人工标注为「在对 Bot 说话」的比例明显高于样本基准。",
            samples=report.ambient_samples,
            evidence={"base_rate": round(base_rate, 4),
                      "codes": [row.as_dict() for row in high[:8]]},
        ))
    if low:
        bottom = "、".join(f"{row.code}({row.lift:+.2f}, n={row.present})" for row in low[:4])
        notes.append(diagnostic(
            title="偏离证据族",
            detail=bottom,
            rationale=(
                "这些证据出现时负例偏多。注意：ChatDynamics 目前没有把逐条证据权重开放成配置项，"
                "所以这属于工程结论，不能转成自动参数建议。"),
            samples=report.ambient_samples,
            evidence={"base_rate": round(base_rate, 4),
                      "codes": [row.as_dict() for row in low[:8]]},
        ))
    if report.model is not None:
        top_weights = sorted(
            ((name, weight) for name, weight in zip(report.model.feature_names, report.model.weights)
             if abs(weight) > 1e-6),
            key=lambda item: (-abs(item[1]), item[0]))[:8]
        notes.append(diagnostic(
            title="拟合权重（样本内）",
            detail="、".join(f"{name} {weight:+.3f}" for name, weight in top_weights) or "全零",
            rationale=(
                "L2 逻辑回归在全部样本上的权重仅用于理解证据方向；它在本批样本上见过标签，"
                "不能当作泛化表现。留出集结论由评测页给出。"),
            samples=report.ambient_samples,
            evidence={"loss": report.model.loss, "iterations": report.model.iterations,
                      "converged": report.model.converged},
        ))
    return notes


__all__ = ["EvidenceLift", "RecipientLearning", "MIN_CODE_SUPPORT", "evidence_lift", "learn"]
