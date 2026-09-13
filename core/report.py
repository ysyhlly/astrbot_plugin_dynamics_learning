"""Assemble one analysis pass into the snapshot the console renders.

This is the only place that decides how a learner's in-sample proposal and the
evaluator's out-of-sample verdict are presented together. The rule is that an
in-sample proposal is always shown next to its verdict, never on its own, so a
number produced while the model could see the labels can never be mistaken for
a validated improvement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .autotune import DECISION_LABEL, TuneRules, TuneRun, run_all as run_tuning
from .config import LearningConfig
from .evaluator import EvaluationReport, VERDICT_ACCEPTED, evaluate_dataset
from .metrics import SAMPLE_NOTE, within_window
from .policy import PolicyCandidate, drift_from
from .recipient_learner import RecipientLearning, learn as learn_recipient
from .recommendation import (
    CONFIDENCE_INSUFFICIENT, KIND_CONFIG_PARAM, Recommendation,
)
from .samples import TASK_RECIPIENT, TASK_REPLY, TASK_TOPIC, LearningSample
from .topic_learner import TopicLearning, learn as learn_topic

OVERVIEW_WINDOW_DAYS = 7

# Which task owns which parameter, so a proposal can be checked against the
# iteration that actually decides it.
TASK_OF_PARAM = {
    "strong_addressivity_threshold": TASK_RECIPIENT,
    "safe_hover_threshold": TASK_RECIPIENT,
    "topic_commit_threshold": TASK_TOPIC,
    "topic_join_threshold": TASK_TOPIC,
    "topic_margin_threshold": TASK_TOPIC,
}


@dataclass
class AnalysisResult:
    generated_at: float = 0.0
    dataset: dict[str, Any] = field(default_factory=dict)
    overview: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    recipient: RecipientLearning = field(default_factory=RecipientLearning)
    topic: TopicLearning = field(default_factory=TopicLearning)
    recommendations: list[Recommendation] = field(default_factory=list)
    evaluation: EvaluationReport | None = None
    tuning: list[TuneRun] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def promoted_runs(self) -> list[TuneRun]:
        return [row for row in self.tuning if row.promoted]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "dataset": dict(self.dataset),
            "overview": dict(self.overview),
            "errors": dict(self.errors),
            "recipient": self.recipient.as_dict(),
            "topic": self.topic.as_dict(),
            "recommendations": [row.as_dict() for row in self.recommendations],
            "evaluation": self.evaluation.as_dict() if self.evaluation else None,
            "tuning": [row.as_dict() for row in self.tuning],
            "notes": list(self.notes),
        }


def dataset_summary(samples: Sequence[LearningSample]) -> dict[str, Any]:
    tasks = {task: sum(1 for sample in samples if sample.task == task)
             for task in (TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY)}
    return {
        "samples": len(samples),
        "sessions": len({sample.session_hash for sample in samples}),
        "tasks": tasks,
        "first_timestamp": min((sample.timestamp for sample in samples), default=None),
        "last_timestamp": max((sample.timestamp for sample in samples), default=None),
        "source_note": SAMPLE_NOTE,
    }


def overview(samples: Sequence[LearningSample], *, now: float) -> dict[str, Any]:
    """Headline numbers, always paired with the sample count behind them."""
    recent = within_window(samples, now=now, days=OVERVIEW_WINDOW_DAYS)
    summary: dict[str, Any] = {"window_days": OVERVIEW_WINDOW_DAYS,
                               "samples_in_window": len(recent),
                               "samples_total": len(samples)}
    for task in (TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY):
        rows = [sample for sample in recent if sample.task == task]
        if not rows:
            summary[task] = {"total": 0, "accuracy": None, "note": SAMPLE_NOTE}
            continue
        correct = sum(1 for sample in rows if sample.correct)
        summary[task] = {"total": len(rows), "correct": correct,
                         "accuracy": round(correct / len(rows), 4), "note": SAMPLE_NOTE}
    return summary


def error_breakdown(samples: Sequence[LearningSample]) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = {}
    for sample in samples:
        if sample.correct:
            continue
        bucket = buckets.setdefault(sample.task, {})
        bucket[sample.error_type or "unknown"] = bucket.get(sample.error_type or "unknown", 0) + 1
    for task, counts in buckets.items():
        buckets[task] = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    return buckets


def _annotate(recommendation: Recommendation, verdict: str | None,
              tuning: TuneRun | None = None) -> Recommendation:
    """Attach the evaluation verdict to a proposal without inventing a new one.

    A parameter proposal is only actionable once something has actually
    validated it on a holdout. Anything else — a rejection, an underpowered
    holdout, an analysis that skipped evaluation, a tuning run that never
    reached a promote tier, or a tuning run that moved the same parameter the
    *other* way — is downgraded to `insufficient` confidence, which makes it
    non-actionable.

    The number is still shown, because hiding it would lose the evidence that
    the direction is at least interesting; it just can no longer be mistaken
    for something that passed a holdout. The conflicting direction is the case
    that matters most: an in-sample sweep and a holdout iteration can disagree,
    and the holdout is the one that wins.
    """
    evidence = dict(recommendation.evidence)
    evidence["evaluation_verdict"] = verdict if verdict is not None else "not_run"
    downgrade = False
    if recommendation.kind == KIND_CONFIG_PARAM:
        downgrade = verdict != VERDICT_ACCEPTED
        if tuning is not None:
            evidence["tuning_decision"] = tuning.decision
            evidence["tuning_label"] = DECISION_LABEL.get(tuning.decision, tuning.decision)
            if not tuning.promoted:
                downgrade = True
                evidence["downgrade_reason"] = "迭代调参未达到采纳门槛"
            elif tuning.final_policy is not None:
                drift = {row["param"]: row["delta"]
                         for row in drift_from(tuning.baseline, tuning.final_policy)}
                moved = drift.get(recommendation.param)
                if (moved is not None and recommendation.before is not None
                        and recommendation.after is not None
                        and (recommendation.after - recommendation.before) * moved < 0):
                    downgrade = True
                    evidence["downgrade_reason"] = "样本内建议方向与留出集迭代结论相反"
    if downgrade:
        return replace(recommendation, evidence=evidence, confidence=CONFIDENCE_INSUFFICIENT)
    return replace(recommendation, evidence=evidence)


def analyze(
    samples: Sequence[LearningSample],
    *,
    config: LearningConfig | None = None,
    existing_versions: Sequence[str] = (),
    baseline_policy: Mapping[str, float] | None = None,
    with_evaluation: bool = True,
    with_tuning: bool = True,
    rules: TuneRules | None = None,
    now: float | None = None,
) -> AnalysisResult:
    config = config or LearningConfig()
    stamp = now if now is not None else time.time()
    result = AnalysisResult(generated_at=stamp)
    result.dataset = dataset_summary(samples)
    result.overview = overview(samples, now=stamp)
    result.errors = error_breakdown(samples)
    result.recipient = learn_recipient(samples, config=config)
    result.topic = learn_topic(samples, config=config)

    evaluation: EvaluationReport | None = None
    if with_evaluation:
        evaluation = evaluate_dataset(samples, config=config, baseline_policy=baseline_policy,
                                      existing_versions=existing_versions, now=stamp)
        result.evaluation = evaluation
    verdict = evaluation.verdict if evaluation is not None else None

    # Tuning runs first: it is what decides promotion, and its verdict is what
    # the in-sample proposals have to be reconciled against below.
    if with_tuning:
        result.tuning = run_tuning(samples, config=config, baseline_policy=baseline_policy,
                                   rules=rules, existing_versions=existing_versions, now=stamp)
        for tuning in result.tuning:
            label = DECISION_LABEL.get(tuning.decision, tuning.decision)
            suffix = f"（{tuning.adopted_steps} 步采纳）" if tuning.adopted_steps else ""
            result.notes.append(f"{tuning.task} 迭代调参结论：{label}{suffix}；{tuning.stop_reason}")
        promoted = result.promoted_runs
        if promoted:
            result.notes.append(
                "达到采纳门槛的任务：" + "、".join(row.task for row in promoted)
                + "。采纳只写入本插件的策略记录，不会改动 ChatDynamics 配置。")
    elif evaluation is not None and evaluation.verdict == VERDICT_ACCEPTED and evaluation.candidate:
        result.notes.append(
            f"离线评测判定候选 {evaluation.candidate.version} 可采纳；"
            "采纳只写入本插件的策略记录，不会改动 ChatDynamics 配置。")

    tuning_by_task = {row.task: row for row in result.tuning}
    proposals: list[Recommendation] = []
    for learner in (result.recipient, result.topic):
        for recommendation in learner.recommendations:
            if recommendation.kind == KIND_CONFIG_PARAM:
                owner = TASK_OF_PARAM.get(recommendation.param or "")
                proposals.append(_annotate(
                    recommendation, verdict,
                    tuning_by_task.get(owner) if owner is not None else None))
            else:
                proposals.append(recommendation)
    result.recommendations = proposals
    if not samples:
        result.notes.append("还没有学习样本。先在 ChatDynamics 回放页做人工标注，再回来执行导入。")
    result.notes.extend(result.recipient.notes[:2])
    result.notes.extend(result.topic.notes[:2])
    return result


def policy_rows(policies: Sequence[PolicyCandidate]) -> list[dict[str, Any]]:
    return [candidate.as_dict() for candidate in sorted(policies, key=lambda row: row.created_at,
                                                        reverse=True)]


__all__ = [
    "OVERVIEW_WINDOW_DAYS", "AnalysisResult", "TuneRules", "TuneRun", "analyze", "dataset_summary",
    "error_breakdown", "overview", "policy_rows",
]
