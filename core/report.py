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

from .attribution import attribution_report
from .autotune import DECISION_LABEL, TuneRules, TuneRun, run_all as run_tuning
from .config import LearningConfig
from .evaluator import (
    VERDICT_ACCEPTED, EvaluationReport, evaluate_dataset, forward_evaluation, promotion_check,
)
from .policy import STATUS_PROPOSED, STATUS_VALIDATED
from .metrics import SAMPLE_NOTE, within_window
from .quality import dataset_gate
from .policy import PolicyCandidate, drift_from
from .recipient_learner import RecipientLearning, learn as learn_recipient
from .recommendation import (
    CONFIDENCE_INSUFFICIENT, KIND_CONFIG_PARAM, Recommendation,
)
from .samples import (
    TASK_RECIPIENT, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME, TASK_TOPIC, LearningSample,
)
from .shadow import evaluate_shadow, rules_from_config
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
    # The time-ordered holdout, and the combined verdict that gates promotion.
    forward: EvaluationReport | None = None
    promotion: dict[str, Any] = field(default_factory=dict)
    # The pre-learning gate: whether a policy may be offered from this corpus.
    dataset_gate: dict[str, Any] = field(default_factory=dict)
    # The shadow A/B result: what a policy's decisions say about it, measured on
    # the turns where it disagreed with the baseline.
    shadow: dict[str, Any] = field(default_factory=dict)
    tuning: list[TuneRun] = field(default_factory=list)
    # The message-level chain. Reported before the per-task numbers because it is
    # the only part that answers "which layer should change", and it is derived
    # from the same samples those numbers come from.
    attribution: dict[str, Any] = field(default_factory=dict)
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
            "forward": self.forward.as_dict() if self.forward else None,
            "promotion": dict(self.promotion),
            "dataset_gate": dict(self.dataset_gate),
            "shadow": dict(self.shadow),
            "tuning": [row.as_dict() for row in self.tuning],
            "attribution": dict(self.attribution),
            "notes": list(self.notes),
        }


ANALYSED_TASKS = (TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME)


def dataset_summary(samples: Sequence[LearningSample]) -> dict[str, Any]:
    tasks = {task: sum(1 for sample in samples if sample.task == task)
             for task in ANALYSED_TASKS}
    return {
        "samples": len(samples),
        "sessions": len({sample.session_hash for sample in samples}),
        # Reported next to sessions rather than merged into it: under the current
        # host contract a scope *is* a session, and a number that silently equals
        # another number teaches a reader to trust a dimension that is not there.
        "scopes": len({sample.scope_hash for sample in samples}),
        "scope_level": "session",
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
    for task in ANALYSED_TASKS:
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
    host_version: str | None = None,
) -> AnalysisResult:
    config = config or LearningConfig()
    stamp = now if now is not None else time.time()
    result = AnalysisResult(generated_at=stamp)
    result.dataset_gate = dataset_gate(samples, config=config, now=stamp)
    result.shadow = evaluate_shadow(samples, rules=rules_from_config(config),
                                    min_samples=config.gate_min_samples)
    result.dataset = dataset_summary(samples)
    result.overview = overview(samples, now=stamp)
    result.errors = error_breakdown(samples)
    result.attribution = attribution_report(samples, min_samples=config.min_samples_for_evaluation)
    result.recipient = learn_recipient(samples, config=config)
    result.topic = learn_topic(samples, config=config)

    evaluation: EvaluationReport | None = None
    forward: EvaluationReport | None = None
    if with_evaluation:
        evaluation = evaluate_dataset(samples, config=config, baseline_policy=baseline_policy,
                                      existing_versions=existing_versions, now=stamp,
                                      host_version=host_version)
        result.evaluation = evaluation
        if config.require_forward_validation:
            forward = forward_evaluation(samples, config=config, baseline_policy=baseline_policy,
                                         existing_versions=existing_versions, now=stamp,
                                         host_version=host_version)
            result.forward = forward
        check = promotion_check(evaluation, forward, config=config)
        result.promotion = check.as_dict()
        if check.reasons:
            result.notes.append("采纳门槛：" + check.reasons[0])
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
        # The tuning runs only saw the session holdout. A candidate they marked
        # validated still has to clear the forward holdout and the interval, so
        # the combination is applied *after* them rather than inside each run —
        # one gate, one answer, and no way for two runs to disagree about it.
        gate = result.promotion or {}
        if gate.get("verdict") != VERDICT_ACCEPTED:
            reason = (gate.get("reasons") or ["未通过采纳门槛"])[0]
            for tuning in result.tuning:
                if tuning.candidate is None:
                    continue
                if tuning.candidate.status != STATUS_VALIDATED:
                    continue
                tuning.candidate = tuning.candidate.with_status(
                    STATUS_PROPOSED, now=stamp, reason=reason[:200])
            if result.evaluation is not None and result.evaluation.candidate is not None \
                    and result.evaluation.candidate.status == STATUS_VALIDATED:
                result.evaluation.candidate = result.evaluation.candidate.with_status(
                    STATUS_PROPOSED, now=stamp, reason=reason[:200])
            if result.tuning:
                result.notes.append("迭代结论本可采纳，但整体门槛未通过：" + reason)
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
    if not result.dataset_gate.get("ok", True):
        # The gate runs *before* the recommendations are handed out, and its
        # effect is to make every parameter proposal non-actionable. The numbers
        # stay visible — a reader still needs to see why the corpus was judged
        # insufficient — but nothing here can be adopted from them.
        blocked_by = result.dataset_gate.get("blocked_by") or []
        result.recommendations = [
            replace(row, confidence=CONFIDENCE_INSUFFICIENT,
                    evidence={**dict(row.evidence), "dataset_gate": "blocked",
                              "dataset_gate_checks": blocked_by})
            for row in proposals
        ]
        result.notes.append("数据门槛未通过（" + "、".join(blocked_by)
                            + "）：本次只出统计与诊断，不出可采纳的策略。")
        for tuning in result.tuning:
            if tuning.candidate is not None:
                tuning.candidate = tuning.candidate.with_status(
                    STATUS_PROPOSED, now=stamp, reason="数据门槛未通过")
    else:
        result.recommendations = proposals
    if not samples:
        result.notes.append("还没有学习样本。先在 ChatDynamics 回放页做人工标注，再回来执行导入。")
    result.notes.extend(result.recipient.notes[:2])
    result.notes.extend(result.topic.notes[:2])
    result.notes.extend(result.attribution.get("notes", [])[:2])
    shadow_gate = (result.shadow or {}).get("gate") or {}
    if result.shadow.get("rows"):
        result.notes.append(
            "shadow A/B：" +
            ("达到进入 active 的门槛。" if shadow_gate.get("ok")
             else "未达进入 active 的门槛（" + "、".join(shadow_gate.get("blocked_by") or [])
                  + "）。"))
    if result.forward is not None:
        result.notes.append(
            "前向验证（按标注时间切分）：" +
            ("通过。" if result.forward.verdict == VERDICT_ACCEPTED
             else (result.forward.reasons[0] if result.forward.reasons else "未通过。")))
    return result


def policy_rows(policies: Sequence[PolicyCandidate]) -> list[dict[str, Any]]:
    return [candidate.as_dict() for candidate in sorted(policies, key=lambda row: row.created_at,
                                                        reverse=True)]


__all__ = [
    "ANALYSED_TASKS", "OVERVIEW_WINDOW_DAYS", "AnalysisResult", "TuneRules", "TuneRun", "analyze",
    "dataset_summary", "error_breakdown", "overview", "policy_rows",
]
