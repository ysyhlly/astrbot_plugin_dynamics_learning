"""Iterative tuning: small steps, error-type evidence, and three tiers of trust.

A single ±5% step that must prove +2% on its own is not how optimisation works.
Real gains compound: three steps of -5% take a threshold from 0.70 to 0.600, and
each of them can be replayed. So the budget is spent on **how far an iteration
may walk** rather than on how far one jump may be:

    per step          ±5%   of the current value
    cumulative        ±15%  of the original baseline
    consecutive steps 3     (0.95^3 = -14.3%, so the two limits agree)

and the trust tiers are:

    Safe candidate   cumulative >= -0.2%, target error improved, no guard broken
                     -> this step is adopted, another one may follow
    Advance          marginal >= +0.5%, or target error down >= 5% relative
                     -> the next step is worth taking
    Promote          cumulative >= +1.0%, or target error down >= 10% relative
                     (with cumulative >= -0.2%)
                     -> may enter the learned policy
    Strong promote   cumulative >= +2.0%  -> high confidence

Every step is judged on the error kind it was aimed at, not only on the global
metric. A change that cuts the target error 37% relative while moving accuracy
+0.6% is a success, and a rule that only reads the global delta cannot see it.

Stops:

    two consecutive marginal gains below +0.2%   -> stop, keep what was earned
    any guard metric down more than 1%           -> stop and roll back
    cumulative drift hits ±15% while improving   -> stop for human review

None of this writes anything. A promoted policy is a *record*; ChatDynamics is
never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .config import LearningConfig
from .evaluator import (
    GUARD_METRICS, PRIMARY_METRIC, evaluate_dataset, score_task, split_by_session,
)
from .metrics import ErrorRate
from .policy import (
    BASE_POLICY, STATUS_PROPOSED, STATUS_VALIDATED, PolicyCandidate, candidate_from,
    clamp_cumulative, drift_from, normalize_policy, policy_deltas, target_error_for,
)
from .recommendation import confidence_for
from .samples import TASK_RECIPIENT, TASK_TOPIC, LearningSample

TASK_PRIMARY_ERROR = {TASK_RECIPIENT: "missed_bot", TASK_TOPIC: "fragmentation"}

DECISION_STRONG_PROMOTE = "strong_promote"
DECISION_PROMOTE = "promote"
DECISION_CANDIDATE = "candidate"
DECISION_REJECT = "reject"
DECISION_ROLLBACK = "rollback"
DECISION_NEEDS_REVIEW = "needs_review"
DECISION_NO_CHANGE = "no_change"
DECISION_INSUFFICIENT = "insufficient"

DECISION_LABEL = {
    DECISION_STRONG_PROMOTE: "强采纳（高置信度）",
    DECISION_PROMOTE: "采纳",
    DECISION_CANDIDATE: "候选（弱正向，未达采纳门槛）",
    DECISION_REJECT: "拒绝",
    DECISION_ROLLBACK: "回滚",
    DECISION_NEEDS_REVIEW: "需人工确认（累计漂移触顶）",
    DECISION_NO_CHANGE: "无可用调整",
    DECISION_INSUFFICIENT: "样本不足",
}
PROMOTED_DECISIONS = (DECISION_STRONG_PROMOTE, DECISION_PROMOTE)


@dataclass(frozen=True)
class TuneRules:
    """The thresholds the whole state machine is parameterised by."""

    step_delta_ratio: float = 0.05
    cumulative_delta_ratio: float = 0.15
    max_steps: int = 3

    safe_floor: float = -0.002
    advance_improvement: float = 0.005
    advance_error_relative: float = 0.05
    promote_improvement: float = 0.010
    promote_error_relative: float = 0.10
    strong_improvement: float = 0.020

    marginal_stall: float = 0.002
    stall_patience: int = 2
    guard_regression: float = 0.010

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class TuneStep:
    index: int
    policy: dict[str, float]
    changes: list[dict[str, Any]] = field(default_factory=list)
    drift: list[dict[str, Any]] = field(default_factory=list)
    clamped: list[str] = field(default_factory=list)
    primary_metric: str = ""
    step_delta: float | None = None
    cumulative_delta: float | None = None
    target_error: str | None = None
    target_error_step: float | None = None
    target_error_cumulative: float | None = None
    collateral: dict[str, Any] = field(default_factory=dict)
    guard_failures: list[str] = field(default_factory=list)
    safe: bool = False
    advanced: bool = False
    decision: str = DECISION_CANDIDATE
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "policy": {key: round(value, 4) for key, value in self.policy.items()},
            "changes": self.changes, "drift": self.drift, "clamped": list(self.clamped),
            "primary_metric": self.primary_metric,
            "step_delta": self.step_delta, "cumulative_delta": self.cumulative_delta,
            "target_error": self.target_error, "target_error_step": self.target_error_step,
            "target_error_cumulative": self.target_error_cumulative,
            "collateral": self.collateral, "guard_failures": list(self.guard_failures),
            "safe": self.safe, "advanced": self.advanced, "decision": self.decision,
            "decision_label": DECISION_LABEL.get(self.decision, self.decision),
            "reasons": list(self.reasons),
        }


@dataclass
class TuneRun:
    task: str
    baseline: dict[str, float] = field(default_factory=lambda: dict(BASE_POLICY))
    rules: TuneRules = field(default_factory=TuneRules)
    steps: list[TuneStep] = field(default_factory=list)
    decision: str = DECISION_INSUFFICIENT
    final_policy: dict[str, float] | None = None
    high_confidence: bool = False
    stop_reason: str = ""
    reasons: list[str] = field(default_factory=list)
    candidate: PolicyCandidate | None = None
    primary_metric: str = ""

    @property
    def promoted(self) -> bool:
        return self.decision in PROMOTED_DECISIONS

    @property
    def adopted_steps(self) -> int:
        return sum(1 for step in self.steps if step.safe)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "decision": self.decision,
            "decision_label": DECISION_LABEL.get(self.decision, self.decision),
            "promoted": self.promoted,
            "high_confidence": self.high_confidence,
            "primary_metric": self.primary_metric,
            "baseline": {key: round(value, 4) for key, value in self.baseline.items()},
            "final_policy": ({key: round(value, 4) for key, value in self.final_policy.items()}
                             if self.final_policy else None),
            "drift": drift_from(self.baseline, self.final_policy) if self.final_policy else [],
            "adopted_steps": self.adopted_steps,
            "steps": [step.as_dict() for step in self.steps],
            "rules": self.rules.as_dict(),
            "stop_reason": self.stop_reason,
            "reasons": list(self.reasons),
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "note": "本插件不会自动修改 ChatDynamics 配置；采纳只写入本插件的策略记录。",
        }


def error_improved(before: ErrorRate, after: ErrorRate) -> bool:
    """Did the target error actually improve?

    A rate of zero has no relative improvement to make, so the only thing that
    matters there is not getting worse.
    """
    if before.count == 0:
        return after.count == 0
    relative = before.relative(after)
    return relative is not None and relative < 0


def target_error_rate(errors: Mapping[str, ErrorRate], kind: str | None) -> ErrorRate | None:
    if kind is None:
        return None
    return errors.get(kind)


def _metric_delta(before: Mapping[str, Any], after: Mapping[str, Any], name: str) -> float | None:
    left, right = before.get(name), after.get(name)
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return round(float(right) - float(left), 4)


def _collateral(errors: Mapping[str, ErrorRate], baseline: Mapping[str, ErrorRate],
                target: str | None) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for kind, before in baseline.items():
        after = errors.get(kind)
        if after is None:
            continue
        relative = before.relative(after)
        rows[kind] = {
            "baseline_rate": _round_or_none(before.rate),
            "candidate_rate": _round_or_none(after.rate),
            "relative": _round_or_none(relative),
            "is_target": kind == target,
            "improved": error_improved(before, after),
        }
    return rows


def run_tuning(
    samples: Sequence[LearningSample],
    *,
    task: str = TASK_RECIPIENT,
    config: LearningConfig | None = None,
    baseline_policy: Mapping[str, float] | None = None,
    rules: TuneRules | None = None,
    existing_versions: Sequence[str] = (),
    now: float | None = None,
) -> TuneRun:
    """Walk small, replayable steps from `baseline_policy` and classify the result.

    Only `task` may propose a parameter move. Every step is re-scored on the same
    session-level holdout, so the numbers in one run are comparable to each other
    and to the single-shot evaluation.
    """
    config = config or LearningConfig()
    rules = rules or TuneRules()
    baseline = normalize_policy(baseline_policy or BASE_POLICY)
    run = TuneRun(task=task, baseline=baseline, rules=rules,
                  primary_metric=PRIMARY_METRIC.get(task, "accuracy"))
    if task not in (TASK_RECIPIENT, TASK_TOPIC):
        run.decision = DECISION_NO_CHANGE
        run.stop_reason = f"任务 {task} 不支持迭代调参（它会随受控参数一起变化）"
        return run

    task_rows = [sample for sample in samples if sample.task == task]
    if not task_rows:
        run.decision = DECISION_INSUFFICIENT
        run.stop_reason = "该任务没有学习样本"
        return run
    split = split_by_session(samples, holdout_ratio=config.holdout_ratio)
    holdout = [sample for sample in split.holdout if sample.task == task]
    if not split.holdout or len(holdout) < config.min_samples_for_evaluation:
        run.decision = DECISION_INSUFFICIENT
        run.stop_reason = (f"留出集该任务只有 {len(holdout)} 条，"
                           f"低于 {config.min_samples_for_evaluation} 条，迭代不下结论")
        return run

    baseline_score = score_task(holdout, baseline, task)

    current = dict(baseline)
    previous_score = baseline_score
    stalled = 0
    for index in range(1, max(1, rules.max_steps) + 1):
        proposal = evaluate_dataset(samples, config=config, baseline_policy=current,
                                    existing_versions=existing_versions, now=now, tasks=(task,))
        proposed_changes = (proposal.candidate.params if proposal.candidate else {})
        if not proposed_changes:
            run.stop_reason = proposal.reasons[0] if proposal.reasons else "本步没有可调整的参数"
            break

        step_before = dict(current)
        proposed, clamped = clamp_cumulative(baseline, proposed_changes,
                                             rules.cumulative_delta_ratio)
        moves = drift_from(step_before, proposed)
        if not moves:
            run.stop_reason = "本步没有产生有效位移"
            break

        step_score = score_task(holdout, proposed, task)
        primary = PRIMARY_METRIC.get(task, "accuracy")
        # moves is a list of drift rows; a mapping is what names the direction.
        target_kind = target_error_for(
            {row["param"]: proposed[row["param"]] for row in moves if row["param"] in proposed},
            step_before)
        if target_kind is None:
            target_kind = TASK_PRIMARY_ERROR.get(task)

        baseline_error = target_error_rate(baseline_score.errors, target_kind)
        step_error = target_error_rate(step_score.errors, target_kind)

        step = TuneStep(
            index=index, policy=dict(proposed), changes=moves,
            drift=drift_from(baseline, proposed), clamped=clamped,
            primary_metric=primary,
            step_delta=_metric_delta(previous_score.metrics, step_score.metrics, primary),
            cumulative_delta=_metric_delta(baseline_score.metrics, step_score.metrics, primary),
            target_error=target_kind,
        )
        step_before_error = target_error_rate(previous_score.errors, target_kind)
        if step_before_error is not None and step_error is not None:
            relative_step = step_before_error.relative(step_error)
            step.target_error_step = _round_or_none(relative_step)
        if baseline_error is not None and step_error is not None:
            step.target_error_cumulative = _round_or_none(baseline_error.relative(step_error))
        step.collateral = _collateral(step_score.errors, baseline_score.errors, target_kind)
        step.guard_failures = _guard_failures(baseline_score.metrics, step_score.metrics, task,
                                              rules.guard_regression)

        # 1) a guard break stops the run and rolls everything back.
        if step.guard_failures:
            step.decision = DECISION_ROLLBACK
            step.reasons.append("核心指标回退超过允许幅度：" + "、".join(step.guard_failures))
            run.steps.append(step)
            run.decision = DECISION_ROLLBACK
            run.stop_reason = "guard 指标回退，整轮迭代作废并回到基线"
            run.final_policy = dict(baseline)
            run.reasons = list(step.reasons)
            return _finalise(run, samples, config, task, baseline, existing_versions, now)

        # 2) Safe gate: the step itself must not make things worse than the
        #    original baseline, and must not leave the target error worse either.
        if baseline_error is None or step_error is None:
            target_ok = True
        else:
            target_ok = error_improved(baseline_error, step_error)
        safe = (step.cumulative_delta is not None and step.cumulative_delta >= rules.safe_floor
                and target_ok)
        step.safe = safe
        step.reasons.append(f"累计 {primary} {_fmt(step.cumulative_delta)}"
                            f"（下限 {rules.safe_floor:+.4f}）")
        step.reasons.append(
            f"目标错误 {target_kind} 相对变化 {_fmt_ratio(step.target_error_cumulative)}"
            f"{'' if target_ok else ' —— 未改善'}")
        if not safe:
            step.decision = DECISION_REJECT
            step.reasons.append("未通过 Safe candidate 门槛，本步不采纳")
            run.steps.append(step)
            run.decision = DECISION_REJECT
            run.stop_reason = "本步未通过 Safe candidate 门槛"
            run.final_policy = dict(current) if current != baseline else None
            run.reasons = list(step.reasons)
            return _finalise(run, samples, config, task, baseline, existing_versions, now)

        # The step is adopted from here on.
        current = dict(proposed)
        previous_score = step_score

        # 3) Strong promote.
        if (step.cumulative_delta is not None
                and step.cumulative_delta >= rules.strong_improvement):
            step.decision = DECISION_STRONG_PROMOTE
            step.reasons.append(
                f"累计提升 {step.cumulative_delta:+.4f} >= {rules.strong_improvement:.2%}，高置信度")
            run.steps.append(step)
            run.decision = DECISION_STRONG_PROMOTE
            run.high_confidence = True
            run.stop_reason = "达到强采纳门槛"
            run.final_policy = dict(current)
            run.reasons = list(step.reasons)
            return _finalise(run, samples, config, task, baseline, existing_versions, now)

        # 4) Promote.
        error_promote = (step.target_error_cumulative is not None
                         and step.target_error_cumulative <= -rules.promote_error_relative
                         and step.cumulative_delta is not None
                         and step.cumulative_delta >= rules.safe_floor)
        if ((step.cumulative_delta is not None
             and step.cumulative_delta >= rules.promote_improvement) or error_promote):
            step.decision = DECISION_PROMOTE
            step.reasons.append(
                f"累计提升 {_fmt(step.cumulative_delta)}，目标错误相对变化 "
                f"{_fmt_ratio(step.target_error_cumulative)}，达到采纳门槛")
            run.steps.append(step)
            run.decision = DECISION_PROMOTE
            run.stop_reason = "达到采纳门槛"
            run.final_policy = dict(current)
            run.reasons = list(step.reasons)
            return _finalise(run, samples, config, task, baseline, existing_versions, now)

        # 5) The cumulative cap bit while still improving: a human decides.
        if clamped and (step.step_delta is None or step.step_delta > 0):
            step.decision = DECISION_NEEDS_REVIEW
            step.reasons.append(
                "累计漂移已达 ±%.0f%% 上限，继续调整需要人工确认"
                % (rules.cumulative_delta_ratio * 100))
            run.steps.append(step)
            run.decision = DECISION_NEEDS_REVIEW
            run.stop_reason = (f"累计漂移已达 ±{rules.cumulative_delta_ratio:.0%} 上限，"
                               "继续调整需要人工确认")
            run.final_policy = dict(current)
            run.reasons = list(step.reasons)
            return _finalise(run, samples, config, task, baseline, existing_versions, now)

        # 6) Is another step worth taking?
        advance = ((step.step_delta is not None and step.step_delta >= rules.advance_improvement)
                   or (step.target_error_step is not None
                       and step.target_error_step <= -rules.advance_error_relative))
        step.advanced = advance
        if advance:
            stalled = 0
            step.decision = "continue"
            step.reasons.append(f"边际提升 {_fmt(step.step_delta)}，继续下一小步")
        else:
            stalled += 1 if (step.step_delta is None
                             or step.step_delta < rules.marginal_stall) else 0
            step.decision = "stall" if stalled else "continue"
            step.reasons.append(
                f"边际提升 {_fmt(step.step_delta)}，未达继续门槛"
                f"（{rules.advance_improvement:.2%}），已停滞 {stalled} 步")
        run.steps.append(step)

        if stalled >= rules.stall_patience:
            run.stop_reason = f"连续 {stalled} 步边际收益低于 {rules.marginal_stall:.2%}，停止"
            break

    if not run.steps:
        # Nothing was ever staked, so there is nothing to promote or roll back.
        run.decision = DECISION_NO_CHANGE
        run.final_policy = None
        return _finalise(run, samples, config, task, baseline, existing_versions, now)

    final_delta = run.steps[-1].cumulative_delta
    if current == baseline or final_delta is None or final_delta <= 0:
        # The parameter moved but no metric did. Recording that as a candidate
        # would claim a validated improvement of exactly nothing.
        run.decision = DECISION_NO_CHANGE
        run.reasons.append(f"停止原因：{run.stop_reason}" if run.stop_reason else "")
        run.stop_reason = "参数发生位移，但留出集上没有任何指标变化，没有可保留的收益"
        run.final_policy = None
        run.reasons.append(
            f"{run.adopted_steps} 步通过 Safe candidate，但累计 {run.primary_metric} "
            f"{'未定义' if final_delta is None else f'{final_delta:+.4f}'}，没有可保留的收益。")
        return _finalise(run, samples, config, task, baseline, existing_versions, now)

    run.decision = DECISION_CANDIDATE
    run.stop_reason = run.stop_reason or "步数用尽，保留已获得的收益"
    run.final_policy = dict(current)
    run.reasons.append(
        f"{run.adopted_steps} 步通过 Safe candidate，累计 {run.primary_metric} "
        f"{final_delta:+.4f}，但未达采纳门槛"
        f"（{rules.promote_improvement:.2%} 或目标错误相对下降 "
        f"{rules.promote_error_relative:.0%}），记录为候选。")
    return _finalise(run, samples, config, task, baseline, existing_versions, now)


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _fmt(value: float | None) -> str:
    return "未定义" if value is None else f"{value:+.4f}"


def _fmt_ratio(value: float | None) -> str:
    return "未定义" if value is None else f"{value:+.2%}"


def _guard_failures(baseline: Mapping[str, Any], candidate: Mapping[str, Any],
                    task: str, allowance: float) -> list[str]:
    failures: list[str] = []
    for name in GUARD_METRICS.get(task, ()):
        delta = _metric_delta(baseline, candidate, name)
        if delta is not None and delta < -allowance:
            failures.append(f"{task}.{name} {delta:+.4f}")
    return failures


def _finalise(run: TuneRun, samples: Sequence[LearningSample], config: LearningConfig,
              task: str, baseline: Mapping[str, float], existing_versions: Sequence[str],
              now: float | None) -> TuneRun:
    """Attach a policy record when the run produced anything worth recording."""
    if run.final_policy is None:
        return run
    if abs(sum(run.final_policy.values()) - sum(baseline.values())) < 1e-9:
        return run
    # `validated`, never `promoted`: a tuning run proves a holdout improvement.
    # The shadow stage — publishing the policy beside live behaviour and watching
    # it — is a different kind of evidence, and this plugin cannot produce it.
    status = STATUS_VALIDATED if run.promoted else STATUS_PROPOSED
    last = run.steps[-1] if run.steps else None
    candidate = candidate_from(
        run.final_policy, baseline=baseline, source="iterative_tuning",
        rationale=(f"{run.task} 迭代调参：{run.adopted_steps} 步采纳，"
                   f"结论 {DECISION_LABEL.get(run.decision, run.decision)}"),
        evidence={
            "decision": run.decision, "steps": len(run.steps),
            "adopted_steps": run.adopted_steps, "rules": run.rules.as_dict(),
            "high_confidence": run.high_confidence,
            "deltas": policy_deltas(baseline, run.final_policy),
        },
        existing_versions=existing_versions, now=now,
    )
    run.candidate = candidate.with_fields(
        status=status,
        target_error=(last.target_error if last is not None else "") or "",
        collateral_regressions=tuple(
            name for name, row in ((last.collateral or {}).items() if last else ())
            if isinstance(row, Mapping) and row.get("is_target") is False
            and row.get("improved") is False),
        confidence=confidence_for(run.rules.max_steps and run.adopted_steps or 0,
                                  min_samples=1),
        holdout_result={
            "decision": run.decision,
            "primary_metric": run.primary_metric,
            "adopted_steps": run.adopted_steps,
            "cumulative_delta": (last.cumulative_delta if last is not None else None),
            "target_error_cumulative": (last.target_error_cumulative
                                        if last is not None else None),
            "steps": [step.as_dict() for step in run.steps],
        },
    ).with_status(status, now=now if now is not None else candidate.created_at,
                  reason=run.stop_reason or run.decision)
    return run


def run_all(
    samples: Sequence[LearningSample],
    *,
    config: LearningConfig | None = None,
    baseline_policy: Mapping[str, float] | None = None,
    rules: TuneRules | None = None,
    existing_versions: Sequence[str] = (),
    now: float | None = None,
) -> list[TuneRun]:
    """Tune each independently controlled task from the same baseline.

    The two tasks touch disjoint parameters, so running them separately gives
    each run one unambiguous target error — which is the whole point of judging
    a change by the error it was aimed at.
    """
    config = config or LearningConfig()
    runs: list[TuneRun] = []
    versions = list(existing_versions)
    for task in (TASK_RECIPIENT, TASK_TOPIC):
        run = run_tuning(samples, task=task, config=config, baseline_policy=baseline_policy,
                         rules=rules, existing_versions=versions, now=now)
        runs.append(run)
        if run.candidate is not None:
            versions.append(run.candidate.version)
    return runs


__all__ = [
    "DECISION_CANDIDATE", "DECISION_INSUFFICIENT", "DECISION_LABEL", "DECISION_NEEDS_REVIEW",
    "DECISION_NO_CHANGE", "DECISION_PROMOTE", "DECISION_REJECT", "DECISION_ROLLBACK",
    "DECISION_STRONG_PROMOTE", "PROMOTED_DECISIONS", "TASK_PRIMARY_ERROR", "TuneRules",
    "TuneRun", "TuneStep", "error_improved", "run_all", "run_tuning", "target_error_rate",
]
