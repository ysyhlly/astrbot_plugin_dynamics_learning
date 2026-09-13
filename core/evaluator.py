"""v0.4 Offline evaluation: baseline vs candidate, on sessions the fit never saw.

The rule the plan sets is enforced here literally: only a candidate that beats
the baseline on unseen sessions may be promoted. So the split is by **session**,
not by sample — topic metrics are within-session, and a scorer that memorised
one group would otherwise look good on a neighbouring message from the same
conversation. Everything that selects a parameter runs on the training split
only; the holdout split is touched exactly once, to score.

Two things are measured for every comparison, because a global number alone
throws away the evidence:

* the **primary metric** of the task, and its guard metrics;
* the **error kind the adjustment was aimed at**, plus everything else as
  collateral. An adjustment that moves accuracy +0.6% while cutting the target
  error 37% relative is a success, and a rule that only reads the global delta
  cannot see that.

What is replayed is the **decision function over recorded traces**, not a rerun
of ChatDynamics' router: no embedding, no parent retrieval, no model call. Every
result carries that sentence so a holdout number is never read as a production
accuracy claim.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .config import LearningConfig
from .features import FEATURE_NAMES, vector
from .logistic import LogisticModel, fit, sweep_threshold
from .metrics import (
    ErrorRate, binary_counts, binary_error_rates, binary_report, compare_error_rates,
    rounded, topic_pair_metrics,
)
from .policy import (
    ERROR_FALSE_BOT, ERROR_FRAGMENTATION, ERROR_MISSED_BOT, ERROR_MISSED_REPLY,
    ERROR_PREMATURE_REPLY, ERROR_WRONG_MERGE, PolicyCandidate, ReplayDecision,
    bounded_target, candidate_from, clamp_param, decide, normalize_policy, target_error_for,
)
from .samples import BOT, LearningSample, REPLY, TASK_RECIPIENT, TASK_REPLY, TASK_TOPIC
from .topic_learner import TopicPairRow, replay_pairs

VERDICT_ACCEPTED = "accepted"
VERDICT_REJECTED = "rejected"
VERDICT_INSUFFICIENT = "insufficient"

REPLAY_NOTE = "回放的是记录轨迹上的决策函数，不是 ChatDynamics 路由器的完整重跑"
SPLIT_NOTE = "按会话切分，训练集与留出集不共享会话"

# Which single metric decides each task, and which ones act as regression
# guards. The guards are the metrics the plan names (recipient accuracy, topic
# accuracy, reply-decision F1) plus recipient F1, which is what stops an
# "always silent" candidate from looking accurate. Precision and recall are
# reported but never guarded: they trade against each other by construction, so
# gating both would reject every balanced move.
PRIMARY_METRIC = {TASK_RECIPIENT: "accuracy", TASK_TOPIC: "pair_accuracy", TASK_REPLY: "f1"}
GUARD_METRICS = {
    TASK_RECIPIENT: ("accuracy", "f1"),
    TASK_TOPIC: ("pair_accuracy", "f1"),
    TASK_REPLY: ("f1", "accuracy"),
}
TASK_ERROR_KINDS = {
    TASK_RECIPIENT: (ERROR_MISSED_BOT, ERROR_FALSE_BOT),
    TASK_TOPIC: (ERROR_FRAGMENTATION, ERROR_WRONG_MERGE),
    TASK_REPLY: (ERROR_MISSED_REPLY, ERROR_PREMATURE_REPLY),
}
REPORTED_METRICS = {
    TASK_RECIPIENT: ("accuracy", "f1", "precision", "recall"),
    TASK_TOPIC: ("pair_accuracy", "f1", "precision", "recall"),
    TASK_REPLY: ("f1", "precision", "recall", "accuracy"),
}


@dataclass(frozen=True)
class Split:
    train: tuple[LearningSample, ...]
    holdout: tuple[LearningSample, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "train_samples": len(self.train), "holdout_samples": len(self.holdout),
            "train_sessions": len({row.session_hash for row in self.train}),
            "holdout_sessions": len({row.session_hash for row in self.holdout}),
        }


def split_by_session(samples: Sequence[LearningSample], *, holdout_ratio: float) -> Split:
    """Deterministic, balanced, session-level split.

    Sessions are ordered by hash and every `1/ratio`-th session goes to the
    holdout, so the split never depends on dict order, wall time or randomness,
    and small datasets still produce a non-empty holdout when one is possible.
    """
    ordered = sorted({sample.session_hash for sample in samples})
    if len(ordered) < 2:
        return Split(train=tuple(samples), holdout=())
    ratio = max(0.05, min(0.9, float(holdout_ratio)))
    wanted = max(1, min(len(ordered) - 1, int(round(len(ordered) * ratio))))
    stride = len(ordered) / wanted
    holdout_hashes = {ordered[min(len(ordered) - 1, int(index * stride))] for index in range(wanted)}
    train = tuple(sample for sample in samples if sample.session_hash not in holdout_hashes)
    holdout = tuple(sample for sample in samples if sample.session_hash in holdout_hashes)
    return Split(train=train, holdout=holdout)


def _ambient(sample: LearningSample) -> bool:
    return (sample.features.get("ctx_explicit", 0.0) == 0.0
            and sample.features.get("ctx_prior_bot", 1.0) >= 1.0)


def _replay_trace(sample: LearningSample) -> Any:
    """Recover the trace a replay needs, preferring the stored snapshot.

    With `store_raw_trace` on (the default) the full host-shaped snapshot is
    present and used as-is. With it off, the evidence is rebuilt from the
    persisted feature map instead, so a replay still works — the explicit marker
    and the structural outcome are the two fields features cannot carry as
    evidence codes and are re-synthesised from `ctx_explicit` /
    `ctx_bot_targeted`, which is exactly how the host reached its answer.
    """
    from .trace import DecisionTrace, EvidenceFact, parse_decision_trace

    trace: DecisionTrace | None = None
    if sample.trace:
        trace = parse_decision_trace(sample.trace)
        if trace.evidence or not sample.features:
            return trace

    facts = []
    for name, value in sample.features.items():
        if not name.startswith("ev_") or value <= 0:
            continue
        code = name[3:]
        facts.append(EvidenceFact(code=code, family="dialogue", source="feature",
                                  strength=float(sample.features.get(f"st_{code}", 0.0))))
    explicit = sample.features.get("ctx_explicit", 0.0) > 0
    if trace is not None and trace.bot_targeted:
        targeted = True
    else:
        targeted = sample.features.get("ctx_bot_targeted", 0.0) > 0
    if explicit:
        facts.append(EvidenceFact(code="bot_mention" if targeted else "other_mention",
                                  family="recipient", strength=1.0, source="feature"))
    return DecisionTrace(
        evidence=tuple(facts),
        family_contributions={name[4:]: float(value) for name, value in sample.features.items()
                              if name.startswith("fam_") and value},
        contribution_total=float(sample.features.get("base_score", 0.0)),
        bot_targeted=targeted,
        topic_id=trace.topic_id if trace is not None else "",
        topic_confidence=trace.topic_confidence if trace is not None else 0.0,
        topic_ambiguous=trace.topic_ambiguous if trace is not None else False,
        participation_score=trace.participation_score if trace is not None else None,
        participation_level=trace.participation_level if trace is not None else None,
        recipient_confidence=float(sample.features.get("rc_recipient_confidence", 0.0)),
        recipient_ambiguous=bool(sample.features.get("rc_recipient_ambiguous", 0.0)),
        state={"intervening_users": int(sample.features.get("st_intervening_many", 0.0)) * 3},
        identity={"mention": sample.features.get("id_mention", 0.0) > 0,
                  "vocative": sample.features.get("id_vocative", 0.0) > 0,
                  "subject": sample.features.get("id_subject", 0.0) > 0},
    )


def _decision(sample: LearningSample, policy: Mapping[str, float],
              model_score: float | None) -> ReplayDecision:
    return decide(_replay_trace(sample), policy, model_score=model_score)


def _topic_rows(samples: Sequence[LearningSample]) -> list[TopicPairRow]:
    return [TopicPairRow(predicted=sample.predicted, expected=sample.expected,
                         confidence=float(sample.confidence),
                         ambiguous=bool(sample.features.get("rc_topic_ambiguous", 0.0)),
                         candidates=sample.topic_candidates, session_hash=sample.session_hash,
                         selected=sample.selected_topic)
            for sample in samples]


@dataclass(frozen=True)
class TaskScore:
    """Every holdout measurement for one policy on one task."""

    task: str
    support: int
    metrics: dict[str, Any]
    errors: dict[str, ErrorRate]

    def as_dict(self) -> dict[str, Any]:
        return {"task": self.task, "support": self.support, "metrics": self.metrics,
                "errors": {key: row.as_dict() for key, row in self.errors.items()}}


def score_task(rows: Sequence[LearningSample], policy: Mapping[str, float], task: str, *,
               model: LogisticModel | None = None) -> TaskScore:
    """Score one policy on one task's rows.

    This is the single place a policy is turned into numbers, so the evaluator
    and the iterative tuner can never drift into measuring different things.
    """
    policy = normalize_policy(policy)
    if task == TASK_TOPIC:
        metrics = topic_pair_metrics(replay_pairs(_topic_rows(rows),
                                                  policy["topic_commit_threshold"]))
        # Topic exposure is measured in labelled pairs, not in messages.
        pair_count = int(metrics.get("pairs") or 0)
        return TaskScore(task, pair_count, metrics, {
            ERROR_FRAGMENTATION: ErrorRate(ERROR_FRAGMENTATION,
                                           int(metrics.get("fragmentation") or 0), pair_count),
            ERROR_WRONG_MERGE: ErrorRate(ERROR_WRONG_MERGE,
                                         int(metrics.get("wrong_merge") or 0), pair_count),
        })

    if task == TASK_RECIPIENT:
        recipient_pairs: list[tuple[bool, bool]] = []
        for sample in rows:
            score = (model.score(vector(sample.features))
                     if model is not None and _ambient(sample) else None)
            decision = _decision(sample, policy, score)
            recipient_pairs.append((decision.recipient_label == BOT, sample.expected == BOT))
        return TaskScore(task, len(recipient_pairs),
                         binary_report(binary_counts(recipient_pairs)),
                         binary_error_rates(recipient_pairs, positive_kind=ERROR_MISSED_BOT,
                                            negative_kind=ERROR_FALSE_BOT))

    reply_pairs: list[tuple[bool, bool]] = []
    for sample in rows:
        decision = _decision(sample, policy, None)
        reply_pairs.append((decision.reply_label == REPLY, sample.expected == REPLY))
    return TaskScore(task, len(reply_pairs), binary_report(binary_counts(reply_pairs)),
                     binary_error_rates(reply_pairs, positive_kind=ERROR_MISSED_REPLY,
                                        negative_kind=ERROR_PREMATURE_REPLY))


@dataclass
class TaskEvaluation:
    task: str
    train: int = 0
    holdout: int = 0
    primary_metric: str = ""
    target_error: str | None = None
    baseline: dict[str, Any] = field(default_factory=dict)
    candidate: dict[str, Any] = field(default_factory=dict)
    deltas: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, Any] = field(default_factory=dict)
    verdict: str = VERDICT_INSUFFICIENT
    reasons: list[str] = field(default_factory=list)
    fitted: dict[str, Any] | None = None
    learned_scorer: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"task": self.task, "train": self.train, "holdout": self.holdout,
                "primary_metric": self.primary_metric, "target_error": self.target_error,
                "baseline": self.baseline, "candidate": self.candidate, "deltas": self.deltas,
                "errors": self.errors, "verdict": self.verdict, "reasons": list(self.reasons),
                "fitted": self.fitted, "learned_scorer": self.learned_scorer}

    @property
    def primary_delta(self) -> float | None:
        row = self.deltas.get(self.primary_metric)
        value = row.get("delta") if isinstance(row, Mapping) else None
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def target_error_relative(self) -> float | None:
        if self.target_error is None:
            return None
        row = self.errors.get(self.target_error)
        value = row.get("relative") if isinstance(row, Mapping) else None
        return float(value) if isinstance(value, (int, float)) else None


@dataclass
class EvaluationReport:
    verdict: str = VERDICT_INSUFFICIENT
    reasons: list[str] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, TaskEvaluation] = field(default_factory=dict)
    candidate: PolicyCandidate | None = None
    baseline_params: dict[str, float] = field(default_factory=dict)
    target_error: str | None = None
    notes: list[str] = field(default_factory=list)
    dataset: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "reasons": list(self.reasons), "split": dict(self.split),
            "tasks": {name: row.as_dict() for name, row in self.tasks.items()},
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "baseline_params": {key: round(value, 4) for key, value in self.baseline_params.items()},
            "target_error": self.target_error,
            "notes": list(self.notes), "dataset": dict(self.dataset),
        }


def _threshold_or_base(*, param: str, trained: Any, baseline: float, config: LearningConfig) -> float:
    if trained is None:
        return clamp_param(param, baseline)
    return bounded_target(param, baseline, float(trained), config.max_param_delta_ratio)


def _evaluate_recipient(
    split: Split,
    config: LearningConfig,
    baseline: Mapping[str, float],
    changes: dict[str, float],
) -> TaskEvaluation | None:
    train_all = [sample for sample in split.train if sample.task == TASK_RECIPIENT]
    holdout_all = [sample for sample in split.holdout if sample.task == TASK_RECIPIENT]
    if not train_all or not holdout_all:
        return None
    evaluation = TaskEvaluation(task=TASK_RECIPIENT, train=len(train_all), holdout=len(holdout_all),
                                primary_metric=PRIMARY_METRIC[TASK_RECIPIENT])
    ambient_train = [sample for sample in train_all if _ambient(sample)]
    labels = [sample.expected == BOT for sample in ambient_train]
    base_threshold = baseline["strong_addressivity_threshold"]

    # --- the exportable candidate ---------------------------------------
    # ChatDynamics exposes a threshold on its own additive score, not a way to
    # replace the scorer. So the gate is applied to a threshold move on the
    # *recorded* score, bounded by max_param_delta_ratio. This is the only
    # thing that could actually be written into the host configuration.
    threshold = base_threshold
    if ambient_train:
        # Sweep on the same metric the gate judges, so the chosen cut is the
        # training optimum for the thing that is actually being decided.
        additive = sweep_threshold([float(sample.features.get("base_score", 0.0))
                                    for sample in ambient_train], labels,
                                   metric=PRIMARY_METRIC[TASK_RECIPIENT])
        if additive.get("threshold") is not None:
            threshold = _threshold_or_base(param="strong_addressivity_threshold",
                                           trained=additive["threshold"],
                                           baseline=base_threshold, config=config)
        evaluation.fitted = {"ambient_train": len(ambient_train), "calibrated_on": "train",
                             "additive_sweep": {key: value for key, value in additive.items()
                                                if key != "curve"}}
    candidate_policy = {**baseline, "strong_addressivity_threshold": threshold}
    _fill(evaluation, TASK_RECIPIENT, holdout_all, baseline, candidate_policy,
          target_error=target_error_for({"strong_addressivity_threshold": threshold}, baseline))
    changes["strong_addressivity_threshold"] = threshold

    # --- the diagnostic scorer ------------------------------------------
    # A fitted ambient scorer is measured too, but it is reported rather than
    # gated: adopting it would mean ChatDynamics replacing its additive policy,
    # which no configuration key can express today. Reporting its holdout
    # numbers is the evidence that such a key would be worth adding.
    if len(ambient_train) >= max(20, config.min_samples_for_evaluation // 2):
        model = fit([vector(sample.features) for sample in ambient_train], labels,
                    feature_names=FEATURE_NAMES, l2_strength=config.l2_strength,
                    learning_rate=config.learning_rate, iterations=config.max_iterations)
        model_sweep = sweep_threshold(
            [model.score(vector(sample.features)) for sample in ambient_train], labels,
            metric=PRIMARY_METRIC[TASK_RECIPIENT])
        cut = model_sweep.get("threshold")
        if cut is not None:
            scored = score_task(holdout_all,
                                {**baseline, "strong_addressivity_threshold": float(cut)},
                                TASK_RECIPIENT, model=model)
            evaluation.learned_scorer = {**scored.metrics, "cut": float(cut),
                                         "converged": model.converged,
                                         "requires_host_support": "环境层评分替换"}
            learned_fitted = dict(evaluation.fitted or {})
            learned_fitted.update({"loss": rounded(model.loss), "iterations": model.iterations})
            evaluation.fitted = learned_fitted
    return evaluation


def _evaluate_reply(
    split: Split,
    config: LearningConfig,
    baseline: Mapping[str, float],
    changes: dict[str, float],
) -> TaskEvaluation | None:
    del config
    train = [sample for sample in split.train if sample.task == TASK_REPLY]
    holdout = [sample for sample in split.holdout if sample.task == TASK_REPLY]
    if not train or not holdout:
        return None
    evaluation = TaskEvaluation(task=TASK_REPLY, train=len(train), holdout=len(holdout),
                                primary_metric=PRIMARY_METRIC[TASK_REPLY])
    candidate_policy = dict(baseline)
    if TASK_RECIPIENT in changes:
        candidate_policy["strong_addressivity_threshold"] = changes["strong_addressivity_threshold"]
    # Reply admission reads the same cut, so the candidate is scored on the
    # additive path that the recipient candidate would actually export. No
    # learned-scorer diagnostic here: the model's cut is calibrated for
    # recipient accuracy, so scoring reply F1 with it would measure the wrong
    # calibration rather than the scorer.
    _fill(evaluation, TASK_REPLY, holdout, baseline, candidate_policy,
          target_error=target_error_for({"strong_addressivity_threshold":
                                         candidate_policy["strong_addressivity_threshold"]},
                                        baseline))
    return evaluation


def _evaluate_topic(
    split: Split,
    config: LearningConfig,
    baseline: Mapping[str, float],
    changes: dict[str, float],
) -> TaskEvaluation | None:
    train = [sample for sample in split.train if sample.task == TASK_TOPIC]
    holdout = [sample for sample in split.holdout if sample.task == TASK_TOPIC]
    if not train or not holdout:
        return None
    evaluation = TaskEvaluation(task=TASK_TOPIC, train=len(train), holdout=len(holdout),
                                primary_metric=PRIMARY_METRIC[TASK_TOPIC])
    train_rows = _topic_rows(train)
    base_threshold = baseline["topic_commit_threshold"]
    best_key: tuple[float, float] | None = None
    tuned = base_threshold
    for candidate_threshold in _topic_candidates(train_rows, base_threshold):
        metrics = topic_pair_metrics(replay_pairs(train_rows, candidate_threshold))
        score = metrics.get("pair_accuracy")
        if score is None:
            continue
        # Ties break toward the current threshold, so an equally good move is
        # never preferred over leaving the parameter alone.
        key = (round(float(score), 6), -abs(candidate_threshold - base_threshold))
        if best_key is None or key > best_key:
            best_key, tuned = key, candidate_threshold
    threshold = _threshold_or_base(param="topic_commit_threshold", trained=tuned,
                                   baseline=base_threshold, config=config)
    evaluation.fitted = {"tuned_threshold": round(float(tuned), 4), "calibrated_on": "train"}
    if abs(threshold - base_threshold) > 1e-9:
        changes["topic_commit_threshold"] = threshold
    else:
        threshold = base_threshold
    _fill(evaluation, TASK_TOPIC, holdout, baseline,
          {**baseline, "topic_commit_threshold": threshold},
          target_error=target_error_for({"topic_commit_threshold": threshold}, baseline))
    return evaluation


def _fill(evaluation: TaskEvaluation, task: str, holdout: Sequence[LearningSample],
          baseline: Mapping[str, float], candidate: Mapping[str, float],
          *, target_error: str | None) -> None:
    """Score both policies on the holdout and record metrics plus error kinds."""
    base_score = score_task(holdout, baseline, task)
    cand_score = score_task(holdout, candidate, task)
    evaluation.baseline = dict(base_score.metrics)
    evaluation.candidate = dict(cand_score.metrics)
    evaluation.deltas = _deltas(base_score.metrics, cand_score.metrics, REPORTED_METRICS[task])
    evaluation.target_error = target_error
    evaluation.errors = compare_error_rates(base_score.errors, cand_score.errors,
                                            target=target_error)


def _topic_candidates(rows: Sequence[TopicPairRow], base: float) -> list[float]:
    values = {round(float(base), 4)}
    for row in rows:
        values.add(round(min(0.95, max(0.30, row.confidence)), 4))
        for candidate in row.candidates:
            if candidate.score_known:
                values.add(round(min(0.95, max(0.30, candidate.score)), 4))
    ordered = sorted(values)
    if len(ordered) > 41:
        stride = (len(ordered) - 1) / 40
        ordered = sorted({ordered[min(len(ordered) - 1, int(round(index * stride)))]
                          for index in range(41)} | {ordered[0], ordered[-1]})
    return ordered


def _deltas(baseline: Mapping[str, Any], candidate: Mapping[str, Any],
            metrics: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in metrics:
        before, after = baseline.get(name), candidate.get(name)
        if before is None or after is None:
            result[name] = {"before": before, "after": after, "delta": None}
        else:
            result[name] = {"before": before, "after": after,
                            "delta": round(float(after) - float(before), 4)}
    return result


def evaluate_dataset(
    samples: Sequence[LearningSample],
    *,
    config: LearningConfig | None = None,
    baseline_policy: Mapping[str, float] | None = None,
    existing_versions: Sequence[str] = (),
    now: float | None = None,
    tasks: Sequence[str] | None = None,
) -> EvaluationReport:
    """Evaluate a candidate against `baseline_policy` on a session-level holdout.

    `tasks` restricts which tasks may *propose* a parameter move. Every task is
    still scored, because a change aimed at one task can move another through a
    shared parameter — reply admission reads the recipient cut — and that has to
    stay visible rather than being filtered away.
    """
    config = config or LearningConfig()
    allowed = tuple(tasks) if tasks else (TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY)
    baseline = normalize_policy(baseline_policy)
    report = EvaluationReport(baseline_params=baseline)
    report.notes.extend([SPLIT_NOTE, REPLAY_NOTE])

    counts = {task: sum(1 for sample in samples if sample.task == task) for task in PRIMARY_METRIC}
    report.dataset = {"samples": len(samples), "tasks": counts,
                      "sessions": len({sample.session_hash for sample in samples})}
    if not samples:
        report.reasons.append("没有学习样本；先在控制台执行一次导入。")
        return report

    split = split_by_session(samples, holdout_ratio=config.holdout_ratio)
    report.split = split.as_dict()
    if not split.holdout:
        report.reasons.append(
            f"只有 {report.split['train_sessions']} 个会话，无法切出留出集；"
            "评测必须跨会话，否则同一段对话会同时出现在训练和验证里。")
        return report

    changes: dict[str, float] = {}
    builders = ((_evaluate_recipient, TASK_RECIPIENT), (_evaluate_reply, TASK_REPLY),
                (_evaluate_topic, TASK_TOPIC))
    for builder, task in builders:
        before = dict(changes)
        evaluation = builder(split, config, baseline, changes)
        if task not in allowed:
            # A task that may not propose is still scored, but anything it wrote
            # into `changes` is rolled back before the next builder reads it — so
            # an existing parameter can never move because of a disallowed task.
            changes.clear()
            changes.update(before)
        if evaluation is not None:
            report.tasks[task] = evaluation

    if not report.tasks:
        report.reasons.append("本批样本不足以评测任何任务（缺少 bot_targeted / 话题 / 回复标注）。")
        return report

    moved = {name: value for name, value in changes.items()
             if abs(value - baseline.get(name, value)) > 1e-9}
    report.target_error = target_error_for(moved, baseline)
    if moved:
        report.candidate = candidate_from(moved, baseline=baseline,
                                          source="offline_evaluation",
                                          rationale="离线回放在留出集上选出的参数",
                                          evidence={name: row.as_dict()
                                                    for name, row in report.tasks.items()},
                                          existing_versions=existing_versions, now=now)
        report.candidate = report.candidate.with_status("candidate")
    _verdict(report, config)
    return report


def guard_failures(report: EvaluationReport, *, max_regression: float) -> list[str]:
    """Names of the guard metrics that regressed beyond the allowance."""
    failures: list[str] = []
    for row in report.tasks.values():
        for name in GUARD_METRICS.get(row.task, ()):
            delta = row.deltas.get(name)
            value = delta.get("delta") if isinstance(delta, Mapping) else None
            if value is not None and value < -max_regression:
                failures.append(f"{row.task}.{name} {value:+.4f}")
    return failures


def primary_task(report: EvaluationReport) -> TaskEvaluation | None:
    return (report.tasks.get(TASK_RECIPIENT) or report.tasks.get(TASK_TOPIC)
            or (next(iter(report.tasks.values())) if report.tasks else None))


def _verdict(report: EvaluationReport, config: LearningConfig) -> None:
    """Apply the plan's gate: improvement required, regression forbidden."""
    evaluated = list(report.tasks.values())
    underpowered = [row for row in evaluated if row.holdout < config.min_samples_for_evaluation]
    if underpowered:
        names = "、".join(f"{row.task}({row.holdout})" for row in underpowered)
        report.verdict = VERDICT_INSUFFICIENT
        report.reasons.append(
            f"留出集样本不足：「{names}」低于 {config.min_samples_for_evaluation} 条，"
            "不下结论。继续积累标注即可。")
        return

    regressions = guard_failures(report, max_regression=config.evaluation_max_regression)
    if regressions:
        report.verdict = VERDICT_REJECTED
        report.reasons.append("核心指标回退超过允许幅度，拒绝候选：" + "、".join(regressions[:6]))
        if report.candidate is not None:
            report.candidate = report.candidate.with_status("rejected")
        return

    primary = primary_task(report)
    if primary is None:
        report.verdict = VERDICT_INSUFFICIENT
        report.reasons.append("没有可评测的任务。")
        return
    metric = PRIMARY_METRIC.get(primary.task, "accuracy")
    delta = primary.primary_delta
    if delta is None:
        report.verdict = VERDICT_INSUFFICIENT
        report.reasons.append(f"{primary.task}.{metric} 在留出集上未定义，不下结论。")
        return
    if delta >= config.evaluation_min_improvement:
        report.verdict = VERDICT_ACCEPTED
        report.reasons.append(
            f"{primary.task}.{metric} 提升 {delta:+.4f}，达到 "
            f"{config.evaluation_min_improvement:.2%} 门槛，且没有核心指标回退。")
        if report.candidate is not None:
            # `evidence` is an immutable-by-copy mapping, so the verdict is
            # written through a replacement rather than mutated in place.
            evidence = dict(report.candidate.evidence)
            evidence["verdict"] = VERDICT_ACCEPTED
            report.candidate = replace(report.candidate.with_status("candidate"),
                                       evidence=evidence)
        return
    report.verdict = VERDICT_REJECTED
    report.reasons.append(
        f"{primary.task}.{metric} 仅变化 {delta:+.4f}，未达到 "
        f"{config.evaluation_min_improvement:.2%} 门槛。")
    if report.candidate is not None:
        report.candidate = report.candidate.with_status("rejected")


def evaluation_fingerprint(report: EvaluationReport) -> str:
    payload = {name: {"baseline": row.baseline, "candidate": row.candidate}
               for name, row in sorted(report.tasks.items())}
    return hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:32]


__all__ = [
    "GUARD_METRICS", "PRIMARY_METRIC", "REPLAY_NOTE", "REPORTED_METRICS", "SPLIT_NOTE", "Split",
    "TASK_ERROR_KINDS", "TaskEvaluation", "TaskScore", "VERDICT_ACCEPTED", "VERDICT_INSUFFICIENT",
    "VERDICT_REJECTED", "EvaluationReport", "evaluate_dataset", "evaluation_fingerprint",
    "guard_failures", "primary_task", "score_task", "split_by_session",
]
