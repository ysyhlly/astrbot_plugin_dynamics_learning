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
from typing import Any, Callable, Mapping, Sequence

from .bootstrap import METRICS, Unit, bootstrap_delta, human_interval
from .config import LearningConfig
from .features import FEATURE_NAMES, vector
from .logistic import LogisticModel, fit, sweep_threshold
from .metrics import (
    ErrorRate, binary_counts, binary_error_rates, binary_report, compare_error_rates,
    rounded, topic_pair_metrics,
)
from .policy import (
    ERROR_FALSE_BOT, ERROR_FRAGMENTATION, ERROR_MISSED_BOT, ERROR_MISSED_REPLY,
    ERROR_PREMATURE_REPLY, ERROR_UNDELIVERED_REPLY, ERROR_UNSOLICITED_REPLY,
    ERROR_WRONG_MERGE, PARAM_NAMES, STATUS_PROPOSED, STATUS_REJECTED, STATUS_VALIDATED,
    PolicyCandidate, ReplayDecision, baseline_config_hash, bounded_target, candidate_from,
    clamp_param, decide, normalize_policy, target_error_for,
)
from .recommendation import confidence_for
from .samples import (
    BOT, SAMPLE_SCHEMA_VERSION, LearningSample, REPLY, TASK_RECIPIENT, TASK_REPLY_ADMISSION,
    TASK_REPLY_OUTCOME, TASK_TOPIC,
)

from .topic_learner import TopicPairRow, replay_pairs
from .trace import declared_schema_value

VERDICT_ACCEPTED = "accepted"
VERDICT_REJECTED = "rejected"
VERDICT_INSUFFICIENT = "insufficient"

REPLAY_NOTE = "回放的是记录轨迹上的决策函数，不是 ChatDynamics 路由器的完整重跑"
SPLIT_NOTE = "按会话切分，训练集与留出集不共享会话"
OUTCOME_NOTE = ("最终发送结果不可回放：门禁、生成与平台发送都不在记录轨迹里。"
                "这一层只报告已记录的事实与停在哪一环，不参与候选参数的门禁结论")

# Which single metric decides each task, and which ones act as regression
# guards. The guards are the metrics the plan names (recipient accuracy, topic
# accuracy, reply-decision F1) plus recipient F1, which is what stops an
# "always silent" candidate from looking accurate. Precision and recall are
# reported but never guarded: they trade against each other by construction, so
# gating both would reject every balanced move.
PRIMARY_METRIC = {
    TASK_RECIPIENT: "accuracy",
    TASK_TOPIC: "pair_accuracy",
    TASK_REPLY_ADMISSION: "f1",
    TASK_REPLY_OUTCOME: "f1",
}
GUARD_METRICS = {
    TASK_RECIPIENT: ("accuracy", "f1"),
    TASK_TOPIC: ("pair_accuracy", "f1"),
    TASK_REPLY_ADMISSION: ("f1", "accuracy"),
    TASK_REPLY_OUTCOME: ("f1", "accuracy"),
}
# The pair of mistakes each task is judged on. The admission and outcome layers
# get different names on purpose: a suppressed send is an `undelivered_reply`
# at the outcome layer and *not* a `missed_reply` at the admission layer, so no
# single counter can be read as "the router should have replied".
TASK_ERROR_KINDS = {
    TASK_RECIPIENT: (ERROR_MISSED_BOT, ERROR_FALSE_BOT),
    TASK_TOPIC: (ERROR_FRAGMENTATION, ERROR_WRONG_MERGE),
    TASK_REPLY_ADMISSION: (ERROR_MISSED_REPLY, ERROR_PREMATURE_REPLY),
    TASK_REPLY_OUTCOME: (ERROR_UNDELIVERED_REPLY, ERROR_UNSOLICITED_REPLY),
}
REPORTED_METRICS = {
    TASK_RECIPIENT: ("accuracy", "f1", "precision", "recall"),
    TASK_TOPIC: ("pair_accuracy", "f1", "precision", "recall"),
    TASK_REPLY_ADMISSION: ("f1", "precision", "recall", "accuracy"),
    TASK_REPLY_OUTCOME: ("f1", "precision", "recall", "accuracy"),
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


TIME_SPLIT_NOTE = ("按标注时间前向切分：旧时间段训练，新时间段验证；"
                   "同一会话可能跨越边界，这一点与按会话切分不同，也正是它要测的东西")
FORWARD_NOTE = ("前向验证回答的是「换到更晚的数据上还成立吗」，"
                "与按会话切分回答的「换到没见过的会话上还成立吗」是两个问题；"
                "推荐的门槛是两个都通过")


def split_by_time(samples: Sequence[LearningSample], *, ratio: float) -> Split:
    """Oldest 1-ratio trains, newest ratio validates.

    Ordering is by (timestamp, session, message, task) rather than by timestamp
    alone: labels written in the same second would otherwise be split differently
    on two runs of the same corpus, and every recorded result would depend on
    dict order. Samples with no timestamp sort first and land in training, which
    is the conservative direction — an undated message cannot be "later data".
    """
    ordered = sorted(samples, key=lambda item: (item.timestamp, item.session_hash,
                                                item.msg_id, item.task))
    if len(ordered) < 2:
        return Split(train=tuple(ordered), holdout=())
    clamped = max(0.05, min(0.9, float(ratio)))
    cut = int(round(len(ordered) * (1.0 - clamped)))
    cut = max(1, min(len(ordered) - 1, cut))
    return Split(train=tuple(ordered[:cut]), holdout=tuple(ordered[cut:]))


def _ambient(sample: LearningSample) -> bool:
    return (sample.features.get("ctx_explicit", 0.0) == 0.0
            and sample.features.get("ctx_prior_bot", 0.0) >= 1.0)


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
    return decide(_replay_trace(sample), policy, model_score=model_score,
                  score_recorded=sample.contribution_total_recorded)


def _unreplayable(sample: LearningSample) -> bool:
    """Exclude missing scores/context; explicit decisions need neither."""
    if sample.features.get("ctx_explicit", 0.0) >= 0.5:
        return False
    return (not sample.contribution_total_recorded
            or sample.features.get("ctx_prior_bot", -1.0) < 0.0)


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
    # Rows excluded from support because the additive score is unavailable.
    unreplayable: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"task": self.task, "support": self.support, "metrics": self.metrics,
                "unreplayable": self.unreplayable,
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
        unreplayable = 0
        for sample in rows:
            if _unreplayable(sample):
                unreplayable += 1
                continue
            score = (model.score(vector(sample.features))
                     if model is not None and _ambient(sample) else None)
            decision = _decision(sample, policy, score)
            recipient_pairs.append((decision.recipient_label == BOT, sample.expected == BOT))
        return TaskScore(task, len(recipient_pairs),
                         binary_report(binary_counts(recipient_pairs)),
                         binary_error_rates(recipient_pairs, positive_kind=ERROR_MISSED_BOT,
                                            negative_kind=ERROR_FALSE_BOT),
                         unreplayable=unreplayable)

    reply_pairs: list[tuple[bool, bool]] = []
    unreplayable = 0
    for sample in rows:
        if _unreplayable(sample):
            unreplayable += 1
            continue
        decision = _decision(sample, policy, None)
        reply_pairs.append((decision.reply_label == REPLY, sample.expected == REPLY))
    return TaskScore(task, len(reply_pairs), binary_report(binary_counts(reply_pairs)),
                     binary_error_rates(reply_pairs, positive_kind=ERROR_MISSED_REPLY,
                                        negative_kind=ERROR_PREMATURE_REPLY),
                     unreplayable=unreplayable)


def score_outcome(rows: Sequence[LearningSample]) -> TaskScore:
    """Score the outcome layer as a **recording**, not as a replay.

    The prediction is what the host recorded; no policy enters, because none
    can. Between admission and delivery sit the decision gate, the generator and
    the platform adapter, and none of the three is in the trace. A candidate
    that lowers `strong_addressivity_threshold` would admit more turns — and
    what would then happen to them is exactly the part the record does not
    contain. Reporting a number anyway would be inventing the gate's behaviour.

    So this layer answers a different question from the admission one: not "was
    the cut right" but "of the turns that should have been answered, how many
    actually went out, and where did the rest stop".
    """
    pairs = [(row.predicted == REPLY, row.expected == REPLY) for row in rows]
    return TaskScore(TASK_REPLY_OUTCOME, len(pairs), binary_report(binary_counts(pairs)),
                     binary_error_rates(pairs, positive_kind=ERROR_UNDELIVERED_REPLY,
                                        negative_kind=ERROR_UNSOLICITED_REPLY))


@dataclass
class TaskEvaluation:
    task: str
    train: int = 0
    holdout: int = 0
    # Holdout rows excluded because additive score or prior-bot evidence is missing.
    # These rows never contribute to metric support or the sample-size gate.
    unreplayable: int = 0
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
    # Paired session-level bootstrap of the primary metric's delta. Carried
    # beside the point estimate, never instead of it.
    bootstrap: dict[str, Any] = field(default_factory=dict)
    # Per-group movement on the same holdout. Diagnostics only — see
    # `group_diagnostics` for why nothing here may become a local policy.
    strata: dict[str, Any] = field(default_factory=dict)

    @property
    def interval_positive(self) -> bool | None:
        """Does the whole interval sit above zero? `None` when undefined."""
        if self.bootstrap.get("crosses_zero") is None:
            return None
        return not self.bootstrap["crosses_zero"]

    def as_dict(self) -> dict[str, Any]:
        return {"task": self.task, "train": self.train, "holdout": self.holdout,
                "unreplayable": self.unreplayable,
                "primary_metric": self.primary_metric, "target_error": self.target_error,
                "baseline": self.baseline, "candidate": self.candidate, "deltas": self.deltas,
                "errors": self.errors, "verdict": self.verdict, "reasons": list(self.reasons),
                "fitted": self.fitted, "learned_scorer": self.learned_scorer,
                "bootstrap": dict(self.bootstrap),
                "interval": human_interval(self.bootstrap),
                "strata": dict(self.strata)}

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
    # The outcome layer, reported beside the gated tasks rather than inside
    # them: it is measured, not replayed, so it must never be able to move the
    # verdict (see `outcome_layer`).
    outcome: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "reasons": list(self.reasons), "split": dict(self.split),
            "tasks": {name: row.as_dict() for name, row in self.tasks.items()},
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "baseline_params": {key: round(value, 4) for key, value in self.baseline_params.items()},
            "target_error": self.target_error,
            "notes": list(self.notes), "dataset": dict(self.dataset),
            "outcome": dict(self.outcome),
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
    ambient_all = [sample for sample in train_all if _ambient(sample)]
    # A turn whose additive total the host never wrote has `base_score` 0.0 — this
    # plugin's default, not a measurement — so it cannot calibrate a cut on that
    # score. The replay keeps its recorded decision (see `core.policy.decide`) and
    # it is counted next to the fitted numbers rather than swept against.
    ambient_train = [sample for sample in ambient_all if sample.contribution_total_recorded]
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
                             "unscored_train": len(ambient_all) - len(ambient_train),
                             "additive_sweep": {key: value for key, value in additive.items()
                                                if key != "curve"}}
    candidate_policy = {**baseline, "strong_addressivity_threshold": threshold}
    _fill(evaluation, TASK_RECIPIENT, holdout_all, baseline, candidate_policy,
          target_error=target_error_for({"strong_addressivity_threshold": threshold}, baseline),
          config=config)
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
    train = [sample for sample in split.train if sample.task == TASK_REPLY_ADMISSION]
    holdout = [sample for sample in split.holdout if sample.task == TASK_REPLY_ADMISSION]
    if not train or not holdout:
        return None
    evaluation = TaskEvaluation(task=TASK_REPLY_ADMISSION, train=len(train), holdout=len(holdout),
                                primary_metric=PRIMARY_METRIC[TASK_REPLY_ADMISSION])
    candidate_policy = dict(baseline)
    # The key is the *parameter*, not the task: `changes` is keyed by
    # parameter name, so testing `TASK_RECIPIENT in changes` was always false
    # and this task was silently scored baseline-against-baseline — every delta
    # exactly 0.0, and a guard check that could never fire.
    if "strong_addressivity_threshold" in changes:
        candidate_policy["strong_addressivity_threshold"] = changes["strong_addressivity_threshold"]
    # Reply admission reads the same cut, so the candidate is scored on the
    # additive path that the recipient candidate would actually export. No
    # learned-scorer diagnostic here: the model's cut is calibrated for
    # recipient accuracy, so scoring reply F1 with it would measure the wrong
    # calibration rather than the scorer.
    _fill(evaluation, TASK_REPLY_ADMISSION, holdout, baseline, candidate_policy,
          target_error=target_error_for({"strong_addressivity_threshold":
                                         candidate_policy["strong_addressivity_threshold"]},
                                        baseline),
          config=config)
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
          target_error=target_error_for({"topic_commit_threshold": threshold}, baseline),
          config=config)
    return evaluation


def _fill(evaluation: TaskEvaluation, task: str, holdout: Sequence[LearningSample],
          baseline: Mapping[str, float], candidate: Mapping[str, float],
          *, target_error: str | None, config: LearningConfig | None = None,
          model: LogisticModel | None = None) -> None:
    """Score both policies on the holdout and record metrics, errors and an interval."""
    base_score = score_task(holdout, baseline, task, model=model)
    cand_score = score_task(holdout, candidate, task, model=model)
    evaluation.baseline = dict(base_score.metrics)
    evaluation.candidate = dict(cand_score.metrics)
    # Same rows on both sides: this is a property of the record, not of the policy.
    evaluation.unreplayable = base_score.unreplayable
    evaluation.deltas = _deltas(base_score.metrics, cand_score.metrics, REPORTED_METRICS[task])
    evaluation.target_error = target_error
    evaluation.errors = compare_error_rates(base_score.errors, cand_score.errors,
                                            target=target_error)
    settings = config or LearningConfig()
    units = _units(holdout, task, baseline, candidate, model=model)
    evaluation.bootstrap = bootstrap_delta(
        units,
        metric=PRIMARY_METRIC[task],
        iterations=settings.bootstrap_iterations,
        seed=settings.bootstrap_seed,
        alpha=settings.bootstrap_alpha,
    )
    evaluation.strata = group_diagnostics(units, task=task, min_support=settings.group_min_support)


def group_diagnostics(units: Sequence[Unit], *, task: str, min_support: int) -> dict[str, Any]:
    """How the candidate moved in each group, and why none of it is a policy.

    The plan's order is deliberate: one global policy first, per-group numbers as
    *diagnostics*. A group that regresses under a globally-good change is the
    most useful thing this report can say, and it is also the most dangerous
    thing to act on: the groups are sessions, their support is a handful of
    labels, and a per-group threshold fitted on a handful of labels is a
    memorised conversation wearing a parameter's name.

    So the numbers are reported, the eligible groups are named (the ones whose
    support would allow a bounded local delta), and nothing is proposed. Whether
    a per-group policy is ever worth fitting is a decision for the day the host
    exposes a cross-session group identity and the corpus is large enough to
    measure one.
    """
    function = METRICS.get(PRIMARY_METRIC.get(task, "accuracy"))
    rows: list[dict[str, Any]] = []
    improved = regressed = unchanged = ineligible = 0
    for unit in units:
        support = int(sum(unit.baseline.values()))
        baseline = function(unit.baseline) if function is not None else None
        candidate = function(unit.candidate) if function is not None else None
        row: dict[str, Any] = {"group": unit.key, "support": support}
        if baseline is None or candidate is None or support < min_support:
            row.update({"eligible": False, "delta": None,
                        "reason": ("样本不足" if support < min_support
                                   else "该组指标未定义")})
            ineligible += 1
            rows.append(row)
            continue
        delta = round(candidate - baseline, 6)
        direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
        if direction == "improved":
            improved += 1
        elif direction == "regressed":
            regressed += 1
        else:
            unchanged += 1
        row.update({"eligible": True, "delta": delta, "baseline": round(baseline, 6),
                    "candidate": round(candidate, 6), "direction": direction})
        rows.append(row)
    rows.sort(key=lambda item: (item["delta"] is None, item.get("delta") or 0.0))
    evaluated = improved + regressed + unchanged
    notes: list[str] = []
    if not evaluated:
        notes.append(f"没有任何分组达到 {min_support} 条支撑，分组诊断不出结论。")
    else:
        notes.append(f"{evaluated} 个分组可比较：{improved} 个改善、{regressed} 个回退、"
                     f"{unchanged} 个持平。")
        if regressed:
            notes.append("回退最多的分组见下表；全局变好不等于每个群都变好。")
    if ineligible:
        notes.append(f"另有 {ineligible} 个分组样本不足，没有给出数值。")
    notes.append("分组诊断只报告，不产生本地策略：当前作用域就是会话，"
                 "在一个会话上拟合出的阈值是记住对话，不是学会相处。")
    return {"task": task, "min_support": min_support, "groups": rows[:64],
            "evaluated": evaluated, "improved": improved, "regressed": regressed,
            "unchanged": unchanged, "ineligible": ineligible, "notes": notes}


def _binary_predictions(rows: Sequence[LearningSample], task: str,
                        policy: Mapping[str, float],
                        model: LogisticModel | None) -> list[tuple[str, bool, bool]]:
    """`(session, predicted, expected)` for a binary task under one policy."""
    result: list[tuple[str, bool, bool]] = []
    for sample in rows:
        if _unreplayable(sample):
            continue
        if task == TASK_RECIPIENT:
            score = (model.score(vector(sample.features))
                     if model is not None and _ambient(sample) else None)
            decision = _decision(sample, policy, score)
            result.append((sample.session_hash, decision.recipient_label == BOT,
                           sample.expected == BOT))
        else:
            decision = _decision(sample, policy, None)
            result.append((sample.session_hash, decision.reply_label == REPLY,
                           sample.expected == REPLY))
    return result


def _count_pairs(rows: Sequence[tuple[str, bool, bool]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for session, predicted, expected in rows:
        counts = grouped.setdefault(session, {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0})
        key = ("tp" if predicted and expected else "fn" if expected
               else "fp" if predicted else "tn")
        counts[key] += 1.0
    return grouped


def _units(rows: Sequence[LearningSample], task: str, baseline: Mapping[str, float],
           candidate: Mapping[str, float],
           model: LogisticModel | None = None) -> list[Unit]:
    """Per-session counts for the paired bootstrap, one unit per session.

    The unit is the session because that is the level the corpus is clustered
    at — see `core/bootstrap.py`. Building them here rather than inside the
    bootstrap keeps the resampling code ignorant of what a learning sample is.
    """
    if task == TASK_TOPIC:
        grouped: dict[str, list[LearningSample]] = {}
        for sample in rows:
            grouped.setdefault(sample.session_hash, []).append(sample)
        units: list[Unit] = []
        for session, group in grouped.items():
            pair_rows = _topic_rows(group)
            base = topic_pair_metrics(replay_pairs(pair_rows,
                                                   baseline["topic_commit_threshold"]))
            cand = topic_pair_metrics(replay_pairs(pair_rows,
                                                   candidate["topic_commit_threshold"]))
            keys = ("pairs", "true_positive", "wrong_merge", "fragmentation")
            units.append(Unit(key=session,
                              baseline={key: float(base.get(key) or 0) for key in keys},
                              candidate={key: float(cand.get(key) or 0) for key in keys}))
        return units

    base_counts = _count_pairs(_binary_predictions(rows, task, baseline, model))
    cand_counts = _count_pairs(_binary_predictions(rows, task, candidate, model))
    return [Unit(key=session, baseline=base_counts[session], candidate=cand_counts[session])
            for session in sorted(base_counts)]


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
    splitter: Callable[[Sequence[LearningSample]], Split] | None = None,
    split_note: str = SPLIT_NOTE,
    host_version: str | None = None,
) -> EvaluationReport:
    """Evaluate a candidate against `baseline_policy` on a session-level holdout.

    `tasks` restricts which tasks may *propose* a parameter move. Every task is
    still scored, because a change aimed at one task can move another through a
    shared parameter — reply admission reads the recipient cut — and that has to
    stay visible rather than being filtered away.
    """
    config = config or LearningConfig()
    allowed = tuple(tasks) if tasks else (TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY_ADMISSION)
    baseline = normalize_policy(baseline_policy)
    report = EvaluationReport(baseline_params=baseline)
    report.notes.extend([split_note, REPLAY_NOTE])

    counts = {task: sum(1 for sample in samples if sample.task == task) for task in PRIMARY_METRIC}
    report.dataset = {"samples": len(samples), "tasks": counts,
                      "sessions": len({sample.session_hash for sample in samples})}
    if not samples:
        report.reasons.append("没有学习样本；先在控制台执行一次导入。")
        return report

    split = (splitter(samples) if splitter is not None
             else split_by_session(samples, holdout_ratio=config.holdout_ratio))
    report.split = {**split.as_dict(), "kind": "time" if splitter is not None else "session"}
    if not split.holdout:
        report.reasons.append(
            f"只有 {report.split['train_sessions']} 个会话，无法切出留出集；"
            "评测必须跨会话，否则同一段对话会同时出现在训练和验证里。")
        return report

    changes: dict[str, float] = {}
    builders = ((_evaluate_recipient, TASK_RECIPIENT),
                (_evaluate_reply, TASK_REPLY_ADMISSION),
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

    report.outcome = outcome_layer(samples, split=split)

    if not report.tasks:
        report.reasons.append("本批样本不足以评测任何任务（缺少 bot_targeted / 话题 / 回复标注）。")
        return report

    moved = {name: value for name, value in changes.items()
             if abs(value - baseline.get(name, value)) > 1e-9}
    report.target_error = target_error_for(moved, baseline)
    if moved:
        candidate = candidate_from(moved, baseline=baseline,
                                   source="offline_evaluation",
                                   rationale="离线回放在留出集上选出的参数",
                                   evidence={name: row.as_dict()
                                             for name, row in report.tasks.items()},
                                   existing_versions=existing_versions, now=now)
        report.candidate = candidate.with_fields(
            status=STATUS_PROPOSED,
            training_dataset=dataset_record(report.dataset),
            holdout_result=holdout_record(report),
            target_error=report.target_error or "",
            collateral_regressions=tuple(guard_failures(
                report, max_regression=config.evaluation_max_regression)),
            confidence=confidence_for(sum(max(0, row.holdout - row.unreplayable)
                                          for row in report.tasks.values()),
                                      min_samples=config.min_samples_for_evaluation),
            compatibility=contract_compatibility(samples),
            target=target_facts(host_version, baseline),
        )
    _verdict(report, config, now=now)
    return report


def dataset_record(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """A fingerprint of the corpus a policy was fitted on.

    Without it, a recorded result cannot be compared to anything later: "F1
    +0.8%" means nothing without knowing which samples produced it, and the
    corpus changes on every import.
    """
    raw_tasks = dataset.get("tasks")
    tasks: dict[str, Any] = dict(raw_tasks) if isinstance(raw_tasks, Mapping) else {}
    payload: dict[str, Any] = {
        "samples": int(dataset.get("samples") or 0),
        "sessions": int(dataset.get("sessions") or 0),
        "tasks": tasks,
    }
    digest = hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
    return {**payload, "fingerprint": digest}


def holdout_record(report: EvaluationReport) -> dict[str, Any]:
    """The per-task holdout numbers, in the shape a policy record keeps."""
    return {
        "split": dict(report.split),
        "tasks": {
            task: {
                "holdout": row.holdout,
                "unreplayable": row.unreplayable,
                "primary_metric": row.primary_metric,
                "baseline": dict(row.baseline),
                "candidate": dict(row.candidate),
                "deltas": dict(row.deltas),
                "target_error": row.target_error,
                "errors": dict(row.errors),
                "verdict": row.verdict,
            }
            for task, row in report.tasks.items()
        },
    }


def contract_compatibility(samples: Sequence[LearningSample]) -> dict[str, Any]:
    """What *data* this policy was fitted on — the provenance half.

    A policy is only as good as the fields it was fitted on, and those fields
    are a property of the host's trace schema. Recording them means a policy
    fitted on schema 2 rows cannot be read later as if it had seen outcomes it
    never had.

    `trace_schema_version` here is the **host's** number, not this plugin's
    reader revision. A mixed corpus is not hidden behind it: the distribution
    travels beside it, because "trained on schema 3" would over-claim for a
    corpus that was ninety percent schema 2.
    """
    schemas: dict[str, int] = {}
    evidence: dict[str, int] = {}
    outcomes = 0
    highest = 0
    for sample in samples:
        trace = sample.trace if isinstance(sample.trace, Mapping) else {}
        key = str(declared_schema_value(trace) if declared_schema_value(trace) is not None
                  else "missing")
        schemas[key] = schemas.get(key, 0) + 1
        evidence[sample.candidate_evidence] = evidence.get(sample.candidate_evidence, 0) + 1
        if key.isdigit():
            highest = max(highest, int(key))
        if sample.outcome.recorded:
            outcomes += 1
    return {
        "trace_schema_version": highest or None,
        "trace_schema_versions": dict(sorted(schemas.items())),
        "sample_schema_version": SAMPLE_SCHEMA_VERSION,
        "candidate_evidence": dict(sorted(evidence.items())),
        "outcome_recorded": outcomes,
        "parameter_specs": list(PARAM_NAMES),
    }


def target_facts(host_version: str | None,
                 baseline: Mapping[str, float]) -> dict[str, Any]:
    """What *host* this policy is aimed at — the adoption-gating half.

    `validated_host_versions` starts as the single version observed during
    training, or as an empty list when the host did not report one. An empty list
    is a finding, not a detail: a consumer that cannot verify which host version
    produced this policy has no basis for `active`, and the published file says
    so instead of implying a match.

    Widening the list later is a *data* change — reparsing the same policy
    against a newer host and appending the version — not a protocol change, and
    deliberately not a SemVer comparison. 1.7.0 -> 1.7.1 can move a scoring
    order or a gate sequence, which changes the distribution every threshold in
    this file was calibrated against; only a replay can establish that it did
    not.
    """
    version = str(host_version)[:64] if isinstance(host_version, str) and host_version else None
    return {
        "chat_dynamics_version": version,
        "baseline_config_hash": baseline_config_hash(baseline),
        "validated_host_versions": [version] if version else [],
    }


def outcome_layer(samples: Sequence[LearningSample], *, split: Split) -> dict[str, Any]:
    """What actually happened, measured on both splits and gated by nothing.

    Deliberately outside `report.tasks`: everything in `tasks` is replayed and
    therefore able to move the verdict, and this layer cannot be replayed. Its
    job is to make the *reason* a should-have-replied turn produced no reply
    visible — 门禁 / 生成 / 发送 / 从未进入 — instead of leaving one
    `missed_reply` count to stand for all four.
    """
    rows = [sample for sample in samples if sample.task == TASK_REPLY_OUTCOME]
    labelled = sum(1 for sample in samples if sample.task == TASK_REPLY_ADMISSION)
    holdout = [sample for sample in split.holdout if sample.task == TASK_REPLY_OUTCOME]
    stages: dict[str, int] = {}
    reasons: list[str] = []
    for sample in rows:
        outcome = sample.outcome
        if outcome.is_delivered:
            continue
        key = outcome.stage if outcome.recorded else "unavailable"
        stages[key] = stages.get(key, 0) + 1
        if outcome.suppression_reason:
            stages[f"reason:{outcome.suppression_reason}"] = (
                stages.get(f"reason:{outcome.suppression_reason}", 0) + 1)
    scored = score_outcome(holdout).as_dict() if holdout else None
    if not rows:
        reasons.append(
            "没有任何消息记录了最终发送结果（schema 3 的 outcome 字段）："
            "被作息/降温压掉的发送与真正的漏回复在这里无法区分，"
            "回复层的错误只能读作「路由准入」，不能读作「最终有没有回复」")
    elif labelled and len(rows) < labelled:
        reasons.append(f"只有 {len(rows)}/{labelled} 条回复标注消息记录了最终结果，"
                       "其余只能按准入层解释")
    reasons.append(OUTCOME_NOTE)
    return {
        "replayable": False,
        "support": len(rows),
        "labelled": labelled,
        "coverage": rounded(len(rows) / labelled) if labelled else None,
        "holdout_support": len(holdout),
        "holdout": scored,
        "stages": dict(sorted(stages.items(), key=lambda item: (-item[1], item[0]))),
        "reasons": reasons,
    }


def forward_evaluation(
    samples: Sequence[LearningSample],
    *,
    config: LearningConfig | None = None,
    baseline_policy: Mapping[str, float] | None = None,
    existing_versions: Sequence[str] = (),
    now: float | None = None,
    host_version: str | None = None,
) -> EvaluationReport:
    """The same evaluation, on a time-ordered split instead of a session one."""
    config = config or LearningConfig()
    return evaluate_dataset(
        samples,
        config=config,
        baseline_policy=baseline_policy,
        existing_versions=existing_versions,
        now=now,
        splitter=lambda rows: split_by_time(rows, ratio=config.forward_holdout_ratio),
        split_note=TIME_SPLIT_NOTE,
        host_version=host_version,
    )


@dataclass
class PromotionCheck:
    """Two holdouts, one interval, one answer."""

    verdict: str = VERDICT_INSUFFICIENT
    reasons: list[str] = field(default_factory=list)
    session_verdict: str = VERDICT_INSUFFICIENT
    forward_verdict: str | None = None
    interval: str = ""
    interval_ok: bool | None = None

    @property
    def accepted(self) -> bool:
        return self.verdict == VERDICT_ACCEPTED

    def as_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "reasons": list(self.reasons),
                "session_verdict": self.session_verdict,
                "forward_verdict": self.forward_verdict,
                "interval": self.interval, "interval_ok": self.interval_ok}


def promotion_check(session: EvaluationReport | None,
                    forward: EvaluationReport | None,
                    *, config: LearningConfig) -> PromotionCheck:
    """The gate the plan asks for: session holdout, forward holdout, and an interval.

    Each clause removes a different way of being wrong:

    `session holdout`   it is not memorising one conversation
    `forward holdout`   it did not stop being true when the group changed habits
    `interval`          the improvement is bigger than the resampling noise

    A check that cannot run reports `insufficient`, never `accepted`. "We did not look"
    and "we looked and it was fine" must not produce the same verdict, or the gate
    is decoration.
    """
    check = PromotionCheck(session_verdict=(session.verdict if session else VERDICT_INSUFFICIENT))
    if session is None:
        check.reasons.append("没有会话留出集评测结果，无法下结论。")
        return check
    if config.require_forward_validation:
        if forward is None or forward.verdict == VERDICT_INSUFFICIENT:
            check.reasons.append("没有可用的前向验证：只有会话留出集通过的候选不下结论。")
            return check
        check.forward_verdict = forward.verdict
        if forward.verdict != VERDICT_ACCEPTED:
            check.verdict = VERDICT_REJECTED
            check.reasons.append("前向验证未通过：" + (forward.reasons[0] if forward.reasons
                                                       else "更晚的数据上没有复现收益"))
            return check
    else:
        check.reasons.append("前向验证被配置关闭：本次结论只基于会话留出集。")
    if session.verdict != VERDICT_ACCEPTED:
        check.verdict = session.verdict
        check.reasons.extend(session.reasons[:2])
        return check

    primary = primary_task(session)
    if primary is not None:
        check.interval = human_interval(primary.bootstrap)
        check.interval_ok = primary.interval_positive
    if config.require_ci_positive and check.interval_ok is False:
        check.verdict = VERDICT_REJECTED
        check.reasons.append(
            f"置信区间跨 0（{check.interval}）：这个提升无法与重采样噪声区分，"
            "不给出可采纳结论。")
        return check
    if config.require_ci_positive and check.interval_ok is None:
        check.verdict = VERDICT_INSUFFICIENT
        check.reasons.append("置信区间无法计算：重采样单元不足，不下结论。")
        return check
    check.verdict = VERDICT_ACCEPTED
    clauses = ["会话留出集通过"]
    if check.forward_verdict == VERDICT_ACCEPTED:
        clauses.append("前向验证通过")
    if check.interval:
        clauses.append(check.interval)
    check.reasons.append("；".join(clauses) + "。")
    return check


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


def _verdict(report: EvaluationReport, config: LearningConfig,
             *, now: float | None = None) -> None:
    """Apply the plan's gate: improvement required, regression forbidden.

    `now` is threaded through rather than read from the clock inside, so two
    evaluations of the same batch under the same `now` produce byte-identical
    records. A verdict that embedded a wall-clock timestamp would make the
    reproducibility test — and any fingerprint built on it — meaningless.
    """
    evaluated = list(report.tasks.values())
    underpowered = [row for row in evaluated if max(0, row.holdout - row.unreplayable)
                    < config.min_samples_for_evaluation]
    if underpowered:
        names = "、".join(f"{row.task}({max(0, row.holdout - row.unreplayable)})" for row in underpowered)
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
            report.candidate = report.candidate.with_status(
                STATUS_REJECTED, now=now, reason="留出集核心指标回退")
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
            # `validated`, not `promoted`: the evaluator can prove an offline
            # improvement and nothing else. Promotion is a separate step that
            # requires the shadow observation this plugin cannot perform.
            report.candidate = replace(
                report.candidate.with_status(STATUS_VALIDATED, now=now,
                                             reason="留出集通过，且无核心指标回退"),
                evidence=evidence)
        return
    report.verdict = VERDICT_REJECTED
    report.reasons.append(
        f"{primary.task}.{metric} 仅变化 {delta:+.4f}，未达到 "
        f"{config.evaluation_min_improvement:.2%} 门槛。")
    if report.candidate is not None:
        report.candidate = report.candidate.with_status(
            STATUS_REJECTED, now=now, reason="留出集提升未达门槛")


def evaluation_fingerprint(report: EvaluationReport) -> str:
    payload = {name: {"baseline": row.baseline, "candidate": row.candidate}
               for name, row in sorted(report.tasks.items())}
    return hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:32]


__all__ = [
    "FORWARD_NOTE", "GUARD_METRICS", "OUTCOME_NOTE", "PRIMARY_METRIC", "REPLAY_NOTE",
    "REPORTED_METRICS", "SPLIT_NOTE", "TIME_SPLIT_NOTE", "PromotionCheck", "Split",
    "TASK_ERROR_KINDS", "TaskEvaluation", "TaskScore", "VERDICT_ACCEPTED", "VERDICT_INSUFFICIENT",
    "VERDICT_REJECTED", "EvaluationReport", "contract_compatibility", "dataset_record",
    "evaluate_dataset", "evaluation_fingerprint", "forward_evaluation", "group_diagnostics",
    "guard_failures", "holdout_record", "outcome_layer", "primary_task", "promotion_check",
    "score_outcome", "score_task", "split_by_session", "split_by_time", "target_facts",
]
