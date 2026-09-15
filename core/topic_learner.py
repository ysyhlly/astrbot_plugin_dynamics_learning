"""v0.3 Topic Learning: merge / split / recall over human labels.

Two directions matter and they are **not** symmetric in what can be replayed:

* **Tightening** (raise the commit threshold) only ever removes an assignment, so
  it is fully replayable from the recorded `topic_confidence`.
* **Relaxing** (lower the threshold) would have to *add* an assignment, which is
  only reconstructible for messages whose annotation snapshot recorded
  `topic_candidates`. Messages without a recorded candidate distribution keep
  their recorded label, so a relaxation replay is a lower bound on the effect,
  never an estimate of the full one.

The learner therefore proposes an evaluable candidate only when the data
supports the direction it wants to move in. A fragmentation-dominant batch
produces a diagnostic naming the missing data instead of a number that cannot
be defended.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .candidates import (
    ATTRIBUTION_CANDIDATE_MISS, ATTRIBUTION_NOT_RECORDED, ATTRIBUTION_RANKING_ERROR,
    KIND_LABELLED, KIND_NEW_TOPIC, KIND_UNLABELLED, CandidateObservation, CandidateRecord,
    parse_candidates, summarise as summarise_candidates,
)
from .config import LearningConfig
from .metrics import PAIR_NOTE, error_distribution, label_confusion, topic_pair_metrics
from .policy import BASE_POLICY, PARAM_SPECS, bounded_target, sweep_values
from .recommendation import Recommendation, config_recommendation, diagnostic
from .samples import LearningSample, TASK_TOPIC, UNASSIGNED

UNASSIGNED_DISPLAY = "(未归属)"


@dataclass(frozen=True)
class TopicPairRow:
    predicted: str
    expected: str
    confidence: float
    ambiguous: bool
    candidates: tuple[CandidateRecord, ...]
    session_hash: str
    selected: str = ""

    @property
    def committed(self) -> bool:
        return bool(self.predicted) and not self.ambiguous


@dataclass
class TopicLearning:
    samples: int = 0
    sessions: int = 0
    unassigned_samples: int = 0
    samples_with_candidates: int = 0
    pair_metrics: dict[str, Any] = field(default_factory=dict)
    error_types: dict[str, int] = field(default_factory=dict)
    confusion: list[dict[str, Any]] = field(default_factory=list)
    threshold_sweep: dict[str, Any] = field(default_factory=dict)
    candidate_metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[Recommendation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": TASK_TOPIC,
            "samples": self.samples, "sessions": self.sessions,
            "unassigned_samples": self.unassigned_samples,
            "samples_with_candidates": self.samples_with_candidates,
            "pair_metrics": self.pair_metrics, "error_types": self.error_types,
            "confusion": self.confusion,
            "threshold_sweep": {key: value for key, value in self.threshold_sweep.items()
                                if key != "curve"},
            "candidate_metrics": self.candidate_metrics,
            "recommendations": [row.as_dict() for row in self.recommendations],
            "notes": list(self.notes),
        }


def pair_rows(samples: Sequence[LearningSample]) -> list[TopicPairRow]:
    """Label / confidence / candidate view of topic samples, for replay.

    Public because contract health asks the same question the sweep does: can a
    threshold move change this row's label? Deriving it a second time in
    `core/quality.py` would let the two answers drift apart.
    """
    rows: list[TopicPairRow] = []
    for sample in samples:
        rows.append(TopicPairRow(
            predicted=sample.predicted,
            expected=sample.expected,
            confidence=float(sample.confidence),
            ambiguous=bool(sample.features.get("rc_topic_ambiguous", 0.0)),
            candidates=sample.topic_candidates,
            session_hash=sample.session_hash,
            selected=sample.selected_topic,
        ))
    return rows


def candidate_observations(samples: Sequence[LearningSample]) -> list[CandidateObservation]:
    """Labeled topic samples reduced to what candidate recall needs.

    Every sample becomes an observation, including the two kinds that cannot be
    scored, so the attribution table totals the corpus instead of silently
    shrinking it:

    * a `NEW` singleton is a topic that by definition did not exist yet, so it
      cannot have been a candidate and blaming generation for it would blame the
      host for being right;
    * a sample with no usable topic label has no candidate question at all.

    `recorded` is taken from the sample, not inferred from a non-empty list: an
    empty list the host really wrote is a generation miss, while a missing field
    is not evidence of anything.
    """
    rows: list[CandidateObservation] = []
    for sample in samples:
        candidates = sample.topic_candidates
        if sample.expected.startswith("NEW:"):
            kind = KIND_NEW_TOPIC
        elif not sample.expected or sample.expected == UNASSIGNED:
            kind = KIND_UNLABELLED
        else:
            kind = KIND_LABELLED
        rows.append(CandidateObservation(
            expected=sample.expected,
            selected=sample.selected_topic,
            candidates=candidates,
            recorded=sample.topic_candidates_recorded,
            kind=kind,
            dropped=parse_candidates(
                sample.trace.get("topic_candidates") if isinstance(sample.trace, Mapping)
                else None).dropped,
        ))
    return rows


def replay_label(row: TopicPairRow, threshold: float, *, allow_relax: bool = True) -> str:
    """The label the commit threshold would produce for one message."""
    if row.committed:
        return row.predicted if row.confidence >= threshold else UNASSIGNED
    if not allow_relax or row.ambiguous:
        return UNASSIGNED
    best: CandidateRecord | None = None
    for candidate in row.candidates:
        # A candidate whose score was never recorded cannot be replayed against
        # a threshold: treating an unknown score as zero would invent a reason
        # the host never had for not choosing it.
        if not candidate.score_known:
            continue
        if candidate.score >= threshold and (best is None or candidate.score > best.score):
            best = candidate
    return best.topic_id if best else UNASSIGNED


def replay_can_move(row: TopicPairRow) -> bool:
    """Whether a commit-threshold move can change this row's label at all.

    Tightening can always drop a committed assignment, so every committed row
    responds to the threshold. Relaxing needs something to relax *to*: a
    candidate whose score the host actually recorded. A row with neither stays at
    its recorded label under every threshold — a fact about the corpus worth
    reporting, not a row to quietly average in.
    """
    if row.committed:
        return True
    if row.ambiguous:
        return False
    return any(candidate.score_known for candidate in row.candidates)


def replay_pairs(rows: Sequence[TopicPairRow], threshold: float, *,
                 allow_relax: bool = True) -> list[list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.session_hash, []).append(
            (replay_label(row, threshold, allow_relax=allow_relax), row.expected))
    return list(grouped.values())


def recorded_pairs(rows: Sequence[TopicPairRow]) -> list[list[tuple[str, str]]]:
    """Observed assignments, without counterfactual threshold relaxation."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.session_hash, []).append((row.predicted, row.expected))
    return list(grouped.values())


def _display_confusion(samples: Sequence[LearningSample]) -> list[dict[str, Any]]:
    rows = label_confusion((sample.predicted, sample.expected) for sample in samples)
    for row in rows:
        if row["predicted"] == UNASSIGNED:
            row["predicted"] = UNASSIGNED_DISPLAY
    return rows


def learn(samples: Sequence[LearningSample], *, config: LearningConfig | None = None) -> TopicLearning:
    config = config or LearningConfig()
    subset = [sample for sample in samples if sample.task == TASK_TOPIC]
    report = TopicLearning(
        samples=len(subset),
        sessions=len({sample.session_hash for sample in subset}),
        unassigned_samples=sum(1 for sample in subset if sample.predicted == UNASSIGNED),
        # Relaxation replay needs actual scored candidates, so this counts
        # non-empty sets. "Did the host record the field at all?" is a different
        # question and lives in candidate_recall.recorded / not_recorded.
        samples_with_candidates=sum(1 for sample in subset if sample.topic_candidates),
        error_types=error_distribution(subset),
    )
    if not subset:
        report.notes.append("没有话题标注样本；先在本体回放页标注 expected_topic。")
        return report

    rows = pair_rows(subset)
    report.confusion = _display_confusion(subset)
    report.pair_metrics = topic_pair_metrics(recorded_pairs(rows))
    report.candidate_metrics = summarise_candidates(candidate_observations(subset))
    report.notes.append(PAIR_NOTE)
    report.notes.extend(_candidate_notes(report))
    if report.unassigned_samples:
        report.notes.append(
            f"{report.unassigned_samples} 条标注的预测为未归属，按本体 routing_metrics 的口径计入误拆分。")
    if report.pair_metrics.get("pairs", 0) == 0:
        report.notes.append("本批标注不足以构成会话内配对，配对指标为空。")

    sweep = _sweep(rows, config)
    report.threshold_sweep = sweep
    report.recommendations = _recommendations(report, rows, config, sweep)
    return report


def _candidate_notes(report: TopicLearning) -> list[str]:
    """Say which layer the residual topic error actually lives in.

    This is the difference between "Embedding 没召回正确 topic" and "Embedding
    召回了但规则选错了" — two problems with opposite fixes.
    """
    metrics = report.candidate_metrics or {}
    recall = metrics.get("candidate_recall") or {}
    selection = metrics.get("selection_accuracy") or {}
    attribution = metrics.get("attribution") or {}
    notes: list[str] = []
    coverage = recall.get("coverage")
    if coverage is not None and coverage < 1.0:
        notes.append(
            f"只有 {recall.get('recorded', 0)}/{recall.get('total', 0)} 条标注记录了候选集"
            f"（覆盖率 {coverage:.1%}），候选指标只在该子集上计算。")
    recall_at_3 = recall.get("recall_at_3")
    selection_accuracy = selection.get("accuracy")
    counts = attribution.get("counts") or {}
    if recall_at_3 is not None:
        # Selection accuracy is undefined when nothing was ever offered, which
        # is not the same as scoring zero: formatting None as a percentage used
        # to raise here, on any batch where candidates existed but none was ever
        # the right one.
        shown = "无法计算" if selection_accuracy is None else f"{selection_accuracy:.1%}"
        notes.append(f"候选召回 Recall@3 {recall_at_3:.1%}；"
                     f"选中准确率 {shown}（仅统计正确话题在候选集内的样本）。")
    miss = int(counts.get(ATTRIBUTION_CANDIDATE_MISS, 0))
    ranking = int(counts.get(ATTRIBUTION_RANKING_ERROR, 0))
    unknown = int(counts.get(ATTRIBUTION_NOT_RECORDED, 0))
    if miss and ranking:
        if miss >= ranking:
            notes.append(
                f"错误归因：候选生成缺失 {miss} 条 >= 排序错误 {ranking} 条，"
                "主要瓶颈在候选生成（embedding / 检索），而不是打分阈值。")
        else:
            notes.append(
                f"错误归因：排序错误 {ranking} 条 > 候选生成缺失 {miss} 条，"
                "正确话题大多进了候选集，瓶颈在打分与阈值。")
    if unknown:
        notes.append(f"{unknown} 条错误无法归因：标注快照没有记录候选集。")
    # The module's own notes carry the facts that decide whether the numbers
    # above can be trusted at all: candidate truncation, entries that could not
    # be parsed, and the two kinds of row that never had a candidate question.
    for note in metrics.get("notes") or []:
        if note not in notes:
            notes.append(note)
    return notes


def _sweep(rows: Sequence[TopicPairRow], config: LearningConfig) -> dict[str, Any]:
    """Sweep the commit threshold over the recorded labels.

    Metrics are computed on the *in-sample* rows; the evaluator re-runs this on
    the training split only and scores the holdout, so an in-sample optimum is
    never treated as validated.
    """
    base = BASE_POLICY["topic_commit_threshold"]
    candidates = sorted({round(value, 4) for value in sweep_values("topic_commit_threshold", steps=25)}
                        | {round(base, 4)})
    curve: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for threshold in candidates:
        metrics = topic_pair_metrics(replay_pairs(rows, threshold))
        row = {"threshold": threshold,
               "pair_accuracy": metrics["pair_accuracy"], "precision": metrics["precision"],
               "recall": metrics["recall"], "f1": metrics["f1"],
               "wrong_merge": metrics["wrong_merge"], "fragmentation": metrics["fragmentation"]}
        curve.append(row)
        score = metrics["pair_accuracy"]
        if score is None:
            continue
        key = (round(float(score), 6), -abs(threshold - base))
        if best is None or key > best["_key"]:
            best = {**row, "_key": key}
    if best is None:
        return {"threshold": None, "metric": "pair_accuracy", "value": None, "curve": []}
    best.pop("_key", None)
    baseline = topic_pair_metrics(recorded_pairs(rows))
    return {"threshold": best["threshold"], "metric": "pair_accuracy", "value": best["pair_accuracy"],
            "evaluated": len(curve), "curve": curve[:48],
            "baseline": {"threshold": None, "source": "recorded", "pair_accuracy": baseline["pair_accuracy"],
                         "wrong_merge": baseline["wrong_merge"],
                         "fragmentation": baseline["fragmentation"]},
            "default_threshold": base}


def _recommendations(
    report: TopicLearning,
    rows: Sequence[TopicPairRow],
    config: LearningConfig,
    sweep: Mapping[str, Any],
) -> list[Recommendation]:
    result: list[Recommendation] = []
    metrics = report.pair_metrics
    merge = int(metrics.get("wrong_merge") or 0)
    fragment = int(metrics.get("fragmentation") or 0)
    total = merge + fragment
    samples = report.samples

    if total == 0:
        result.append(diagnostic(
            title="没有可判定的合并/拆分错误",
            detail=f"{samples} 条标注中未出现主题配对冲突。",
            rationale="标注样本尚未覆盖有争议的话题边界，暂时没有可学习的方向。",
            samples=samples,
        ))
        return result

    imbalance = (fragment - merge) / total
    detail = f"误拆分 {fragment}，误合并 {merge}（配对冲突 {total}）"

    if imbalance > 0.15:
        result.append(diagnostic(
            title="系统倾向过度拆分",
            detail=detail,
            rationale=(
                "误拆分明显多于误合并，方向是**放宽**话题确定归属阈值。但放宽意味着要新增归属，"
                "只有标注快照里存了 topic_candidates 的消息才能重构；"
                f"本批 {report.samples_with_candidates}/{samples} 条带候选分布。"
                "因此这里只给方向，不给可评测的阈值建议——给出数字会是无法验证的猜测。"),
            samples=samples,
            evidence={"wrong_merge": merge, "fragmentation": fragment,
                      "imbalance": round(imbalance, 4),
                      "samples_with_candidates": report.samples_with_candidates,
                      "required_host_field": "routing.topic_candidates"},
        ))
    elif imbalance < -0.15:
        threshold = sweep.get("threshold")
        if threshold is None:
            return result
        base = BASE_POLICY["topic_commit_threshold"]
        target = bounded_target("topic_commit_threshold", base, float(threshold),
                                config.max_param_delta_ratio)
        result.append(config_recommendation(
            param="topic_commit_threshold",
            label=PARAM_SPECS["topic_commit_threshold"]["label"],
            before=base, after=target,
            title="收紧话题确定归属阈值",
            rationale=(
                f"{detail}，误合并占主导，方向是收紧归属。样本内最优阈值 {float(threshold):.3f}"
                f"（配对准确率 {sweep.get('value')}），当前 {base:.2f}。"
                f"建议值按 ±{config.max_param_delta_ratio:.0%} 上限截断，并需要留出集评测通过。"),
            samples=samples, min_samples=config.min_samples_for_recommendation,
            evidence={"sweep": {key: value for key, value in sweep.items() if key != "curve"},
                      "imbalance": round(imbalance, 4)},
        ))
    else:
        result.append(diagnostic(
            title="合并与拆分基本平衡",
            detail=detail,
            rationale="两个方向的错误量级接近，当前阈值没有明确的移动方向。",
            samples=samples,
            evidence={"wrong_merge": merge, "fragmentation": fragment,
                      "imbalance": round(imbalance, 4)},
        ))

    if report.samples_with_candidates < samples:
        result.append(diagnostic(
            title="放宽方向暂不可回放",
            detail=f"{samples - report.samples_with_candidates} 条标注没有记录候选话题分布",
            rationale=(
                "ChatDynamics 只在部分消息上写入 routing.topic_candidates。"
                "要让本插件能评测‘该不该放宽归属’，本体需要在该字段上保持稳定覆盖。"),
            samples=samples,
            evidence={"covered": report.samples_with_candidates, "total": samples},
        ))
    return result


def threshold_curve_rows(report: TopicLearning) -> list[dict[str, Any]]:
    curve = report.threshold_sweep.get("curve")
    return list(curve) if isinstance(curve, list) else []


__all__ = [
    "TopicLearning", "TopicPairRow", "UNASSIGNED_DISPLAY", "candidate_observations", "learn",
    "pair_rows", "recorded_pairs", "replay_can_move", "replay_label", "replay_pairs", "threshold_curve_rows",
]
