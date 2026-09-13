"""Supervised metrics that never invent a denominator.

Every ratio returns `None` when its denominator is zero, matching the host
plugin's convention: an undefined metric is reported as undefined, not as 0.0
and not as 1.0. Every count is labelled as coming from **human-selected**
samples, which is not a population accuracy estimate.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from .samples import (
    LearningSample, TASK_RECIPIENT, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME, TASK_TOPIC,
)
from .trace import known_topic_label

SAMPLE_NOTE = "仅统计人工标注样本，不代表真实准确率"
PAIR_NOTE = "话题指标按会话内标注配对计算，对标签重命名不变"


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def rounded(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def binary_counts(pairs: Iterable[tuple[Any, Any]]) -> dict[str, int]:
    """`(predicted, expected)` boolean pairs to a confusion matrix."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for predicted, expected in pairs:
        if not isinstance(predicted, bool) or not isinstance(expected, bool):
            continue
        if expected and predicted:
            counts["tp"] += 1
        elif expected and not predicted:
            counts["fn"] += 1
        elif predicted and not expected:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return counts


def binary_report(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, tn, fn = (int(counts.get(key, 0)) for key in ("tp", "fp", "tn", "fn"))
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
    total = tp + fp + tn + fn
    return {
        "support": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": rounded(precision), "recall": rounded(recall), "f1": rounded(f1),
        "accuracy": rounded(ratio(tp + tn, total)),
        "positive_rate": rounded(ratio(tp + fp, total)),
    }


def label_confusion(pairs: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    """Sorted `(expected, predicted)` counts; `__other__` aggregates the tail."""
    counts: Counter[tuple[str, str]] = Counter()
    for predicted, expected in pairs:
        counts[(str(expected)[:160], str(predicted)[:160])] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in ordered[:200]]


def topic_pair_metrics(sessions: Sequence[Sequence[tuple[str, str]]]) -> dict[str, Any]:
    """Equality-relation metrics over labelled messages, per session.

    `sessions` is a sequence of per-session `(predicted_label, expected_label)`
    pairs. Comparisons never cross sessions, and labels are compared only by
    equality, so any one-to-one renaming of either label set is invisible.
    """
    tp = merge = fragment = pairs = 0
    for rows in sessions:
        for left, right in combinations(rows, 2):
            pairs += 1
            same_truth = left[1] == right[1]
            same_prediction = bool(left[0]) and left[0] == right[0]
            tp += bool(same_truth and same_prediction)
            merge += bool(not same_truth and same_prediction)
            fragment += bool(same_truth and not same_prediction)
    precision = ratio(tp, tp + merge)
    recall = ratio(tp, tp + fragment)
    f1 = None if not precision or not recall else 2 * precision * recall / (precision + recall)
    return {
        "pairs": pairs, "true_positive": tp, "wrong_merge": merge, "fragmentation": fragment,
        "precision": rounded(precision), "recall": rounded(recall), "f1": rounded(f1),
        # Fraction of labelled pairs whose predicted relation matches the truth.
        "pair_accuracy": rounded(ratio(tp, pairs)),
        "note": PAIR_NOTE,
    }


@dataclass(frozen=True)
class ErrorRate:
    """One error kind as a rate over its own exposure.

    A global accuracy move of +0.6% can hide an error kind dropping 37%
    relative; judging an adjustment only on the global number discards exactly
    the evidence that says it worked. So every adjustment is scored on the error
    kind it was aimed at, and on the collateral it caused.
    """

    kind: str
    count: int
    support: int

    @property
    def rate(self) -> float | None:
        return ratio(self.count, self.support)

    def relative(self, other: "ErrorRate") -> float | None:
        """Relative change from this rate to `other`, as a signed fraction.

        `-0.375` reads as "the target error fell by 37.5% relative to itself",
        which is what a "target error down >= 10%" rule has to measure.
        """
        before, after = self.rate, other.rate
        if before is None or after is None or before == 0:
            return None
        return (after - before) / before

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "count": self.count, "support": self.support,
                "rate": rounded(self.rate)}


def binary_error_rates(pairs: Iterable[tuple[Any, Any]], *,
                       positive_kind: str, negative_kind: str) -> dict[str, ErrorRate]:
    """Name the two binary mistakes from the **expected** side.

    `positive_kind` counts missed expected positives (false negatives);
    `negative_kind` counts firings on expected negatives (false positives).
    """
    counts = binary_counts(pairs)
    support = counts["tp"] + counts["fp"] + counts["tn"] + counts["fn"]
    return {
        positive_kind: ErrorRate(positive_kind, counts["fn"], support),
        negative_kind: ErrorRate(negative_kind, counts["fp"], support),
    }


def compare_error_rates(baseline: Mapping[str, ErrorRate],
                        candidate: Mapping[str, ErrorRate],
                        *, target: str | None) -> dict[str, Any]:
    """Relative movement of the target error, with everything else as collateral."""
    rows: dict[str, Any] = {}
    for kind, before in baseline.items():
        after = candidate.get(kind)
        if after is None:
            continue
        rows[kind] = {
            "baseline_count": before.count, "candidate_count": after.count,
            "baseline_rate": rounded(before.rate), "candidate_rate": rounded(after.rate),
            "relative": rounded(before.relative(after)),
            "absolute": rounded((after.rate or 0.0) - (before.rate or 0.0)),
            "is_target": kind == target,
        }
    return rows


def target_error_relative(errors: Mapping[str, Any], target: str | None) -> float | None:
    """Pull the target error's relative change out of a compare result."""
    if target is None:
        return None
    row = errors.get(target)
    if not isinstance(row, Mapping):
        return None
    value = row.get("relative")
    return float(value) if isinstance(value, (int, float)) else None


def sample_accuracy(samples: Sequence[LearningSample]) -> dict[str, Any]:
    total = len(samples)
    correct = sum(1 for sample in samples if sample.correct)
    return {"total": total, "correct": correct, "incorrect": total - correct,
            "accuracy": rounded(ratio(correct, total)), "note": SAMPLE_NOTE}


def error_distribution(samples: Sequence[LearningSample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in samples:
        if sample.correct:
            continue
        counts[str(sample.error_type or "unknown")] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def predicted_distribution(samples: Sequence[LearningSample]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(sample.predicted) for sample in samples)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:64])


def task_report(samples: Sequence[LearningSample]) -> dict[str, Any]:
    """Per-task aggregation shared by the console and the learner reports."""
    report: dict[str, Any] = {"total": len(samples)}
    recipients = [s for s in samples if s.task == TASK_RECIPIENT]
    if recipients:
        report["recipient"] = sample_accuracy(recipients)
        report["recipient"]["confusion"] = binary_report(binary_counts(
            (sample.predicted == "bot", sample.expected == "bot") for sample in recipients))
        report["recipient"]["error_types"] = error_distribution(recipients)
    admissions = [s for s in samples if s.task == TASK_REPLY_ADMISSION]
    if admissions:
        report["reply_admission"] = binary_report(binary_counts(
            (sample.predicted == "reply", sample.expected == "reply") for sample in admissions))
        report["reply_admission"]["note"] = (
            "回复准入：预测目标是 participation.level == strong，"
            "回答「该不该进入回复流程」，不是最终是否发送")
    outcomes = [s for s in samples if s.task == TASK_REPLY_OUTCOME]
    if outcomes:
        report["reply_outcome"] = binary_report(binary_counts(
            (sample.predicted == "reply", sample.expected == "reply") for sample in outcomes))
        report["reply_outcome"]["note"] = (
            "最终发送结果：预测目标是 schema 3 记录的 outcome.delivered，"
            "回答「最终是否真的发出去了」；被门禁压制计入这里，不计入回复准入")
    topics = [s for s in samples if s.task == TASK_TOPIC]
    if topics:
        grouped: dict[str, list[tuple[str, str]]] = {}
        for sample in topics:
            if known_topic_label(sample.predicted) or known_topic_label(sample.expected):
                grouped.setdefault(sample.session_hash, []).append((sample.predicted, sample.expected))
        report["topic"] = topic_pair_metrics(list(grouped.values()))
        report["topic"]["sampled_messages"] = len(topics)
        report["topic"]["error_types"] = error_distribution(topics)
        report["topic"]["confusion"] = label_confusion(
            (sample.predicted, sample.expected) for sample in topics)
    return report


def within_window(samples: Sequence[LearningSample], *, now: float, days: int) -> list[LearningSample]:
    if days <= 0:
        return list(samples)
    cutoff = now - days * 86_400
    return [sample for sample in samples if sample.timestamp >= cutoff]


__all__ = [
    "ErrorRate", "PAIR_NOTE", "SAMPLE_NOTE", "binary_counts", "binary_error_rates",
    "binary_report", "compare_error_rates", "error_distribution", "label_confusion",
    "predicted_distribution", "ratio", "rounded", "sample_accuracy", "target_error_relative",
    "task_report", "topic_pair_metrics", "within_window",
]
