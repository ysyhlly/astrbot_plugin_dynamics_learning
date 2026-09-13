"""Topic candidate records, candidate recall and error attribution.

A wrong topic decision is not one failure mode, it is two:

* the correct topic never entered the candidate set -> **candidate generation**
  is the problem, and no scorer change can fix it;
* the correct topic was a candidate but another one won -> **ranking / scoring**
  is the problem, and that is what a threshold or weight move can fix.

Collapsing both into "话题识别错了" makes Topic Learning learn the wrong thing.
This module keeps them apart, and keeps a third group honest: the errors that
are simply unattributable because the host recorded no candidate set.

Two distinctions decide whether an error can be attributed at all, and neither
survives being flattened:

* a **missing** candidate key means the host recorded nothing, so nothing can be
  said about whether the correct topic was offered. It becomes `not_recorded`
  and is excluded from recall instead of being counted as a miss;
* an **empty list** means the host looked and proposed nothing, which *is* a
  candidate-generation miss whenever a labelled topic existed.

The host contract is accepted in two shapes: the structured form the host is
being upgraded to, and the legacy `[[score, topic_id], ...]` pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

MAX_CANDIDATES = 8
RANK_LIMITS = (1, 3, 5)

# Why an observation is in the corpus at all. Only labelled rows can be scored;
# the other two are counted so the attribution table totals the whole corpus
# instead of quietly shrinking it.
KIND_LABELLED = "labelled"
KIND_NEW_TOPIC = "new_topic"
KIND_UNLABELLED = "unlabelled"

ATTRIBUTION_CORRECT = "correct"
ATTRIBUTION_CANDIDATE_MISS = "candidate_miss"
ATTRIBUTION_RANKING_ERROR = "ranking_error"
ATTRIBUTION_NOT_RECORDED = "not_recorded"
ATTRIBUTION_NEW_TOPIC = "new_topic_expected"
ATTRIBUTION_UNATTRIBUTABLE = "unattributable"

ATTRIBUTION_LABEL = {
    ATTRIBUTION_CORRECT: "正确",
    ATTRIBUTION_CANDIDATE_MISS: "候选生成缺失（正确话题没进候选集）",
    ATTRIBUTION_RANKING_ERROR: "候选排序错误（正确话题在候选集里但没被选中）",
    ATTRIBUTION_NOT_RECORDED: "无法归因（本体没有记录候选集）",
    ATTRIBUTION_NEW_TOPIC: "新话题（按定义不可能是候选）",
    ATTRIBUTION_UNATTRIBUTABLE: "无可用真值标签",
}

SAMPLE_NOTE = "仅统计已标注样本；未记录候选集的行不计入召回"


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class CandidateRecord:
    topic_id: str
    score: float
    rank: int
    evidence: Mapping[str, float] = field(default_factory=dict)
    score_known: bool = True

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"topic_id": self.topic_id, "rank": self.rank}
        # A candidate whose score the host never recorded still counts for
        # recall — the topic *was* a candidate — but it must never be treated as
        # scoring zero during a threshold replay.
        if self.score_known:
            payload["final_score"] = round(self.score, 6)
        if self.evidence:
            payload["evidence"] = {key: round(float(value), 6)
                                   for key, value in self.evidence.items()}
        return payload


@dataclass(frozen=True)
class Candidates:
    """What one message's candidate field said, including whether it said anything."""

    recorded: bool
    items: tuple[CandidateRecord, ...] = ()
    dropped: int = 0

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(row.topic_id for row in self.items)

    def top(self, limit: int) -> tuple[str, ...]:
        return self.ids[:max(0, limit)]


def _evidence_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, raw in list(value.items())[:32]:
        if not isinstance(key, str):
            continue
        number = _finite(raw)
        if number is not None:
            result[key[:64]] = number
    return result


def parse_candidates(raw: Any, *, limit: int = MAX_CANDIDATES) -> Candidates:
    """Accept both recorded shapes and normalise to ranked records.

    `recorded` says whether a candidate list was there at all. A value that is
    present but neither a list nor parseable is reported as **not** recorded:
    the store could not be read, which is not evidence that the host looked and
    found nothing.
    """
    if not isinstance(raw, (list, tuple)):
        return Candidates(False)
    # (explicit_rank, rank, -score, record): an explicit rank always wins, and
    # everything without one falls back to score order. The legacy pair form
    # carries no rank at all, so ordering it by list position would read a
    # score-sorted list backwards.
    parsed: list[tuple[int, int, float, CandidateRecord]] = []
    dropped = 0
    for item in list(raw)[:limit]:
        if isinstance(item, Mapping):
            topic_id = item.get("topic_id")
            if not isinstance(topic_id, str) or not topic_id:
                dropped += 1
                continue
            score = _finite(item.get("final_score"))
            if score is None:
                score = _finite(item.get("score"))
            raw_rank = item.get("rank")
            rank_value = (raw_rank if isinstance(raw_rank, int)
                          and not isinstance(raw_rank, bool) and raw_rank > 0 else 0)
            explicit = rank_value > 0
            parsed.append((
                0 if explicit else 1,
                rank_value,
                -(score if score is not None else 0.0),
                CandidateRecord(topic_id=topic_id[:160],
                                score=score if score is not None else 0.0,
                                rank=rank_value,
                                evidence=_evidence_map(item.get("evidence")),
                                score_known=score is not None),
            ))
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            score, topic_id = _finite(item[0]), item[1]
            if score is None or not isinstance(topic_id, str) or not topic_id:
                dropped += 1
                continue
            parsed.append((1, 0, -score,
                           CandidateRecord(topic_id=topic_id[:160], score=score, rank=0)))
            continue
        dropped += 1
    parsed.sort(key=lambda row: row[:3])
    return Candidates(
        True,
        tuple(CandidateRecord(topic_id=row.topic_id, score=row.score, rank=index,
                              evidence=row.evidence, score_known=row.score_known)
              for index, (_, _, _, row) in enumerate(parsed, start=1)),
        dropped)


def parse_topic_candidates(raw: Any, *, limit: int = MAX_CANDIDATES) -> tuple[CandidateRecord, ...]:
    """The ranked records alone, for callers that do not need the presence flag."""
    return parse_candidates(raw, limit=limit).items


def candidate_ids(records: Sequence[CandidateRecord]) -> tuple[str, ...]:
    return tuple(row.topic_id for row in records)


@dataclass(frozen=True)
class CandidateObservation:
    """One labelled message reduced to what the candidate metrics need."""

    expected: str
    selected: str
    candidates: tuple[CandidateRecord, ...] = ()
    recorded: bool = False
    kind: str = KIND_LABELLED
    dropped: int = 0

    @property
    def scorable(self) -> bool:
        """Only a labelled row about an existing topic has a candidate question."""
        return self.kind == KIND_LABELLED

    @property
    def hit_at(self) -> int | None:
        """1-based rank of the expected topic in the candidate list, if present."""
        if not self.scorable:
            return None
        for index, row in enumerate(self.candidates, start=1):
            if row.topic_id == self.expected:
                return index
        return None

    @property
    def attribution(self) -> str:
        if self.kind == KIND_NEW_TOPIC:
            return ATTRIBUTION_NEW_TOPIC
        if self.kind == KIND_UNLABELLED:
            return ATTRIBUTION_UNATTRIBUTABLE
        if self.selected and self.selected == self.expected:
            return ATTRIBUTION_CORRECT
        if not self.recorded:
            return ATTRIBUTION_NOT_RECORDED
        if self.hit_at is not None:
            return ATTRIBUTION_RANKING_ERROR
        return ATTRIBUTION_CANDIDATE_MISS


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _scorable(observations: Sequence[CandidateObservation]) -> list[CandidateObservation]:
    return [row for row in observations if row.scorable]


def candidate_recall(observations: Sequence[CandidateObservation],
                     *, limits: Sequence[int] = RANK_LIMITS) -> dict[str, Any]:
    """`Recall@K` over labelled observations that actually recorded a candidate set.

    An empty recorded list counts here and is a miss; a missing field does not
    count at all. The two are reported separately so the exclusion is visible.
    """
    scorable = _scorable(observations)
    recorded = [row for row in scorable if row.recorded]
    not_recorded = len(scorable) - len(recorded)
    result: dict[str, Any] = {"recorded": len(recorded), "total": len(scorable),
                              "not_recorded": not_recorded,
                              "excluded_new_topic": len(observations) - len(scorable),
                              "coverage": _ratio(len(recorded), len(scorable))}
    for limit in limits:
        hits = sum(1 for row in recorded
                   if row.hit_at is not None and row.hit_at <= limit)
        result[f"recall_at_{limit}"] = _ratio(hits, len(recorded))
        result[f"hits_at_{limit}"] = hits
    result["missed"] = sum(1 for row in recorded if row.hit_at is None)
    return result


def selection_accuracy(observations: Sequence[CandidateObservation]) -> dict[str, Any]:
    """How often the host picked the right topic **when it was available**.

    Conditioning on availability is the point: this number only moves when the
    scoring or ranking layer is the problem, so it can be compared against
    `candidate_recall` without the two masking each other.
    """
    available = [row for row in _scorable(observations)
                 if row.recorded and row.hit_at is not None]
    hits = sum(1 for row in available if row.selected == row.expected)
    return {"eligible": len(available), "correct": hits,
            "accuracy": _ratio(hits, len(available)),
            "note": "仅在正确话题进入候选集的样本上计算"}


def attribution(observations: Sequence[CandidateObservation]) -> dict[str, Any]:
    counts: dict[str, int] = {key: 0 for key in ATTRIBUTION_LABEL}
    for row in observations:
        counts[row.attribution] += 1
    total = len(observations)
    return {
        "total": total,
        "counts": counts,
        "rates": {key: _ratio(value, total) for key, value in counts.items()},
        "labels": dict(ATTRIBUTION_LABEL),
    }


def _notes(observations: Sequence[CandidateObservation], recall: dict[str, Any],
           lengths: Mapping[str, int], dropped: int) -> list[str]:
    """What a reader has to know before trusting the numbers above."""
    notes: list[str] = []
    if not recall["recorded"]:
        notes.append("没有任何标注记录了候选集（routing.topic_candidates），"
                     "候选生成与排序错误无法区分。")
    if lengths and max(int(key) for key in lengths) <= 3:
        notes.append("本体最多记录 3 个候选，Recall@5 与 Recall@3 在截断改变前必然相同。")
    if recall["not_recorded"]:
        notes.append(f"{recall['not_recorded']} 条标注没有记录候选集，"
                     "既不计入召回也不算候选生成失败。")
    if dropped:
        notes.append(f"{dropped} 个候选条目无法解析（缺 topic_id 或结构不符），"
                     "已丢弃且不计入召回。")
    new_topic = sum(1 for row in observations if row.kind == KIND_NEW_TOPIC)
    unlabelled = sum(1 for row in observations if row.kind == KIND_UNLABELLED)
    if new_topic or unlabelled:
        notes.append(f"另有 {new_topic} 条新话题样本与 {unlabelled} 条无标签样本不参与候选指标："
                     "按定义它们没有可判定的候选问题。")
    return notes


def summarise(observations: Sequence[CandidateObservation]) -> dict[str, Any]:
    recall = candidate_recall(observations)
    lengths: dict[str, int] = {}
    for row in observations:
        if row.recorded:
            key = str(len(row.candidates))
            lengths[key] = lengths.get(key, 0) + 1
    dropped = sum(row.dropped for row in observations)
    return {
        "sample_note": SAMPLE_NOTE,
        "candidate_recall": recall,
        "selection_accuracy": selection_accuracy(observations),
        "attribution": attribution(observations),
        "candidate_lengths": {key: lengths[key] for key in sorted(lengths, key=int)},
        "dropped_entries": dropped,
        "notes": _notes(observations, recall, lengths, dropped),
    }


def records_to_payload(records: Sequence[CandidateRecord]) -> list[dict[str, Any]]:
    return [row.as_dict() for row in records]


__all__ = [
    "ATTRIBUTION_CANDIDATE_MISS", "ATTRIBUTION_CORRECT", "ATTRIBUTION_LABEL",
    "ATTRIBUTION_NEW_TOPIC", "ATTRIBUTION_NOT_RECORDED", "ATTRIBUTION_RANKING_ERROR",
    "ATTRIBUTION_UNATTRIBUTABLE", "CandidateObservation", "CandidateRecord", "Candidates",
    "KIND_LABELLED", "KIND_NEW_TOPIC", "KIND_UNLABELLED", "MAX_CANDIDATES", "RANK_LIMITS",
    "SAMPLE_NOTE", "attribution", "candidate_ids", "candidate_recall",
    "parse_candidates", "parse_topic_candidates", "records_to_payload",
    "selection_accuracy", "summarise",
]
