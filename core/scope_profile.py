"""Scope Review Profile: where one conversation's reviewed errors actually are.

The first layer answers one question and refuses the other:

    本作用域被人工检查过的样本里，系统经常错在哪里？      <- this module
    这个作用域实际有多少比例的消息会出错？                <- not answerable

Labels are chosen by a human, so they are a *selected* sample. A scope where the
operator went looking for failures will look worse than one nobody reviewed, and
a rate read as a population estimate would turn "where I looked" into "where the
bot is bad". Every sentence this module emits therefore says 被检查样本.

The second layer is the comparison, and it is deliberately leave-one-out:

    scope A vs everything except A

Comparing A against a global that contains A biases the difference toward zero —
for a large scope it is nearly a comparison with itself — and the smoothing prior
has the same problem: a scope's own errors must not inform the rate it is shrunk
toward.

Small scopes are shrunk toward the LOO rate with a Beta-Binomial prior:

    smoothed = (errors + prior * loo_rate) / (support + prior)

so three samples cannot produce a "66.7%" that reads like a trait. The raw and
the smoothed value both travel with every rate, because a number whose movement
cannot be explained is a number nobody should trust.

Confidence is a tier, not a score, and the stable tier asks for more than one long
evening of labelling: 100+ samples *and* 3+ sessions *and* 3+ annotation days.
Note what those days are: `LearningSample.timestamp` is the moment a human
labelled the message, not when it was sent, so this says "the review was spread
out", never "the conversation behaved this way over time".
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import buckets
from .candidates import summarise as summarise_candidates
from .metrics import SAMPLE_NOTE, error_distribution, ratio, rounded, topic_pair_metrics
from .policy import (
    ERROR_FALSE_BOT, ERROR_FRAGMENTATION, ERROR_MISSED_BOT, ERROR_MISSED_REPLY,
    ERROR_PREMATURE_REPLY, ERROR_WRONG_MERGE,
)
from .samples import (
    TASK_RECIPIENT, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME, TASK_TOPIC, LearningSample,
)
from .scope import SCOPE_LABEL_CHARS, SCOPE_SPANS_SESSIONS
from .topic_learner import candidate_observations
from .trace import known_topic_label

# ---- confidence tiers ---------------------------------------------------

CONFIDENCE_INSUFFICIENT = "insufficient"
CONFIDENCE_LOW = "low"
CONFIDENCE_MODERATE = "moderate"
CONFIDENCE_STABLE = "stable"

CONFIDENCE_LABEL = {
    CONFIDENCE_INSUFFICIENT: "样本不足",
    CONFIDENCE_LOW: "置信度低",
    CONFIDENCE_MODERATE: "置信度中",
    CONFIDENCE_STABLE: "稳定",
}

MIN_CONFIDENCE_SAMPLES = 20
MODERATE_SAMPLES = 100
STABLE_MIN_SESSIONS = 3
STABLE_MIN_DAYS = 3

# ---- comparison ---------------------------------------------------------

# The prior's weight, in samples: a scope with `prior_strength` reviewed samples
# sits halfway between its own rate and the rest of the corpus.
PRIOR_STRENGTH = 20
# Below this many reviewed samples a rate difference is not reported at all —
# smoothed or not, the comparison would be noise with a percentage sign.
MIN_RATE_SUPPORT = 5
# How far above the LOO rate a kind has to be before it is called dominant.
DOMINANT_DELTA = 0.05
DOMINANT_LIMIT = 3

# ---- diagnosis gates ----------------------------------------------------

DIAGNOSIS_INSUFFICIENT = "insufficient_candidate_evidence"
DIAGNOSIS_CANDIDATE_GENERATION = "candidate_generation"
DIAGNOSIS_RANKING = "ranking_or_scoring"
DIAGNOSIS_NONE = "no_clear_bottleneck"

DIAGNOSIS_LABEL = {
    DIAGNOSIS_INSUFFICIENT: "候选证据不足，无法归因",
    DIAGNOSIS_CANDIDATE_GENERATION: "候选生成是主要瓶颈",
    DIAGNOSIS_RANKING: "候选排序/打分为主要瓶颈",
    DIAGNOSIS_NONE: "候选链上没有明显瓶颈",
}

MIN_CANDIDATE_COVERAGE = 0.70
MIN_RECALL_ELIGIBLE = 20
MIN_SELECTION_ELIGIBLE = 20
RECALL_LOW = 0.80
SELECTION_HIGH = 0.90

RECOMMENDED_CANDIDATE_GENERATION = "topic_candidate_generation"
RECOMMENDED_RANKING = "topic_ranking_or_scoring"

# Which task's samples carry each error kind. The reply chain is deliberately
# two entries: an admission error ("should this have been answered?") and an
# outcome event ("it was answered, but 作息压掉了") are counted over different
# populations, and giving them one denominator would divide a gate decision by a
# routing decision.
KIND_TASK = {
    ERROR_MISSED_BOT: TASK_RECIPIENT,
    ERROR_FALSE_BOT: TASK_RECIPIENT,
    ERROR_MISSED_REPLY: TASK_REPLY_ADMISSION,
    ERROR_PREMATURE_REPLY: TASK_REPLY_ADMISSION,
    ERROR_FRAGMENTATION: TASK_TOPIC,
    ERROR_WRONG_MERGE: TASK_TOPIC,
    buckets.GATE_SUPPRESSION: TASK_REPLY_OUTCOME,
    buckets.GENERATION_FAILURE: TASK_REPLY_OUTCOME,
    buckets.DELIVERY_FAILURE: TASK_REPLY_OUTCOME,
}

KIND_LABEL = {
    ERROR_MISSED_BOT: "漏识别 Bot",
    ERROR_FALSE_BOT: "误判指向 Bot",
    ERROR_MISSED_REPLY: "漏回复",
    ERROR_PREMATURE_REPLY: "抢话",
    ERROR_FRAGMENTATION: "话题误拆分",
    ERROR_WRONG_MERGE: "话题误合并",
    buckets.GATE_SUPPRESSION: "门禁压制（非路由错误）",
    buckets.GENERATION_FAILURE: "生成失败（非路由错误）",
    buckets.DELIVERY_FAILURE: "发送失败（非路由错误）",
}

# Which of the kinds a policy could actually move. The scope view uses this to
# keep "this group is mostly silent because of 作息" from reading as "this group
# is where the router is worst".
MODEL_KINDS = (ERROR_MISSED_BOT, ERROR_FALSE_BOT, ERROR_MISSED_REPLY,
               ERROR_PREMATURE_REPLY, ERROR_FRAGMENTATION, ERROR_WRONG_MERGE)
SYSTEM_KINDS = (buckets.GATE_SUPPRESSION, buckets.GENERATION_FAILURE,
                buckets.DELIVERY_FAILURE)

KIND_ORDER = (*MODEL_KINDS, *SYSTEM_KINDS)

REVIEW_NOTE = "仅统计人工标注（被检查过）的样本，不代表该作用域的真实错误率"
SCOPE_LEVEL_NOTE = "作用域层级是会话：本体契约没有提供跨会话的群身份"


# ---- per-scope counting -------------------------------------------------

@dataclass(frozen=True)
class Support:
    """One error kind's exposure: how many chances it had in one population."""

    count: int = 0
    support: int = 0

    def __add__(self, other: "Support") -> "Support":
        return Support(self.count + other.count, self.support + other.support)

    def minus(self, other: "Support") -> "Support":
        return Support(max(0, self.count - other.count), max(0, self.support - other.support))

    def as_dict(self) -> dict[str, int]:
        return {"count": self.count, "support": self.support}


def empty_counts() -> dict[str, Support]:
    return {kind: Support() for kind in KIND_ORDER}


def rate_counts(rows: Sequence[LearningSample]) -> dict[str, Support]:
    """Error-kind counts with their own denominators, for one population.

    Recipient and reply kinds count samples; topic kinds count **pairs**, because
    that is what the topic metrics are defined over. Giving them one shared
    denominator would divide a pair count by a sample count.
    """
    counts = empty_counts()
    recipient = [row for row in rows if row.task == TASK_RECIPIENT]
    admission = [row for row in rows if row.task == TASK_REPLY_ADMISSION]
    outcome = [row for row in rows if row.task == TASK_REPLY_OUTCOME]
    topic = [row for row in rows if row.task == TASK_TOPIC]

    for kind in (ERROR_MISSED_BOT, ERROR_FALSE_BOT):
        counts[kind] = Support(sum(1 for row in recipient if row.error_type == kind),
                               len(recipient))
    for kind in (ERROR_MISSED_REPLY, ERROR_PREMATURE_REPLY):
        counts[kind] = Support(sum(1 for row in admission if row.error_type == kind),
                               len(admission))
    for kind in SYSTEM_KINDS:
        counts[kind] = Support(sum(1 for row in outcome if row.error_type == kind),
                               len(outcome))

    pairs = topic_pairs(topic)
    metrics = topic_pair_metrics(pairs)
    pair_support = int(metrics["pairs"])
    counts[ERROR_FRAGMENTATION] = Support(int(metrics["fragmentation"]), pair_support)
    counts[ERROR_WRONG_MERGE] = Support(int(metrics["wrong_merge"]), pair_support)
    return counts


def topic_pairs(rows: Sequence[LearningSample]) -> list[list[tuple[str, str]]]:
    """Per-session label pairs. Comparisons never cross a session boundary.

    Grouping by session even when one scope covers several of them is the point: a
    scope that pooled two conversations before pairing them would report
    fragmentation for every topic that simply did not continue across them.
    """
    grouped: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        if not (known_topic_label(row.predicted) or known_topic_label(row.expected)):
            continue
        grouped.setdefault(row.session_hash, []).append((row.predicted, row.expected))
    return list(grouped.values())


def add_counts(target: dict[str, Support], other: Mapping[str, Support]) -> None:
    for kind in KIND_ORDER:
        target[kind] = target[kind] + other.get(kind, Support())


def scope_samples(samples: Sequence[LearningSample]) -> dict[str, list[LearningSample]]:
    grouped: dict[str, list[LearningSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.scope_hash, []).append(sample)
    return grouped


def counts_by_scope(
    samples: Sequence[LearningSample],
) -> tuple[dict[str, list[LearningSample]], dict[str, dict[str, Support]], dict[str, Support]]:
    """Group once, count once. Every scope and the total share one pass.

    `scope A vs everything except A` is then a subtraction, so listing N scopes
    costs one counting pass instead of N of them.
    """
    grouped = scope_samples(samples)
    per_scope = {key: rate_counts(rows) for key, rows in grouped.items()}
    totals = empty_counts()
    for counts in per_scope.values():
        add_counts(totals, counts)
    return grouped, per_scope, totals


@dataclass(frozen=True)
class RateRow:
    """One error kind, in one scope, next to the same kind outside it."""

    kind: str
    task: str
    label: str
    count: int
    support: int
    loo_count: int
    loo_support: int
    prior_strength: int = PRIOR_STRENGTH
    comparable: bool = True
    reason: str = ""

    @property
    def raw_rate(self) -> float | None:
        return ratio(self.count, self.support)

    @property
    def loo_rate(self) -> float | None:
        """The baseline: the same kind everywhere *except* this scope."""
        return ratio(self.loo_count, self.loo_support)

    @property
    def smoothed_rate(self) -> float | None:
        baseline, raw = self.loo_rate, self.raw_rate
        if baseline is None:
            return raw
        if raw is None:
            return baseline
        return (self.count + self.prior_strength * baseline) / (self.support + self.prior_strength)

    @property
    def delta(self) -> float | None:
        """Smoothed scope rate minus the LOO rate, in rate units."""
        if not self.comparable:
            return None
        smoothed, baseline = self.smoothed_rate, self.loo_rate
        if smoothed is None or baseline is None:
            return None
        return smoothed - baseline

    @property
    def delta_pp(self) -> float | None:
        """The delta in percentage points, which is how a reader compares it."""
        delta = self.delta
        return None if delta is None else round(delta * 100, 1)

    def as_dict(self) -> dict[str, Any]:
        delta = self.delta
        return {
            "kind": self.kind,
            "task": self.task,
            "label": self.label,
            "count": self.count,
            "support": self.support,
            "raw_rate": rounded(self.raw_rate),
            "smoothed_rate": rounded(self.smoothed_rate),
            "loo_count": self.loo_count,
            "loo_support": self.loo_support,
            "loo_rate": rounded(self.loo_rate),
            "delta": rounded(delta),
            "delta_pp": self.delta_pp,
            "prior_strength": self.prior_strength,
            "comparable": self.comparable,
            "reason": self.reason,
        }


def rate_rows(scope: Mapping[str, Support], loo: Mapping[str, Support], *,
              prior_strength: int = PRIOR_STRENGTH) -> list[RateRow]:
    rows: list[RateRow] = []
    for kind in KIND_ORDER:
        own = scope.get(kind, Support())
        other = loo.get(kind, Support())
        reason = ""
        comparable = True
        if own.support == 0:
            comparable, reason = False, "本作用域没有该类样本"
        elif own.support < MIN_RATE_SUPPORT:
            comparable = False
            reason = f"本作用域该类样本只有 {own.support} 条，不足以对比"
        elif other.support < MIN_RATE_SUPPORT:
            comparable = False
            reason = f"其余作用域该类样本只有 {other.support} 条，基线不可用"
        rows.append(RateRow(kind=kind, task=KIND_TASK[kind], label=KIND_LABEL[kind],
                            count=own.count, support=own.support,
                            loo_count=other.count, loo_support=other.support,
                            prior_strength=prior_strength,
                            comparable=comparable, reason=reason))
    return rows


# ---- profiles -----------------------------------------------------------

def annotation_days(rows: Sequence[LearningSample]) -> int:
    """Distinct UTC days on which these samples were *labelled*.

    UTC so the number does not move with the machine's timezone, and named after
    labelling rather than conversation because that is what the timestamp is.
    """
    days = {time.strftime("%Y-%m-%d", time.gmtime(row.timestamp))
            for row in rows if row.timestamp}
    return len(days)


def required_sessions() -> int:
    """How many distinct sessions a stable profile demands.

    Today a scope *is* a session, so demanding three of them would make the
    stable tier unreachable — and a gate nobody can pass is not a safety
    feature, it is a decoration. The annotation-day requirement carries the "not
    one sitting" intent, and this number rises to `STABLE_MIN_SESSIONS` the
    moment `core/scope.py` stops equating scope with session.
    """
    return STABLE_MIN_SESSIONS if SCOPE_SPANS_SESSIONS else 1


def confidence_for(*, labelled_samples: int, labelled_sessions: int,
                   days: int) -> tuple[str, str]:
    """A tier plus the sentence that justifies it, never a bare score."""
    if labelled_samples < MIN_CONFIDENCE_SAMPLES:
        return (CONFIDENCE_INSUFFICIENT,
                f"只有 {labelled_samples} 条被检查样本，低于 {MIN_CONFIDENCE_SAMPLES} 条，"
                "本作用域的比例还不构成特征。")
    if labelled_samples < MODERATE_SAMPLES:
        return (CONFIDENCE_LOW,
                f"{labelled_samples} 条被检查样本（{labelled_sessions} 个会话、"
                f"{days} 个标注日），只够看方向，不够下结论。")
    needed = required_sessions()
    missing: list[str] = []
    if labelled_sessions < needed:
        missing.append(f"只有 {labelled_sessions} 个会话（需要 {needed} 个）")
    if days < STABLE_MIN_DAYS:
        missing.append(f"只跨 {days} 个标注日（需要 {STABLE_MIN_DAYS} 个）")
    if missing:
        return (CONFIDENCE_MODERATE,
                f"{labelled_samples} 条被检查样本，但" + "、".join(missing)
                + "：样本量够看，覆盖面不够，暂不标为稳定。")
    return (CONFIDENCE_STABLE,
            f"{labelled_samples} 条被检查样本，跨 {labelled_sessions} 个会话、"
            f"{days} 个标注日。")


@dataclass
class ScopeReviewProfile:
    scope_hash: str
    labelled_samples: int = 0
    labelled_sessions: int = 0
    days: int = 0
    scope_source: str = ""
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    tasks: dict[str, int] = field(default_factory=dict)
    error_counts: dict[str, int] = field(default_factory=dict)
    recipient: dict[str, Any] = field(default_factory=dict)
    topic: dict[str, Any] = field(default_factory=dict)
    reply: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    candidate_metrics: dict[str, Any] = field(default_factory=dict)
    confidence: str = CONFIDENCE_INSUFFICIENT
    confidence_reason: str = ""

    @property
    def scope_label(self) -> str:
        """Display only. Comparisons and lookups use the full digest."""
        return self.scope_hash[:SCOPE_LABEL_CHARS]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_hash": self.scope_hash,
            "scope_label": self.scope_label,
            "scope_source": self.scope_source,
            "labelled_samples": self.labelled_samples,
            "labelled_sessions": self.labelled_sessions,
            "annotation_days": self.days,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "tasks": dict(self.tasks),
            "error_counts": dict(self.error_counts),
            "recipient": dict(self.recipient),
            "topic": dict(self.topic),
            "reply": dict(self.reply),
            "outcome": dict(self.outcome),
            "candidate_metrics": dict(self.candidate_metrics),
            "confidence": self.confidence,
            "confidence_label": CONFIDENCE_LABEL.get(self.confidence, self.confidence),
            "confidence_reason": self.confidence_reason,
            "note": REVIEW_NOTE,
        }


def build_profile(scope_hash: str, rows: Sequence[LearningSample]) -> ScopeReviewProfile:
    """The review layer: what was looked at here, and what went wrong inside it."""
    recipient = [row for row in rows if row.task == TASK_RECIPIENT]
    admission = [row for row in rows if row.task == TASK_REPLY_ADMISSION]
    outcome = [row for row in rows if row.task == TASK_REPLY_OUTCOME]
    topic = [row for row in rows if row.task == TASK_TOPIC]
    sessions = len({row.session_hash for row in rows})
    days = annotation_days(rows)
    confidence, reason = confidence_for(labelled_samples=len(rows),
                                        labelled_sessions=sessions, days=days)
    timestamps = [row.timestamp for row in rows if row.timestamp]
    return ScopeReviewProfile(
        scope_hash=scope_hash,
        scope_source=rows[0].scope_source if rows else "",
        labelled_samples=len(rows),
        labelled_sessions=sessions,
        days=days,
        first_timestamp=min(timestamps, default=None),
        last_timestamp=max(timestamps, default=None),
        tasks={TASK_RECIPIENT: len(recipient), TASK_TOPIC: len(topic),
               TASK_REPLY_ADMISSION: len(admission), TASK_REPLY_OUTCOME: len(outcome)},
        error_counts=error_distribution(rows),
        recipient={
            "samples": len(recipient),
            "missed_bot": sum(1 for row in recipient if row.error_type == ERROR_MISSED_BOT),
            "false_bot": sum(1 for row in recipient if row.error_type == ERROR_FALSE_BOT),
            "error_types": error_distribution(recipient),
        },
        topic={
            "samples": len(topic),
            **topic_pair_metrics(topic_pairs(topic)),
            "error_types": error_distribution(topic),
        },
        reply={
            "samples": len(admission),
            "missed_reply": sum(1 for row in admission
                                if row.error_type == ERROR_MISSED_REPLY),
            "premature_reply": sum(1 for row in admission
                                   if row.error_type == ERROR_PREMATURE_REPLY),
            "error_types": error_distribution(admission),
            "note": "回复准入：判定目标是 level == strong，不是最终是否发送",
        },
        outcome={
            "samples": len(outcome),
            "delivered": sum(1 for row in outcome if row.predicted == "reply"),
            "suppressed": sum(1 for row in outcome
                              if row.error_type == buckets.GATE_SUPPRESSION),
            "generation_failed": sum(1 for row in outcome
                                     if row.error_type == buckets.GENERATION_FAILURE),
            "delivery_failed": sum(1 for row in outcome
                                   if row.error_type == buckets.DELIVERY_FAILURE),
            "error_types": error_distribution(outcome),
            "unavailable": not outcome,
        },
        candidate_metrics=summarise_candidates(candidate_observations(topic)),
        confidence=confidence,
        confidence_reason=reason,
    )


# ---- diagnosis ----------------------------------------------------------

@dataclass(frozen=True)
class Diagnosis:
    code: str
    detail: str
    recommended_target: str | None = None

    @property
    def label(self) -> str:
        return DIAGNOSIS_LABEL.get(self.code, self.code)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "label": self.label, "detail": self.detail,
                "recommended_target": self.recommended_target}


def diagnose_candidates(candidate_metrics: Mapping[str, Any]) -> Diagnosis:
    """Name the bottleneck on the topic chain — or refuse to.

    The gates come first. A recall number over eight rows is not a measurement,
    and a ranking verdict drawn from a corpus that barely records candidates is a
    guess wearing a label. Both cases return `insufficient_candidate_evidence`
    with the reason, which is a finding in its own right.
    """
    recall = candidate_metrics.get("candidate_recall") or {}
    selection = candidate_metrics.get("selection_accuracy") or {}
    eligible = int(recall.get("recorded") or 0)
    total = int(recall.get("total") or 0)
    selection_eligible = int(selection.get("eligible") or 0)
    coverage = ratio(eligible, total)

    missing: list[str] = []
    if coverage is None or coverage < MIN_CANDIDATE_COVERAGE:
        shown = "没有可归因样本" if coverage is None else f"覆盖率 {coverage:.1%}"
        missing.append(f"候选集记录{shown}，低于 {MIN_CANDIDATE_COVERAGE:.0%}")
    if eligible < MIN_RECALL_ELIGIBLE:
        missing.append(f"可算召回的样本 {eligible} 条，低于 {MIN_RECALL_ELIGIBLE}")
    if selection_eligible < MIN_SELECTION_ELIGIBLE:
        missing.append(f"可算选中准确率的样本 {selection_eligible} 条，"
                       f"低于 {MIN_SELECTION_ELIGIBLE}")
    if missing:
        return Diagnosis(DIAGNOSIS_INSUFFICIENT,
                         "候选证据不足，不下结论：" + "；".join(missing) + "。")

    recall_at_3 = recall.get("recall_at_3")
    accuracy = selection.get("accuracy")
    if recall_at_3 is None or accuracy is None:
        return Diagnosis(DIAGNOSIS_INSUFFICIENT,
                         "候选证据不足，不下结论：该子集上召回或选中准确率无法计算。")
    if recall_at_3 < RECALL_LOW and accuracy >= SELECTION_HIGH:
        return Diagnosis(
            DIAGNOSIS_CANDIDATE_GENERATION,
            f"候选召回 Recall@3 {recall_at_3:.1%}，而正确话题进了候选集时选中准确率 "
            f"{accuracy:.1%}：瓶颈在候选生成（embedding / 检索），"
            "暂不建议继续调整 topic_commit_threshold。",
            RECOMMENDED_CANDIDATE_GENERATION)
    if recall_at_3 >= RECALL_LOW and accuracy < SELECTION_HIGH:
        return Diagnosis(
            DIAGNOSIS_RANKING,
            f"候选召回 Recall@3 {recall_at_3:.1%} 已经不低，但选中准确率只有 "
            f"{accuracy:.1%}：瓶颈在打分与阈值，"
            "topic_commit_threshold / topic_margin_threshold 才是可动的旋钮。",
            RECOMMENDED_RANKING)
    return Diagnosis(DIAGNOSIS_NONE,
                     f"召回 Recall@3 {recall_at_3:.1%}、选中准确率 {accuracy:.1%}，"
                     "两者都还在可接受区间，候选链上没有明显瓶颈。")


# ---- comparison ---------------------------------------------------------

@dataclass
class ScopeComparison:
    profile: ScopeReviewProfile
    rates: list[RateRow] = field(default_factory=list)
    diagnosis: Diagnosis = field(default_factory=lambda: Diagnosis(DIAGNOSIS_INSUFFICIENT, ""))
    diagnostics: list[str] = field(default_factory=list)
    totals: dict[str, int] = field(default_factory=dict)

    @property
    def dominant_errors(self) -> list[str]:
        rows = [row for row in self.rates
                if row.comparable and row.delta is not None and row.delta >= DOMINANT_DELTA]
        rows.sort(key=lambda row: (-(row.delta or 0.0), KIND_ORDER.index(row.kind)))
        return [row.kind for row in rows[:DOMINANT_LIMIT]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.as_dict(),
            "global": {
                "baseline": "leave_one_out",
                "note": "全局列是 leave-one-out 基线：不含本作用域，避免自己稀释自己",
                "rates": {row.kind: {"count": row.loo_count, "support": row.loo_support,
                                     "rate": rounded(row.loo_rate),
                                     "label": row.label}
                          for row in self.rates},
            },
            "deltas": [row.as_dict() for row in self.rates],
            "dominant_errors": self.dominant_errors,
            "dominant_labels": [KIND_LABEL.get(kind, kind) for kind in self.dominant_errors],
            "diagnosis": self.diagnosis.as_dict(),
            "diagnostics": list(self.diagnostics),
            "totals": dict(self.totals),
            "note": REVIEW_NOTE,
        }


def _delta_sentence(row: RateRow) -> str:
    raw, baseline, delta = row.raw_rate, row.loo_rate, row.delta or 0.0
    if raw is None or baseline is None:  # pragma: no cover - guarded by comparable
        return f"{row.label} 高于其余作用域。"
    return (f"在被检查的样本里，{row.label} 比其余作用域高 {delta * 100:+.1f}pp"
            f"（本作用域 {raw:.1%}，其余作用域 {baseline:.1%}）——"
            "这是被检查样本中的分布差异，不是该作用域的真实错误率。")


def _diagnostics(profile: ScopeReviewProfile, rates: Sequence[RateRow],
                 diagnosis: Diagnosis) -> list[str]:
    lines: list[str] = [profile.confidence_reason, SCOPE_LEVEL_NOTE]
    for row in rates:
        if not row.comparable and row.support:
            lines.append(f"{row.label}：{row.reason}。")
    dominated = sorted((row for row in rates
                        if row.comparable and (row.delta or 0.0) >= DOMINANT_DELTA),
                       key=lambda row: -(row.delta or 0.0))
    if dominated:
        lines.append(_delta_sentence(dominated[0]))
    if profile.confidence in (CONFIDENCE_INSUFFICIENT, CONFIDENCE_LOW):
        lines.append("置信度不足时，这些差异只能用来决定下一步看哪里，不能写进结论。")
    lines.append(f"候选链诊断：{diagnosis.label}。{diagnosis.detail}")
    return lines


def _comparison(grouped: Mapping[str, list[LearningSample]], scope_hash: str,
                per_scope: Mapping[str, dict[str, Support]], totals: Mapping[str, Support],
                *, prior_strength: int) -> "ScopeComparison | None":
    rows = grouped.get(scope_hash)
    if not rows:
        return None
    own = per_scope.get(scope_hash, empty_counts())
    loo = {kind: totals[kind].minus(own.get(kind, Support())) for kind in KIND_ORDER}
    profile = build_profile(scope_hash, rows)
    rates = rate_rows(own, loo, prior_strength=prior_strength)
    diagnosis = diagnose_candidates(profile.candidate_metrics)
    return ScopeComparison(
        profile=profile, rates=rates, diagnosis=diagnosis,
        diagnostics=_diagnostics(profile, rates, diagnosis),
        totals={
            "scopes": len(grouped),
            "labelled_samples": sum(len(item) for item in grouped.values()),
            "scope_samples": len(rows),
            "other_scope_samples": sum(len(item) for item in grouped.values()) - len(rows),
        },
    )


def compare_scope(samples: Sequence[LearningSample], scope_hash: str, *,
                  prior_strength: int = PRIOR_STRENGTH) -> "ScopeComparison | None":
    """Profile one scope against every *other* scope, and diagnose the topic chain."""
    grouped, per_scope, totals = counts_by_scope(samples)
    return _comparison(grouped, scope_hash, per_scope, totals, prior_strength=prior_strength)


def profile_rows(samples: Sequence[LearningSample], *,
                 prior_strength: int = PRIOR_STRENGTH) -> list[dict[str, Any]]:
    """One row per scope, for the scope list. Sorted by reviewed volume."""
    grouped, per_scope, totals = counts_by_scope(samples)
    rows: list[dict[str, Any]] = []
    for scope_hash in grouped:
        comparison = _comparison(grouped, scope_hash, per_scope, totals,
                                 prior_strength=prior_strength)
        if comparison is None:  # pragma: no cover - the scope came from the samples
            continue
        profile = comparison.profile
        rows.append({
            "scope_hash": profile.scope_hash,
            "scope_label": profile.scope_label,
            "samples": profile.labelled_samples,
            "sessions": profile.labelled_sessions,
            "annotation_days": profile.days,
            "confidence": profile.confidence,
            "confidence_label": CONFIDENCE_LABEL.get(profile.confidence, profile.confidence),
            "dominant_errors": comparison.dominant_errors,
            "dominant_labels": [KIND_LABEL.get(kind, kind)
                                for kind in comparison.dominant_errors],
            "diagnosis": comparison.diagnosis.code,
            "diagnosis_label": comparison.diagnosis.label,
        })
    rows.sort(key=lambda row: (-int(row["samples"]), str(row["scope_hash"])))
    return rows


def scopes_payload(samples: Sequence[LearningSample]) -> dict[str, Any]:
    rows = profile_rows(samples)
    return {
        "total": len(rows),
        "scope_level": "session",
        "rows": rows,
        "notes": [REVIEW_NOTE, SCOPE_LEVEL_NOTE,
                  "作用域身份来自 core/scope.py：当前等于会话身份。"],
    }


def scope_payload(samples: Sequence[LearningSample], scope_hash: str) -> dict[str, Any] | None:
    comparison = compare_scope(samples, scope_hash)
    if comparison is None:
        return None
    payload = comparison.as_dict()
    payload["notes"] = [REVIEW_NOTE, SCOPE_LEVEL_NOTE, SAMPLE_NOTE]
    return payload


__all__ = [
    "CONFIDENCE_INSUFFICIENT", "CONFIDENCE_LABEL", "CONFIDENCE_LOW", "CONFIDENCE_MODERATE",
    "CONFIDENCE_STABLE", "DIAGNOSIS_CANDIDATE_GENERATION", "DIAGNOSIS_INSUFFICIENT",
    "DIAGNOSIS_LABEL", "DIAGNOSIS_NONE", "DIAGNOSIS_RANKING", "DOMINANT_DELTA", "DOMINANT_LIMIT",
    "Diagnosis", "KIND_LABEL", "KIND_ORDER", "KIND_TASK", "MIN_CANDIDATE_COVERAGE",
    "MIN_RATE_SUPPORT", "MODERATE_SAMPLES", "PRIOR_STRENGTH", "RECOMMENDED_CANDIDATE_GENERATION",
    "RECOMMENDED_RANKING", "REVIEW_NOTE", "RateRow", "SCOPE_LEVEL_NOTE", "ScopeComparison",
    "ScopeReviewProfile", "Support", "add_counts", "annotation_days", "build_profile",
    "compare_scope", "confidence_for", "counts_by_scope", "diagnose_candidates", "empty_counts",
    "profile_rows", "rate_counts", "rate_rows", "required_sessions", "scope_payload",
    "scope_samples", "scopes_payload", "topic_pairs",
]
