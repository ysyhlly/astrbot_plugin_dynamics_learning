"""Contract health: what the recorded data can and cannot answer.

A single "data health 82" number is worse than useless here: it looks precise
while hiding the only thing a reader needs to know, which is *which capability*
the corpus does not support. So health is reported as a **capability matrix** —
one row per analysis the learners actually perform, each with its own eligible
count, total, coverage and the reason for every exclusion.

Two planes, and the split is not cosmetic:

* **contract plane** — facts about the raw annotation records, counted while
  they are still raw. The host writes `annotation_schema_version` and
  `decision_trace.routing_schema_version`, and whether a field is present at all
  is only visible before normalisation. It is counted at import time and cached,
  because nothing downstream can recover it.
* **sample plane** — facts about the samples the learners actually read,
  recomputed from the store on every request.

The division is mandatory, not stylistic: `core/trace.py` normalises every trace
into schema 2 and re-emits it, so a version counter read back from a stored
sample reports 100% schema 2 no matter what the host wrote, and a missing
`contribution_total` reads back as `0.0`. Anything whose presence can be erased
by the round trip is either counted on the contract plane or carried through
normalisation as an explicit flag (`topic_candidates_recorded`,
`contribution_total_recorded`); it is never inferred afterwards.

Capabilities are asked of the learners' own predicates — `policy.decide`'s
conditions for the threshold-sensitive split, `topic_learner.replay_can_move`
for the topic one — so the matrix cannot drift away from what tuning does.

Scope: the learning identity is the *session* (`core/scope.py`). The host
contract exposes no cross-session conversation identity, so a matrix that
claimed to describe "groups" would be describing sessions under a more
confident name.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import outcome as outcome_module
from .candidates import CandidateObservation, candidate_recall
from .config import LearningConfig
from .metrics import SAMPLE_NOTE, ratio, rounded
from .samples import (
    TASK_RECIPIENT, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME, TASK_TOPIC, LearningSample,
)
from .scope import SCOPE_SOURCE_SESSION_UMO_EQUAL, SCOPE_SOURCE_SESSION_UMO_MISMATCH, resolve_scope
from .topic_learner import candidate_observations, pair_rows, replay_can_move
from .trace import (
    CANDIDATE_EVIDENCE_FULL, CANDIDATE_EVIDENCE_NONE, CANDIDATE_EVIDENCE_PARTIAL,
    LATEST_TRACE_SCHEMA, SUPPORTED_SCHEMAS as SUPPORTED_TRACE_SCHEMAS, declared_schema_value,
    trace_from_sample_record,
)

# ---- status vocabulary -------------------------------------------------

STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_UNSUPPORTED = "unsupported"
STATUS_INSUFFICIENT = "insufficient"

STATUS_LABEL = {
    STATUS_OK: "正常",
    STATUS_WARNING: "警告",
    STATUS_UNSUPPORTED: "不支持",
    STATUS_INSUFFICIENT: "样本不足",
}

# Below this many samples a coverage number is noise, not a measurement.
MIN_CAPABILITY_SAMPLES = 20
# A capability is "ok" when nearly every sample can exercise it.
COVERAGE_OK = 0.9

PLANE_SAMPLE = "sample"
PLANE_CONTRACT = "contract"

CAPABILITY_RECIPIENT_REPLAY = "recipient_replay"
CAPABILITY_TOPIC_ATTRIBUTION = "topic_attribution"
CAPABILITY_TOPIC_THRESHOLD_REPLAY = "topic_threshold_replay"
CAPABILITY_REPLY_ADMISSION_REPLAY = "reply_admission_replay"
CAPABILITY_SCOPE_IDENTITY = "scope_identity"
CAPABILITY_FINAL_REPLY_OUTCOME = "final_reply_outcome"
CAPABILITY_CANDIDATE_EVIDENCE = "candidate_evidence"

DEFINITION_RECIPIENT_REPLAY = (
    "定向阈值回放：这条样本的判定确实由记录下来的加性分数决定，"
    "移动 strong_addressivity_threshold 能改变它"
)
DEFINITION_REPLY_ADMISSION_REPLAY = (
    "回复准入回放：同上，判定目标是 participation.level == strong"
)
DEFINITION_TOPIC_ATTRIBUTION = "话题候选归因：能否区分「候选生成缺失」与「排序错误」"
DEFINITION_TOPIC_THRESHOLD_REPLAY = "话题阈值回放：移动 topic_commit_threshold 能否改变这条样本的归属"
DEFINITION_SCOPE_IDENTITY = "作用域身份：样本能否归到一个稳定的会话作用域"
DEFINITION_FINAL_REPLY_OUTCOME = (
    "最终发送结果：这条消息最后到底有没有被回复出去（不是路由准入，是真实发送）"
)
DEFINITION_CANDIDATE_EVIDENCE = (
    "候选逐条证据：每个候选话题是否带有本体记录的分项得分（语义/回复边/参与者重叠…）"
)

SCOPE_LEVEL_NOTE = (
    "本体契约没有提供跨会话的群身份，因此作用域层级是会话："
    "一个会话就是一个作用域，画像最多只能做到这个粒度。"
)
REPLY_LEVEL_NOTE = (
    "回复准入回放的是路由准入判定 level == strong，不是「最终是否回复」；"
    "最终发送结果单独作为 final_reply_outcome 一行报告"
)
OUTCOME_LEVEL_NOTE = (
    "最终发送结果不可回放：门禁（作息/降温/媒体）、生成与平台发送都不在记录轨迹里，"
    "所以它只作为事实报告，不参与任何阈值回放"
)
TIMESTAMP_NOTE = "样本时间戳是人工标注时刻（annotated_at），不是消息发生时间"


@dataclass(frozen=True)
class CapabilityHealth:
    """One analysis, and whether the corpus can actually support it."""

    name: str
    status: str
    eligible: int
    total: int
    definition: str = ""
    plane: str = PLANE_SAMPLE
    coverage: float | None = None
    reasons: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return STATUS_LABEL.get(self.status, self.status)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "status_label": self.label,
            "plane": self.plane,
            "definition": self.definition,
            "coverage": rounded(self.coverage),
            "eligible": self.eligible,
            "total": self.total,
            "reasons": list(self.reasons),
            "detail": dict(self.detail),
        }


def _status_for(eligible: int, total: int, *, min_samples: int) -> str:
    """Availability first, sample size second.

    "Nothing in this corpus can exercise the capability" and "the corpus is too
    small to judge" are different findings with different fixes, so they get
    different statuses instead of collapsing into one low number.
    """
    if total == 0 or eligible == 0:
        return STATUS_UNSUPPORTED
    if total < min_samples:
        return STATUS_INSUFFICIENT
    # Both counts are positive here, so the ratio is defined by construction
    # rather than by a `None` check the reader would have to trust.
    return STATUS_OK if eligible / total >= COVERAGE_OK else STATUS_WARNING


def _health(name: str, eligible: int, total: int, *, definition: str, reasons: Sequence[str],
            detail: Mapping[str, Any] | None = None,
            min_samples: int = MIN_CAPABILITY_SAMPLES) -> CapabilityHealth:
    return CapabilityHealth(
        name=name,
        status=_status_for(eligible, total, min_samples=min_samples),
        eligible=eligible, total=total,
        definition=definition,
        coverage=ratio(eligible, total),
        reasons=tuple(reason for reason in reasons if reason),
        detail=dict(detail or {}),
    )


# ---- sample plane: what the learners can do with these samples ----------

def _replay_split(samples: Sequence[LearningSample]) -> dict[str, int]:
    """Why a threshold move can or cannot change these samples.

    Mirrors `core.policy.decide`: an explicit (structural) turn is answered from
    the evidence codes, a turn with no prior bot message hits the host's early
    return, and an ambient turn is scored from the recorded additive total.
    """
    counts = {"total": len(samples), "eligible": 0, "explicit": 0, "no_prior_bot": 0,
              "no_score": 0}
    for sample in samples:
        features = sample.features
        if features.get("ctx_explicit", 0.0) >= 0.5:
            counts["explicit"] += 1
        elif features.get("ctx_prior_bot", 0.0) < 0.5:
            counts["no_prior_bot"] += 1
        elif not sample.contribution_total_recorded:
            counts["no_score"] += 1
        else:
            counts["eligible"] += 1
    return counts


def _replay_reasons(counts: Mapping[str, int]) -> list[str]:
    reasons: list[str] = []
    if counts["explicit"]:
        reasons.append(f"{counts['explicit']} 条是结构化直判（明确指代、回复、称呼等短路证据），"
                       "判定不经过分数，阈值移动对它没有影响")
    if counts["no_prior_bot"]:
        reasons.append(f"{counts['no_prior_bot']} 条没有前置机器人消息，"
                       "本体在该条件下提前返回，分数不参与判定")
    if counts["no_score"]:
        reasons.append(f"{counts['no_score']} 条没有记录 participation.contribution_total，"
                       "加性分数无法复现，阈值回放只能沿用原判定")
    return reasons


def recipient_replay(samples: Sequence[LearningSample], *,
                     min_samples: int = MIN_CAPABILITY_SAMPLES) -> CapabilityHealth:
    rows = [sample for sample in samples if sample.task == TASK_RECIPIENT]
    counts = _replay_split(rows)
    reasons = _replay_reasons(counts)
    if not rows:
        reasons.append("还没有收件人标注样本：本体记录 bot_targeted 才会产生。")
    return _health(CAPABILITY_RECIPIENT_REPLAY, counts["eligible"], counts["total"],
                   definition=DEFINITION_RECIPIENT_REPLAY,
                   reasons=reasons, detail=counts, min_samples=min_samples)


def reply_admission_replay(samples: Sequence[LearningSample], *,
                           min_samples: int = MIN_CAPABILITY_SAMPLES) -> CapabilityHealth:
    """Reply health, with the standing reminder that this is not the final send."""
    rows = [sample for sample in samples if sample.task == TASK_REPLY_ADMISSION]
    counts = _replay_split(rows)
    reasons = _replay_reasons(counts)
    reasons.append(REPLY_LEVEL_NOTE)
    if not rows:
        reasons.append("还没有回复准入标注样本：本体记录 expected_reply 才会产生。")
    return _health(CAPABILITY_REPLY_ADMISSION_REPLAY, counts["eligible"], counts["total"],
                   definition=DEFINITION_REPLY_ADMISSION_REPLAY,
                   reasons=reasons, detail=counts, min_samples=min_samples)


def topic_attribution(samples: Sequence[LearningSample], *,
                      min_samples: int = MIN_CAPABILITY_SAMPLES) -> CapabilityHealth:
    """Can the corpus tell a generation miss from a ranking mistake?

    The denominator is the learner's own: `candidate_recall` scopes itself to
    labelled rows about an existing topic, so NEW singletons and unlabelled rows
    are excluded from both sides rather than counted as failures of a capability
    they never exercised.
    """
    rows = [sample for sample in samples if sample.task == TASK_TOPIC]
    observations: list[CandidateObservation] = candidate_observations(rows)
    recall = candidate_recall(observations)
    total = int(recall["total"])
    eligible = int(recall["recorded"])
    reasons: list[str] = []
    if recall["not_recorded"]:
        reasons.append(f"{recall['not_recorded']} 条话题标注没有记录候选集"
                       "（routing.topic_candidates），正确话题是否被提出无从判断，"
                       "既不计入召回也不算候选生成失败")
    if recall["excluded_new_topic"]:
        reasons.append(f"{recall['excluded_new_topic']} 条新话题/无标签样本没有候选问题，"
                       "不计入该能力的分母")
    if not rows:
        reasons.append("还没有话题标注样本。")
    return _health(CAPABILITY_TOPIC_ATTRIBUTION, eligible, total,
                   definition=DEFINITION_TOPIC_ATTRIBUTION,
                   reasons=reasons,
                   detail={"recorded": eligible, "not_recorded": int(recall["not_recorded"]),
                           "excluded": int(recall["excluded_new_topic"])},
                   min_samples=min_samples)


def topic_threshold_replay(samples: Sequence[LearningSample], *,
                           min_samples: int = MIN_CAPABILITY_SAMPLES) -> CapabilityHealth:
    """Whether a commit-threshold move can change a topic sample's label at all.

    Asked of the learner's own replay predicate rather than re-derived here, so
    the matrix cannot disagree with what tuning actually does.
    """
    pairs = pair_rows([sample for sample in samples if sample.task == TASK_TOPIC])
    movable = [row for row in pairs if replay_can_move(row)]
    frozen = len(pairs) - len(movable)
    reasons: list[str] = []
    if frozen:
        reasons.append(f"{frozen} 条样本的归属判定不受阈值影响：既未确定归属、"
                       "也没有记录带分数的候选，放宽阈值无法重建它们该归到哪里")
    if not pairs:
        reasons.append("还没有话题标注样本。")
    return _health(CAPABILITY_TOPIC_THRESHOLD_REPLAY, len(movable), len(pairs),
                   definition=DEFINITION_TOPIC_THRESHOLD_REPLAY,
                   reasons=reasons,
                   detail={"movable": len(movable), "frozen": frozen},
                   min_samples=min_samples)


def scope_identity(samples: Sequence[LearningSample], *,
                   min_samples: int = MIN_CAPABILITY_SAMPLES) -> CapabilityHealth:
    """Every sample must land in a stable scope; the *level* is reported, not implied."""
    total = len(samples)
    resolvable = sum(1 for sample in samples
                     if sample.scope_hash and sample.scope_hash == sample.session_hash)
    confirmed = sum(1 for sample in samples
                    if sample.scope_source == SCOPE_SOURCE_SESSION_UMO_EQUAL)
    mismatched = sum(1 for sample in samples
                     if sample.scope_source == SCOPE_SOURCE_SESSION_UMO_MISMATCH)
    distinct_hints = {sample.group_hint_hash for sample in samples if sample.group_hint_hash}
    reasons: list[str] = [SCOPE_LEVEL_NOTE]
    if total and not confirmed:
        reasons.append("没有任何样本的作用域得到本体运行时快照的确认"
                       "（数据来自导出导入，或运行时快照里没有对应会话）")
    if mismatched:
        reasons.append(f"{mismatched} 条样本的运行时快照里 umo 与 session_key 不一致，"
                       "本体自己的恢复逻辑把这种快照视为非法，这里按会话身份处理并报出")
    return _health(CAPABILITY_SCOPE_IDENTITY, resolvable, total,
                   definition=DEFINITION_SCOPE_IDENTITY,
                   reasons=reasons,
                   detail={"host_confirmed": confirmed,
                           "fallback": total - confirmed - mismatched,
                           "umo_mismatch": mismatched,
                           "distinct_group_hints": len(distinct_hints)},
                   min_samples=min_samples)


def final_reply_outcome(samples: Sequence[LearningSample], *,
                        min_samples: int = MIN_CAPABILITY_SAMPLES) -> CapabilityHealth:
    """Can the corpus say whether the bot actually sent anything?

    The denominator is every message a human left a reply label on, and the
    numerator is the subset where the host also recorded a final outcome. That
    is the honest pairing: "should have replied" without "did it" is exactly the
    schema 2 gap, and stating it as a row — rather than as an error, or as
    silence — is what turns the gap into an ordinary working capability the day
    the host starts writing `outcome`.
    """
    labelled = [sample for sample in samples if sample.task == TASK_REPLY_ADMISSION]
    recorded = sum(1 for sample in labelled if sample.outcome.recorded)
    stages: dict[str, int] = {}
    reasons: list[str] = []
    for sample in labelled:
        found = sample.outcome
        if found.recorded and not found.is_delivered:
            stages[found.stage] = stages.get(found.stage, 0) + 1
    if not recorded:
        reasons.append("没有任何标注记录过最终发送结果：schema 2 的 decision_trace 里"
                       "没有 outcome 字段，本体也没有写 participation.should_reply，"
                       "因此在这个契约下该能力不可用")
    elif recorded < len(labelled):
        reasons.append(f"{len(labelled) - recorded} 条回复标注没有对应的最终结果记录，"
                       "它们只能按路由准入解释")
    reasons.append(OUTCOME_LEVEL_NOTE)
    return _health(CAPABILITY_FINAL_REPLY_OUTCOME, recorded, len(labelled),
                   definition=DEFINITION_FINAL_REPLY_OUTCOME, reasons=reasons,
                   detail={"recorded": recorded, "total": len(labelled),
                           "not_delivered_by_stage": dict(sorted(stages.items()))},
                   min_samples=min_samples)


def candidate_evidence(samples: Sequence[LearningSample], *,
                       min_samples: int = MIN_CAPABILITY_SAMPLES) -> CapabilityHealth:
    """Whether each candidate topic carries the host's own per-candidate scores.

    The candidate *list* is enough to tell a generation miss from a ranking
    mistake. It is not enough to answer the next question — "which scoring term
    put the wrong one first" — and that is what per-candidate evidence is for.
    Schema 2 recorded pairs of `[score, topic_id]`, so the level is `partial`
    there by construction, and saying so keeps the reader from reading a flat
    "candidates are recorded" as "the ranking is explainable".
    """
    rows = [sample for sample in samples if sample.task == TASK_TOPIC]
    levels: dict[str, int] = dict.fromkeys(
        (CANDIDATE_EVIDENCE_FULL, CANDIDATE_EVIDENCE_PARTIAL, CANDIDATE_EVIDENCE_NONE), 0)
    for sample in rows:
        levels[sample.candidate_evidence] = levels.get(sample.candidate_evidence, 0) + 1
    eligible = levels[CANDIDATE_EVIDENCE_FULL]
    reasons: list[str] = []
    if rows and not eligible:
        reasons.append(
            f"{levels[CANDIDATE_EVIDENCE_PARTIAL]} 条话题标注只有候选列表、没有逐条候选证据"
            "（schema 2 只记 [score, topic_id]）：可以区分候选生成与排序错误，"
            "但无法回答「是哪个分项把错的候选排到了前面」")
    if levels[CANDIDATE_EVIDENCE_NONE]:
        reasons.append(f"{levels[CANDIDATE_EVIDENCE_NONE]} 条话题标注根本没有候选集，"
                       "连候选生成与排序都无法区分")
    if not rows:
        reasons.append("还没有话题标注样本。")
    return _health(CAPABILITY_CANDIDATE_EVIDENCE, eligible, len(rows),
                   definition=DEFINITION_CANDIDATE_EVIDENCE, reasons=reasons,
                   detail=dict(levels), min_samples=min_samples)


CAPABILITY_BUILDERS = (
    recipient_replay, topic_attribution, topic_threshold_replay,
    reply_admission_replay, scope_identity, final_reply_outcome, candidate_evidence,
)


def capabilities(samples: Sequence[LearningSample], *,
                 min_samples: int = MIN_CAPABILITY_SAMPLES) -> dict[str, CapabilityHealth]:
    found: dict[str, CapabilityHealth] = {}
    for builder in CAPABILITY_BUILDERS:
        row = builder(samples, min_samples=min_samples)
        found[row.name] = row
    return found


# ---- dataset facts ------------------------------------------------------

def dataset_health(samples: Sequence[LearningSample]) -> dict[str, Any]:
    """Headline counts, each one a fact about the samples that exist.

    `timestamp_semantics` is stated rather than implied: the stored timestamp is
    the moment a human labelled the message, not when the message was sent, so
    every window built on it is an annotation window.
    """
    timestamps = [sample.timestamp for sample in samples if sample.timestamp]
    return {
        "samples": len(samples),
        "sessions": len({sample.session_hash for sample in samples}),
        "scopes": len({sample.scope_hash for sample in samples}),
        "scope_level": "session",
        "tasks": {task: sum(1 for sample in samples if sample.task == task)
                  for task in (TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY_ADMISSION,
                               TASK_REPLY_OUTCOME)},
        "degraded_traces": sum(1 for sample in samples if _degraded(sample)),
        # The two plan markers, counted on the sample plane as well as the
        # contract plane: the same record can be readable and still unable to
        # answer these two questions, and a reader needs the count next to the
        # capability rows it explains.
        "outcome_unavailable": sum(1 for sample in samples if sample.outcome_unavailable),
        "candidate_evidence": {
            level: sum(1 for sample in samples if sample.candidate_evidence == level)
            for level in (CANDIDATE_EVIDENCE_FULL, CANDIDATE_EVIDENCE_PARTIAL,
                          CANDIDATE_EVIDENCE_NONE)
        },
        "first_timestamp": min(timestamps, default=None),
        "last_timestamp": max(timestamps, default=None),
        "timestamp_semantics": "annotated_at",
        "source_note": SAMPLE_NOTE,
    }


def _degraded(sample: LearningSample) -> bool:
    trace = sample.trace if isinstance(sample.trace, Mapping) else {}
    summary = trace.get("evidence_summary")
    if isinstance(summary, Mapping) and summary.get("degraded") is True:
        return True
    return trace.get("learning_degraded") is True


# ---- contract plane: raw records ----------------------------------------

MISSING = "missing"
INVALID = "invalid"

_COUNTER_FIELDS = (
    "annotation_keys", "unreadable_keys", "annotations_seen", "annotations_kept",
    "malformed", "oversized", "unknown_session", "truncated", "decision_trace_present",
    "decision_trace_absent", "topic_candidates_missing", "topic_candidates_empty",
    "topic_candidates_nonempty", "outcome_present", "outcome_absent",
    "shadow_present", "shadow_absent",
    "contribution_total_present", "contribution_total_absent",
    "contribution_total_unknown", "sessions", "distinct_group_id_sessions",
)
_COUNTER_MAPS = ("annotation_schema_versions", "routing_schema_versions", "scope_sources",
                 "suppression_reasons", "candidate_evidence_levels")


def _version_bucket(value: Any) -> str:
    """A stable string key for a version field, whatever shape it arrived in."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return INVALID
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value.strip()[:32] if value.strip() else MISSING
    return INVALID


def _bump(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


@dataclass
class RawContractStats:
    """What the host actually wrote, counted before anything normalises it.

    Counted while the records are still raw because the round trip is lossy: a
    stored sample always reads back as schema 2, and a missing
    `contribution_total` reads back as `0.0`. Both facts exist only here.

    Two populations, deliberately different:

    * the **accounting** counters (seen / kept / malformed / oversized /
      unknown_session / truncated) cover every row the host stored, so the
      identity `seen == kept + malformed + unknown_session` holds and a row can
      never disappear unnoticed. It is exposed as `balanced` and asserted in
      tests: counters that do not total the corpus would repeat, inside the
      health report, the very problem that report exists to find.
    * the **field** counters (schema versions, candidate buckets, additive
      totals) describe the records that were kept, i.e. the corpus the learners
      actually see. A row that never became a sample is accounted for in the
      first group and excluded from the second, because a ratio whose numerator
      and denominator came from different populations is the next silent bug.
    """

    source: str = ""
    annotation_keys: int = 0
    unreadable_keys: int = 0
    annotations_seen: int = 0
    annotations_kept: int = 0
    malformed: int = 0
    oversized: int = 0
    unknown_session: int = 0
    truncated: int = 0

    annotation_schema_versions: dict[str, int] = field(default_factory=dict)
    decision_trace_present: int = 0
    decision_trace_absent: int = 0
    routing_schema_versions: dict[str, int] = field(default_factory=dict)

    topic_candidates_missing: int = 0
    topic_candidates_empty: int = 0
    topic_candidates_nonempty: int = 0
    # Schema 3's two questions, counted while the record is still raw. The
    # sample layer stores its own copy of both (`candidate_evidence`,
    # `outcome`), so these counters exist to answer a different question: what
    # did the *host* write, before any of this plugin's reading is involved.
    candidate_evidence_levels: dict[str, int] = field(default_factory=dict)

    outcome_present: int = 0
    outcome_absent: int = 0
    suppression_reasons: dict[str, int] = field(default_factory=dict)
    # The shadow decision is recorded by the host for every turn it processes,
    # but only turns a human labelled become samples. Counting it on the raw
    # plane is what lets the report say "the host recorded 4,000 comparisons and
    # 320 of them are in the corpus" instead of letting 320 be read as the
    # population.
    shadow_present: int = 0
    shadow_absent: int = 0

    contribution_total_present: int = 0
    contribution_total_absent: int = 0
    contribution_total_unknown: int = 0

    sessions: int = 0
    scope_sources: dict[str, int] = field(default_factory=dict)
    distinct_group_id_sessions: int = 0

    @property
    def balanced(self) -> bool:
        return (self.annotations_seen
                == self.annotations_kept + self.malformed + self.unknown_session)

    # ---- observation ---------------------------------------------------

    def observe_record(self, raw: Any) -> None:
        """Count one record row. Never raises: the row may be anything at all."""
        if not isinstance(raw, Mapping):
            return
        _bump(self.annotation_schema_versions,
              _version_bucket(raw.get("annotation_schema_version")))
        trace = raw.get("decision_trace")
        if not isinstance(trace, Mapping):
            self.decision_trace_absent += 1
            self.contribution_total_unknown += 1
        else:
            self.decision_trace_present += 1
            _bump(self.routing_schema_versions,
                  _version_bucket(declared_schema_value(trace)))
            self._observe_participation(trace.get("participation"))
        self._observe_topic_candidates(raw)
        # The two schema 3 facts are read through the *same* reader the sample
        # layer uses, so the contract plane and the sample plane cannot disagree
        # about what counts as "recorded": a second implementation here would be
        # a second answer to the same question.
        observed = trace_from_sample_record(raw)
        if observed.shadow.recorded:
            self.shadow_present += 1
        else:
            self.shadow_absent += 1
        _bump(self.candidate_evidence_levels, observed.candidate_evidence)
        self._observe_outcome(observed.outcome)

    def _observe_participation(self, participation: Any) -> None:
        value = (participation.get("contribution_total")
                 if isinstance(participation, Mapping) else None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # A null here separates "the host scored it zero" from "the host never
            # scored it"; normalisation erases the difference, so it is counted now.
            self.contribution_total_absent += 1
            return
        self.contribution_total_present += 1

    def _observe_outcome(self, found: Any) -> None:
        """Count the final outcome once per record, wherever the host put it."""
        if not isinstance(found, outcome_module.FinalOutcome) or not found.recorded:
            self.outcome_absent += 1
            return
        self.outcome_present += 1
        if found.suppression_reason:
            _bump(self.suppression_reasons, found.suppression_reason)

    def _observe_topic_candidates(self, record: Mapping[str, Any]) -> None:
        """Count the candidate field's presence, in either location.

        Schema 2 writes it into `record["routing"]`, schema 3 into
        `decision_trace["routing"]`; the *presence* question is the same one
        either way, so both are checked before calling it missing.
        """
        trace = record.get("decision_trace")
        raw = None
        for container in (record.get("routing"), trace.get("routing")
                          if isinstance(trace, Mapping) else None):
            if not isinstance(container, Mapping):
                continue
            candidate = container.get("topic_candidates")
            if not isinstance(candidate, list):
                candidate = container.get("candidates")
            if isinstance(candidate, list):
                raw = candidate
                break
        if not isinstance(raw, list):
            # Present but unreadable is not evidence that the host looked and
            # found nothing, so it counts as missing — the reading
            # `candidates.parse_candidates` also gives it.
            self.topic_candidates_missing += 1
        elif raw:
            self.topic_candidates_nonempty += 1
        else:
            self.topic_candidates_empty += 1

    def observe_sessions(self, sessions: Mapping[str, Mapping[str, Any]],
                         used: Iterable[str]) -> None:
        """Count identity facts for the sessions that actually produced records."""
        keys = [key for key in dict.fromkeys(used) if key]
        self.sessions = len(keys)
        for key in keys:
            scope = resolve_scope(key, sessions.get(key))
            _bump(self.scope_sources, scope.scope_source)
            if scope.group_hint_hash:
                self.distinct_group_id_sessions += 1

    def merge(self, other: "RawContractStats") -> None:
        for name in _COUNTER_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in _COUNTER_MAPS:
            target = getattr(self, name)
            for key, value in getattr(other, name).items():
                target[key] = target.get(key, 0) + value

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "balanced": self.balanced,
            "annotation_keys": self.annotation_keys,
            "unreadable_keys": self.unreadable_keys,
            "annotations_seen": self.annotations_seen,
            "annotations_kept": self.annotations_kept,
            "malformed": self.malformed,
            "oversized": self.oversized,
            "unknown_session": self.unknown_session,
            "truncated": self.truncated,
            "annotation_schema_versions": dict(self.annotation_schema_versions),
            "decision_trace_present": self.decision_trace_present,
            "decision_trace_absent": self.decision_trace_absent,
            "routing_schema_versions": dict(self.routing_schema_versions),
            "topic_candidates": {
                "missing": self.topic_candidates_missing,
                "empty": self.topic_candidates_empty,
                "nonempty": self.topic_candidates_nonempty,
            },
            "candidate_evidence": {
                level: self.candidate_evidence_levels.get(level, 0)
                for level in (CANDIDATE_EVIDENCE_FULL, CANDIDATE_EVIDENCE_PARTIAL,
                              CANDIDATE_EVIDENCE_NONE)
            },
            "outcome": {
                "present": self.outcome_present,
                "absent": self.outcome_absent,
                "suppression_reasons": dict(self.suppression_reasons),
            },
            "shadow": {"present": self.shadow_present, "absent": self.shadow_absent},
            "contribution_total": {
                "present": self.contribution_total_present,
                "absent": self.contribution_total_absent,
                "no_trace": self.contribution_total_unknown,
            },
            "sessions": self.sessions,
            "scope_sources": dict(self.scope_sources),
            "distinct_group_id_sessions": self.distinct_group_id_sessions,
        }


def contract_findings(contract: Mapping[str, Any] | None) -> list[str]:
    """The few things a reader must know before trusting the raw plane."""
    if not isinstance(contract, Mapping) or not contract:
        return ["还没有导入记录：契约面（本体到底写了什么字段）暂无数据，先执行一次导入。"]
    findings: list[str] = []
    if contract.get("balanced") is False:
        findings.append("原始记录计数不守恒"
                        "（seen != kept + malformed + unknown_session）："
                        "这是本插件自己的统计缺陷，先修它再用这些数字下结论。")
    versions = contract.get("annotation_schema_versions")
    if isinstance(versions, Mapping) and versions:
        rendered = "、".join(f"{key}×{value}" for key, value in sorted(versions.items()))
        findings.append(f"标注 schema 分布：{rendered}")
    absent = int(contract.get("decision_trace_absent") or 0)
    if absent:
        findings.append(f"{absent} 条记录没有 decision_trace（本体升级前写入的旧记录），"
                        "它们的证据与参与度不可回放")
    candidates = contract.get("topic_candidates")
    if isinstance(candidates, Mapping):
        missing = int(candidates.get("missing") or 0)
        total = sum(int(candidates.get(key) or 0) for key in ("missing", "empty", "nonempty"))
        if missing:
            findings.append(f"{missing}/{total} 条记录的 routing.topic_candidates 字段缺失，"
                            "这部分样本无法区分候选生成与排序错误")
    evidence = contract.get("candidate_evidence")
    if isinstance(evidence, Mapping):
        partial = int(evidence.get(CANDIDATE_EVIDENCE_PARTIAL) or 0)
        none = int(evidence.get(CANDIDATE_EVIDENCE_NONE) or 0)
        if partial:
            findings.append(f"{partial} 条记录的候选集只有分数没有分项证据"
                            "（schema 2 的 [score, topic_id] 形式），"
                            "候选生成与排序可以区分，但排序原因不可解释")
        if none:
            findings.append(f"{none} 条记录完全没有候选集字段")
    outcome = contract.get("outcome")
    if isinstance(outcome, Mapping):
        present = int(outcome.get("present") or 0)
        absent = int(outcome.get("absent") or 0)
        if absent:
            findings.append(
                f"{absent} 条记录没有最终发送结果（schema 2 不写 outcome）："
                "在这些记录上，「该回但被作息压掉」与「该回而路由没回」是同一条记录，"
                "只能按路由准入解释")
        if present:
            reasons = outcome.get("suppression_reasons")
            rendered = "、".join(f"{key}×{value}" for key, value in
                                 sorted(dict(reasons or {}).items())[:6])
            findings.append(f"{present} 条记录带最终结果"
                            + (f"，压制原因：{rendered}" if rendered else ""))
    totals = contract.get("contribution_total")
    if isinstance(totals, Mapping):
        absent_scores = int(totals.get("absent") or 0)
        if absent_scores:
            findings.append(f"{absent_scores} 条记录的 participation.contribution_total 为 null"
                            "（不是 0），阈值回放对它们只能沿用原判定")
    truncated = int(contract.get("truncated") or 0)
    if truncated:
        findings.append(f"{truncated} 条记录因超过每会话上限被截断，没有进入学习样本")
    return findings


# ---- assembly -----------------------------------------------------------

QUALITY_SCHEMA_VERSION = 1


def trace_schema_block(samples: Sequence[LearningSample],
                       contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Which trace schemas the host writes, and which ones this reader knows.

    Reported as its own block because it is the *host's* number. A reader
    upgrade must never move it, and a host upgrade must never be inferred from a
    reader's version — the two are separate protocols and the whole point of
    naming them is that neither can be read off the other.
    """
    observed: dict[str, int] = {}
    for sample in samples:
        trace = sample.trace if isinstance(sample.trace, Mapping) else {}
        declared = declared_schema_value(trace)
        key = str(declared) if declared is not None else "missing"
        observed[key] = observed.get(key, 0) + 1
    if isinstance(contract, Mapping) and not observed:
        # Fall back to the raw plane when the sample plane carries no version at
        # all (an empty dataset), so the block still describes what was imported.
        raw = contract.get("routing_schema_versions")
        if isinstance(raw, Mapping):
            observed = {str(key): int(value) for key, value in raw.items()}
    known = {str(schema) for schema in SUPPORTED_TRACE_SCHEMAS}
    return {
        "supported": list(SUPPORTED_TRACE_SCHEMAS),
        "latest": LATEST_TRACE_SCHEMA,
        "observed": dict(sorted(observed.items())),
        "unreadable": sorted(key for key in observed
                             if key not in known and key != "missing"),
    }


def quality_report(
    samples: Sequence[LearningSample],
    *,
    contract: Mapping[str, Any] | None = None,
    reader_version: int | None = None,
    ingest_at: float | None = None,
    min_samples: int = MIN_CAPABILITY_SAMPLES,
    now: float | None = None,
) -> dict[str, Any]:
    """One response for `GET /quality`.

    The sample plane is always live; the contract plane is a snapshot from the
    last import, so it carries its own timestamp instead of pretending to be
    current.
    """
    stamp = now if now is not None else time.time()
    found = capabilities(samples, min_samples=min_samples)
    return {
        "quality_schema_version": QUALITY_SCHEMA_VERSION,
        # This plugin's own reader revision, so a reader can tell "the host never
        # wrote this field" from "we were not reading it yet". Deliberately not
        # called a contract version: the host's trace schema lives in the
        # `trace` block, and one word for both is how they get confused.
        "reader_version": reader_version,
        "trace": trace_schema_block(samples, contract),
        "generated_at": stamp,
        "dataset": dataset_health(samples),
        "capabilities": {name: row.as_dict() for name, row in found.items()},
        "blocked": _blocked_lines(found),
        "contract": dict(contract) if isinstance(contract, Mapping) else None,
        "contract_at": ingest_at,
        "contract_findings": contract_findings(contract),
        "notes": [
            SAMPLE_NOTE,
            SCOPE_LEVEL_NOTE,
            TIMESTAMP_NOTE,
            OUTCOME_LEVEL_NOTE,
            "契约面来自最近一次导入的原始记录，样本面每次请求实时重算；两者计数口径不同"
            "（去重、样本上限、每会话截断），不应相互对齐。",
        ],
    }


# ---- the dataset gate ---------------------------------------------------
#
# Everything above describes the corpus. This decides whether a policy may be
# offered from it at all, and it is deliberately a *pre*-learning check: a
# recommendation produced from a corpus that cannot support it is worse than no
# recommendation, because it arrives with the same confident formatting.

GATE_OK = "ok"
GATE_WARN = "warn"
GATE_BLOCK = "block"

GATE_MIN_POSITIVE_NOTE = ("正类比例过低时，F1 的分子几乎恒为 0，"
                          "任何阈值移动都只会改变分母")

# Checks that stop a policy, and checks that only qualify one. The split matters:
# sparse candidate evidence makes the *topic* direction unusable and says nothing
# about whether the recipient direction is sound, so it is reported and does not
# veto a recommendation it cannot speak to.
GATE_BLOCKING = ("samples", "sessions", "label_age")
GATE_QUALIFYING = ("balance", "degraded", "candidate_coverage", "outcome_coverage")


def _gate_row(name: str, status: str, detail: str, *,
              value: Any = None, threshold: Any = None) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail,
            "value": value, "threshold": threshold, "blocking": name in GATE_BLOCKING}


def _positive_rate(rows: Sequence[LearningSample]) -> float | None:
    if not rows:
        return None
    positives = sum(1 for row in rows if row.expected in ("reply", "bot"))
    return positives / len(rows)


def dataset_gate(
    samples: Sequence[LearningSample],
    *,
    config: LearningConfig | None = None,
    contract: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Whether this corpus is good enough to learn from, and where it is not.

    Stated as a list of named checks rather than one score, because the fixes
    differ: too few labels, too few conversations, a label set that is almost
    entirely negative, and a corpus annotated months ago that no longer
    describes the group are four different problems.
    """
    settings = config or LearningConfig()
    stamp = now if now is not None else time.time()
    rows = list(samples)
    checks: list[dict[str, Any]] = []

    total = len(rows)
    checks.append(_gate_row(
        "samples", GATE_OK if total >= settings.gate_min_samples else GATE_BLOCK,
        f"标注样本 {total} 条（门槛 {settings.gate_min_samples}）："
        + ("足够开始学习。" if total >= settings.gate_min_samples
           else "样本不足时任何指标都只是噪声，先继续标注。"),
        value=total, threshold=settings.gate_min_samples))

    sessions = len({row.session_hash for row in rows})
    checks.append(_gate_row(
        "sessions", GATE_OK if sessions >= settings.gate_min_sessions else GATE_BLOCK,
        f"会话 {sessions} 个（门槛 {settings.gate_min_sessions}）："
        + ("可以切出会话留出集。" if sessions >= settings.gate_min_sessions
           else "会话太少，切不出不共享会话的留出集，评测无法进行。"),
        value=sessions, threshold=settings.gate_min_sessions))

    balance_rows: list[str] = []
    balance_status = GATE_OK
    for task in (TASK_RECIPIENT, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME):
        subset = [row for row in rows if row.task == task]
        rate = _positive_rate(subset)
        if rate is None:
            continue
        if min(rate, 1.0 - rate) < settings.gate_min_positive_rate:
            balance_status = GATE_WARN
            balance_rows.append(f"{task} 正类占比 {rate:.1%}")
    checks.append(_gate_row(
        "balance", balance_status,
        ("；".join(balance_rows) + "。" + GATE_MIN_POSITIVE_NOTE) if balance_rows
        else "各任务的正负样本比例都在可用区间内。"))

    degraded = sum(1 for row in rows if _degraded(row))
    ratio = (degraded / total) if total else 0.0
    checks.append(_gate_row(
        "degraded", GATE_OK if ratio <= settings.gate_max_degraded_ratio else GATE_WARN,
        f"{degraded}/{total} 条样本的决策轨迹降级（门槛 {settings.gate_max_degraded_ratio:.0%}）："
        + ("证据可回放。" if ratio <= settings.gate_max_degraded_ratio
           else "降级过多说明本体写入的字段与读取契约不一致，先对齐契约。"),
        value=round(ratio, 4), threshold=settings.gate_max_degraded_ratio))

    topic_rows = [row for row in rows if row.task == TASK_TOPIC]
    recorded = sum(1 for row in topic_rows if row.topic_candidates_recorded)
    coverage = (recorded / len(topic_rows)) if topic_rows else None
    checks.append(_gate_row(
        "candidate_coverage",
        GATE_WARN if (coverage is not None and coverage < 0.7) else GATE_OK,
        (f"{recorded}/{len(topic_rows)} 条话题标注记录了候选集（{coverage:.1%}）："
         + ("可以区分候选生成与排序错误。" if coverage >= 0.7
            else "覆盖率偏低，放宽阈值的回放只是下界，话题方向的可信度随之下降。"))
        if coverage is not None else "没有话题标注，候选覆盖无从统计。",
        value=round(coverage, 4) if coverage is not None else None, threshold=0.7))

    labelled = [row for row in rows if row.task == TASK_REPLY_ADMISSION]
    outcomes = sum(1 for row in labelled if row.outcome.recorded)
    outcome_coverage = (outcomes / len(labelled)) if labelled else None
    checks.append(_gate_row(
        "outcome_coverage",
        GATE_WARN if (outcome_coverage is not None and outcome_coverage < 0.5) else GATE_OK,
        (f"{outcomes}/{len(labelled)} 条回复标注记录了最终发送结果（{outcome_coverage:.1%}）："
         + ("门禁压制与真正的漏回复可以分开。" if outcome_coverage >= 0.5
            else "缺少最终结果，回复层只能按路由准入解释，门禁压制会被算成路由错误。"))
        if outcome_coverage is not None else "没有回复标注，最终结果覆盖无从统计。",
        value=round(outcome_coverage, 4) if outcome_coverage is not None else None,
        threshold=0.5))

    timestamps = [row.timestamp for row in rows if row.timestamp]
    age_days = ((stamp - max(timestamps)) / 86_400.0) if timestamps else None
    checks.append(_gate_row(
        "label_age",
        GATE_OK if (age_days is not None
                    and age_days <= settings.gate_max_label_age_days) else GATE_BLOCK,
        (f"最新一条标注在 {age_days:.1f} 天前"
         + ("（门槛 {:.0f} 天）。".format(settings.gate_max_label_age_days)
            if age_days <= settings.gate_max_label_age_days
            else "，超过 {} 天：群里的行为习惯可能已经变了，"
                 "先补一段新标注再学习。".format(settings.gate_max_label_age_days)))
        if age_days is not None else "没有任何带时间戳的标注。",
        value=round(age_days, 2) if age_days is not None else None,
        threshold=settings.gate_max_label_age_days))

    versions: dict[str, int] = {}
    for row in rows:
        trace = row.trace if isinstance(row.trace, Mapping) else {}
        declared = declared_schema_value(trace)
        key = str(declared) if declared is not None else "missing"
        versions[key] = versions.get(key, 0) + 1
    checks.append(_gate_row(
        "schema", GATE_OK,
        "schema 分布：" + ("、".join(f"{key}×{value}" for key, value in sorted(versions.items()))
                           or "无"),
        value=versions))

    blocked = [row["name"] for row in checks if row["blocking"] and row["status"] == GATE_BLOCK]
    return {
        "ok": not blocked,
        "checks": checks,
        "blocked_by": blocked,
        "contract_available": bool(contract),
        "summary": ("数据可以支撑学习。" if not blocked
                    else "数据不满足学习门槛（" + "、".join(blocked) + "），本次不出策略。"),
    }


def _blocked_lines(found: Mapping[str, CapabilityHealth]) -> list[str]:
    """Plain sentences naming what the corpus currently cannot support."""
    lines: list[str] = []
    for name in (CAPABILITY_TOPIC_ATTRIBUTION, CAPABILITY_TOPIC_THRESHOLD_REPLAY,
                 CAPABILITY_CANDIDATE_EVIDENCE, CAPABILITY_RECIPIENT_REPLAY,
                 CAPABILITY_REPLY_ADMISSION_REPLAY, CAPABILITY_FINAL_REPLY_OUTCOME):
        row = found.get(name)
        if row is None or row.status in (STATUS_OK, STATUS_INSUFFICIENT):
            continue
        if row.status == STATUS_UNSUPPORTED and row.total == 0:
            lines.append(f"{name}：还没有样本。")
            continue
        if row.coverage is None:
            lines.append(f"{name}：{row.eligible}/{row.total} 条可用。")
        else:
            lines.append(f"{name}：{row.eligible}/{row.total} 条可用，"
                         f"覆盖率 {row.coverage:.1%}。")
    if not lines and found:
        lines.append("各项能力的覆盖率都在阈值以上：本批样本可以支撑候选归因与阈值回放。")
    return lines


__all__ = [
    "CAPABILITY_BUILDERS", "CAPABILITY_CANDIDATE_EVIDENCE", "CAPABILITY_FINAL_REPLY_OUTCOME",
    "GATE_BLOCK", "GATE_BLOCKING", "GATE_OK", "GATE_QUALIFYING", "GATE_WARN", "dataset_gate",
    "CAPABILITY_RECIPIENT_REPLAY", "CAPABILITY_REPLY_ADMISSION_REPLAY", "CAPABILITY_SCOPE_IDENTITY",
    "CAPABILITY_TOPIC_ATTRIBUTION", "CAPABILITY_TOPIC_THRESHOLD_REPLAY", "COVERAGE_OK",
    "CapabilityHealth", "DEFINITION_CANDIDATE_EVIDENCE", "DEFINITION_FINAL_REPLY_OUTCOME",
    "MIN_CAPABILITY_SAMPLES", "OUTCOME_LEVEL_NOTE", "PLANE_CONTRACT", "PLANE_SAMPLE",
    "QUALITY_SCHEMA_VERSION", "RawContractStats", "SCOPE_LEVEL_NOTE", "STATUS_INSUFFICIENT",
    "STATUS_LABEL", "STATUS_OK", "STATUS_UNSUPPORTED", "STATUS_WARNING", "TIMESTAMP_NOTE",
    "candidate_evidence", "capabilities", "contract_findings", "dataset_health",
    "final_reply_outcome", "quality_report", "recipient_replay", "reply_admission_replay",
    "scope_identity", "topic_attribution", "topic_threshold_replay", "trace_schema_block",
]
