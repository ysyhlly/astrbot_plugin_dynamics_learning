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

from .candidates import CandidateObservation, candidate_recall
from .metrics import SAMPLE_NOTE, ratio, rounded
from .samples import TASK_RECIPIENT, TASK_REPLY, TASK_TOPIC, LearningSample
from .scope import SCOPE_SOURCE_SESSION_UMO_EQUAL, SCOPE_SOURCE_SESSION_UMO_MISMATCH, resolve_scope
from .topic_learner import candidate_observations, pair_rows, replay_can_move

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

SCOPE_LEVEL_NOTE = (
    "本体契约没有提供跨会话的群身份，因此作用域层级是会话："
    "一个会话就是一个作用域，画像最多只能做到这个粒度。"
)
REPLY_LEVEL_NOTE = (
    "本体的最终发送决策（should_reply）恒为 null，这里回放的是路由准入判定 "
    "level == strong，不是「最终是否回复」"
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
    rows = [sample for sample in samples if sample.task == TASK_REPLY]
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
    """The capability the host contract cannot offer, stated as a row.

    ChatDynamics keeps `participation.should_reply` null on every record, so no
    sample knows whether the bot actually sent anything. Showing that as a
    permanently unsupported row — rather than an error, or silence — is the point
    of a capability matrix: the gap is a fact about the data, and it becomes an
    ordinary working capability the day the host records the field.
    """
    rows = [sample for sample in samples if sample.task == TASK_REPLY]
    recorded = sum(1 for sample in rows if _should_reply_recorded(sample))
    reasons = []
    if not recorded:
        reasons.append("本体把 participation.should_reply 恒置为 null："
                       "没有任何样本记录过最终是否发送，这个能力在当前契约下不可用")
    return _health(CAPABILITY_FINAL_REPLY_OUTCOME, recorded, len(rows),
                   definition=DEFINITION_FINAL_REPLY_OUTCOME, reasons=reasons,
                   detail={"recorded": recorded, "total": len(rows)},
                   min_samples=min_samples)


def _should_reply_recorded(sample: LearningSample) -> bool:
    trace = sample.trace if isinstance(sample.trace, Mapping) else {}
    participation = trace.get("participation")
    if isinstance(participation, Mapping) and isinstance(participation.get("should_reply"), bool):
        return True
    summary = trace.get("evidence_summary")
    flagged = summary.get("should_reply") if isinstance(summary, Mapping) else None
    return isinstance(flagged, bool)


CAPABILITY_BUILDERS = (
    recipient_replay, topic_attribution, topic_threshold_replay,
    reply_admission_replay, scope_identity, final_reply_outcome,
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
                  for task in (TASK_RECIPIENT, TASK_TOPIC, TASK_REPLY)},
        "degraded_traces": sum(1 for sample in samples if _degraded(sample)),
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
    "topic_candidates_nonempty", "contribution_total_present", "contribution_total_absent",
    "contribution_total_unknown", "sessions", "distinct_group_id_sessions",
)
_COUNTER_MAPS = ("annotation_schema_versions", "routing_schema_versions", "scope_sources")


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
                  _version_bucket(trace.get("routing_schema_version")))
            self._observe_participation(trace.get("participation"))
        self._observe_topic_candidates(raw.get("routing"))

    def _observe_participation(self, participation: Any) -> None:
        value = (participation.get("contribution_total")
                 if isinstance(participation, Mapping) else None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # A null here separates "the host scored it zero" from "the host never
            # scored it"; normalisation erases the difference, so it is counted now.
            self.contribution_total_absent += 1
            return
        self.contribution_total_present += 1

    def _observe_topic_candidates(self, routing: Any) -> None:
        raw = routing.get("topic_candidates") if isinstance(routing, Mapping) else None
        if not isinstance(raw, list):
            # Present but unreadable is not evidence that the host looked and
            # found nothing, so it counts as missing — the reading
            # `candidates.parse_candidates` also gives it.
            raw = routing.get("candidates") if isinstance(routing, Mapping) else None
        if not isinstance(raw, list):
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


def quality_report(
    samples: Sequence[LearningSample],
    *,
    contract: Mapping[str, Any] | None = None,
    contract_version: int | None = None,
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
        # The *read* contract this snapshot was taken under, so a reader can tell
        # "the host never wrote this field" from "we were not reading it yet".
        "contract_version": contract_version,
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
            "契约面来自最近一次导入的原始记录，样本面每次请求实时重算；两者计数口径不同"
            "（去重、样本上限、每会话截断），不应相互对齐。",
        ],
    }


def _blocked_lines(found: Mapping[str, CapabilityHealth]) -> list[str]:
    """Plain sentences naming what the corpus currently cannot support."""
    lines: list[str] = []
    for name in (CAPABILITY_TOPIC_ATTRIBUTION, CAPABILITY_TOPIC_THRESHOLD_REPLAY,
                 CAPABILITY_RECIPIENT_REPLAY, CAPABILITY_REPLY_ADMISSION_REPLAY,
                 CAPABILITY_FINAL_REPLY_OUTCOME):
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
    "CAPABILITY_BUILDERS", "CAPABILITY_FINAL_REPLY_OUTCOME", "CAPABILITY_RECIPIENT_REPLAY",
    "CAPABILITY_REPLY_ADMISSION_REPLAY", "CAPABILITY_SCOPE_IDENTITY", "CAPABILITY_TOPIC_ATTRIBUTION",
    "CAPABILITY_TOPIC_THRESHOLD_REPLAY", "COVERAGE_OK", "CapabilityHealth",
    "DEFINITION_FINAL_REPLY_OUTCOME",
    "MIN_CAPABILITY_SAMPLES", "PLANE_CONTRACT", "PLANE_SAMPLE", "QUALITY_SCHEMA_VERSION",
    "RawContractStats", "SCOPE_LEVEL_NOTE", "STATUS_INSUFFICIENT", "STATUS_LABEL", "STATUS_OK",
    "STATUS_UNSUPPORTED", "STATUS_WARNING", "TIMESTAMP_NOTE", "capabilities",
    "contract_findings", "dataset_health", "final_reply_outcome", "quality_report",
    "recipient_replay", "reply_admission_replay", "scope_identity", "topic_attribution",
    "topic_threshold_replay",
]
