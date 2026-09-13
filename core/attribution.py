"""The error attribution chain: which layer should be changed?

A dataset that only says "reply F1 is 0.71" cannot tell a reviewer what to do.
The plan asks for something more useful, and it is the single most valuable
thing this plugin can produce:

    recipient_error        机器人没被认成收件人
    topic_candidate_miss   正确话题没进候选集（生成 / 检索）
    topic_ranking_error    正确话题进了候选集但没被选中（打分 / 排序）
    participation_error    该不该进入回复流程判断错了（参与准入）
    gate_suppression       准入没错，被作息 / 降温 / 媒体压掉（不是路由错误）
    generation_failure     进了流程，生成没产出可用回复（不是路由错误）
    delivery_failure       生成了，平台发送失败（不是路由错误）
    unattributable         记录不足以判断哪一环出错

The chain is a **review order**, not a causal claim, and the difference is
stated rather than implied. "Who is this addressed to", "what is being
discussed", "should the bot join" and "did the reply go out" are four largely
independent layers; an error in one does not cause an error in the next. What
the order buys is a deterministic answer to "where do I look first" — and a
message that fails two layers is counted **once**, in the earlier one, with the
other recorded in also_failed. Counting it twice would make the buckets sum to
more than the corpus, which is how an attribution table starts lying.

Three properties are load-bearing:

* **the buckets partition the corpus.** Every message lands in exactly one, so
  the table totals the dataset instead of quietly shrinking it;
* **the topic link reuses the learner predicate.** core.topic_learner.
  candidate_observations decides candidate-miss vs ranking-error, so the chain
  cannot disagree with the topic learner sitting next to it;
* **"no outcome recorded" is not "no reply".** A schema 2 message whose
  admission was right and whose delivery is unknown lands in ok with an explicit
  outcome_unavailable flag, never in a failure bucket. Guessing there is exactly
  the mistake the two-layer reply split exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import buckets
from .candidates import (
    ATTRIBUTION_CANDIDATE_MISS, ATTRIBUTION_CORRECT, ATTRIBUTION_NEW_TOPIC,
    ATTRIBUTION_NOT_RECORDED, ATTRIBUTION_RANKING_ERROR,
)
from .outcome import EMPTY as OUTCOME_EMPTY, STAGE_DELIVERY, STAGE_GATE, STAGE_GENERATION
from .outcome import FinalOutcome
from .recommendation import CONFIDENCE_INSUFFICIENT, confidence_for
from .samples import (
    BOT, REPLY, TASK_RECIPIENT, TASK_REPLY_ADMISSION, TASK_REPLY_OUTCOME, TASK_TOPIC,
    LearningSample,
)
from .topic_learner import candidate_observations

ATTRIBUTION_SCHEMA_VERSION = 1

# Minimum number of attributed messages before the table is worth reading at
# all. Below it the report still prints every count, and says what it is.
MIN_ATTRIBUTED_MESSAGES = 20

# candidates.py outcome -> chain bucket. None means "this link did not fail",
# which is a different statement from "this link failed in an unknown way".
TOPIC_LINK: dict[str, str | None] = {
    ATTRIBUTION_CORRECT: None,
    ATTRIBUTION_NEW_TOPIC: None,
    ATTRIBUTION_CANDIDATE_MISS: buckets.TOPIC_CANDIDATE_MISS,
    ATTRIBUTION_RANKING_ERROR: buckets.TOPIC_RANKING_ERROR,
    ATTRIBUTION_NOT_RECORDED: buckets.UNATTRIBUTABLE,
}

CHOICE_UNKNOWN = "unrecorded"


@dataclass(frozen=True)
class MessageChain:
    """One message, seen as the links that decided it.

    The four slots are optional because the host labels are optional: a record
    with only bot_targeted produces a chain with one link, and reporting that as
    "the rest was fine" would be reading a missing label as a passed check.
    present names what was actually there.
    """

    session_hash: str
    msg_id: str
    recipient: LearningSample | None = None
    topic: LearningSample | None = None
    admission: LearningSample | None = None
    outcome: LearningSample | None = None

    @property
    def slots(self) -> tuple[LearningSample | None, ...]:
        return (self.recipient, self.topic, self.admission, self.outcome)

    @property
    def present(self) -> tuple[str, ...]:
        named = ((TASK_RECIPIENT, self.recipient), (TASK_TOPIC, self.topic),
                 (TASK_REPLY_ADMISSION, self.admission), (TASK_REPLY_OUTCOME, self.outcome))
        return tuple(name for name, sample in named if sample is not None)

    @property
    def final(self) -> FinalOutcome:
        """The recorded final outcome, from whichever sample carries it.

        Every sample of a message is written with the same outcome block, so the
        search order only matters for rows stored before that became true.
        """
        for sample in self.slots:
            if sample is None:
                continue
            found = sample.outcome
            if found.recorded:
                return found
        return OUTCOME_EMPTY


@dataclass(frozen=True)
class Attribution:
    """One message, reduced to the layer a reviewer should open first."""

    session_hash: str
    msg_id: str
    bucket: str
    reason: str = ""
    also_failed: tuple[str, ...] = ()
    outcome_unavailable: bool = True
    present: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return buckets.label_for(self.bucket)

    @property
    def action(self) -> str:
        return buckets.ACTION.get(self.bucket, "")

    @property
    def is_model_error(self) -> bool:
        return buckets.is_model_error(self.bucket)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_hash": self.session_hash,
            "msg_id": self.msg_id,
            "bucket": self.bucket,
            "bucket_label": self.label,
            "reason": self.reason,
            "action": self.action,
            "also_failed": list(self.also_failed),
            "outcome_unavailable": self.outcome_unavailable,
            "present": list(self.present),
            "evidence": dict(self.evidence),
        }


def build_chains(samples: Sequence[LearningSample]) -> list[MessageChain]:
    """Group samples into one chain per (session, message).

    The key is the pair, never the message id alone: message ids are only unique
    inside a session, and merging two sessions on a shared id would attribute
    one conversation's failure to another conversation's evidence.
    """
    grouped: dict[tuple[str, str], dict[str, LearningSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.session_hash, sample.msg_id), {})[sample.task] = sample
    chains = [
        MessageChain(
            session_hash=session_hash,
            msg_id=msg_id,
            recipient=by_task.get(TASK_RECIPIENT),
            topic=by_task.get(TASK_TOPIC),
            admission=by_task.get(TASK_REPLY_ADMISSION),
            outcome=by_task.get(TASK_REPLY_OUTCOME),
        )
        for (session_hash, msg_id), by_task in grouped.items()
    ]
    chains.sort(key=lambda chain: (chain.session_hash, chain.msg_id))
    return chains


def _topic_failure(topic: LearningSample) -> tuple[str, str] | None:
    """Which topic layer failed, asked of the topic learner's own predicate.

    Re-deriving "was the right topic in the candidate set" here would let the
    chain and the topic learner disagree about the same message while both
    printed a number. candidate_observations is the one implementation, and a
    NEW singleton is not a candidate problem at all — the topic did not exist
    yet — so it never becomes a chain bucket.
    """
    observation = candidate_observations([topic])[0]
    found = observation.attribution
    bucket = TOPIC_LINK.get(found, buckets.UNATTRIBUTABLE)
    if bucket is None:
        return None
    if found == ATTRIBUTION_NOT_RECORDED:
        return bucket, "话题判定与标注不一致，但记录里没有候选集，无法区分候选生成与排序错误"
    if found == ATTRIBUTION_CANDIDATE_MISS:
        return bucket, "正确话题没有进入候选集：问题在候选生成（embedding / 检索）"
    if found == ATTRIBUTION_RANKING_ERROR:
        return bucket, "正确话题在候选集里但没有被选中：问题在打分与排序"
    return bucket, "话题标注为可归属，但既没有预测也没有候选信息"


def _execution_failure(chain: MessageChain, admission_ok: bool) -> tuple[str, str] | None:
    """Where the reply stopped, once the admission layer is known to be right."""
    final = chain.final
    if not final.recorded:
        return None
    admission = chain.admission
    should_reply = admission is not None and admission.expected == REPLY
    if final.is_delivered:
        if admission is not None and not should_reply and admission_ok:
            return (buckets.UNATTRIBUTABLE,
                    "标注为不该回复，但记录显示已经发送；准入判定与标注一致，"
                    "记录轨迹里没有任何一环该为这次发送负责")
        return None
    if not should_reply or not admission_ok:
        # Either nothing was expected, or the admission layer already owns this
        # failure. Neither is an execution finding.
        return None
    reason = final.suppression_reason or "未记录的原因"
    if final.stage == STAGE_GATE:
        return buckets.GATE_SUPPRESSION, f"准入判定正确，但门禁以 {reason} 压掉了发送"
    if final.stage == STAGE_GENERATION:
        return buckets.GENERATION_FAILURE, f"准入判定正确，生成阶段失败（{reason}）"
    if final.stage == STAGE_DELIVERY:
        return buckets.DELIVERY_FAILURE, f"准入判定正确，发送阶段失败（{reason}）"
    return (buckets.UNATTRIBUTABLE,
            "该回复但没有发送出去，而记录没有说明停在哪一环"
            "（既没有可识别的压制原因，也没有阶段字段）")


def classify(chain: MessageChain) -> Attribution:
    """The first failing link, plus every other link that also failed."""
    failures: list[tuple[str, str]] = []

    recipient = chain.recipient
    if recipient is not None and not recipient.correct:
        failures.append((buckets.RECIPIENT_ERROR,
                         f"人工标注 bot_targeted={recipient.expected == BOT}，"
                         f"记录判定 {recipient.predicted}"))

    topic = chain.topic
    if topic is not None and not topic.correct:
        found = _topic_failure(topic)
        if found is not None:
            failures.append(found)

    admission = chain.admission
    admission_ok = admission is None or admission.correct
    if admission is not None and not admission.correct:
        failures.append((buckets.PARTICIPATION_ERROR,
                         f"标注 expected_reply={admission.expected == REPLY}，"
                         f"记录的路由准入判定为 {admission.predicted}"))

    execution = _execution_failure(chain, admission_ok)
    if execution is not None:
        failures.append(execution)

    final = chain.final
    primary = failures[0] if failures else (buckets.OK, "")
    also = tuple(dict.fromkeys(bucket for bucket, _reason in failures[1:]))
    return Attribution(
        session_hash=chain.session_hash,
        msg_id=chain.msg_id,
        bucket=primary[0],
        reason=primary[1],
        also_failed=also,
        outcome_unavailable=not final.recorded,
        present=chain.present,
        evidence={
            "recipient": recipient.predicted if recipient is not None else CHOICE_UNKNOWN,
            "recipient_expected": recipient.expected if recipient is not None else CHOICE_UNKNOWN,
            "topic": topic.predicted if topic is not None else CHOICE_UNKNOWN,
            "topic_expected": topic.expected if topic is not None else CHOICE_UNKNOWN,
            "admission": admission.predicted if admission is not None else CHOICE_UNKNOWN,
            "admission_expected": admission.expected if admission is not None else CHOICE_UNKNOWN,
            "outcome": final.value if final.recorded else None,
            "outcome_stage": final.stage if final.recorded else None,
            "suppression_reason": final.suppression_reason,
        },
    )


def attribute(samples: Sequence[LearningSample]) -> list[Attribution]:
    return [classify(chain) for chain in build_chains(samples)]


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def attribution_report(
    samples: Sequence[LearningSample],
    *,
    min_messages: int = MIN_ATTRIBUTED_MESSAGES,
    min_samples: int = MIN_ATTRIBUTED_MESSAGES,
    examples: int = 8,
) -> dict[str, Any]:
    """The attribution table, with the facts that decide how to read it."""
    chains = build_chains(samples)
    rows = [classify(chain) for chain in chains]
    total = len(rows)
    counts: dict[str, int] = dict.fromkeys(buckets.ORDER, 0)
    also: dict[str, int] = {}
    stages: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for row in rows:
        counts[row.bucket] = counts.get(row.bucket, 0) + 1
        for bucket in row.also_failed:
            also[bucket] = also.get(bucket, 0) + 1
        if row.bucket in buckets.SYSTEM_EVENTS:
            stage = row.evidence.get("outcome_stage")
            if isinstance(stage, str):
                stages[stage] = stages.get(stage, 0) + 1
            suppression = row.evidence.get("suppression_reason")
            if isinstance(suppression, str) and suppression:
                reasons[suppression] = reasons.get(suppression, 0) + 1
    model_errors = sum(counts.get(bucket, 0) for bucket in buckets.MODEL_ERRORS)
    system_events = sum(counts.get(bucket, 0) for bucket in buckets.SYSTEM_EVENTS)
    unavailable = sum(1 for row in rows if row.outcome_unavailable)
    coverage = {
        "recipient": sum(1 for chain in chains if chain.recipient is not None),
        "topic": sum(1 for chain in chains if chain.topic is not None),
        "admission": sum(1 for chain in chains if chain.admission is not None),
        "outcome": total - unavailable,
    }
    return {
        "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "messages": total,
        "counts": counts,
        "rates": {bucket: _rate(counts.get(bucket, 0), total) for bucket in buckets.ORDER},
        "labels": dict(buckets.LABEL),
        "actions": dict(buckets.ACTION),
        "model_errors": model_errors,
        "model_error_rate": _rate(model_errors, total),
        "system_events": system_events,
        "system_event_rate": _rate(system_events, total),
        "also_failed": dict(sorted(also.items(), key=lambda item: (-item[1], item[0]))),
        "execution_stages": dict(sorted(stages.items(), key=lambda item: (-item[1], item[0]))),
        "suppression_reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        "outcome_unavailable": unavailable,
        "coverage": coverage,
        "confidence": (confidence_for(total, min_samples=min_samples)
                       if total else CONFIDENCE_INSUFFICIENT),
        "examples": [row.as_dict() for row in rows if row.bucket != buckets.OK][:examples],
        "notes": _notes(total, counts, unavailable, coverage, min_messages),
    }


def _notes(total: int, counts: Mapping[str, int], unavailable: int,
           coverage: Mapping[str, int], min_messages: int) -> list[str]:
    notes: list[str] = []
    if not total:
        notes.append("还没有可归因的消息：先导入标注。")
        return notes
    if total < min_messages:
        notes.append(f"只有 {total} 条消息可归因，低于 {min_messages} 条："
                     "下面每一格都是计数，不是比例估计。")
    if unavailable:
        notes.append(f"{unavailable}/{total} 条消息没有记录最终发送结果（schema 2）："
                     "它们只能归因到路由层。")
    if not coverage.get("topic"):
        notes.append("没有任何消息带话题标注，话题两格必然为空。")
    if not coverage.get("admission"):
        notes.append("没有任何消息带回复标注，参与准入与执行两段必然为空。")
    suppressed = counts.get(buckets.GATE_SUPPRESSION, 0)
    if suppressed:
        notes.append(f"{suppressed} 条是门禁压制：准入判定是对的，"
                     "把它读成「路由漏回复」会去动一个没有错的阈值。")
    notes.append("归因表是复查顺序而不是因果结论：一条消息可能同时错在多环，"
                 "只记入最先的一格，其余在 also_failed 里。")
    return notes


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION", "Attribution", "CHOICE_UNKNOWN", "MIN_ATTRIBUTED_MESSAGES",
    "MessageChain", "TOPIC_LINK", "attribute", "attribution_report", "build_chains", "classify",
]
