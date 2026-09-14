"""Per-message post-mortem of the reply decision, written by a model.

The capability matrix answers "can this corpus support a threshold replay". This
module answers a different question, one message at a time: *should the bot have
replied to that one*. The evidence is different too — a judgement about one
message needs the message, and the learning layer has never stored text.

So the text is read from the host shared preferences for a single call and never
written back: not into the sample store, not into the review cache. What the page
shows afterwards is the model verdict beside the facts the corpus already had,
and the text snippet is only as long as it takes to recognise the message.

What the model is **not** shown is the point. It does not see the human label,
the host admission decision, or whether anything was delivered — otherwise it is
not a second opinion, it is a reviewer who has already been told the answer. It
sees the text, whether the message referred to the bot, and which of the messages
belong to the same conversation (as an anonymous group label). Every comparison
is therefore worth reading: model against human, model against what actually
went out, and the two against each other.
"""
from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, Mapping, Sequence

from .outcome import parse_record_outcome

REPLY_REVIEW_SCHEMA_VERSION = 1
REPLY_REVIEW_PROMPT_VERSION = 1

MAX_MESSAGES = 40
DEFAULT_MAX_MESSAGES = 12
MAX_TEXT_IN_DIGEST = 600
MAX_TEXT_IN_PANEL = 200
MAX_REASON = 400
MAX_SUMMARY = 400
MAX_PATTERNS = 5

# The host routing cut: `strong` is the level that enters the reply flow.
ADMITTED_LEVEL = "strong"
LEVELS = ("strong", "hover", "weak")

VERDICT_MISSED = "missed"
VERDICT_OVER_REPLIED = "over_replied"
VERDICT_AGREED_REPLY = "agreed_reply"
VERDICT_AGREED_SILENT = "agreed_silent"
VERDICT_UNDECIDED = "undecided"
VERDICT_LABEL = {
    VERDICT_MISSED: "模型认为漏回",
    VERDICT_OVER_REPLIED: "模型认为多回",
    VERDICT_AGREED_REPLY: "与实际一致（回）",
    VERDICT_AGREED_SILENT: "与实际一致（没回）",
    VERDICT_UNDECIDED: "模型没有判断",
}

DISAGREE_LABEL = {
    "agree": "与人工一致",
    "disagree": "与人工不一致",
    "unknown": "没有人工标注",
}

_NUMBER = re.compile("[0-9]+(?:[.][0-9]+)?")
FENCE = chr(96) * 3


# ---- reading the host records -------------------------------------------

def _text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def message_facts(session_key: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """One host annotation record, reduced to the facts a post-mortem needs.

    The record is the same shape the sample builder reads, so this stays a
    reader of the host contract rather than a second parser for it.
    """
    trace = _mapping(record.get("decision_trace"))
    participation = _mapping(trace.get("participation"))
    identity = _mapping(trace.get("identity"))
    outcome = parse_record_outcome(record)
    level = participation.get("level")
    mentioned = bool(identity.get("mention")) or bool(identity.get("vocative"))
    reference = identity.get("bot_reference")
    if isinstance(reference, str) and reference not in ("", "none"):
        mentioned = True
    expected = record.get("expected_reply")
    return {
        "session_key": str(session_key or ""),
        "msg_id": _text(record.get("msg_id"), 128),
        "annotated_at": float(record.get("annotated_at") or 0.0),
        "text": _text(record.get("text"), MAX_TEXT_IN_DIGEST),
        "label_expected_reply": expected if isinstance(expected, bool) else None,
        "host_level": level if level in LEVELS else None,
        "host_should_reply": (participation.get("should_reply")
                              if isinstance(participation.get("should_reply"), bool) else None),
        "mentions_bot": mentioned,
        "outcome": outcome.as_dict(),
        "recorded": outcome.recorded,
        "delivered": outcome.delivered if outcome.recorded else None,
    }


def select_messages(annotations: Sequence[tuple[str, Any]], *,
                    limit: int = DEFAULT_MAX_MESSAGES) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The messages worth reviewing, newest first, plus what the batch looks like.

    A message qualifies when the host recorded *something* about the turn: a
    human reply label, a routing level, or a final outcome. Rows where nothing
    at all was recorded have nothing to compare a judgement against, and
    reviewing them would produce prose with no possible disagreement.
    """
    facts: list[dict[str, Any]] = []
    for session_key, record in annotations:
        if not isinstance(record, Mapping):
            continue
        item = message_facts(session_key, record)
        if not item["msg_id"]:
            continue
        if (item["label_expected_reply"] is None and not item["recorded"]
                and item["host_level"] is None):
            continue
        facts.append(item)
    facts.sort(key=lambda item: (item["annotated_at"], item["msg_id"]), reverse=True)
    cap = max(1, min(MAX_MESSAGES, int(limit)))
    chosen = facts[:cap]
    stats = {
        "candidates": len(facts),
        "selected": len(chosen),
        "with_text": sum(1 for item in chosen if item["text"]),
        "with_label": sum(1 for item in chosen if item["label_expected_reply"] is not None),
        "with_outcome": sum(1 for item in chosen if item["recorded"]),
        "sessions": len({item["session_key"] for item in chosen}),
        "mismatched": sum(1 for item in chosen
                          if item["label_expected_reply"] is not None and item["recorded"]
                          and item["label_expected_reply"] != bool(item["delivered"])),
    }
    return chosen, stats


# ---- the digest: what the model may see ---------------------------------

def _session_labels(messages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Stable, anonymous conversation labels for one call.

    The model needs to know which messages came from the same conversation; it
    does not need to know which conversation. The label is a hash of the session
    key truncated for reading, and it is never written anywhere.
    """
    labels: dict[str, str] = {}
    for item in messages:
        key = str(item.get("session_key") or "")
        if key and key not in labels:
            digest = sha256(key.encode("utf-8")).hexdigest()[:6]
            labels[key] = "c" + digest
    return labels


def build_digest(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The bounded facts a reply post-mortem is allowed to be based on.

    Deliberately without the human label, the host decision and the recorded
    outcome: a judge that has been shown the answer agrees with it.
    """
    labels = _session_labels(messages)
    return {
        "reply_review_schema_version": REPLY_REVIEW_SCHEMA_VERSION,
        "prompt_version": REPLY_REVIEW_PROMPT_VERSION,
        "order": "newest_first",
        "messages": [
            {
                "msg_id": str(item.get("msg_id") or ""),
                "conversation": labels.get(str(item.get("session_key") or ""), ""),
                "text": str(item.get("text") or "") or None,
                "mentions_bot": bool(item.get("mentions_bot")),
            }
            for item in messages
        ],
    }


def digest_fingerprint(digest: Mapping[str, Any]) -> str:
    """Stable identity of one batch, so an identical batch is not re-asked."""
    encoded = json.dumps(digest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


# ---- the prompt ---------------------------------------------------------

SYSTEM_PROMPT = """你是 ChatDynamics 的回复复盘器。

输入是一批「机器人处理过的群消息」，按时间倒序排列。逐条回答同一个问题：**这条消息，机器人当时应该回复吗？**

硬性规则：
1. 只根据给你的正文判断。没有正文（text 为 null）的条目，confidence 写 0，reason 写「没有正文」。
2. 你看不到人工标注，也看不到机器人当时是怎么判的。不要猜「标注者想要什么」，给出你自己的判断。
3. 你只看到被标注过的若干条，不是完整对话，也不一定连续。上下文不足时降低 confidence，并在 reason 里说清缺什么。
4. should_reply 必须是布尔值；confidence 是 0~1 的数字；reason 用一句话说明依据，不要复述规则。
5. 每条 msg_id 必须原样引用，不要新增、不要漏。漏掉的条目会被记成「没有判断」。
6. 只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字。

输出结构：
{"summary": "整批的一句话结论",
 "rows": [{"msg_id": "原样引用", "should_reply": true, "confidence": 0.8,
           "reason": "为什么该回或不回"}],
 "patterns": ["反复出现的判断模式，最多 5 条"]}"""


def build_prompt(digest: Mapping[str, Any]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one batch."""
    body = json.dumps(digest, ensure_ascii=False, indent=1)
    return SYSTEM_PROMPT, "待复盘消息（按时间倒序）：\n" + body + "\n\n只输出 JSON 对象。"


# ---- reading the reply --------------------------------------------------

def _extract_json(text: str) -> Any:
    """The first JSON object in a reply, fenced or bare; None when there is none."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    if FENCE in stripped:
        for part in stripped.split(FENCE)[1::2]:
            candidates.append(part.strip().removeprefix("json").strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, round(float(value), 3)))


def compare(item: Mapping[str, Any], should_reply: bool, decided: bool) -> dict[str, Any]:
    """The model judgement beside the facts that were hidden from it."""
    human = item.get("label_expected_reply")
    delivered = item.get("delivered")
    admitted = item.get("host_level") == ADMITTED_LEVEL
    if not decided:
        return {"verdict": VERDICT_UNDECIDED, "verdict_label": VERDICT_LABEL[VERDICT_UNDECIDED],
                "vs_human": "unknown", "vs_human_label": DISAGREE_LABEL["unknown"],
                "host_admitted": admitted if item.get("host_level") else None,
                "delivered": delivered, "outcome_recorded": bool(item.get("recorded"))}
    if human is None:
        vs_human = "unknown"
    else:
        vs_human = "agree" if bool(human) == bool(should_reply) else "disagree"
    if item.get("recorded"):
        if bool(should_reply) and not bool(delivered):
            verdict = VERDICT_MISSED
        elif not bool(should_reply) and bool(delivered):
            verdict = VERDICT_OVER_REPLIED
        elif bool(should_reply):
            verdict = VERDICT_AGREED_REPLY
        else:
            verdict = VERDICT_AGREED_SILENT
    else:
        verdict = (VERDICT_AGREED_REPLY if bool(should_reply) == bool(admitted)
                   else (VERDICT_MISSED if should_reply else VERDICT_OVER_REPLIED))
    return {
        "verdict": verdict,
        "verdict_label": VERDICT_LABEL[verdict],
        "vs_human": vs_human,
        "vs_human_label": DISAGREE_LABEL[vs_human],
        "host_admitted": admitted if item.get("host_level") else None,
        "delivered": delivered,
        "outcome_recorded": bool(item.get("recorded")),
    }


def parse_reply_review(text: str, messages: Sequence[Mapping[str, Any]], *,
                       stats: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Validate a model reply against the batch; None when nothing survives.

    The judgement is the model; the facts beside it are the corpus. A row the
    model skipped is kept and marked undecided, because dropping it would make
    the batch look smaller than it was.
    """
    payload = _extract_json(text)
    if not isinstance(payload, Mapping):
        return None
    by_id = {str(item.get("msg_id") or ""): item for item in messages}
    order = [str(item.get("msg_id") or "") for item in messages]
    verdicts: dict[str, dict[str, Any]] = {}
    invented: list[str] = []
    for raw in payload.get("rows") or []:
        if not isinstance(raw, Mapping):
            continue
        identifier = raw.get("msg_id")
        identifier = identifier.strip() if isinstance(identifier, str) else ""
        if not identifier or identifier in verdicts:
            continue
        if identifier not in by_id:
            if identifier not in invented and len(invented) < 5:
                invented.append(identifier[:64])
            continue
        decision = raw.get("should_reply")
        decided = isinstance(decision, bool)
        verdicts[identifier] = {
            "should_reply": decision if decided else None,
            # A confidence attached to no decision is not a fact about anything.
            "confidence": _confidence(raw.get("confidence")) if decided else 0.0,
            "reason": _text(raw.get("reason"), MAX_REASON),
            "decided": decided,
        }
    if not verdicts and not _text(payload.get("summary"), MAX_SUMMARY):
        return None

    rows: list[dict[str, Any]] = []
    for identifier in order:
        item = by_id[identifier]
        answer = verdicts.get(identifier) or {
            "should_reply": None, "confidence": 0.0, "reason": "", "decided": False}
        comparison = compare(item, bool(answer["should_reply"]), bool(answer["decided"]))
        rows.append({
            "msg_id": identifier,
            "session": sha256(str(item.get("session_key") or "").encode("utf-8")).hexdigest()[:12],
            "text": _text(item.get("text"), MAX_TEXT_IN_PANEL),
            "has_text": bool(item.get("text")),
            "mentions_bot": bool(item.get("mentions_bot")),
            "human_expected_reply": item.get("label_expected_reply"),
            "host_level": item.get("host_level"),
            "outcome": dict(item.get("outcome") or {}),
            "model_should_reply": answer["should_reply"],
            "model_confidence": answer["confidence"],
            "model_reason": answer["reason"],
            "decided": answer["decided"],
            "annotated_at": item.get("annotated_at"),
            **comparison,
        })

    counts = {
        "decided": sum(1 for row in rows if row["decided"]),
        "undecided": sum(1 for row in rows if not row["decided"]),
        "model_reply": sum(1 for row in rows if row["model_should_reply"] is True),
        "model_silent": sum(1 for row in rows if row["model_should_reply"] is False),
        "with_human": sum(1 for row in rows if row["vs_human"] != "unknown"),
        "human_agree": sum(1 for row in rows if row["vs_human"] == "agree"),
        "human_disagree": sum(1 for row in rows if row["vs_human"] == "disagree"),
        "missed": sum(1 for row in rows if row["verdict"] == VERDICT_MISSED),
        "over_replied": sum(1 for row in rows if row["verdict"] == VERDICT_OVER_REPLIED),
    }
    return {
        "reply_review_schema_version": REPLY_REVIEW_SCHEMA_VERSION,
        "prompt_version": REPLY_REVIEW_PROMPT_VERSION,
        "summary": _text(payload.get("summary"), MAX_SUMMARY),
        "patterns": [_text(item, MAX_REASON) for item in (payload.get("patterns") or [])
                     if _text(item, MAX_REASON)][:MAX_PATTERNS],
        "rows": rows,
        "counts": counts,
        "invented_ids": invented,
        "stats": dict(stats or {}),
        "fingerprint": digest_fingerprint(build_digest(messages)),
    }


__all__ = [
    "DEFAULT_MAX_MESSAGES", "MAX_MESSAGES", "REPLY_REVIEW_PROMPT_VERSION",
    "REPLY_REVIEW_SCHEMA_VERSION", "SYSTEM_PROMPT", "VERDICT_LABEL", "VERDICT_MISSED",
    "VERDICT_OVER_REPLIED", "VERDICT_UNDECIDED", "build_digest", "build_prompt", "compare",
    "digest_fingerprint", "message_facts", "parse_reply_review", "select_messages",
]
