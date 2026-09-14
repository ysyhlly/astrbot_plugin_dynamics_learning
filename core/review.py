"""The contract panel, reread by a model.

The capability matrix is a table of facts: denominators, coverage, and the
predicate the learner itself would run. What it cannot do is tell the person
holding the batch what that batch means *for them* — which missing label is
blocking which analysis, which row is worth acting on first, and whether a zero
in a numerator is a hole in the host contract or simply nothing labelled yet.
That reading is what a model is for.

Two rules keep it honest:

* **Every number comes from the digest.** build_digest() is the only thing the
  model is shown, parse_review() takes the counts from the digest rather than
  from the reply, and a number the reply cites that the digest does not contain
  is reported in unverified_numbers() instead of being printed as a finding. A
  review that invents a threshold is not a review; it is a plausible-looking
  wrong answer.
* **Failure is ordinary.** No provider, a timeout, prose instead of JSON, an
  invented capability id — each one falls back to the deterministic table with
  the reason attached, because a panel that goes blank whenever a model is busy
  is worse than one that was never interpreted at all.

Nothing here calls a model. The runtime lives in main.py; this module owns what
the model may see, and what of its reply survives contact with the numbers.
"""
from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, Mapping

from .quality import (
    MIN_CAPABILITY_SAMPLES, STATUS_INSUFFICIENT, STATUS_LABEL, STATUS_OK, STATUS_UNSUPPORTED,
    STATUS_WARNING,
)

REVIEW_SCHEMA_VERSION = 1
REVIEW_PROMPT_VERSION = 1

# The status vocabulary is imported from the matrix rather than restated: a
# second list here would be a second vocabulary, and the panel would render a
# label no other page knows.
REVIEW_STATUSES = (STATUS_OK, STATUS_WARNING, STATUS_INSUFFICIENT, STATUS_UNSUPPORTED)
REVIEW_VERDICTS = ("empty", "usable", "partial", "blocked")
REVIEW_VERDICT_LABEL = {
    "empty": "还没有数据",
    "usable": "可以开始",
    "partial": "部分可用",
    "blocked": "先补数据",
}

MAX_HEADLINE = 200
MAX_LABEL = 40
MAX_TEXT = 600
MAX_ACTIONS = 8
MAX_CAVEATS = 6
MAX_REASONS_PER_ROW = 4
MAX_UNVERIFIED = 8
MAX_INVENTED_IDS = 5

_DATASET_KEYS = (
    "samples", "sessions", "scopes", "scope_level", "degraded_traces", "outcome_unavailable",
    "tasks", "candidate_evidence", "timestamp_semantics",
)
_TRACE_KEYS = ("supported", "latest", "observed", "unreadable")

# Written without backslash escapes on purpose: this module ships inside a
# plugin that is edited through tools where escaping is a silent failure mode.
_NUMBER = re.compile("[0-9]+(?:[.][0-9]+)?")
FENCE = chr(96) * 3


# ---- small coercions ----------------------------------------------------

def _text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _strings(value: Any, limit: int, count: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        text = _text(item, limit)
        if text and text not in out:
            out.append(text)
        if len(out) >= count:
            break
    return out


def _number(value: Any) -> int | float | None:
    # Positive form: `bool` is an `int` subclass, and reading the number this way
    # is also what narrows `Any` to a real type for the checker.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _mapping_of_numbers(value: Any) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int | float] = {}
    for key, item in value.items():
        number = _number(item)
        if number is not None:
            out[str(key)] = number
    return out


# ---- the digest: everything the model may see ---------------------------

def _dataset_digest(dataset: Any) -> dict[str, Any]:
    if not isinstance(dataset, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in _DATASET_KEYS:
        value = dataset.get(key)
        if isinstance(value, str):
            out[key] = value[:120]
        elif isinstance(value, bool):
            out[key] = value
        elif isinstance(value, Mapping):
            out[key] = _mapping_of_numbers(value)
        elif _number(value) is not None:
            out[key] = value
    return out


def _trace_digest(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in _TRACE_KEYS:
        value = trace.get(key)
        if isinstance(value, (list, tuple)):
            out[key] = [item for item in value
                        if isinstance(item, (str, int, float)) and not isinstance(item, bool)][:12]
        elif isinstance(value, Mapping):
            out[key] = _mapping_of_numbers(value)
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            out[key] = value
    return out


def _capabilities_digest(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, Mapping):
        return []
    out: list[dict[str, Any]] = []
    for name, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        status = row.get("status")
        out.append({
            "id": str(name)[:64],
            "definition": _text(row.get("definition"), 200),
            "deterministic_status": (
                status if status in REVIEW_STATUSES else STATUS_UNSUPPORTED),
            "eligible": _number(row.get("eligible")),
            "total": _number(row.get("total")),
            "coverage": _number(row.get("coverage")),
            "reasons": _strings(row.get("reasons"), MAX_TEXT, MAX_REASONS_PER_ROW),
        })
    return out


def _gate_digest(gate: Any) -> dict[str, Any]:
    if not isinstance(gate, Mapping):
        return {}
    checks: list[dict[str, Any]] = []
    for row in gate.get("checks") or []:
        if not isinstance(row, Mapping):
            continue
        checks.append({
            "name": _text(row.get("name"), 40),
            "status": _text(row.get("status"), 16),
            "detail": _text(row.get("detail"), MAX_TEXT),
            "value": _number(row.get("value")),
            "threshold": _number(row.get("threshold")),
            "blocking": bool(row.get("blocking")),
        })
        if len(checks) >= 10:
            break
    return {
        "ok": bool(gate.get("ok")),
        "summary": _text(gate.get("summary"), MAX_TEXT),
        "checks": checks,
    }


def build_digest(quality: Mapping[str, Any], *,
                 gate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The bounded, number-only facts a review is allowed to be based on.

    Built by whitelist rather than by trimming the quality payload: the payload
    grows a field every time the plugin learns something new, and every one of
    them would otherwise start travelling to a model by default.
    """
    source = gate if gate is not None else quality.get("dataset_gate")
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "prompt_version": REVIEW_PROMPT_VERSION,
        "reader_version": _number(quality.get("reader_version")),
        "status_vocabulary": dict(STATUS_LABEL),
        "min_samples_for_capability": MIN_CAPABILITY_SAMPLES,
        "dataset": _dataset_digest(quality.get("dataset")),
        "trace": _trace_digest(quality.get("trace")),
        "capabilities": _capabilities_digest(quality.get("capabilities")),
        "blocked": _strings(quality.get("blocked"), MAX_TEXT, 8),
        "contract_findings": _strings(quality.get("contract_findings"), MAX_TEXT, 8),
        "notes": _strings(quality.get("notes"), MAX_TEXT, 6),
        "gate": _gate_digest(source),
    }


def digest_fingerprint(digest: Mapping[str, Any]) -> str:
    """Stable identity of one digest, so a cached review cannot outlive its data."""
    encoded = json.dumps(digest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()[:16]


# ---- the prompt ---------------------------------------------------------

SYSTEM_PROMPT = """你是 ChatDynamics 学习插件的数据契约解读器。

你会收到一份 JSON 摘要，它描述「这批数据现在能支撑哪些分析」。把这份摘要写给持有数据的人看：
现在能做什么、下一步做什么。

硬性规则：
1. 只使用摘要里出现过的数字。摘要里没有的数字一律不要写，包括你自己算出来的比例。
2. 每条能力都要有一行，capability 用摘要里的 id，status 只能是 ok / warning / insufficient / unsupported 之一。
3. 摘要给了 deterministic_status。你不同意时可以改，但要在 explanation 里说清依据是摘要里的哪个计数。
4. 样本为 0 时不要描述成系统故障：说清缺哪一类标注、大概需要多少条才能开始。
5. 不要复述摘要原文，不要输出样本内容或任何身份信息（摘要里也没有）。
6. 只输出一个 JSON 对象，不要 markdown 代码块，不要任何解释文字。

输出结构：
{"headline": "一句话结论",
 "verdict": "empty|usable|partial|blocked",
 "rows": [{"capability": "id", "status": "ok|warning|insufficient|unsupported",
           "label": "不超过 20 字的短标签", "explanation": "为什么是这个状态",
           "next_action": "要改变它该做什么；已经正常就留空字符串"}],
 "actions": ["按优先级排序的下一步动作"],
 "caveats": ["读这份结论时要注意的前提"]}"""


def build_prompt(digest: Mapping[str, Any]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one digest."""
    body = json.dumps(digest, ensure_ascii=False, indent=1)
    return SYSTEM_PROMPT, "数据摘要：\n" + body + "\n\n只输出 JSON 对象。"


# ---- reading the reply --------------------------------------------------

def _extract_json(text: str) -> Any:
    """The first JSON object in a reply, fenced or bare; None when there is none.

    Models wrap JSON in prose and code fences often enough that refusing those
    replies would be refusing the feature. What is not tolerated is a guess: the
    object has to parse, and everything downstream is validated against the
    digest anyway.
    """
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


def _variants(value: float) -> set[str]:
    """How one digest number may legitimately be spelled in a reply."""
    out: set[str] = set()

    def add(candidate: float) -> None:
        if abs(candidate) >= 1e15:
            return
        out.add(f"{candidate:g}")
        if float(candidate).is_integer():
            out.add(str(int(candidate)))

    add(value)
    # A rate may be written as a percentage (0.8 -> 80%). A count may not be
    # written as its hundredfold: 20 -> 2000 is a different claim about the
    # corpus, and accepting it would let a made-up threshold through the check
    # that exists to catch exactly that.
    if 0.0 <= value <= 1.0:
        add(value * 100.0)
    return out


def _digest_numbers(digest: Mapping[str, Any]) -> set[str]:
    """Every number the digest contains, in the spellings a reply might use."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.update(_variants(float(value)))
        elif isinstance(value, str):
            for token in _NUMBER.findall(value):
                found.update(_variants(float(token)))
        elif isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(digest)
    return found


def unverified_numbers(reply: str, digest: Mapping[str, Any]) -> list[str]:
    """Numbers the reply cites that the digest does not contain."""
    if not isinstance(reply, str) or not reply:
        return []
    known = _digest_numbers(digest)
    out: list[str] = []
    for token in _NUMBER.findall(reply):
        if token in out or (_variants(float(token)) & known):
            continue
        out.append(token)
        if len(out) >= MAX_UNVERIFIED:
            break
    return out


def parse_review(text: str, digest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate a model reply against the digest; None when nothing survives.

    The counts, coverage and denominators are copied from the digest, never from
    the reply: the model decides what a row *means*, the corpus decides what it
    *is*. Rows the model skipped are filled from the deterministic verdict and
    marked as such, so the table is always complete.
    """
    payload = _extract_json(text)
    if not isinstance(payload, Mapping):
        return None
    capabilities = [row for row in digest.get("capabilities") or [] if isinstance(row, Mapping)]
    known = {str(row.get("id")): row for row in capabilities}
    order = [str(row.get("id")) for row in capabilities]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    invented: list[str] = []
    from_model = 0
    for raw in payload.get("rows") or []:
        if not isinstance(raw, Mapping):
            continue
        identifier = raw.get("capability")
        identifier = identifier.strip() if isinstance(identifier, str) else ""
        if not identifier or identifier in seen:
            continue
        base = known.get(identifier)
        if base is None:
            if identifier not in invented and len(invented) < MAX_INVENTED_IDS:
                invented.append(identifier[:64])
            continue
        seen.add(identifier)
        status = raw.get("status")
        status = status if status in REVIEW_STATUSES else str(base.get("deterministic_status"))
        rows.append({
            "capability": identifier,
            "status": status,
            "status_label": STATUS_LABEL.get(status, status),
            "deterministic_status": base.get("deterministic_status"),
            "agrees": status == base.get("deterministic_status"),
            "label": _text(raw.get("label"), MAX_LABEL),
            "explanation": _text(raw.get("explanation"), MAX_TEXT),
            "next_action": _text(raw.get("next_action"), MAX_TEXT),
            "eligible": base.get("eligible"),
            "total": base.get("total"),
            "coverage": base.get("coverage"),
            "source": "model",
        })
        from_model += 1

    headline = _text(payload.get("headline"), MAX_HEADLINE)
    if not from_model and not headline:
        return None

    filled = 0
    for identifier in order:
        if identifier in seen:
            continue
        base = known[identifier]
        status = str(base.get("deterministic_status"))
        rows.append({
            "capability": identifier,
            "status": status,
            "status_label": STATUS_LABEL.get(status, status),
            "deterministic_status": status,
            "agrees": True,
            "label": "",
            "explanation": "；".join(str(reason) for reason in base.get("reasons") or []),
            "next_action": "",
            "eligible": base.get("eligible"),
            "total": base.get("total"),
            "coverage": base.get("coverage"),
            "source": "deterministic",
        })
        filled += 1
    rows.sort(key=lambda row: order.index(row["capability"]) if row["capability"] in order
              else len(order))

    verdict = payload.get("verdict")
    verdict = verdict if verdict in REVIEW_VERDICTS else None
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "prompt_version": REVIEW_PROMPT_VERSION,
        "headline": headline,
        "verdict": verdict,
        "verdict_label": REVIEW_VERDICT_LABEL.get(verdict or "", ""),
        "rows": rows,
        "actions": _strings(payload.get("actions"), MAX_TEXT, MAX_ACTIONS),
        "caveats": _strings(payload.get("caveats"), MAX_TEXT, MAX_CAVEATS),
        "unverified_numbers": unverified_numbers(text, digest),
        "invented_capabilities": invented,
        "model_rows": from_model,
        "deterministic_rows": filled,
        "disagreements": sum(1 for row in rows if row["source"] == "model" and not row["agrees"]),
        "fingerprint": digest_fingerprint(digest),
    }


__all__ = [
    "MAX_ACTIONS", "REVIEW_PROMPT_VERSION", "REVIEW_SCHEMA_VERSION", "REVIEW_STATUSES",
    "REVIEW_VERDICT_LABEL", "REVIEW_VERDICTS", "SYSTEM_PROMPT", "build_digest", "build_prompt",
    "digest_fingerprint", "parse_review", "unverified_numbers",
]
