"""What can still be labelled, and for how long.

The host keeps a bounded message graph per session — a node cap and a TTL, both
reported in its runtime snapshot — and a message is annotatable only while it is
still in that graph. Once pruning drops it the replay page cannot show it, and no
label can be made any more. The annotation records themselves are permanent, and
they are what keeps a message text, so the window is a deadline for *labelling*,
not for the data.

This module answers the question a reader has before that deadline: which
sessions still have something to label, how much, how old it is, and how long the
oldest piece of it has left. Everything here is a pure function of the host
snapshot and the annotation keys; nothing is written anywhere.
"""
from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Any, Mapping, Sequence

ANNOTATION_WINDOW_SCHEMA_VERSION = 1

# Legacy public constants retained for import compatibility. Missing host limits
# are unavailable; these defaults must never be used to invent a deadline.
FALLBACK_MAX_NODES = 500
FALLBACK_TTL_SECONDS = 3600.0

MAX_TEXT_IN_EXPORT = 300
MAX_MESSAGES_PER_SESSION = 600


def _text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mask(value: Any) -> str:
    """A short, recognisable label for an identifier. Never the identifier."""
    text = str(value or "")
    if len(text) <= 8:
        return text[:8]
    return text[:4] + "…" + text[-3:]


def session_digest(session_key: str) -> str:
    """The key the host files a session annotations under."""
    return sha256(str(session_key).encode("utf-8")).hexdigest()


def _wall_clock(runtime: Mapping[str, Any]) -> Callable[[float], float] | None:
    """A converter from the host monotonic clock to wall time, or None.

    Node timestamps come from the host monotonic clock; the snapshot carries one
    pair of (wall, monotonic) readings taken at the same moment, which is enough
    to place every node on the civil clock.
    """
    saved_wall = _number(runtime.get("saved_wall"))
    saved_clock = _number(runtime.get("saved_clock"))
    if saved_wall is None or saved_clock is None:
        return None
    return lambda stamp: saved_wall + (float(stamp) - saved_clock)


def window_payload(runtime: Any, annotated: Mapping[str, set[str]], *,
                   now_wall: float, include_messages: bool = False,
                   fallback_max_nodes: int = FALLBACK_MAX_NODES,
                   fallback_ttl_seconds: float = FALLBACK_TTL_SECONDS) -> dict[str, Any]:
    """Per-session window facts, newest session first.

    `annotated` maps a session digest to the message ids already labelled, so a
    reader can tell "the window is empty" from "everything in it is already
    labelled" — the two have completely different next actions.
    """
    runtime = runtime if isinstance(runtime, Mapping) else {}
    graph = runtime.get("graph") if isinstance(runtime.get("graph"), Mapping) else {}
    max_nodes = _number(graph.get("max_nodes"))
    max_nodes = int(max_nodes) if max_nodes is not None and max_nodes > 0 else None
    ttl_seconds = _number(graph.get("ttl_seconds"))
    ttl_seconds = ttl_seconds if ttl_seconds is not None and ttl_seconds >= 0 else None
    to_wall = _wall_clock(runtime)

    rows: list[dict[str, Any]] = []
    for session in runtime.get("sessions") or []:
        if not isinstance(session, Mapping):
            continue
        key = str(session.get("session_key") or "")
        nodes = [node for node in (session.get("nodes") or []) if isinstance(node, Mapping)]
        if not key or not nodes:
            continue
        digest = session_digest(key)
        labelled = annotated.get(digest) or set()
        stamps = [stamp for stamp in (_number(node.get("timestamp")) for node in nodes)
                  if stamp is not None]
        walls = [to_wall(stamp) for stamp in stamps] if to_wall is not None else []
        walls = [wall for wall in walls if wall is not None]
        with_text = sum(1 for node in nodes if _text(node.get("text"), 1))
        labelled_here = sum(1 for node in nodes
                            if str(node.get("msg_id") or "") in labelled)
        oldest = min(walls) if walls else None
        newest = max(walls) if walls else None
        row: dict[str, Any] = {
            "session": _mask(key),
            "session_hash": digest[:12],
            "messages": len(nodes),
            "with_text": with_text,
            "annotated": labelled_here,
            "unlabelled": len(nodes) - labelled_here,
            "oldest_wall": oldest,
            "newest_wall": newest,
            "span_seconds": (newest - oldest) if (oldest is not None and newest is not None) else None,
            "idle_seconds": (now_wall - newest) if newest is not None else None,
            "expires_in_seconds": ((oldest + ttl_seconds) - now_wall) if oldest is not None and ttl_seconds is not None else None,
            "cap_pressure": round(len(nodes) / max_nodes, 3) if max_nodes else None,
        }
        if include_messages:
            row["messages_detail"] = _message_rows(nodes, labelled, to_wall)
        rows.append(row)
    rows.sort(key=lambda item: item.get("messages") or 0, reverse=True)

    totals = {
        "sessions": len(rows),
        "messages": sum(row["messages"] for row in rows),
        "with_text": sum(row["with_text"] for row in rows),
        "annotated": sum(row["annotated"] for row in rows),
        "unlabelled": sum(row["unlabelled"] for row in rows),
    }
    return {
        "annotation_window_schema_version": ANNOTATION_WINDOW_SCHEMA_VERSION,
        "generated_at": now_wall,
        "limits": {"max_nodes": max_nodes, "ttl_seconds": ttl_seconds,
                   "reported_by_host": max_nodes is not None and ttl_seconds is not None},
        "sessions": rows,
        "totals": totals,
        "hint": _hint(rows, totals, ttl_seconds),
        "text_policy": ("导出的是本体此刻仍保留的消息；下载到你的机器上，本插件不保存正文，也不写回任何东西。"),
    }


def _message_rows(nodes: Sequence[Mapping[str, Any]], labelled: set[str],
                  to_wall: Callable[[float], float] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes[-MAX_MESSAGES_PER_SESSION:]:
        stamp = _number(node.get("timestamp"))
        wall = to_wall(stamp) if (to_wall is not None and stamp is not None) else None
        msg_id = str(node.get("msg_id") or "")
        out.append({
            "msg_id": msg_id,
            "wall": wall,
            "user": _mask(node.get("user_id")),
            "reply_to": _mask(node.get("reply_to_id")),
            "has_text": bool(_text(node.get("text"), 1)),
            "text": _text(node.get("text"), MAX_TEXT_IN_EXPORT),
            "annotated": msg_id in labelled,
        })
    return out


def _hint(rows: Sequence[Mapping[str, Any]], totals: Mapping[str, int],
          ttl_seconds: float | None) -> str:
    if not rows:
        return ("本体的运行快照里还没有会话：确认 ChatDynamics 正在运行、生效群里有消息，再刷新一次。")
    if not totals["unlabelled"]:
        return ("这段窗口里的消息都已经标注过了。等新消息进来再回来看。")
    soonest = min((row["expires_in_seconds"] for row in rows
                   if row.get("expires_in_seconds") is not None), default=None)
    where = "本体未报告保留期限，无法计算清理时间；" if ttl_seconds is None else ("" if soonest is None else _deadline_phrase(soonest, ttl_seconds))
    return (f"窗口里有 {totals['unlabelled']} 条还没标注（带正文 {totals['with_text']} 条）。"
            + where
            + "标注记录是永久的，未标注的消息会随本体的消息图一起被清理 —— 要标就现在标。")


def _deadline_phrase(seconds: float, ttl_seconds: float) -> str:
    if seconds <= 0:
        return f"最旧的一条已经超过本体的保留期（{int(ttl_seconds / 60)} 分钟），下一次清理就会消失；"
    minutes = int(seconds // 60)
    if minutes <= 0:
        return "最旧的一条不到一分钟就会被清理；"
    return f"最旧的一条大约还有 {minutes} 分钟就会被清理；"


__all__ = [
    "ANNOTATION_WINDOW_SCHEMA_VERSION", "FALLBACK_MAX_NODES", "FALLBACK_TTL_SECONDS",
    "MAX_TEXT_IN_EXPORT", "session_digest", "window_payload",
]
