"""Pure relation-judgment helpers (zero astrbot imports).

Used by main.py write/read paths and by stdlib unittest without the plugin runtime.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Word lists — keep in parity with main.py FRIENDLY/HOSTILE/NEGATION/_tokens
# ---------------------------------------------------------------------------

FRIENDLY_WORDS: frozenset[str] = frozenset(
    {
        "谢谢",
        "感谢",
        "爱你",
        "喜欢",
        "可爱",
        "厉害",
        "牛逼",
        "太强",
        "靠谱",
        "辛苦",
        "抱抱",
        "贴贴",
        "赞",
        "好耶",
        "支持",
        "同意",
        "没事",
        "加油",
        "欢迎",
        "开心",
        "thank",
        "thanks",
        "love",
        "nice",
        "good",
        "great",
        "awesome",
        "cute",
        "cool",
    }
)
HOSTILE_WORDS: frozenset[str] = frozenset(
    {
        "讨厌",
        "傻逼",
        "sb",
        "滚",
        "闭嘴",
        "烦死",
        "恶心",
        "垃圾",
        "废物",
        "蠢",
        "笨蛋",
        "有病",
        "爬",
        "拉黑",
        "吵架",
        "别叫",
        "别说",
        "生气",
        "破防",
        "离谱",
        "hate",
        "stupid",
        "idiot",
        "shut",
        "trash",
        "annoying",
        "bad",
    }
)
NEGATION_WORDS: frozenset[str] = frozenset(
    {"不", "没", "没有", "别", "不是", "不太", "并不"}
)

_TOKEN_STOP: frozenset[str] = frozenset(
    {
        "哈哈",
        "不是",
        "这个",
        "那个",
        "什么",
        "可以",
        "就是",
        "没有",
        "一下",
        "一个",
        "我们",
        "你们",
        "他们",
    }
)

_BROADCAST_VALUES: frozenset[str] = frozenset(
    {
        "all",
        "全体",
        "全体成员",
    }
)

# reason priority: lower rank = higher priority
_REASON_PRIORITY: dict[str, int] = {
    "回复": 0,
    "@": 1,
    "提到QQ": 2,
    "提到昵称": 3,
    "邻接": 4,
    "同话题": 5,
    "近邻": 6,
    "友善": 7,
    "敌对": 7,
    "玩笑语气": 7,
}
_EXTERNAL_MEMORY_PREFIX = "外部记忆"
_EXTERNAL_MEMORY_RANK = 8
_UNKNOWN_REASON_RANK = 50

DEFAULT_SCORING: dict[str, Any] = {
    "reply_weight": 8.0,
    "at_weight": 5.0,
    "adjacency_weight": 1.5,
    "adjacency_window_seconds": 90,
    "ambient_tone_scale": 0.15,
    "min_hits_tie": 12,
    "min_active_days_tie": 3,
    "min_directed_tie": 3,
    "bot_user_ids": [],
    "directed_event_retain_days": 90,
    "debug_log_components": False,
    "pair_base_weight": 1.0,
    "mention_weight": 5.0,
    "same_topic_weight": 0.8,
    "max_recent_per_group": 80,
    "min_message_chars": 1,
    "context_window_seconds": 300,
}

DIRECTED_EVENTS_DDL: str = """
CREATE TABLE IF NOT EXISTS directed_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id TEXT NOT NULL,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0,
    ts INTEGER NOT NULL
);
""".strip()

DIRECTED_EVENTS_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_directed_events_group_ts ON directed_events(group_id, ts);",
    "CREATE INDEX IF NOT EXISTS idx_directed_events_pair ON directed_events(group_id, src, dst, ts);",
    "CREATE INDEX IF NOT EXISTS idx_directed_events_kind ON directed_events(group_id, kind, ts);",
]

RELATIONS_COUNTER_ALTERS: list[str] = [
    "ALTER TABLE relations ADD COLUMN directed_hits INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE relations ADD COLUMN reply_hits INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE relations ADD COLUMN at_hits INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE relations ADD COLUMN ab_hits INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE relations ADD COLUMN ba_hits INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE relations ADD COLUMN active_days INTEGER NOT NULL DEFAULT 0;",
    "ALTER TABLE relations ADD COLUMN last_active_date TEXT NOT NULL DEFAULT '';",
]


def is_broadcast_at_value(v: Any) -> bool:
    """True when At target is broadcast (all / 全体 / 全体成员), case-insensitive."""
    try:
        s = str(v if v is not None else "").strip()
    except Exception:
        return False
    if not s:
        return False
    return s.casefold() in {x.casefold() for x in _BROADCAST_VALUES}


def _norm_id(value: Any) -> str | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
    except Exception:
        return None
    if not s or s == "0" or is_broadcast_at_value(s):
        return None
    return s


def _class_name(comp: Any) -> str:
    try:
        return type(comp).__name__ or ""
    except Exception:
        return ""


def _getattr_chain(comp: Any, names: Sequence[str]) -> Any:
    for name in names:
        try:
            if hasattr(comp, name):
                val = getattr(comp, name, None)
            elif isinstance(comp, dict) and name in comp:
                val = comp[name]
            else:
                continue
            if val is None:
                continue
            s = str(val).strip()
            if s and s != "0":
                return val
        except Exception:
            continue
    return None


def comp_kind_and_target(comp: Any) -> tuple[str | None, str | None]:
    """Map a message component to (kind, target_id) without raising.

    Reply-like: class name contains Reply OR has sender_id.
    At-like: class name contains At (not AtAll) OR has qq/user_id and is not reply.
    Broadcast / AtAll / empty → (None, None).
    """
    try:
        if comp is None:
            return (None, None)
        cls = _class_name(comp)
        cls_l = cls.lower()

        if "atall" in cls_l:
            return (None, None)

        has_sender_attr = False
        try:
            has_sender_attr = (
                hasattr(comp, "sender_id")
                and getattr(comp, "sender_id", None) is not None
            )
            if has_sender_attr:
                sid_raw = getattr(comp, "sender_id", None)
                if sid_raw is None or str(sid_raw).strip() == "":
                    has_sender_attr = False
        except Exception:
            has_sender_attr = False

        is_reply = ("reply" in cls_l) or has_sender_attr
        if is_reply:
            raw = _getattr_chain(comp, ("sender_id", "qq", "user_id"))
            tid = _norm_id(raw)
            if tid is None:
                return (None, None)
            return ("reply", tid)

        is_at_name = ("at" in cls_l) and ("atall" not in cls_l)
        raw_at = _getattr_chain(comp, ("qq", "user_id"))
        if is_at_name:
            tid = _norm_id(raw_at)
            if tid is None:
                return (None, None)
            return ("at", tid)

        # Duck-typed At: qq/user_id present, not classified as reply.
        tid = _norm_id(raw_at)
        if tid is not None:
            return ("at", tid)
        return (None, None)
    except Exception:
        return (None, None)


def extract_directed_targets(
    comps: Iterable[Any] | None,
    sender_id: Any,
    self_id: Any,
    bot_ids: Iterable[Any] | None,
    *,
    reply_weight: float = 8.0,
    at_weight: float = 5.0,
) -> list[tuple[str, str, float]]:
    """Extract directed (target_id, kind, weight_factor) edges from components.

    weight_factor: reply always 1.0; at is 1.0 (single) or 1/n (2-5 unique);
    ≥6 unique ats → no at edges. Skips self/bot/empty/broadcast.
    Same target keeps max absolute weight (reply > at at defaults).
    """
    try:
        sender = _norm_id(sender_id) or str(sender_id or "").strip()
        self_s = _norm_id(self_id) or str(self_id or "").strip()
        bots: set[str] = set()
        for b in bot_ids or ():
            nb = _norm_id(b) or str(b or "").strip()
            if nb:
                bots.add(nb)
        if self_s:
            bots.add(self_s)

        reply_targets: list[str] = []
        at_targets: list[str] = []
        for comp in comps or ():
            kind, tid = comp_kind_and_target(comp)
            if not kind or not tid:
                continue
            if sender and tid == sender:
                continue
            if tid in bots:
                continue
            if kind == "reply":
                reply_targets.append(tid)
            else:
                at_targets.append(tid)

        # unique ats preserving first-seen order
        seen_at: dict[str, None] = {}
        for tid in at_targets:
            if tid not in seen_at:
                seen_at[tid] = None
        unique_ats = list(seen_at.keys())
        n_at = len(unique_ats)
        at_factor: float | None
        if n_at <= 0:
            at_factor = None
            unique_ats = []
        elif n_at == 1:
            at_factor = 1.0
        elif n_at <= 5:
            at_factor = 1.0 / float(n_at)
        else:
            at_factor = None
            unique_ats = []

        # tid -> (kind, factor, abs_weight)
        best: dict[str, tuple[str, float, float]] = {}

        for tid in reply_targets:
            abs_w = float(reply_weight) * 1.0
            prev = best.get(tid)
            if prev is None or abs_w > prev[2]:
                best[tid] = ("reply", 1.0, abs_w)

        if at_factor is not None:
            for tid in unique_ats:
                abs_w = float(at_weight) * float(at_factor)
                prev = best.get(tid)
                if prev is None or abs_w > prev[2]:
                    best[tid] = ("at", float(at_factor), abs_w)

        return [(tid, kind, factor) for tid, (kind, factor, _) in best.items()]
    except Exception:
        return []


def _reason_rank(item: str) -> int:
    s = (item or "").strip()
    if not s:
        return _UNKNOWN_REASON_RANK
    if s in _REASON_PRIORITY:
        return _REASON_PRIORITY[s]
    if s.startswith(_EXTERNAL_MEMORY_PREFIX):
        return _EXTERNAL_MEMORY_RANK
    return _UNKNOWN_REASON_RANK


def merge_reasons_priority(
    old: str,
    new_items: list[str],
    max_items: int = 5,
) -> str:
    """Merge reason tags, sort by locked priority, keep at most max_items."""
    items: list[str] = []
    seen: set[str] = set()
    for part in (old or "").split("、"):
        p = part.strip()
        if p and p not in seen:
            items.append(p)
            seen.add(p)
    for item in new_items or []:
        p = str(item or "").strip()
        if p and p not in seen:
            items.append(p)
            seen.add(p)
    items.sort(key=lambda x: (_reason_rank(x), x))
    return "、".join(items[: max(0, int(max_items))])


def evidence_tier(
    directed_hits: int,
    reply_hits: int,
    at_hits: int,
    hits: int,
    reasons: str,
) -> str:
    """Read-time evidence tier: directed | legacy | ambient."""
    try:
        d = int(directed_hits or 0) + int(reply_hits or 0) + int(at_hits or 0)
    except Exception:
        d = 0
    if d > 0:
        return "directed"
    reasons_s = str(reasons or "")
    try:
        h = int(hits or 0)
    except Exception:
        h = 0
    if (_EXTERNAL_MEMORY_PREFIX + ":") in reasons_s or "外部记忆:" in reasons_s:
        return "legacy"
    if h > 0:
        return "legacy"
    return "ambient"


def confidence(
    directed_hits: int,
    active_days: int,
    hits: int,
    evidence_tier_value: str,
) -> float:
    """Read-time confidence in [0, 1]."""
    try:
        if str(evidence_tier_value or "") == "directed":
            dh = max(0, int(directed_hits or 0))
            ad = max(0, int(active_days or 0))
            return min(1.0, dh / 3.0) * min(1.0, max(ad, 1) / 3.0)
        h = max(0, int(hits or 0))
        return min(1.0, h / 12.0)
    except Exception:
        return 0.0


def _base_strength_band(score: float) -> str:
    if score >= 80:
        return "铁关系"
    if score >= 35:
        return "熟人圈"
    if score >= 12:
        return "常互动"
    return "新互动"


def strength_label_v2(
    score: float,
    hits: int,
    directed_hits: int,
    active_days: int,
    reciprocity: float,
    *,
    evidence_tier: str,
    min_hits_tie: int = 12,
    min_active_days_tie: int = 3,
    min_directed_tie: int = 3,
) -> str:
    """Evidence-gated strength label with legacy grace and single-pass demote."""
    try:
        sc = float(score or 0.0)
    except Exception:
        sc = 0.0
    base = _base_strength_band(sc)
    tier = str(evidence_tier or "")
    if tier != "directed":
        return base

    try:
        h = int(hits or 0)
        dh = int(directed_hits or 0)
        ad = int(active_days or 0)
        rec = float(reciprocity or 0.0)
    except Exception:
        return base

    if base == "铁关系":
        ok = (
            h >= int(min_hits_tie)
            and ad >= int(min_active_days_tie)
            and (dh >= int(min_directed_tie) or rec >= 0.35)
        )
        if not ok:
            return "熟人圈"  # single-pass: do not cascade
        return base

    if base == "熟人圈":
        ok = h >= 6 and (dh >= 1 or ad >= 2)
        if not ok:
            return "常互动"
        return base

    return base


def tokens(text: str) -> set[str]:
    """Tokenize message text — parity with main.py _tokens."""
    text = (text or "").lower()
    words = set(re.findall(r"[a-zA-Z0-9_]{3,}|[\u4e00-\u9fff]{2,4}", text))
    return {w for w in words if w not in _TOKEN_STOP}


def tone_signal(text: str) -> tuple[float, float, list[str]]:
    """Observe explicit tone with local negation and English word boundaries."""
    low = (text or "").lower()
    friendly = 0.0
    hostile = 0.0
    tags: list[str] = []
    for words, positive in ((FRIENDLY_WORDS, True), (HOSTILE_WORDS, False)):
        for word in words:
            pattern = (
                r"\b" + re.escape(word) + r"\b" if word.isascii() else re.escape(word)
            )
            matches = list(re.finditer(pattern, low))
            for match in matches[:1]:
                prefix = low[max(0, match.start() - 12) : match.start()]
                negated = bool(
                    re.search(
                        r"(?:不|没|没有|并不|不是|别).{0,2}$|(?:not|never|don't)\s*(?:very\s+)?$",
                        prefix,
                    )
                )
                if negated:
                    if positive and word in {"喜欢", "爱你", "love"}:
                        hostile += 0.5
                    continue
                if positive:
                    friendly += 1
                elif word == "爬" and re.search(r"爬山|爬虫|爬楼", low):
                    continue
                elif re.search(r"(?:游戏|软件|代码|网络|天气|服务器).{0,5}$", prefix):
                    continue
                else:
                    hostile += 1
    friendly, hostile = min(3.0, friendly), min(3.0, hostile)
    if (friendly or hostile) and re.search(r"哈+|草|笑死|doge|233|hhh", low):
        friendly *= 0.45
        hostile *= 0.45
        if "玩笑语气" not in tags:
            tags.append("玩笑语气")
    if friendly:
        tags.append("友善")
    if hostile:
        tags.append("敌对")
    return friendly, hostile, tags


__all__ = [
    "FRIENDLY_WORDS",
    "HOSTILE_WORDS",
    "NEGATION_WORDS",
    "DEFAULT_SCORING",
    "DIRECTED_EVENTS_DDL",
    "DIRECTED_EVENTS_INDEXES",
    "RELATIONS_COUNTER_ALTERS",
    "is_broadcast_at_value",
    "comp_kind_and_target",
    "extract_directed_targets",
    "merge_reasons_priority",
    "evidence_tier",
    "confidence",
    "strength_label_v2",
    "tokens",
    "tone_signal",
]
