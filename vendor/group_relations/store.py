"""Embedded MIT relation statistics from Hermes Group Relation Analyzer 2.1.2.
Original behavior retained; native source-id deduplication is added locally.
"""
from __future__ import annotations
import asyncio
import datetime as _dt
import json
import math
import re
import sqlite3
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from .relation_metrics import activity_observation,day_key,network_metrics,pair_metrics,window_start
from .relation_presentation import representative_relations
from .relation_model import model_tone
from .relation_signals import (DEFAULT_SCORING,DIRECTED_EVENTS_DDL,DIRECTED_EVENTS_INDEXES,RELATIONS_COUNTER_ALTERS,
    merge_reasons_priority,confidence as rs_confidence,evidence_tier as rs_evidence_tier,
    strength_label_v2 as rs_strength_label_v2,tokens as rs_tokens,tone_signal as rs_tone_signal)


_REL_COUNTER_SELECT = (
    "COALESCE(directed_hits, 0) AS directed_hits, "
    "COALESCE(reply_hits, 0) AS reply_hits, "
    "COALESCE(at_hits, 0) AS at_hits, "
    "COALESCE(ab_hits, 0) AS ab_hits, "
    "COALESCE(ba_hits, 0) AS ba_hits, "
    "COALESCE(active_days, 0) AS active_days"
)


def _now() -> int:
    return int(time.time())


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:500]


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args: Any) -> bool:
        try:
            return super().__exit__(*args)
        finally:
            self.close()


class _SharedConnection:
    def __init__(self, store: Any) -> None:
        self._store = store
        self._nested_stack: list[bool] = []

    def __enter__(self) -> sqlite3.Connection:
        self._store.lock.acquire()
        try:
            nested = bool(self._store.db.in_transaction)
            if not nested:
                self._store.db.execute("BEGIN")
            self._nested_stack.append(nested)
            return self._store.db
        except BaseException:
            self._store.lock.release()
            raise

    def __exit__(self, *args: Any) -> bool:
        nested = self._nested_stack.pop() if self._nested_stack else False
        try:
            if not nested:
                if args[0] is None:
                    if self._store.db.in_transaction:
                        self._store.db.commit()
                else:
                    if self._store.db.in_transaction:
                        self._store.db.rollback()
        finally:
            self._store.lock.release()
        return False


class RelationStore:
    def __init__(
        self,
        db_path: Path,
        *,
        min_hits_tie: int | None = None,
        min_active_days_tie: int | None = None,
        min_directed_tie: int | None = None,
        analysis_window_days: int = 30,
        relation_growth_scale: float = 120.0,
        pair_daily_budget: float = 80.0,
        activity_daily_cap: int = 80,
        shared_store: Any = None,
    ) -> None:
        self.db_path = db_path
        self._shared_store = shared_store
        # 铁关系证据门槛：可选注入，缺省回退到 DEFAULT_SCORING（与评分规则同源）。
        self.min_hits_tie = int(
            min_hits_tie
            if min_hits_tie is not None
            else DEFAULT_SCORING["min_hits_tie"]
        )
        self.min_active_days_tie = int(
            min_active_days_tie
            if min_active_days_tie is not None
            else DEFAULT_SCORING["min_active_days_tie"]
        )
        self.min_directed_tie = int(
            min_directed_tie
            if min_directed_tie is not None
            else DEFAULT_SCORING["min_directed_tie"]
        )
        self._lock = asyncio.Lock()
        self.analysis_window_days = max(1, min(90, analysis_window_days))
        self.relation_growth_scale = max(1.0, relation_growth_scale)
        self.pair_daily_budget = max(1.0, pair_daily_budget)
        self.activity_daily_cap = max(1, activity_daily_cap)
        self._last_history_prune_ts: int | None = None
        self._history_prune_interval_seconds: int = 3600
        self._init_db()

    def _connect(self) -> sqlite3.Connection | _SharedConnection:
        if self._shared_store is not None:
            return _SharedConnection(self._shared_store)
        conn = sqlite3.connect(
            str(self.db_path), timeout=10, factory=_ClosingConnection
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        wal_prefix = "" if self._shared_store is not None else "PRAGMA journal_mode=WAL;\n                PRAGMA synchronous=NORMAL;\n"
        with self._connect() as conn:
            conn.executescript(
                f"""
                {wal_prefix}
                CREATE TABLE IF NOT EXISTS users (
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    last_seen INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (group_id, user_id)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    tokens TEXT NOT NULL DEFAULT '',
                    ts INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_group_ts ON messages(group_id, ts);
                CREATE TABLE IF NOT EXISTS relations (
                    group_id TEXT NOT NULL,
                    a TEXT NOT NULL,
                    b TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0,
                    hits INTEGER NOT NULL DEFAULT 0,
                    friendly_score REAL NOT NULL DEFAULT 0,
                    hostile_score REAL NOT NULL DEFAULT 0,
                    sentiment_hits INTEGER NOT NULL DEFAULT 0,
                    last_ts INTEGER NOT NULL DEFAULT 0,
                    reasons TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (group_id, a, b)
                );
                CREATE INDEX IF NOT EXISTS idx_relations_group_score ON relations(group_id, score DESC);
                CREATE INDEX IF NOT EXISTS idx_relations_group_a ON relations(group_id, a);
                CREATE INDEX IF NOT EXISTS idx_relations_group_b ON relations(group_id, b);
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS relation_daily (
                    group_id TEXT NOT NULL, day TEXT NOT NULL, a TEXT NOT NULL, b TEXT NOT NULL,
                    raw_evidence REAL NOT NULL DEFAULT 0, evidence REAL NOT NULL DEFAULT 0,
                    friendly REAL NOT NULL DEFAULT 0, hostile REAL NOT NULL DEFAULT 0,
                    ab_friendly REAL NOT NULL DEFAULT 0, ab_hostile REAL NOT NULL DEFAULT 0,
                    ba_friendly REAL NOT NULL DEFAULT 0, ba_hostile REAL NOT NULL DEFAULT 0,
                    hits INTEGER NOT NULL DEFAULT 0, focused_hits INTEGER NOT NULL DEFAULT 0, directed_hits INTEGER NOT NULL DEFAULT 0,
                    reply_hits INTEGER NOT NULL DEFAULT 0, at_hits INTEGER NOT NULL DEFAULT 0,
                    ab_hits INTEGER NOT NULL DEFAULT 0, ba_hits INTEGER NOT NULL DEFAULT 0,
                    sentiment_hits INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(group_id, day, a, b)
                );
                CREATE TABLE IF NOT EXISTS member_daily (
                    group_id TEXT NOT NULL, day TEXT NOT NULL, user_id TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    estimated INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(group_id, day, user_id)
                );
                """
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS relation_source_receipts(group_id TEXT,source_id TEXT,fingerprint TEXT,created INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(group_id,source_id))"
            )
            # directed edge events (idempotent; table + indexes use IF NOT EXISTS)
            conn.execute(DIRECTED_EVENTS_DDL)
            for stmt in DIRECTED_EVENTS_INDEXES:
                conn.execute(stmt)
            # legacy + directed counters: ALTER may fail when the column already
            # exists on a fresh table or a re-run — each is best-effort.
            for stmt in (
                "ALTER TABLE relations ADD COLUMN friendly_score REAL NOT NULL DEFAULT 0",
                "ALTER TABLE relations ADD COLUMN hostile_score REAL NOT NULL DEFAULT 0",
                "ALTER TABLE relations ADD COLUMN sentiment_hits INTEGER NOT NULL DEFAULT 0",
                *RELATIONS_COUNTER_ALTERS,
                "ALTER TABLE relation_daily ADD COLUMN focused_hits INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE relation_source_receipts ADD COLUMN created INTEGER NOT NULL DEFAULT 0",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relation_source_receipts_created ON relation_source_receipts(created)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_directed_events_ts ON directed_events(ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_relation_daily_day ON relation_daily(day)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_member_daily_day ON member_daily(day)"
            )
            # Backfill directed history exactly once. Historical tone is unknown.
            if not conn.execute(
                "SELECT 1 FROM meta WHERE key='network_v2_started'"
            ).fetchone():
                now = _now()
                cutoff = now - 90 * 86400
                events = conn.execute(
                    "SELECT group_id,src,dst,kind,weight,ts FROM directed_events WHERE ts>=? ORDER BY ts",
                    (cutoff,),
                )
                messages: dict[tuple[str, str, str], set[int]] = defaultdict(set)
                for event in events:
                    src, dst = str(event["src"]), str(event["dst"])
                    if not src or not dst or src == dst or src == "0" or dst == "0":
                        continue
                    a, b = self._pair(src, dst)
                    day = day_key(int(event["ts"]))
                    messages[(str(event["group_id"]), day, src)].add(int(event["ts"]))
                    for uid in (src, dst):
                        conn.execute(
                            "INSERT OR IGNORE INTO users(group_id,user_id,name,message_count,last_seen) VALUES(?,?,?,0,?)",
                            (event["group_id"], uid, uid, event["ts"]),
                        )
                    old = conn.execute(
                        "SELECT raw_evidence FROM relation_daily WHERE group_id=? AND day=? AND a=? AND b=?",
                        (event["group_id"], day, a, b),
                    ).fetchone()
                    credit = min(
                        max(0.0, float(event["weight"])),
                        max(
                            0.0, self.pair_daily_budget - (float(old[0]) if old else 0)
                        ),
                    )
                    if credit <= 0:
                        continue
                    conn.execute(
                        """INSERT INTO relation_daily(group_id,day,a,b,raw_evidence,evidence,hits,directed_hits,reply_hits,at_hits,ab_hits,ba_hits)
                        VALUES(?,?,?,?,?,?,1,1,?,?,?,?) ON CONFLICT(group_id,day,a,b) DO UPDATE SET
                        raw_evidence=raw_evidence+excluded.raw_evidence,evidence=evidence+excluded.evidence,
                        hits=hits+1,directed_hits=directed_hits+1,reply_hits=reply_hits+excluded.reply_hits,
                        at_hits=at_hits+excluded.at_hits,ab_hits=ab_hits+excluded.ab_hits,ba_hits=ba_hits+excluded.ba_hits""",
                        (
                            event["group_id"],
                            day,
                            a,
                            b,
                            credit,
                            credit,
                            int(event["kind"] == "reply"),
                            int(event["kind"] == "at"),
                            int(src == a),
                            int(src == b),
                        ),
                    )
                for (gid, day, uid), timestamps in messages.items():
                    conn.execute(
                        "INSERT OR IGNORE INTO member_daily VALUES(?,?,?,?,1)",
                        (gid, day, uid, len(timestamps)),
                    )
                conn.execute(
                    "INSERT INTO meta(key,value) VALUES('network_v2_started',?)",
                    (str(now),),
                )

    @staticmethod
    def _pair(a: str, b: str) -> tuple[str, str]:
        return (a, b) if str(a) <= str(b) else (b, a)

    async def apply_daily_decay(
        self, factor: float, min_score: float = 0.05
    ) -> dict[str, Any]:
        """按自然日对库内关系分数做持久化衰减；同一天只执行一次，支持补算漏天。"""
        async with self._lock:
            return await asyncio.to_thread(
                self._apply_daily_decay_sync, factor, min_score
            )

    def _apply_daily_decay_sync(
        self, factor: float, min_score: float = 0.05
    ) -> dict[str, Any]:
        try:
            factor = float(factor)
        except Exception:
            factor = 1.0
        # 无效或无衰减：不改库
        if factor <= 0 or factor >= 1.0:
            return {"days": 0, "updated": 0, "deleted": 0, "skipped": True}

        today = _dt.date.today()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key=?",
                ("last_daily_decay",),
            ).fetchone()
            if not row or not str(row["value"] or "").strip():
                # 首次启用只记锚点，避免对历史数据一次性按“上线前多日”暴减
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("last_daily_decay", today.isoformat()),
                )
                return {"days": 0, "updated": 0, "deleted": 0, "initialized": True}

            try:
                last = _dt.date.fromisoformat(str(row["value"]).strip()[:10])
            except Exception:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    ("last_daily_decay", today.isoformat()),
                )
                return {"days": 0, "updated": 0, "deleted": 0, "reinitialized": True}

            days = (today - last).days
            if days <= 0:
                return {"days": 0, "updated": 0, "deleted": 0}

            # 防止异常时钟/损坏日期导致极端倍率
            days = min(days, 365)
            mult = float(factor) ** days
            cur = conn.execute(
                """
                UPDATE relations SET
                  score = score * ?,
                  friendly_score = friendly_score * ?,
                  hostile_score = hostile_score * ?
                """,
                (mult, mult, mult),
            )
            updated = int(cur.rowcount or 0)
            deleted = 0
            try:
                min_score = float(min_score)
            except Exception:
                min_score = 0.0
            if min_score > 0:
                cur = conn.execute(
                    """
                    DELETE FROM relations
                    WHERE score < ? AND friendly_score < ? AND hostile_score < ?
                    """,
                    (min_score, min_score, min_score),
                )
                deleted = int(cur.rowcount or 0)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                ("last_daily_decay", today.isoformat()),
            )
            return {
                "days": days,
                "updated": updated,
                "deleted": deleted,
                "mult": mult,
                "factor": factor,
            }

    async def record_message(
        self,
        group_id: str,
        user_id: str,
        name: str,
        text: str,
        *,
        window_seconds: int,
        max_recent: int,
        pair_weight: float,
        mention_weight: float,
        same_topic_weight: float,
        directed: list[tuple[str, str, float]] | None = None,
        reply_weight: float = 8.0,
        at_weight: float = 5.0,
        adjacency_weight: float = 1.5,
        adjacency_window_seconds: int = 90,
        ambient_tone_scale: float = 0.15,
        directed_event_retain_days: int = 90,
        sentiment_probabilities: dict | None = None,
        source_id: str = "",
    ) -> None:
        ts = _now()
        text = _clean_text(text)
        toks = rs_tokens(text)
        async with self._lock:
            await asyncio.to_thread(
                self._record_message_sync,
                group_id,
                user_id,
                name,
                text,
                toks,
                ts,
                window_seconds,
                max_recent,
                pair_weight,
                mention_weight,
                same_topic_weight,
                list(directed or []),
                reply_weight,
                at_weight,
                adjacency_weight,
                adjacency_window_seconds,
                ambient_tone_scale,
                directed_event_retain_days,
                sentiment_probabilities,
                source_id,
            )

    def _record_message_sync(
        self,
        group_id: str,
        user_id: str,
        name: str,
        text: str,
        toks: set[str],
        ts: int,
        window_seconds: int,
        max_recent: int,
        pair_weight: float,
        mention_weight: float,
        same_topic_weight: float,
        directed: list[tuple[str, str, float]] | None = None,
        reply_weight: float = 8.0,
        at_weight: float = 5.0,
        adjacency_weight: float = 1.5,
        adjacency_window_seconds: int = 90,
        ambient_tone_scale: float = 0.15,
        directed_event_retain_days: int = 90,
        sentiment_probabilities: dict | None = None,
        source_id: str = "",
    ) -> None:
        group_id = str(group_id)
        user_id = str(user_id)
        name = str(name or "")
        directed = list(directed or [])
        cutoff = ts - window_seconds
        today = _dt.date.today().isoformat()
        haystack = (text or "").lower()

        with self._connect() as conn:
            if source_id:
                import hashlib
                stamp = hashlib.sha256(json.dumps([user_id,text,directed],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                old = conn.execute("SELECT fingerprint FROM relation_source_receipts WHERE group_id=? AND source_id=?", (group_id,source_id)).fetchone()
                if old:
                    if old[0] != stamp:
                        raise ValueError("group message source id collision")
                    return
                conn.execute(
                    "INSERT INTO relation_source_receipts(group_id,source_id,fingerprint,created) VALUES(?,?,?,?)",
                    (group_id, source_id, stamp, int(ts or _now())),
                )
            conn.execute(
                """INSERT INTO member_daily(group_id,day,user_id,message_count) VALUES(?,?,?,1)
                ON CONFLICT(group_id,day,user_id) DO UPDATE SET message_count=message_count+1""",
                (group_id, day_key(ts), user_id),
            )
            conn.execute(
                """
                INSERT INTO users(group_id, user_id, name, message_count, last_seen)
                VALUES(?, ?, ?, 1, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET
                  name=excluded.name,
                  message_count=message_count+1,
                  last_seen=excluded.last_seen
                """,
                (group_id, user_id, name, ts),
            )
            recent = conn.execute(
                """
                SELECT user_id, name, text, tokens, ts FROM messages
                WHERE group_id=? AND ts>=? AND user_id<>?
                ORDER BY ts DESC LIMIT ?
                """,
                (group_id, cutoff, user_id, max_recent),
            ).fetchall()

            # latest tokens per other user (for optional directed C8 topic)
            latest_tokens: dict[str, set[str]] = {}
            for row in recent:
                oid = str(row["user_id"])
                if oid not in latest_tokens:
                    latest_tokens[oid] = {
                        tok for tok in str(row["tokens"] or "").split(",") if tok
                    }

            directed_targets: set[str] = set()
            for raw_target, kind, factor in directed:
                target = str(raw_target or "").strip()
                if not target or target == user_id:
                    continue
                kind_s = str(kind or "").strip().lower()
                if kind_s not in {"reply", "at"}:
                    continue
                try:
                    factor_f = float(factor)
                except Exception:
                    factor_f = 1.0
                if kind_s == "reply":
                    base_w = float(reply_weight) * factor_f
                    reason_tag = "回复"
                else:
                    base_w = float(at_weight) * factor_f
                    reason_tag = "@"

                weight = float(base_w)
                reasons: list[str] = [reason_tag]
                other_toks = latest_tokens.get(target) or set()
                overlap = toks & other_toks
                if overlap:
                    weight += float(same_topic_weight) * min(3, len(overlap))
                    reasons.append("同话题")

                conn.execute(
                    """
                    INSERT INTO directed_events(group_id, src, dst, kind, weight, ts)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (group_id, user_id, target, kind_s, float(base_w), ts),
                )

                friendly, hostile, tone_tags = rs_tone_signal(text)
                if len(directed) == 1:
                    friendly, hostile, tone_tags = model_tone(
                        text, sentiment_probabilities, (friendly, hostile, tone_tags)
                    )
                friendly /= max(1, len(directed))
                hostile /= max(1, len(directed))
                # directed edges: full tone scale 1.0
                if friendly or hostile:
                    reasons.extend(tone_tags)

                self._upsert_relation_credit(
                    conn,
                    group_id=group_id,
                    src=user_id,
                    dst=target,
                    score_delta=weight,
                    hits_delta=1,
                    directed_hits_delta=1,
                    reply_hits_delta=1 if kind_s == "reply" else 0,
                    at_hits_delta=1 if kind_s == "at" else 0,
                    ab_ba=True,
                    friendly=float(friendly),
                    hostile=float(hostile),
                    reasons_new=reasons,
                    ts=ts,
                    today=today,
                )
                directed_targets.add(target)

            # Ambient / adjacency / text-mention — skip targets already directed this msg
            seen_ambient: set[str] = set()
            ambient_count = max(
                1, len({str(r["user_id"]) for r in recent} - directed_targets)
            )
            for row in recent:
                other_id = str(row["user_id"])
                if not other_id or other_id == user_id:
                    continue
                if other_id in directed_targets or other_id in seen_ambient:
                    continue
                seen_ambient.add(other_id)

                other_name = str(row["name"] or "")
                age = max(0, ts - int(row["ts"]))
                other_tokens = {
                    tok for tok in str(row["tokens"] or "").split(",") if tok
                }
                overlap = toks & other_tokens
                reasons = []
                weight = 0.0

                name_hit = bool(
                    other_name
                    and len(other_name) >= 2
                    and other_name.lower() in haystack
                )
                adj_window = max(0, int(adjacency_window_seconds))
                is_adjacency = age <= adj_window and (name_hit or len(overlap) >= 2)
                if is_adjacency:
                    weight += float(adjacency_weight)
                    reasons.append("邻接")
                else:
                    decay = max(0.15, 1.0 - age / max(window_seconds, 1))
                    weight += float(pair_weight) * decay / ambient_count
                    reasons.append("近邻")

                # text mention *0.6 only when not directed to that id (already gated)
                if other_id and re.search(
                    r"(?<![A-Za-z0-9_])" + re.escape(other_id) + r"(?![A-Za-z0-9_])",
                    text,
                ):
                    weight += float(mention_weight) * 0.6
                    reasons.append("提到QQ")
                if name_hit:
                    weight += float(mention_weight) * 0.6
                    reasons.append("提到昵称")

                # C8 ambient topic *0.5
                if overlap:
                    weight += float(same_topic_weight) * 0.5 * min(3, len(overlap))
                    reasons.append("同话题")

                if weight <= 0:
                    continue

                friendly, hostile, tone_tags = rs_tone_signal(text)
                direct_tone = any(x in reasons for x in ("提到QQ", "提到昵称"))
                # Ambient mood is not attributed to a particular relationship.
                tone_scale = 1.0 if direct_tone else 0.0
                friendly *= tone_scale
                hostile *= tone_scale
                if friendly or hostile:
                    reasons.extend(tone_tags)

                self._upsert_relation_credit(
                    conn,
                    group_id=group_id,
                    src=user_id,
                    dst=other_id,
                    score_delta=weight,
                    hits_delta=1,
                    directed_hits_delta=0,
                    reply_hits_delta=0,
                    at_hits_delta=0,
                    ab_ba=False,
                    friendly=float(friendly),
                    hostile=float(hostile),
                    reasons_new=reasons,
                    ts=ts,
                    today=today,
                )

            if (
                self._last_history_prune_ts is None
                or abs(ts - self._last_history_prune_ts) >= self._history_prune_interval_seconds
            ):
                self._last_history_prune_ts = ts
                self._prune_directed_events_sync(conn, directed_event_retain_days, batch_size=500)
                self._prune_source_receipts_sync(conn, ts, retain_days=30, batch_size=500)
                self._prune_daily_table_sync(conn, "relation_daily", window_start(ts, 90), batch_size=500)
                self._prune_daily_table_sync(conn, "member_daily", window_start(ts, 90), batch_size=500)

            conn.execute(
                "INSERT INTO messages(group_id, user_id, name, text, tokens, ts) VALUES(?, ?, ?, ?, ?, ?)",
                (group_id, user_id, name, text, ",".join(sorted(toks)), ts),
            )
            conn.execute(
                "DELETE FROM messages WHERE group_id=? AND ts<?",
                (group_id, ts - max(window_seconds * 12, 3600)),
            )

    def _upsert_relation_credit(
        self,
        conn: sqlite3.Connection,
        *,
        group_id: str,
        src: str,
        dst: str,
        score_delta: float,
        hits_delta: int,
        directed_hits_delta: int,
        reply_hits_delta: int,
        at_hits_delta: int,
        ab_ba: bool,
        friendly: float,
        hostile: float,
        reasons_new: list[str],
        ts: int,
        today: str,
    ) -> None:
        """One relations upsert for a directed or ambient credit this message."""
        a, b = self._pair(str(src), str(dst))
        conn.execute(
            "INSERT OR IGNORE INTO users(group_id,user_id,name,message_count,last_seen) VALUES(?,?,?,0,?)",
            (group_id, dst, dst, ts),
        )
        day = day_key(ts)
        used = conn.execute(
            "SELECT raw_evidence FROM relation_daily WHERE group_id=? AND day=? AND a=? AND b=?",
            (group_id, day, a, b),
        ).fetchone()
        granted = min(
            max(0.0, score_delta),
            max(0.0, self.pair_daily_budget - (float(used[0]) if used else 0)),
        )
        ratio = granted / score_delta if score_delta > 0 else 0.0
        friendly, hostile = min(3.0, friendly) * ratio, min(3.0, hostile) * ratio
        quality = (
            1.0
            if directed_hits_delta
            else (
                0.6
                if any(r in reasons_new for r in ("提到QQ", "提到昵称"))
                else (0.2 if "邻接" in reasons_new else 0.05)
            )
        )
        conn.execute(
            """INSERT INTO relation_daily(group_id,day,a,b,raw_evidence,evidence,friendly,hostile,
                ab_friendly,ab_hostile,ba_friendly,ba_hostile,hits,directed_hits,reply_hits,at_hits,ab_hits,ba_hits,sentiment_hits)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(group_id,day,a,b) DO UPDATE SET
                raw_evidence=raw_evidence+excluded.raw_evidence,evidence=evidence+excluded.evidence,
                friendly=friendly+excluded.friendly,hostile=hostile+excluded.hostile,
                ab_friendly=ab_friendly+excluded.ab_friendly,ab_hostile=ab_hostile+excluded.ab_hostile,
                ba_friendly=ba_friendly+excluded.ba_friendly,ba_hostile=ba_hostile+excluded.ba_hostile,
                hits=hits+excluded.hits,directed_hits=directed_hits+excluded.directed_hits,
                reply_hits=reply_hits+excluded.reply_hits,at_hits=at_hits+excluded.at_hits,
                ab_hits=ab_hits+excluded.ab_hits,ba_hits=ba_hits+excluded.ba_hits,sentiment_hits=sentiment_hits+excluded.sentiment_hits""",
            (
                group_id,
                day,
                a,
                b,
                granted,
                granted * quality,
                friendly,
                hostile,
                friendly if src == a else 0,
                hostile if src == a else 0,
                friendly if src == b else 0,
                hostile if src == b else 0,
                int(granted > 0),
                directed_hits_delta if granted > 0 else 0,
                reply_hits_delta if granted > 0 else 0,
                at_hits_delta if granted > 0 else 0,
                int(ab_ba and src == a and granted > 0),
                int(ab_ba and src == b and granted > 0),
                int(bool(friendly or hostile)),
            ),
        )
        score_delta = granted
        if quality >= 0.2 and granted > 0:
            conn.execute(
                "UPDATE relation_daily SET focused_hits=focused_hits+1 WHERE group_id=? AND day=? AND a=? AND b=?",
                (group_id, day, a, b),
            )
        old = conn.execute(
            """
            SELECT reasons, active_days, last_active_date
            FROM relations WHERE group_id=? AND a=? AND b=?
            """,
            (group_id, a, b),
        ).fetchone()
        if old:
            try:
                active_days = int(old["active_days"] or 0)
            except Exception:
                active_days = 0
            last_ad = str(old["last_active_date"] or "")
            if last_ad != today:
                active_days += 1
                last_ad = today
            elif active_days <= 0:
                active_days = 1
                last_ad = today
            old_reasons = str(old["reasons"] or "")
        else:
            active_days = 1
            last_ad = today
            old_reasons = ""

        ab_delta = 0
        ba_delta = 0
        if ab_ba:
            if str(src) == a and str(dst) == b:
                ab_delta = 1
            elif str(src) == b and str(dst) == a:
                ba_delta = 1

        merged_reasons = merge_reasons_priority(old_reasons, list(reasons_new or []))
        sentiment = 1 if (friendly or hostile) else 0
        conn.execute(
            """
            INSERT INTO relations(
              group_id, a, b, score, hits, friendly_score, hostile_score, sentiment_hits,
              last_ts, reasons, directed_hits, reply_hits, at_hits, ab_hits, ba_hits,
              active_days, last_active_date
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(group_id, a, b) DO UPDATE SET
              score=score+excluded.score,
              hits=hits+excluded.hits,
              friendly_score=friendly_score+excluded.friendly_score,
              hostile_score=hostile_score+excluded.hostile_score,
              sentiment_hits=sentiment_hits+excluded.sentiment_hits,
              last_ts=excluded.last_ts,
              reasons=excluded.reasons,
              directed_hits=directed_hits+excluded.directed_hits,
              reply_hits=reply_hits+excluded.reply_hits,
              at_hits=at_hits+excluded.at_hits,
              ab_hits=ab_hits+excluded.ab_hits,
              ba_hits=ba_hits+excluded.ba_hits,
              active_days=excluded.active_days,
              last_active_date=excluded.last_active_date
            """,
            (
                group_id,
                a,
                b,
                float(score_delta),
                int(hits_delta),
                float(friendly),
                float(hostile),
                int(sentiment),
                int(ts),
                merged_reasons,
                int(directed_hits_delta),
                int(reply_hits_delta),
                int(at_hits_delta),
                int(ab_delta),
                int(ba_delta),
                int(active_days),
                last_ad,
            ),
        )

    @staticmethod
    def _merge_reasons(old: str, new_items: list[str]) -> str:
        return merge_reasons_priority(old, list(new_items or []))

    @staticmethod
    def _reciprocity(ab_hits: int, ba_hits: int) -> float:
        """Pair reciprocity: min(ab,ba)/max(ab,ba,1). One-way → 0.0."""
        try:
            ab = max(0, int(ab_hits or 0))
            ba = max(0, int(ba_hits or 0))
        except (TypeError, ValueError):
            return 0.0
        return float(min(ab, ba)) / float(max(ab, ba, 1))

    @staticmethod
    def _median_of(values: Sequence[float]) -> float | None:
        """Median of numeric samples; None when empty (no ZeroDivisionError)."""
        nums = [float(v) for v in values]
        if not nums:
            return None
        return float(statistics.median(nums))

    @staticmethod
    def _pair_median_latency_s(
        events: Sequence[tuple[str, str, int]],
        context_window_seconds: int,
    ) -> float | None:
        """Median direction-flip latency for one unordered pair.

        Walk reply/at events ordered by ts; when direction flips, keep
        dt = curr.ts - prev.ts if 0 < dt <= context_window_seconds * 2.
        Display-only — never multiplies score or labels.
        """
        try:
            max_dt = max(0, int(context_window_seconds)) * 2
        except (TypeError, ValueError):
            max_dt = 0
        dts: list[float] = []
        prev_src: str | None = None
        prev_dst: str | None = None
        prev_ts: int | None = None
        for src, dst, ts in events:
            src_s, dst_s = str(src), str(dst)
            try:
                ts_i = int(ts)
            except (TypeError, ValueError):
                continue
            if prev_src is not None and prev_dst is not None and prev_ts is not None:
                if src_s == prev_dst and dst_s == prev_src:
                    dt = ts_i - prev_ts
                    if 0 < dt <= max_dt:
                        dts.append(float(dt))
            prev_src, prev_dst, prev_ts = src_s, dst_s, ts_i
        return RelationStore._median_of(dts)

    def _pair_latencies_map(
        self,
        conn: sqlite3.Connection,
        group_id: str,
        context_window_seconds: int,
    ) -> dict[tuple[str, str], float | None]:
        """Map unordered pair → median_latency_s from directed_events (reply/at)."""
        rows = conn.execute(
            """
            SELECT src, dst, ts FROM directed_events
            WHERE group_id=? AND kind IN ('reply', 'at')
            ORDER BY ts ASC
            """,
            (group_id,),
        ).fetchall()
        by_pair: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
        for row in rows:
            src, dst = str(row["src"]), str(row["dst"])
            if not src or not dst or src == dst:
                continue
            pa, pb = self._pair(src, dst)
            by_pair[(pa, pb)].append((src, dst, int(row["ts"] or 0)))
        return {
            pair: self._pair_median_latency_s(evs, context_window_seconds)
            for pair, evs in by_pair.items()
        }

    @staticmethod
    def _latency_ratio(
        median_latency_s: float | None,
        group_baseline: float | None,
    ) -> float | None:
        """median / baseline when both > 0; else null (no div0)."""
        if median_latency_s is None or group_baseline is None:
            return None
        try:
            med = float(median_latency_s)
            base = float(group_baseline)
        except (TypeError, ValueError):
            return None
        if med <= 0 or base <= 0:
            return None
        return med / base

    @staticmethod
    def _group_latency_baseline(pair_medians: Sequence[float | None]) -> float | None:
        """Median of non-null pair medians when count ≥ 2; else null."""
        vals = [float(m) for m in pair_medians if m is not None]
        if len(vals) < 2:
            return None
        return RelationStore._median_of(vals)

    @staticmethod
    def _counter_fields(row: sqlite3.Row | dict[str, Any]) -> dict[str, int]:
        """Extract directed counter columns with safe defaults."""

        def _i(key: str) -> int:
            try:
                if isinstance(row, dict):
                    return int(row.get(key) or 0)
                return int(row[key] or 0)
            except (KeyError, TypeError, ValueError, IndexError):
                return 0

        return {
            "directed_hits": _i("directed_hits"),
            "reply_hits": _i("reply_hits"),
            "at_hits": _i("at_hits"),
            "ab_hits": _i("ab_hits"),
            "ba_hits": _i("ba_hits"),
            "active_days": _i("active_days"),
        }

    def _enrich_relation_row(
        self,
        row_dict: dict[str, Any],
        *,
        reciprocity: float,
        median_latency_s: float | None,
        latency_ratio: float | None,
        group_baseline: float | None,
    ) -> dict[str, Any]:
        """Attach C6 read-enrichment keys. Latency is display-only (not used in labels)."""
        out = dict(row_dict)
        counters = RelationStore._counter_fields(out)
        out.update(counters)
        try:
            hits = int(out.get("hits") or 0)
        except (TypeError, ValueError):
            hits = 0
        try:
            score = float(out.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        reasons = str(out.get("reasons") or "")
        try:
            rec = float(reciprocity)
        except (TypeError, ValueError):
            rec = 0.0

        tier = rs_evidence_tier(
            counters["directed_hits"],
            counters["reply_hits"],
            counters["at_hits"],
            hits,
            reasons,
        )
        conf = rs_confidence(
            counters["directed_hits"],
            counters["active_days"],
            hits,
            tier,
        )
        label = rs_strength_label_v2(
            score,
            hits,
            counters["directed_hits"],
            counters["active_days"],
            rec,
            evidence_tier=tier,
            min_hits_tie=self.min_hits_tie,
            min_active_days_tie=self.min_active_days_tie,
            min_directed_tie=self.min_directed_tie,
        )
        # group_baseline accepted for API symmetry / future display; not a C6 key.
        _ = group_baseline
        out["reciprocity"] = rec
        out["median_latency_s"] = median_latency_s
        out["latency_ratio"] = latency_ratio
        out["evidence_tier"] = tier
        out["confidence"] = conf
        out["strength_label"] = label
        if out.get("algorithm_version") == 2:
            for field in (
                "evidence_tier",
                "confidence",
                "reciprocity",
                "strength_label",
            ):
                out[field] = row_dict[field]
        return out

    def _enrich_pair_list(
        self,
        items: list[dict[str, Any]],
        latencies: dict[tuple[str, str], float | None],
        *,
        pair_keys: tuple[str, str] = ("a", "b"),
    ) -> list[dict[str, Any]]:
        """Enrich a list of relation dicts that each have pair user-id keys."""
        k0, k1 = pair_keys
        medians_for_baseline: list[float | None] = []
        pair_meds: list[float | None] = []
        for item in items:
            pa, pb = self._pair(str(item.get(k0, "")), str(item.get(k1, "")))
            med = latencies.get((pa, pb))
            pair_meds.append(med)
            medians_for_baseline.append(med)
        baseline = self._group_latency_baseline(medians_for_baseline)
        enriched: list[dict[str, Any]] = []
        for item, med in zip(items, pair_meds, strict=True):
            ab = int(item.get("ab_hits") or 0)
            ba = int(item.get("ba_hits") or 0)
            rec = self._reciprocity(ab, ba)
            ratio = self._latency_ratio(med, baseline)
            enriched.append(
                self._enrich_relation_row(
                    item,
                    reciprocity=rec,
                    median_latency_s=med,
                    latency_ratio=ratio,
                    group_baseline=baseline,
                )
            )
        return enriched

    async def overview(
        self, group_id: str, *, top_limit: int, daily_decay: float
    ) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(
                self._overview_sync, group_id, top_limit, daily_decay
            )

    def _overview_sync(
        self, group_id: str, top_limit: int, daily_decay: float
    ) -> dict[str, Any]:
        now = _now()
        cw = int(DEFAULT_SCORING.get("context_window_seconds", 300) or 300)
        with self._connect() as conn:
            cutoff_day = window_start(now, self.analysis_window_days)
            counts = {
                str(r["user_id"]): int(r["c"])
                for r in conn.execute(
                    "SELECT user_id,SUM(message_count) AS c FROM member_daily WHERE group_id=? AND day>=? GROUP BY user_id",
                    (group_id, cutoff_day),
                )
            }
            member_days: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in conn.execute(
                "SELECT * FROM member_daily WHERE group_id=? AND day>=?",
                (group_id, cutoff_day),
            ):
                member_days[str(row["user_id"])].append(dict(row))
            users = {
                str(r["user_id"]): {
                    "name": str(r["name"] or r["user_id"]),
                    "count": counts.get(str(r["user_id"]), 0),
                    "lifetime_count": int(r["message_count"]),
                }
                for r in conn.execute(
                    "SELECT user_id, name, message_count FROM users WHERE group_id=?",
                    (group_id,),
                )
            }
            for uid, user in users.items():
                user.update(
                    activity_observation(
                        member_days[uid],
                        now=now,
                        window=self.analysis_window_days,
                        daily_cap=self.activity_daily_cap,
                    )
                )
                user["activity_daily_cap"] = self.activity_daily_cap
            rows = conn.execute(
                "SELECT * FROM relations WHERE group_id=?",
                (group_id,),
            ).fetchall()
            daily: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for r in conn.execute(
                "SELECT * FROM relation_daily WHERE group_id=? AND day>=?",
                (group_id, cutoff_day),
            ):
                daily[(str(r["a"]), str(r["b"]))].append(dict(r))
            started = conn.execute(
                "SELECT value FROM meta WHERE key='network_v2_started'"
            ).fetchone()
            latencies = self._pair_latencies_map(conn, group_id, cw)
        relations = []
        for row in rows:
            item = pair_metrics(
                dict(row),
                daily[(str(row["a"]), str(row["b"]))],
                now=now,
                window=self.analysis_window_days,
                scale=self.relation_growth_scale,
            )
            if item["score"] <= 0:
                continue
            a, b = str(row["a"]), str(row["b"])
            item.update(
                a_name=users.get(a, {}).get("name", a),
                b_name=users.get(b, {}).get("name", b),
            )
            relations.append(item)
        relations.sort(
            key=lambda x: (
                x["directed_hits"] + x["focused_hits"] > 0,
                x["score"],
                x["recent_score"],
            ),
            reverse=True,
        )
        # Network statistics use every pair; top_limit only affects presentation.
        network = network_metrics(users, relations)
        activity_by_id = {str(p["user_id"]): p for p in network["member_profiles"]}
        for uid, user in users.items():
            user["activity_score"] = activity_by_id.get(uid, {}).get(
                "activity_score", 0
            )
        all_pairs = self._enrich_pair_list(relations, latencies)
        current_pairs = [
            r
            for r in all_pairs
            if r["recent_score"] >= 5 and r["directed_hits"] + r["focused_hits"] > 0
        ]
        top = current_pairs[:top_limit]
        active_users = sorted(
            users.items(),
            key=lambda kv: (kv[1]["activity_score"], kv[1]["count"]),
            reverse=True,
        )[:top_limit]
        centers = [
            (p["user_id"], p["social_score"])
            for p in network["member_profiles"]
            if p["partner_count"]
        ][:top_limit]
        return {
            "user_count": network["active_member_count"],
            "window_days": self.analysis_window_days,
            "window_started_at": int(started[0]) if started else now,
            "relation_count": len(current_pairs),
            "historical_relation_count": len(rows),
            "all_relations": all_pairs,
            "top_relations": top,
            "active_users": active_users,
            "centers": centers,
            "users": users,
            **network,
        }

    async def reset_group(self, group_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._reset_group_sync, group_id)

    def _reset_group_sync(self, group_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM relation_daily WHERE group_id=?", (group_id,))
            conn.execute("DELETE FROM member_daily WHERE group_id=?", (group_id,))
            conn.execute("DELETE FROM directed_events WHERE group_id=?", (group_id,))
            conn.execute("DELETE FROM relations WHERE group_id=?", (group_id,))
            conn.execute("DELETE FROM messages WHERE group_id=?", (group_id,))
            conn.execute("DELETE FROM users WHERE group_id=?", (group_id,))

    async def prune_history(
        self,
        *,
        directed_event_retain_days: int = 90,
        receipt_retain_days: int = 30,
        daily_retain_days: int = 90,
        batch_size: int = 500,
        now_ts: int | None = None,
    ) -> dict[str, int]:
        """Batched, indexed history pruning releasing shared lock between chunks."""
        async with self._lock:
            return await asyncio.to_thread(
                self._prune_history_batched_sync,
                directed_event_retain_days,
                receipt_retain_days,
                daily_retain_days,
                batch_size,
                now_ts,
            )

    def _prune_history_batched_sync(
        self,
        directed_event_retain_days: int,
        receipt_retain_days: int,
        daily_retain_days: int,
        batch_size: int,
        now_ts: int | None,
    ) -> dict[str, int]:
        stamp = _now() if now_ts is None else int(now_ts)
        limit = max(1, int(batch_size))
        day_cutoff = window_start(stamp, max(1, int(daily_retain_days)))
        totals = {
            "directed_events": 0,
            "relation_source_receipts": 0,
            "relation_daily": 0,
            "member_daily": 0,
        }
        for key, fn in (
            ("directed_events", lambda c: self._prune_directed_events_sync(c, directed_event_retain_days, batch_size=limit, now_ts=stamp)),
            ("relation_source_receipts", lambda c: self._prune_source_receipts_sync(c, stamp, retain_days=receipt_retain_days, batch_size=limit)),
            ("relation_daily", lambda c: self._prune_daily_table_sync(c, "relation_daily", day_cutoff, batch_size=limit)),
            ("member_daily", lambda c: self._prune_daily_table_sync(c, "member_daily", day_cutoff, batch_size=limit)),
        ):
            while True:
                with self._connect() as conn:
                    deleted = fn(conn)
                totals[key] += deleted
                if deleted < limit:
                    break
        self._last_history_prune_ts = stamp
        return totals

    @staticmethod
    def _prune_daily_table_sync(
        conn: sqlite3.Connection, table: str, cutoff_day: str, *, batch_size: int | None = None
    ) -> int:
        if table not in {"relation_daily", "member_daily"}:
            return 0
        if batch_size is not None and batch_size > 0:
            cur = conn.execute(
                f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} WHERE day < ? LIMIT ?)",
                (cutoff_day, int(batch_size)),
            )
        else:
            cur = conn.execute(f"DELETE FROM {table} WHERE day < ?", (cutoff_day,))
        return int(cur.rowcount or 0)

    @staticmethod
    def _prune_directed_events_sync(
        conn: sqlite3.Connection,
        retain_days: int,
        *,
        batch_size: int | None = None,
        now_ts: int | None = None,
    ) -> int:
        """Delete directed_events older than retain_days natural days.

        Caller owns the connection (same transaction as the write path).
        Returns the number of rows deleted; 0 when retain_days is invalid.
        """
        try:
            retain_days = int(retain_days)
        except Exception:
            return 0
        if retain_days <= 0:
            return 0
        stamp = _now() if now_ts is None else int(now_ts)
        cutoff = stamp - retain_days * 86400
        if batch_size is not None and batch_size > 0:
            cur = conn.execute(
                "DELETE FROM directed_events WHERE rowid IN (SELECT rowid FROM directed_events WHERE ts < ? LIMIT ?)",
                (cutoff, int(batch_size)),
            )
        else:
            cur = conn.execute("DELETE FROM directed_events WHERE ts < ?", (cutoff,))
        return int(cur.rowcount or 0)

    @staticmethod
    def _prune_source_receipts_sync(
        conn: sqlite3.Connection,
        now_ts: int | float | None = None,
        *,
        retain_days: int = 30,
        batch_size: int | None = None,
    ) -> int:
        try:
            days = int(retain_days)
        except Exception:
            return 0
        if days <= 0:
            return 0
        stamp = int(now_ts) if now_ts is not None else _now()
        cutoff = stamp - days * 86400
        if batch_size is not None and batch_size > 0:
            cur = conn.execute(
                "DELETE FROM relation_source_receipts WHERE rowid IN (SELECT rowid FROM relation_source_receipts WHERE created < ? LIMIT ?)",
                (cutoff, int(batch_size)),
            )
        else:
            cur = conn.execute("DELETE FROM relation_source_receipts WHERE created < ?", (cutoff,))
        return int(cur.rowcount or 0)
