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

    async def user_relations(
        self, group_id: str, keyword: str, *, top_limit: int, daily_decay: float
    ) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(
                self._user_relations_sync, group_id, keyword, top_limit, daily_decay
            )

    def _user_relations_sync(
        self, group_id: str, keyword: str, top_limit: int, daily_decay: float
    ) -> dict[str, Any]:
        keyword = (keyword or "").strip().lower()
        with self._connect() as conn:
            candidates = conn.execute(
                "SELECT user_id,name,message_count FROM users WHERE group_id=?",
                (group_id,),
            ).fetchall()
            target = None
            exact_name_matches = []
            fuzzy_matches = []
            for row in candidates:
                uid = str(row["user_id"])
                name = str(row["name"] or "")
                low_name = name.lower()
                if keyword == uid.lower():
                    target = uid
                    break
                if keyword and keyword == low_name:
                    exact_name_matches.append((uid, name))
                elif keyword and keyword in low_name:
                    fuzzy_matches.append((uid, name))
            if not target:
                matches = exact_name_matches or fuzzy_matches
                if len(matches) > 1:
                    return {
                        "found": False,
                        "ambiguous": True,
                        "keyword": keyword,
                        "matches": matches[:8],
                    }
                if len(matches) == 1:
                    target = matches[0][0]
            if not target:
                return {"found": False, "keyword": keyword}
            users = {
                str(r["user_id"]): str(r["name"] or r["user_id"]) for r in candidates
            }
        overview = self._overview_sync(group_id, top_limit, daily_decay)
        enriched = [
            dict(
                r,
                other=r["b"] if r["a"] == target else r["a"],
                name=users.get(r["b"] if r["a"] == target else r["a"], ""),
            )
            for r in overview["all_relations"]
            if target in (r["a"], r["b"])
        ][:top_limit]
        # Drop internal pair keys used only for latency lookup (keep other/name).
        for it in enriched:
            it.pop("a", None)
            it.pop("b", None)
        profile = next(
            (p for p in overview["member_profiles"] if p["user_id"] == target), {}
        )
        return {
            "found": True,
            "target": target,
            "name": users.get(target, target),
            "items": enriched,
            "profile": profile,
            "window_days": self.analysis_window_days,
        }

    async def pair_detail(
        self, group_id: str, left: str, right: str, *, daily_decay: float
    ) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(
                self._pair_detail_sync, group_id, left, right, daily_decay
            )

    def _pair_detail_sync(
        self, group_id: str, left: str, right: str, daily_decay: float
    ) -> dict[str, Any]:
        left = (left or "").strip().lower()
        right = (right or "").strip().lower()
        if not left or not right:
            return {"found": False, "reason": "empty"}
        with self._connect() as conn:
            candidates = conn.execute(
                "SELECT user_id,name,message_count,last_seen FROM users WHERE group_id=?",
                (group_id,),
            ).fetchall()

            def resolve(keyword: str) -> dict[str, Any]:
                exact_id = []
                exact_name = []
                fuzzy = []
                for row in candidates:
                    uid = str(row["user_id"])
                    name = str(row["name"] or "")
                    low_name = name.lower()
                    item = {
                        "user_id": uid,
                        "name": name or uid,
                        "message_count": int(row["message_count"] or 0),
                        "last_seen": int(row["last_seen"] or 0),
                    }
                    if keyword == uid.lower():
                        exact_id.append(item)
                    elif keyword == low_name:
                        exact_name.append(item)
                    elif keyword in low_name:
                        fuzzy.append(item)
                matches = exact_id or exact_name or fuzzy
                if len(matches) == 1:
                    return {"ok": True, "user": matches[0]}
                if len(matches) > 1:
                    return {
                        "ok": False,
                        "ambiguous": True,
                        "keyword": keyword,
                        "matches": matches[:8],
                    }
                return {"ok": False, "keyword": keyword, "matches": []}

            a_res = resolve(left)
            b_res = resolve(right)
            if not a_res.get("ok") or not b_res.get("ok"):
                return {"found": False, "left": a_res, "right": b_res}
            a_uid = a_res["user"]["user_id"]
            b_uid = b_res["user"]["user_id"]
            if a_uid == b_uid:
                return {
                    "found": False,
                    "reason": "same_user",
                    "left": a_res,
                    "right": b_res,
                }
            pa, pb = self._pair(a_uid, b_uid)
            rel = conn.execute(
                f"SELECT score,hits,friendly_score,hostile_score,sentiment_hits,last_ts,reasons, {_REL_COUNTER_SELECT} "
                "FROM relations WHERE group_id=? AND a=? AND b=?",
                (group_id, pa, pb),
            ).fetchone()
            recent = conn.execute(
                """
                SELECT user_id,name,text,tokens,ts FROM messages
                WHERE group_id=? AND user_id IN (?, ?)
                ORDER BY ts DESC LIMIT 30
                """,
                (group_id, a_uid, b_uid),
            ).fetchall()
        if not rel:
            return {
                "found": True,
                "has_relation": False,
                "left": a_res["user"],
                "right": b_res["user"],
                "recent": [],
            }
        topics: dict[str, int] = defaultdict(int)
        samples = []
        for row in recent:
            for tok in str(row["tokens"] or "").split(","):
                if tok:
                    topics[tok] += 1
            text = str(row["text"] or "")
            if text:
                samples.append(
                    {
                        "name": str(row["name"] or row["user_id"]),
                        "user_id": str(row["user_id"]),
                        "text": _clean_text(text)[:80],
                        "ts": int(row["ts"] or 0),
                    }
                )
        top_topics = [
            k for k, _ in sorted(topics.items(), key=lambda kv: kv[1], reverse=True)[:6]
        ]
        overview = self._overview_sync(group_id, 1, daily_decay)
        enriched = dict(
            next(
                (r for r in overview["all_relations"] if r["a"] == pa and r["b"] == pb),
                {},
            )
        )
        enriched.pop("a", None)
        enriched.pop("b", None)
        return {
            "found": True,
            "has_relation": True,
            "left": a_res["user"],
            "right": b_res["user"],
            "topics": top_topics,
            "recent": samples[:8],
            **enriched,
        }

    async def export_group(
        self, group_id: str, *, limit: int, daily_decay: float
    ) -> dict[str, Any]:
        data = await self.overview(group_id, top_limit=limit, daily_decay=daily_decay)
        graph_profiles = sorted(
            data.get("member_profiles", []),
            key=lambda p: (
                0.65 * p["social_score"] + 0.35 * p["activity_score"],
                p["activity_score"],
            ),
            reverse=True,
        )[: min(60, limit)]
        graph_relations = representative_relations(
            data.get("all_relations", []), {str(p["user_id"]) for p in graph_profiles}
        )
        return {
            "group_id": group_id,
            "generated_at": _now(),
            "user_count": data.get("active_member_count", 0),
            "relation_count": data.get("relation_count", 0),
            "top_relations": data.get("top_relations", [])[:limit],
            "active_users": [
                {
                    "user_id": uid,
                    "name": info.get("name", uid),
                    "message_count": info.get("count", 0),
                    "activity_score": info.get("activity_score", 0),
                    "active_days": info.get("active_days", 0),
                    "effective_messages": info.get("effective_messages", 0),
                }
                for uid, info in data.get("active_users", [])[:limit]
            ],
            "social_centers": [
                {
                    "user_id": uid,
                    "name": data.get("users", {}).get(uid, {}).get("name", uid),
                    "score": score,
                }
                for uid, score in data.get("centers", [])[:limit]
            ],
            "member_profiles": data.get("member_profiles", [])[:limit],
            "graph_profiles": graph_profiles,
            "graph_relations": graph_relations,
            "network_summary": {
                k: data.get(k)
                for k in (
                    "breadth",
                    "balance",
                    "reciprocity",
                    "affinity",
                    "conflict_risk",
                    "affinity_confidence",
                    "active_member_count",
                    "connected_member_count",
                    "tone_member_count",
                    "tone_pair_count",
                    "tone_pair_total",
                    "tone_member_coverage",
                    "tone_pair_coverage",
                )
            },
            "window_days": self.analysis_window_days,
        }

    async def group_report(
        self, group_id: str, *, limit: int, daily_decay: float
    ) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(
                self._group_report_sync, group_id, limit, daily_decay
            )

    def _group_report_sync(
        self, group_id: str, limit: int, daily_decay: float
    ) -> dict[str, Any]:
        overview = self._overview_sync(group_id, limit, daily_decay)
        pairs = overview["all_relations"]
        known = [r for r in pairs if r["affinity_confidence"] >= 0.2]
        friendly = sorted(
            (dict(r, tone_score=r["affinity"]) for r in known if r["affinity"] > 0),
            key=lambda r: r["tone_score"],
            reverse=True,
        )
        hostile = sorted(
            (dict(r, tone_score=-r["affinity"]) for r in known if r["affinity"] < 0),
            key=lambda r: r["tone_score"],
            reverse=True,
        )
        with self._connect() as conn:
            token_rows = conn.execute(
                "SELECT tokens FROM messages WHERE group_id=? ORDER BY ts DESC LIMIT 800",
                (group_id,),
            ).fetchall()
        topics: dict[str, int] = defaultdict(int)
        for row in token_rows:
            for tok in str(row["tokens"] or "").split(","):
                if tok:
                    topics[tok] += 1
        top_topics = [
            {"topic": k, "count": v}
            for k, v in sorted(topics.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ]
        total_messages = sum(u["count"] for u in overview["users"].values())
        active = overview["active_member_count"]
        activity = 100 * (
            1
            - math.exp(-total_messages / max(1, active * self.analysis_window_days * 4))
        )
        coverage = 100 * overview["connected_member_count"] / max(1, active)
        cohesion = (
            0.45 * overview["breadth"] + 0.3 * overview["balance"] + 0.25 * coverage
        )
        friendliness = 50 + overview["affinity"] / 2
        conflict = overview["conflict_risk"]
        # Interaction distribution is the principal component. A very strong
        # single edge has a bounded contribution through each member profile.
        overall = (
            0.2 * activity
            + 0.35 * cohesion
            + 0.15 * overview["reciprocity"]
            + 0.2 * friendliness
            + 0.1 * (100 - conflict)
        )
        tone_known = overview["affinity_confidence"] >= 0.2
        if not tone_known:
            overall = (
                0.2 * activity + 0.35 * cohesion + 0.15 * overview["reciprocity"]
            ) / 0.7
        if not active:
            verdict = "近期样本不足"
            overall = 0.0
        elif conflict >= 35:
            verdict = "局部冲突需留意"
        elif cohesion >= 65 and friendliness >= 65:
            verdict = "广泛友好互动"
        elif overview["balance"] < 35:
            verdict = "互动集中在少数关系"
        else:
            verdict = "整体互动稳定"
        started = overview["window_started_at"]
        tone_days = min(
            self.analysis_window_days, max(1, (_now() - started) // 86400 + 1)
        )
        note = f"近 {self.analysis_window_days} 天；历史有向互动已有限回填，正负语气从新版启用后累计（新版已运行 {tone_days} 天）"
        highlights = [
            f"近期活跃 {active} 人，建立有效互动的成员 {overview['connected_member_count']} 人",
            f"平均互动广度 {overview['breadth']:.1f}，互动均衡 {overview['balance']:.1f}",
            (
                f"整体好感倾向 {overview['affinity']:+.1f}；冲突分开统计为 {conflict:.1f}"
                if tone_known
                else "整体正负倾向待观察；综合分暂按互动数据计算"
            ),
            f"可判断语气的成员 {overview['tone_member_count']}/{active} 人，关系 {overview['tone_pair_count']}/{overview['tone_pair_total']} 条",
        ]
        return {
            "group_id": group_id,
            "generated_at": _now(),
            "window_days": self.analysis_window_days,
            "verdict": verdict,
            "overall_score": round(overall, 1),
            "data_note": note,
            "metrics": {
                "活跃度": round(activity, 1),
                "凝聚力": round(cohesion, 1),
                "互动广度": round(overview["breadth"], 1),
                "互动均衡": round(overview["balance"], 1),
                "互惠程度": round(overview["reciprocity"], 1),
                "友善度": round(friendliness, 1),
                "冲突风险": round(conflict, 1),
            },
            "summary": {
                "user_count": active,
                "message_count": total_messages,
                "relation_count": overview["relation_count"],
                "topic_count": len(topics),
                "friendly_score": sum(r["friendly_score"] for r in pairs),
                "hostile_score": sum(r["hostile_score"] for r in pairs),
                "affinity": overview["affinity"],
                "affinity_confidence": overview["affinity_confidence"],
                "tone_member_count": overview["tone_member_count"],
                "tone_pair_count": overview["tone_pair_count"],
                "tone_pair_total": overview["tone_pair_total"],
                "tone_member_coverage": overview["tone_member_coverage"],
                "overall_basis": "互动与正负倾向"
                if tone_known
                else "互动数据（正负待观察）",
            },
            "highlights": highlights,
            "top_topics": top_topics,
            "top_relations": overview["top_relations"][:limit],
            "active_users": [
                {
                    "user_id": uid,
                    "name": u["name"],
                    "message_count": u["count"],
                    "activity_score": u["activity_score"],
                    "active_days": u["active_days"],
                    "effective_messages": u["effective_messages"],
                }
                for uid, u in overview["active_users"]
                if u["count"] > 0
            ][:limit],
            "social_centers": [
                {
                    "user_id": uid,
                    "name": overview["users"].get(uid, {}).get("name", uid),
                    "score": score,
                }
                for uid, score in overview["centers"][:limit]
            ],
            "member_profiles": overview["member_profiles"][:limit],
            "friendly_rank": friendly[:limit],
            "hostile_rank": hostile[:limit],
            "network_summary": {
                k: overview[k]
                for k in (
                    "breadth",
                    "balance",
                    "reciprocity",
                    "affinity",
                    "conflict_risk",
                    "affinity_confidence",
                    "active_member_count",
                    "tone_member_count",
                    "tone_pair_count",
                    "tone_pair_total",
                    "tone_member_coverage",
                    "tone_pair_coverage",
                )
            },
        }

    async def import_external_memory(
        self,
        db_paths: list[str],
        *,
        group_id: str = "",
        limit: int = 500,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(
                self._import_external_memory_sync,
                db_paths,
                group_id,
                limit,
                weight,
            )

    def _import_external_memory_sync(
        self,
        db_paths: list[str],
        group_id: str,
        limit: int,
        weight: float,
    ) -> dict[str, Any]:
        imported = 0
        scanned = 0
        errors: list[str] = []
        for raw_path in db_paths:
            if imported >= limit:
                break
            path = Path(str(raw_path)).expanduser()
            if not path.exists() or not path.is_file():
                errors.append(f"不存在: {path}")
                continue
            try:
                # Read the live WAL as well as the database without modifying it.
                uri = f"file:{path.as_posix()}?mode=ro"
                ext = sqlite3.connect(uri, uri=True, timeout=3)
                ext.row_factory = sqlite3.Row
            except Exception as exc:
                errors.append(f"打开失败 {path}: {exc}")
                continue
            try:
                tables = {
                    str(r["name"])
                    for r in ext.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "social_relations" in tables:
                    rows = ext.execute(
                        """
                        SELECT group_id, from_user, to_user, relation_type,
                               affection_score, interaction_count, strength, frequency,
                               last_interaction, updated_at, created_at
                        FROM social_relations
                        WHERE COALESCE(from_user, '')<>'' AND COALESCE(to_user, '')<>''
                          AND (?='' OR group_id=?)
                        ORDER BY COALESCE(updated_at, last_interaction, created_at, 0) DESC
                        LIMIT ?
                        """,
                        (group_id, group_id, max(0, limit - imported)),
                    ).fetchall()
                    for row in rows:
                        scanned += 1
                        score = self._external_score(row, weight)
                        imported += self._upsert_external_relation(
                            row["group_id"],
                            row["from_user"],
                            row["to_user"],
                            score,
                            row["relation_type"] or "外部记忆",
                            int(
                                row["last_interaction"]
                                or row["updated_at"]
                                or row["created_at"]
                                or _now()
                            ),
                        )
                        if imported >= limit:
                            break
                if imported < limit and "social_network_edges" in tables:
                    rows = ext.execute(
                        """
                        SELECT group_id, from_user_id, to_user_id, edge_type, weight, updated_at, created_at
                        FROM social_network_edges
                        WHERE COALESCE(from_user_id, '')<>'' AND COALESCE(to_user_id, '')<>''
                          AND (?='' OR group_id=?)
                        ORDER BY COALESCE(updated_at, created_at, 0) DESC
                        LIMIT ?
                        """,
                        (group_id, group_id, max(0, limit - imported)),
                    ).fetchall()
                    for row in rows:
                        scanned += 1
                        score = max(0.1, float(row["weight"] or 1.0)) * weight
                        imported += self._upsert_external_relation(
                            row["group_id"],
                            row["from_user_id"],
                            row["to_user_id"],
                            score,
                            row["edge_type"] or "外部记忆",
                            int(row["updated_at"] or row["created_at"] or _now()),
                        )
                        if imported >= limit:
                            break
                if imported < limit and "social_relation_history" in tables:
                    rows = ext.execute(
                        """
                        SELECT group_id, from_user_id, to_user_id, relation_type, new_value, timestamp
                        FROM social_relation_history
                        WHERE COALESCE(from_user_id, '')<>'' AND COALESCE(to_user_id, '')<>''
                          AND (?='' OR group_id=?)
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (group_id, group_id, max(0, limit - imported)),
                    ).fetchall()
                    for row in rows:
                        scanned += 1
                        score = max(0.1, float(row["new_value"] or 1.0)) * weight
                        imported += self._upsert_external_relation(
                            row["group_id"],
                            row["from_user_id"],
                            row["to_user_id"],
                            score,
                            row["relation_type"] or "外部记忆",
                            int(row["timestamp"] or _now()),
                        )
                        if imported >= limit:
                            break
                if imported < limit and "user_social_relation_components" in tables:
                    rows = ext.execute(
                        """
                        SELECT group_id, from_user_id, to_user_id, relation_type, value, frequency, last_interaction, created_at
                        FROM user_social_relation_components
                        WHERE COALESCE(from_user_id, '')<>'' AND COALESCE(to_user_id, '')<>''
                          AND (?='' OR group_id=?)
                        ORDER BY COALESCE(last_interaction, created_at, 0) DESC
                        LIMIT ?
                        """,
                        (group_id, group_id, max(0, limit - imported)),
                    ).fetchall()
                    for row in rows:
                        scanned += 1
                        score = (
                            max(0.1, float(row["value"] or 1.0))
                            * max(1, int(row["frequency"] or 1))
                            * weight
                        )
                        imported += self._upsert_external_relation(
                            row["group_id"],
                            row["from_user_id"],
                            row["to_user_id"],
                            score,
                            row["relation_type"] or "外部记忆",
                            int(row["last_interaction"] or row["created_at"] or _now()),
                        )
                        if imported >= limit:
                            break
            except Exception as exc:
                errors.append(f"读取失败 {path}: {exc}")
            finally:
                try:
                    ext.close()
                except Exception:
                    pass
        return {"imported": imported, "scanned": scanned, "errors": errors}

    @staticmethod
    def _external_score(row: sqlite3.Row, weight: float) -> float:
        values = []
        for key in ("affection_score", "strength"):
            try:
                if row[key] is not None:
                    values.append(float(row[key]))
            except Exception:
                pass
        base = max(values) if values else 1.0
        try:
            base += min(20.0, float(row["interaction_count"] or 0) * 0.2)
        except Exception:
            pass
        try:
            base += min(10.0, float(row["frequency"] or 0) * 0.1)
        except Exception:
            pass
        return max(0.1, base) * weight

    def _upsert_external_relation(
        self, group_id: Any, left: Any, right: Any, score: float, reason: str, ts: int
    ) -> int:
        group = str(group_id or "").strip()
        a0 = str(left or "").strip()
        b0 = str(right or "").strip()
        if not group or not a0 or not b0 or a0 == b0:
            return 0
        a, b = self._pair(a0, b0)
        reason_text = f"外部记忆:{str(reason or '关系').strip()[:20]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users(group_id, user_id, name, message_count, last_seen)
                VALUES(?, ?, ?, 0, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET last_seen=max(last_seen, excluded.last_seen)
                """,
                (group, a, a, ts),
            )
            conn.execute(
                """
                INSERT INTO users(group_id, user_id, name, message_count, last_seen)
                VALUES(?, ?, ?, 0, ?)
                ON CONFLICT(group_id, user_id) DO UPDATE SET last_seen=max(last_seen, excluded.last_seen)
                """,
                (group, b, b, ts),
            )
            old = conn.execute(
                "SELECT reasons FROM relations WHERE group_id=? AND a=? AND b=?",
                (group, a, b),
            ).fetchone()
            merged_reasons = self._merge_reasons(
                old["reasons"] if old else "", [reason_text]
            )
            conn.execute(
                """
                INSERT INTO relations(group_id, a, b, score, hits, friendly_score, hostile_score, sentiment_hits, last_ts, reasons)
                VALUES(?, ?, ?, ?, 1, 0, 0, 0, ?, ?)
                ON CONFLICT(group_id, a, b) DO UPDATE SET
                  score=score+excluded.score,
                  hits=hits+1,
                  last_ts=max(last_ts, excluded.last_ts),
                  reasons=excluded.reasons
                """,
                (group, a, b, float(score), ts, merged_reasons),
            )
        return 1

    async def sentiment_rank(
        self, group_id: str, mode: str, *, top_limit: int, daily_decay: float
    ) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(
                self._sentiment_rank_sync, group_id, mode, top_limit, daily_decay
            )

    def _sentiment_rank_sync(
        self, group_id: str, mode: str, top_limit: int, daily_decay: float
    ) -> dict[str, Any]:
        mode = (
            "hostile"
            if str(mode).lower() in {"hostile", "敌对", "negative"}
            else "friendly"
        )
        overview = self._overview_sync(group_id, top_limit, daily_decay)
        sign = -1 if mode == "hostile" else 1
        items = [
            dict(r, tone_score=sign * r["affinity"])
            for r in overview["all_relations"]
            if sign * r["affinity"] > 0 and r["affinity_confidence"] >= 0.2
        ]
        items.sort(key=lambda r: r["tone_score"], reverse=True)
        return {
            "mode": mode,
            "items": items[:top_limit],
            "window_days": self.analysis_window_days,
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
