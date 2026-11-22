# ruff: noqa: E402 -- owned module refresh must precede imports on hot reload.
from __future__ import annotations

# Refresh owned modules before importing domain code on a scoped hot reload.
from .reload_bootstrap import ensure_current

ensure_current(
    __package__,
    __name__,
    __file__,
    version="2.0.0-rc.6",
    checks=[
        ("core.engine", "Engine", "_review_contract_v1"),
        ("core.engine", "Engine", "_automatic_review_contract_v1"),
    ],
)

import asyncio
import hashlib
import inspect
import json
import math
import time
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import json_response, request

from .core.contract import (
    MODEL_INPUT_LIMIT,
    PERSONALITY_POLICY,
    MalformedModelResponse,
    decode_model_object,
    encode,
    text,
)
from .core.engine import Engine
from .core.native import current_persona, group_projection, private_projection, replacement_allowed, role_provider
from .core.profiles import native_message
from .core.review_batches import ReviewUnavailable
from .core.review_evidence import provider_model
from .core.social import Social, character_key, observed_event
from .core.source_withdrawals import recall_source
from .core.store import Store
from .series_panels import (
    get_series_defaults,
    inherit_umos,
    is_provider_budget_busy_error,
    is_series_group_excluded,
    resolve_series_provider,
    resolve_series_timezone,
    series_llm_call,
)

PLUGIN = "astrbot_plugin_dynamics_learning"
MIN_SAFE_MODEL_MAX_TOKENS = 1536
REASONING_EFFORT_OPTIONS = frozenset({"none", "minimal", "low", "medium", "high"})
_CATEGORY_LABELS = {
    "expression_style": "本群表达习惯",
    "expression_adjustment": "表达反馈调整",
    "scene_pattern": "群内接话场景",
    "scene_adjustment": "接话反馈调整",
    "expression_example": "原话样例",
    "style": "群友表达方式",
    "typing_habit": "群友打字习惯",
    "preference": "群友喜恶",
    "jargon": "群聊黑话",
    "affection": "数值好感度",
    "relationship": "社交关系",
    "state": "短期状态",
    "decision_guidance": "决策指导",
    "participation": "接话阈值",
}


_TZ_SHANGHAI = timezone(timedelta(hours=8))
_UNBOUND_PERSONA_SENTINELS = frozenset({"__none__", "[%None]"})


def _iso_timestamp(ts: float | None = None, *, tz=None) -> str:
    zone = tz or _TZ_SHANGHAI
    stamp = time.time() if ts is None else float(ts)
    return datetime.fromtimestamp(stamp, tz=zone).isoformat(timespec="seconds")


def _date_window(
    date_str: str, *, now_ts: float | None = None, tz=None
) -> tuple[float, float] | None:
    zone = tz or _TZ_SHANGHAI
    clean = str(date_str or "today").strip().lower()
    if clean in {"", "today"}:
        stamp = time.time() if now_ts is None else float(now_ts)
        now = datetime.fromtimestamp(stamp, tz=zone)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp(), (start + timedelta(days=1)).timestamp()
    try:
        parsed = datetime.strptime(clean, "%Y-%m-%d").replace(tzinfo=zone)
        start = parsed.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp(), (start + timedelta(days=1)).timestamp()
    except ValueError:
        return None


def _resolve_journal_window(
    date_str: str,
    start_ts: float | None = None,
    end_ts: float | None = None,
    *,
    now_ts: float | None = None,
    tz=None,
) -> tuple[bool, tuple[float, float] | None]:
    if start_ts is not None or end_ts is not None:
        if (
            isinstance(start_ts, bool)
            or isinstance(end_ts, bool)
            or not isinstance(start_ts, (int, float))
            or not isinstance(end_ts, (int, float))
        ):
            return False, None
        s_val = float(start_ts)
        e_val = float(end_ts)
        if not (math.isfinite(s_val) and math.isfinite(e_val) and e_val > s_val):
            return False, None
        return True, (s_val, e_val)
    clean = str(date_str or "today").strip().lower()
    if clean in {"all", "*"}:
        return True, None
    bounds = _date_window(clean, now_ts=now_ts, tz=tz)
    if bounds is None:
        return False, None
    return True, bounds


def _value_snippet(raw, limit: int = 80) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        text_val = raw
    elif isinstance(raw, dict):
        if raw.get("kind") == "expression_example":
            text_val = str(raw.get("expression") or raw.get("situation") or "")
        elif raw.get("stance"):
            text_val = f"{raw.get('topic', '')}：{raw.get('quote', '')}"
        elif "affection_level" in raw:
            try:
                level = float(raw.get("affection_level") or 0)
                text_val = f"好感 {level:.1f}"
            except (TypeError, ValueError):
                text_val = "好感变化"
        else:
            text_val = encode(raw)
    else:
        text_val = str(raw)
    flat = " ".join(text_val.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _accepts_kwarg(fn, name):
    """Return True if callable fn accepts keyword argument `name` (explicitly or via **kwargs)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    param = params.get(name)
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _cache_anchor_contexts(instruction):
    """Build a deterministic 2-message static anchor so Gemini/Responses implicit caching hits across calls."""
    if not isinstance(instruction, str) or not instruction.strip():
        return []
    prefix = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:12]
    anchor = f"[cache_anchor task=dynamics_learning prefix={prefix}]"
    return [
        {"role": "user", "content": anchor},
        {"role": "assistant", "content": "OK"},
    ]


def _text_chat_extra_kwargs(provider, config):
    """Build optional max_tokens / reasoning_effort kwargs accepted by provider.text_chat."""
    if not isinstance(config, dict) and not hasattr(config, "get"):
        return {}
    candidates = {}
    raw_max = config.get("model_max_tokens", 0)
    if not isinstance(raw_max, bool):
        try:
            parsed_max = int(raw_max or 0)
        except (TypeError, ValueError):
            parsed_max = 0
        if parsed_max > 0:
            candidates["max_tokens"] = max(MIN_SAFE_MODEL_MAX_TOKENS, parsed_max)
    raw_effort = str(config.get("model_reasoning_effort", "") or "").strip().lower()
    if raw_effort in REASONING_EFFORT_OPTIONS:
        candidates["reasoning_effort"] = raw_effort
    if not candidates:
        return {}
    text_chat = getattr(provider, "text_chat", None)
    if not callable(text_chat):
        return {}
    try:
        params = inspect.signature(text_chat).parameters
    except (TypeError, ValueError):
        return {}
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    allowed_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {
        key: val
        for key, val in candidates.items()
        if has_varkw or (key in params and params[key].kind in allowed_kinds)
    }


class DynamicsLearning(Star):
    """Public synchronous ingress + cached read; model and SQLite work stays off reply path."""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.engine = None
        self.tasks = []
        self.closed = False
        self.ready = False
        self.jobs = set()
        self.projection_error = ""
        self.legacy_kv_v1 = {"present": False, "count": 0}

    async def _detect_legacy_kv_v1(self):
        """Read-only check for residual v1.4.1 learning_*_v1 shared KV keys; never migrates or applies them."""
        try:
            from astrbot.core import sp
        except Exception:
            return {"present": False, "count": 0}
        if getattr(sp, "db_helper", None) is None:
            return {"present": False, "count": 0}
        count = 0
        plugin_id = getattr(self, "name", None) or PLUGIN
        for scope, scope_id in (("plugin", plugin_id), ("global", "global"), ("global", None)):
            try:
                rows = await sp.range_get_async(scope, scope_id, None)
            except Exception:
                continue
            for pref in rows or ():
                key = str(getattr(pref, "key", "") or "")
                if key.startswith("learning_") and key.endswith("_v1"):
                    count += 1
            if count:
                break
        return {"present": count > 0, "count": count}

    def _effective_sessions(self) -> list[str]:
        fn = getattr(self.engine, "effective_sessions", None)
        if callable(fn):
            return fn()
        explicit = self.config.get("sessions")
        if explicit:
            return list(explicit)
        return inherit_umos([], self.context) if self.context is not None else []

    def _effective_provider(self, role: str) -> str:
        if self.engine is not None:
            return str(self.engine.provider_for(role) or "").strip()
        explicit = str(self.config.get(role, "") or "").strip()
        if explicit:
            return explicit
        if role == "analysis_provider":
            return resolve_series_provider(
                self.context, "", role="reply"
            ) or Engine._host_default_provider_id(self.context)
        if role == "review_provider":
            defaults = get_series_defaults(self.context)
            candidate = str(defaults.get("decision_provider") or "").strip()
            if candidate:
                getter = getattr(self.context, "get_provider_by_id", None)
                provider = getter(candidate) if callable(getter) else None
                if provider is None or not getattr(provider, "is_native_decision_provider", False):
                    return candidate
            return str(defaults.get("reply_provider") or "").strip() or Engine._host_default_provider_id(
                self.context
            )
        return ""

    def _provider_status(self):
        status = {}
        get_provider = getattr(self.context, "get_provider_by_id", None)
        for role in ("analysis_provider", "review_provider"):
            pid = self._effective_provider(role)
            if not pid:
                status[role] = {"configured": False, "ready": False, "provider_id": ""}
                continue
            provider = get_provider(pid) if callable(get_provider) else None
            ready = bool(
                provider is not None
                and callable(getattr(provider, "text_chat", None))
                and not getattr(provider, "is_native_decision_provider", False)
            )
            status[role] = {"configured": True, "ready": ready, "provider_id": pid}
        return status

    async def initialize(self):
        from importlib import reload

        from . import series_panels
        from .series_workspace import register_panel_api

        reload(series_panels).register_series_panels(self, "astrbot_plugin_dynamics_learning")
        path = StarTools.get_data_dir(PLUGIN) / "learning.sqlite3"
        store = await asyncio.to_thread(Store, path)
        await asyncio.to_thread(store.enforce_personality_policy)
        await asyncio.to_thread(store.native.bootstrap_scopes)
        self.legacy_kv_v1 = await self._detect_legacy_kv_v1()
        self.engine = Engine(store, self.config, self._model, host_context=self.context, provider_tracks_reviews=True)
        await self.engine.refresh()
        self.engine.social = await asyncio.to_thread(Social, self.engine, self.context, logger)
        await self.engine.social.initialize()
        await self.engine.apply_pending_affection()
        await self.engine.apply_approved_profiles()
        self.tasks = [
            asyncio.create_task(self.engine.consume()),
            asyncio.create_task(self.engine.automatic_reviews.run()),
            asyncio.create_task(self.engine.maintenance()),
            asyncio.create_task(self.engine.social.consume()),
        ]
        for endpoint, handler, methods in (
            ("overview", self.overview, ["GET"]),
            ("analyze", self.analyze, ["POST"]),
            ("review", self.review, ["POST"]),
            ("action", self.action, ["POST"]),
            ("amend", self.amend, ["POST"]),
            ("purge", self.purge, ["POST"]),
            ("social", self.social, ["POST"]),
            ("jargon_propose", self.jargon_propose, ["POST"]),
            ("profile", self.profile, ["POST"]),
            ("history", self.history, ["POST"]),
            ("review_group", self.review_group, ["POST"]),
            ("batch_action", self.batch_action, ["POST"]),
            ("review_queue", self.review_queue, ["POST"]),
            ("group_batch_action", self.group_batch_action, ["POST"]),
            ("group_batch_review", self.group_batch_review, ["POST"]),
            ("group_batch_reject", self.group_batch_reject, ["POST"]),
            ("review_batch_jobs", self.review_batch_jobs, ["POST"]),
            ("review_batch_control", self.review_batch_control, ["POST"]),
            ("reply_preview", self.reply_preview, ["POST"]),
            ("governance_action", self.governance_action, ["POST"]),
            ("library", self.library, ["POST"]),
            ("material_detail", self.material_detail, ["POST"]),
            ("entries", self.entries, ["POST"]),
            ("proposal_detail", self.proposal_detail, ["POST"]),
            ("members", self.members, ["POST"]),
            ("member_detail", self.member_detail, ["POST"]),
            ("usage", self.usage, ["POST"]),
        ):
            register_panel_api(self, self.context.register_web_api, f"/{PLUGIN}/{endpoint}", handler, methods, "群间学习审查中心")
        logger.info("[DynamicsLearning] Ready; configured scopes=%s", len(self._effective_sessions()))
        self.ready = True

    def get_dynamics_capabilities(self):
        return {
            "schema_version": 1,
            "display_name": "群间学习",
            "enabled": replacement_allowed(self),
            "ready": self.ready and not self.closed,
            "capabilities": {"social": 1},
        }

    def get_series_summary(self) -> dict:
        """Synchronous, read-only badge and pulse summary for series_presence v2."""
        tz = resolve_series_timezone(self.context)
        updated_at = _iso_timestamp(tz=tz)
        if self.closed or not self.ready or self.engine is None:
            return {
                "badge": {"count": 0, "tone": "muted", "label": "未就绪"},
                "pulse": {"headline": "群间学习尚未就绪", "updated_at": updated_at},
            }
        try:
            counts = self.engine.store.proposal_counts()
            pending = int(counts.get("pending", 0)) + int(counts.get("reviewed", 0))
            applied = int(counts.get("applied", 0))
            master_enabled = bool(self.config.get("enabled", True))
            if not master_enabled:
                label = f"总开关已关闭 · {pending} 项待审" if pending > 0 else "总开关已关闭"
                headline = f"学习总开关已关闭 · 待审 {pending} 项 · 已应用 {applied} 条"
                return {
                    "badge": {"count": pending, "tone": "muted", "label": label},
                    "pulse": {"headline": headline, "updated_at": updated_at},
                }
            tone = "warm" if pending > 0 else "accent"
            label = f"{pending} 项待审" if pending > 0 else "队列空闲"
            job_note = f"处理中 {len(self.jobs)} 项" if self.jobs else "队列空闲"
            headline = f"待审 {pending} 项 · 已应用 {applied} 条 · {job_note}"
            return {
                "badge": {"count": pending, "tone": tone, "label": label},
                "pulse": {"headline": headline, "updated_at": updated_at},
            }
        except Exception:
            return {
                "badge": {"count": 0, "tone": "muted", "label": "暂不可用"},
                "pulse": {"headline": "状态读取暂不可用", "updated_at": updated_at},
            }

    def get_series_scope_status(self, umo: str) -> dict:
        """Read-only per-scope status for the cross-plugin group control matrix."""
        scope = str(umo or "").strip()[:512]
        master_enabled = bool(self.config.get("enabled", True))
        if self.closed or not self.ready or self.engine is None:
            return {
                "enabled": False,
                "configured": False,
                "master_enabled": master_enabled,
                "supported": False,
                "persona_id": "",
                "activity_today": None,
                "summary": "群间学习尚未就绪",
                "version": 0,
            }
        try:
            sessions = self._effective_sessions()
            configured = bool(scope and scope in sessions)
            excluded_by_core = bool(scope and is_series_group_excluded(self.context, umo=scope))
            active = bool(scope and not excluded_by_core and self.engine.allowed(scope))
            persona_id = str(self.engine.current_personas.get(scope, "") or "")
            version = int(self.engine.scope_projection_generation(scope)) if scope else 0
            if not scope:
                summary = (
                    f"已配置 {len(sessions)} 个会话"
                    if master_enabled
                    else f"学习总开关已关闭（已配置 {len(sessions)} 个会话）"
                )
                return {
                    "enabled": False,
                    "configured": False,
                    "excluded_by_core": False,
                    "master_enabled": master_enabled,
                    "supported": True,
                    "persona_id": "",
                    "activity_today": 0,
                    "summary": summary,
                    "version": version,
                }
            now = time.time()
            tz = resolve_series_timezone(self.context)
            today_bounds = _date_window("today", tz=tz)
            day_start, day_end = today_bounds if today_bounds is not None else (0.0, now + 1.0)
            store = self.engine.store
            with store.lock:
                pending_row = store.db.execute(
                    "SELECT count(*) FROM proposals WHERE scope=? AND state IN ('pending','reviewed')",
                    (scope,),
                ).fetchone()
                active_row = store.db.execute(
                    "SELECT count(*) FROM entries WHERE json_extract(body,'$.scope')=? "
                    "AND json_extract(body,'$.active')=1 "
                    "AND (json_extract(body,'$.category')='affection' OR json_extract(body,'$.expires')>?)",
                    (scope, now),
                ).fetchone()
                today_row = store.db.execute(
                    "SELECT count(*) FROM proposals WHERE scope=? AND updated>=? AND updated<?",
                    (scope, day_start, day_end),
                ).fetchone()
            pending_count = int(pending_row[0]) if pending_row else 0
            active_count = int(active_row[0]) if active_row else 0
            activity_today = int(today_row[0]) if today_row else 0
            if excluded_by_core:
                summary = "主控排除"
            elif active:
                summary = f"待审 {pending_count} · 有效资料 {active_count}"
            elif configured and not master_enabled:
                summary = f"总开关已关闭 · 已配置该群（待审 {pending_count} · 有效资料 {active_count}）"
            elif not master_enabled:
                summary = "总开关已关闭 · 未纳入生效会话"
            else:
                summary = "未纳入生效会话"
            return {
                "enabled": active,
                "configured": configured,
                "excluded_by_core": excluded_by_core,
                "master_enabled": master_enabled,
                "supported": True,
                "persona_id": persona_id,
                "activity_today": activity_today,
                "summary": summary,
                "version": version,
            }
        except Exception:
            return {
                "enabled": None,
                "configured": False,
                "master_enabled": master_enabled,
                "supported": True,
                "persona_id": "",
                "activity_today": None,
                "summary": "状态读取失败",
                "version": 0,
            }

    def series_search_items(self, query: str, scope: str = "", limit: int = 8, persona: str = "") -> list[dict]:
        """Lightweight read-only search across pending proposals and active learned entries."""
        q = str(query or "").strip()[:128]
        target_scope = str(scope or "").strip()[:512]
        target_persona = str(persona or "").strip()[:256]
        try:
            max_items = max(1, min(20, int(limit or 8)))
        except (TypeError, ValueError):
            max_items = 8
        if self.closed or not self.ready or self.engine is None or not q:
            return []
        if target_persona in _UNBOUND_PERSONA_SENTINELS:
            return []
        try:
            allowed = tuple(self._effective_sessions())
            if not allowed:
                return []
            if target_scope and target_scope not in allowed:
                return []
            q_lower = q.lower()
            results: list[dict] = []
            seen_ids: set[str] = set()
            store = self.engine.store
            allowed_json = encode(list(allowed))
            with store.lock:
                prop_rows = store.db.execute(
                    """SELECT body FROM proposals
                    WHERE state IN ('pending','reviewed')
                    AND (?='' OR scope=?)
                    AND (?='' OR persona=?)
                    AND scope IN (SELECT value FROM json_each(?))
                    AND instr(lower(body), ?) > 0
                    ORDER BY updated DESC LIMIT ?""",
                    (target_scope, target_scope, target_persona, target_persona, allowed_json, q_lower, max_items),
                ).fetchall()
                for row in prop_rows:
                    p = json.loads(row[0])
                    pid = str(p.get("id") or "")
                    if not pid or pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    cat = _CATEGORY_LABELS.get(p.get("category", ""), p.get("category", ""))
                    member = p.get("member") or "全群"
                    snippet = _value_snippet(p.get("value"), 64)
                    results.append(
                        {
                            "kind": "learning_proposal",
                            "title": f"待审 · {cat} · {p.get('key', '')}",
                            "subtitle": f"{member} · {snippet}" if snippet else str(member),
                            "plugin_id": PLUGIN,
                            "page": "review",
                            "deep_link": {"tab": "proposals", "scope": p.get("scope", ""), "id": pid},
                        }
                    )
                    if len(results) >= max_items:
                        return results

                remaining = max_items - len(results)
                if remaining > 0:
                    now = time.time()
                    entry_rows = store.db.execute(
                        """SELECT body FROM entries
                        WHERE json_extract(body,'$.active')=1
                        AND (json_extract(body,'$.category')='affection' OR json_extract(body,'$.expires')>?)
                        AND (?='' OR json_extract(body,'$.scope')=?)
                        AND (?='' OR json_extract(body,'$.persona')=?)
                        AND json_extract(body,'$.scope') IN (SELECT value FROM json_each(?))
                        AND instr(lower(body), ?) > 0
                        ORDER BY rowid DESC LIMIT ?""",
                        (now, target_scope, target_scope, target_persona, target_persona, allowed_json, q_lower, remaining),
                    ).fetchall()
                    for row in entry_rows:
                        e = json.loads(row[0])
                        eid = str(e.get("id") or "")
                        if not eid or eid in seen_ids:
                            continue
                        seen_ids.add(eid)
                        cat = _CATEGORY_LABELS.get(e.get("category", ""), e.get("category", ""))
                        member = e.get("member") or "全群"
                        snippet = _value_snippet(e.get("value"), 64)
                        results.append(
                            {
                                "kind": "learning_entry",
                                "title": f"{cat} · {e.get('key', '')}",
                                "subtitle": f"{member} · {snippet}" if snippet else str(member),
                                "plugin_id": PLUGIN,
                                "page": "review",
                                "deep_link": {"tab": "entries", "scope": e.get("scope", ""), "id": eid},
                            }
                        )
            return results
        except Exception:
            return []

    def get_journal_events(
        self,
        persona: str = "",
        scope: str = "",
        date: str = "today",
        limit: int = 20,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> list[dict]:
        """Read-only journal events for Dynamics World's Today's Living Journal feed."""
        if self.closed or not self.ready or self.engine is None:
            return []
        try:
            max_items = max(1, min(50, int(limit or 20)))
        except (TypeError, ValueError):
            max_items = 20
        target_scope = str(scope or "").strip()[:512]
        target_persona = str(persona or "").strip()[:256]
        if target_persona in _UNBOUND_PERSONA_SENTINELS:
            return []
        try:
            allowed = tuple(self._effective_sessions())
            if not allowed:
                return []
            if target_scope and target_scope not in allowed:
                return []
            if target_scope and not target_persona:
                bound_persona = str(self.engine.current_personas.get(target_scope, "") or "").strip()
                if not bound_persona or bound_persona in _UNBOUND_PERSONA_SENTINELS:
                    return []
                target_persona = bound_persona
            tz = resolve_series_timezone(self.context)
            ok, bounds = _resolve_journal_window(date, start_ts, end_ts, tz=tz)
            if not ok:
                return []
            allowed_json = encode(list(allowed))
            store = self.engine.store
            with store.lock:
                sql = (
                    "SELECT body FROM proposals "
                    "WHERE state IN ('pending','reviewed','applied') "
                )
                params: list[object] = []
                if bounds is not None:
                    sql += "AND updated >= ? AND updated < ? "
                    params.extend(bounds)
                sql += (
                    "AND (?='' OR scope=?) "
                    "AND (?='' OR persona=?) "
                    "AND scope IN (SELECT value FROM json_each(?)) "
                    "ORDER BY updated DESC LIMIT ?"
                )
                params.extend(
                    (
                        target_scope,
                        target_scope,
                        target_persona,
                        target_persona,
                        allowed_json,
                        max_items,
                    )
                )
                rows = store.db.execute(sql, params).fetchall()
            events: list[dict] = []
            for row in rows:
                p = json.loads(row[0])
                pid = str(p.get("id") or "")
                if not pid:
                    continue
                cat_key = str(p.get("category") or "")
                cat = _CATEGORY_LABELS.get(cat_key, cat_key or "学习调整")
                member = p.get("member") or "全群"
                state = str(p.get("state") or "")
                snippet = _value_snippet(p.get("value"), 72)
                if state in {"pending", "reviewed"}:
                    tone = "warm"
                    title = f"观察到新提案：{cat} · {p.get('key', '')}"
                    subtitle = f"{member} · 点击进入调整审查核对"
                elif cat_key == "affection":
                    tone = "info"
                    title = f"好感度更新：{member}"
                    subtitle = snippet or "已记录新的互动好感变化"
                else:
                    tone = "accent"
                    title = f"已学会：{cat} · {p.get('key', '')}"
                    subtitle = f"{member} · {snippet}" if snippet else str(member)
                events.append(
                    {
                        "id": pid,
                        "category": "learning",
                        "source_label": "群间学习",
                        "tone": tone,
                        "occurred_at": _iso_timestamp(p.get("updated") or p.get("created"), tz=tz),
                        "title": title,
                        "subtitle": subtitle,
                        "target_plugin": PLUGIN,
                        "target_route": "review",
                        "target_id": pid,
                    }
                )
            return events
        except Exception:
            return []

    def peek_context_revision(self, scope):
        scope_key = str(scope.get("umo", ""))
        return {
            "available": bool(self.ready and replacement_allowed(self)),
            "scope_key": scope_key,
            "generation": self.engine.scope_projection_generation(scope_key) if self.engine else 0,
        }

    async def observe_events(self, events):
        if not self.ready or not replacement_allowed(self):
            return {"schema_version": 1, "available": False, "accepted": 0}
        if not isinstance(events, list) or len(events) > 100:
            raise ValueError("invalid native event batch")
        async with self.engine.lock:
            accepted = await asyncio.to_thread(
                self.engine.store.native.observe, events, self._effective_sessions()
            )
        return {"schema_version": 1, "available": True, "accepted": accepted}

    def learning_capabilities(self):
        return {
            "schema_version": 1,
            "ready": self.ready and self.engine is not None and not self.closed,
            "capabilities": [
                "turn_feedback",
                "reviewed_context",
                "participation_threshold",
                "group_social",
                "affection",
                "jargon",
                "member_profiles",
                "member_learning_layers",
                "reply_projection_v1",
                "fixed_personality",
            ]
            + (
                ["scene_learning_v1"]
                + (["scene_disambiguation_v1"] if self.config.get("scene_disambiguation_enabled", False) else [])
                if self.config.get("scene_learning_enabled", False)
                else []
            )
            + (["expression_style_v1"] if self.config.get("expression_style_enabled", False) else []),
        }

    def get_scene_context(self, **scope):
        return self.engine.scenes.context(**scope) if self.ready and self.engine and not self.closed else {}

    def get_expression_style_context(self, **scope):
        return self.engine.expressions.context(**scope) if self.ready and self.engine and not self.closed else {}

    async def disambiguate_reply_scene(self, **request):
        if not self.ready or not self.engine or self.closed:
            return {}
        return await self.engine.scenes.disambiguate(**request)

    def report_scene_event(self, record):
        if (
            not self.ready
            or not self.engine
            or self.closed
            or not isinstance(record, dict)
            or not self.engine.expression_feedback_enabled(record.get("scope", ""), record.get("persona"))
        ):
            return False
        try:
            payload = json.loads(encode(record))
            if len(encode(payload)) > 8000:
                return False
            return self.engine.enqueue(
                {"kind": "scene_receipt", "scope": payload["scope"], "record": payload}
            )
        except (ValueError, TypeError):
            return False

    def ingest_dynamics_event(self, event):
        return bool(self.ready and self.engine and not self.closed and self.engine.ingest(event))

    def report_reply_usage(self, record):
        if not self.ready or not isinstance(record, dict) or not self.engine.allowed(record.get("scope")):
            return False
        try:
            payload = json.loads(encode(record))
            if len(encode(payload)) > 16000:
                return False
            return self.engine.enqueue(
                {"kind": "reply_usage", "scope": payload["scope"], "record": payload}
            )
        except (ValueError, TypeError):
            return False

    def get_culture_jargon_v1(self, *, scope, query="", limit=10):
        """Read reviewed, current-persona jargon as discovery clues, never origin evidence."""
        import time

        empty = {"schema_version": 1, "scope": scope, "terms": []}
        if self.closed or not self.ready or not self.engine or not self.engine.allowed(scope):
            return empty
        if not self.config.get("jargon_enabled", True) or ":GroupMessage:" not in scope:
            return empty
        persona = self.engine.current_personas.get(scope)
        if not persona or not isinstance(query, str) or len(query) > 16000:
            return empty
        terms = []
        now = time.time()
        for entry in self.engine._entries_for(scope, persona):
            if (
                entry["category"] != "jargon"
                or not entry["active"]
                or entry["expires"] <= now
            ):
                continue
            if query and entry["key"] not in query:
                continue
            terms.append({"name": entry["key"], "meaning": entry["value"], "revision": entry["revision"]})
            if len(terms) >= max(1, min(20, int(limit))):
                break
        return {"schema_version": 1, "scope": scope, "terms": terms, "evidence": False}

    def get_learning_context(self, **scope):
        return self.engine.context(**scope) if self.ready and self.engine and not self.closed else {}

    def get_social_ownership(self, scope, participants):
        return (
            self.engine.social.ownership(scope, participants)
            if self.engine and self.engine.social and not self.closed
            else {}
        )

    def get_affection_for_decision(self, *, scope, persona, member, **kwargs):
        if self.closed or not self.engine or not self.engine.social or not self.engine.allowed(scope):
            return None
        entry = self.engine.social.current_affection(scope, persona, member)
        if not entry:
            return None
        return {
            "source": "dynamics_learning",
            "group_id": scope.rsplit(":", 1)[-1],
            "scope": scope,
            "persona": persona,
            "user_id": member,
            "affection_level": entry["value"]["affection_level"],
            "max_affection": None,
            "reference_affection": 100,
            "revision": entry["revision"],
        }

    async def get_social_context(
        self, scope, participants, purpose="reply", card_version="", card_context=None, **kwargs
    ):
        if isinstance(scope, dict) and is_series_group_excluded(
            self.context, umo=str(scope.get("umo", "") or "")
        ):
            return {"schema_version": 1, "available": False, "items": []}
        if self.config.get("replace_selflearning", False):
            if (
                not self.ready
                or not replacement_allowed(self)
                or not isinstance(scope, dict)
                or not isinstance(participants, list)
            ):
                return {"schema_version": 1, "available": False, "items": []}
            if ":GroupMessage:" in str(scope.get("umo", "")):
                available = current_persona(self.engine, scope) is not None
                items = group_projection(self.engine, scope, participants)
            else:
                items = []
                for member in participants[:1]:
                    items.extend(private_projection(self.engine, scope, str(member)))
                available = bool(items)
            return {
                "schema_version": 1,
                "available": available,
                "items": items,
                "purpose": purpose,
                "card_version": card_version,
                "diagnostics": {"relationship_restore_status": "complete" if available else "pending"},
            }
        owner = self.get_social_ownership(scope, participants)
        if not self.engine or not owner:
            return {"schema_version": 1, "available": False, "items": []}
        key = character_key(scope)
        personas = [p for (s, p), v in self.engine.social.bindings.items() if s == scope.get("umo") and v == key]
        items = []
        for persona in personas[:1]:
            for member in participants[:8]:
                content, versions = self.engine.social.context(scope["umo"], persona, str(member))
                if content:
                    items.append(
                        {
                            "text": PERSONALITY_POLICY + "\n" + content,
                            "subject_id": str(member),
                            "scope": dict(scope),
                            "source_id": "social:" + persona + ":" + str(member),
                            "fact_kind": "subjective",
                            "versions": versions,
                        }
                    )
        return {
            "schema_version": 1,
            "available": bool(personas),
            "items": items,
            "purpose": purpose,
            "card_version": card_version,
        }

    async def get_social_snapshot(self, scope, participants, purpose="reply", card_version=""):
        return await self.get_social_context(scope, participants, purpose, card_version)

    async def _character_for_event(self, event):
        world = role_provider(self.context, "character", "astrbot_plugin_worldbook")
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        if world is None or ":" not in umo:
            return None
        scope = {"platform": umo.rsplit(":", 2)[0], "umo": umo, "persona_id": "", "instance_id": ""}
        card = await asyncio.wait_for(world.get_character_context(scope, event), 2)
        resolved = card.get("scope", {}) if isinstance(card, dict) else {}
        if (
            not isinstance(card, dict)
            or not card.get("available")
            or card.get("disabled")
            or resolved.get("umo") != umo
            or resolved.get("platform") != scope["platform"]
        ):
            return None
        return resolved

    @filter.on_llm_request(priority=2)
    async def inject_social_context(self, event: AstrMessageEvent, req=None):
        if req is None or not self.ready or not replacement_allowed(self):
            return
        if is_series_group_excluded(
            self.context, umo=str(getattr(event, "unified_msg_origin", "") or "")
        ):
            return
        owned = getattr(event, "_chat_dynamics_owned_request", False) is True
        if owned and ":GroupMessage:" in str(getattr(event, "unified_msg_origin", "")):
            return  # Chat Dynamics installs the filtered structured projection.
        extra = getattr(event, "get_extra", None)
        if callable(extra):
            if (
                extra("_narrative_internal_v1", False)
                or extra("_narrative_managed_v1", False)
                or extra("_dynamics_learning_injected_v1", False)
            ):
                return
            blocking = extra("_autonomous_blocking_context_v1", None)
            if isinstance(blocking, dict) and blocking.get("projection_restricted") is True:
                return
        try:
            scope = await self._character_for_event(event)
            if not scope:
                return
            if owned:
                event._dynamics_learning_private_scope_v1 = {"owner": self, "event": event, "scope": dict(scope)}
                return
            member = str(event.get_sender_id())
            if ":GroupMessage:" in scope["umo"]:
                persona = current_persona(self.engine, scope)
                if not persona:
                    return
                content = self.engine.context(
                    scope=scope["umo"],
                    persona=persona,
                    member=member,
                    baseline="",
                    host_version="",
                    query=str(event.get_message_str() or ""),
                )["context"]
            else:
                content = "\n".join(row["text"] for row in private_projection(self.engine, scope, member))
            if content and isinstance(getattr(req, "extra_user_content_parts", None), list):
                from astrbot.core.agent.message import TextPart

                part = TextPart(text=content)
                if hasattr(part, "mark_as_temp"):
                    part = part.mark_as_temp() or part
                req.extra_user_content_parts.append(part)
                setter = getattr(event, "set_extra", None)
                if callable(setter):
                    setter("_dynamics_learning_injected_v1", True)
        except Exception as exc:
            self.projection_error = type(exc).__name__
            return

    @filter.llm_tool(name="query_affection")
    async def query_affection(
        self, event: AstrMessageEvent, mode: str = "user", user_id: str = "", group_id: str = "", limit: int = 10
    ) -> str:
        """读取当前群的角色好感度，私聊只读取明确授权的群来源，不合并分数。

        Args:
            mode(str): user 查询本人；ranking 当前群排名，仅管理员。
            user_id(str): 留空查本人；指定其他成员仅管理员。
            group_id(str): 留空使用当前范围；管理员私聊可指定授权来源群。
            limit(int): 排名数量，1至20。
        """
        if not self.ready or not replacement_allowed(self):
            return encode({"available": False})
        sender = str(event.get_sender_id())
        member = str(user_id or sender)
        check = getattr(event, "is_admin", None)
        admin = bool(check()) if callable(check) else False
        if (member != sender or mode == "ranking") and not admin:
            return encode({"available": False, "reason": "查询其他成员或排名需要管理员权限"})
        scope = await self._character_for_event(event)
        if not scope:
            return encode({"available": False, "reason": "当前角色未就绪"})
        if ":GroupMessage:" not in scope["umo"]:
            rows = private_projection(self.engine, scope, member)
            if group_id:
                if not admin:
                    return encode({"available": False, "reason": "指定群查询需要管理员权限"})
                rows = [r for r in rows if r["source_umo"].rsplit(":", 1)[-1] == group_id]
            if mode != "user":
                return encode({"available": False, "reason": "私聊不提供群成员排名"})
            return encode({"available": bool(rows), "sources": rows})
        if group_id and group_id != scope["umo"].rsplit(":", 1)[-1]:
            return encode({"available": False, "reason": "不能跨群查询"})
        persona = current_persona(self.engine, scope)
        if not persona:
            return encode({"available": False})
        if mode == "user":
            return encode(self.engine.social.query(scope["umo"], member, "affection"))
        if mode != "ranking":
            return encode({"available": False, "reason": "未知查询模式"})
        entries = [
            e
            for e in self.engine.cache
            if (e["scope"], e["persona"], e["category"]) == (scope["umo"], persona, "affection") and e["active"]
        ]
        entries.sort(key=lambda e: e["value"]["affection_level"], reverse=True)
        return encode(
            {
                "available": True,
                "scope": scope,
                "ranking": [
                    {"member": e["member"], "affection": e["value"]["affection_level"]}
                    for e in entries[: max(1, min(20, int(limit)))]
                ],
            }
        )

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=201)
    async def observe_source_recall(self, event: AstrMessageEvent):
        if self.closed or self.engine is None:
            return
        source = recall_source(event)
        if source is None:
            return
        scope, member, message_id = source
        async with self.engine.lock:
            authorized = self.engine.allowed(scope)
            changed = await self.engine.review_write(
                lambda: self.engine.store.source_withdrawals.record(scope, member, message_id, authorized=authorized)
            )
            if changed:
                self.engine.automatic_reviews.notify_context_change()

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=200)
    async def observe_group_message(self, event: AstrMessageEvent):
        if self.closed or not self.engine or not self.engine.social:
            return
        if is_series_group_excluded(
            self.context, umo=str(getattr(event, "unified_msg_origin", "") or "")
        ):
            return
        try:
            self.engine.capture_message(native_message(event))
            observed = observed_event(
                event,
                reply_weight=self.engine.social.settings["reply_weight"],
                at_weight=self.engine.social.settings["at_weight"],
            )
            if observed:
                self.engine.social.observe(observed)
        except Exception as exc:
            self.engine.social.last_error = type(exc).__name__

    @filter.llm_tool(name="query_group_social")
    async def query_group_social(
        self, event: AstrMessageEvent, kind: str = "overview", member_id: str = "", term: str = ""
    ) -> str:
        """查询当前群的群友习惯与喜恶、互动分布、角色好感度或已确认黑话。

        Args:
            kind(str): profile 为群友表达方式、打字习惯与喜恶，overview 为群聊分布，affection 为角色对成员的好感，relations 为成员往来，jargon 为具体黑话。
            member_id(str): 当前群成员的账号，留空使用当前发言者。不能查询其他群。
            term(str): kind 为 jargon 时填写需要解释的完整词语。
        """
        if self.closed or not self.ready or not self.engine or not self.engine.social:
            return encode({"available": False, "reason": "群间学习尚未就绪"})
        scope = str(getattr(event, "unified_msg_origin", "") or "")
        member = str(member_id or event.get_sender_id())
        if kind == "profile":
            if ":GroupMessage:" not in scope or len(member) > 128:
                return encode({"available": False, "reason": "仅支持当前群成员"})
            return encode(self.engine.member_profile(scope, member))
        return encode(self.engine.social.query(scope, member, kind, str(term or "").strip()))

    async def _model(self, role, instruction, data, *, review_dispatch=None):
        provider_id = self._effective_provider(role)
        if not provider_id:
            raise ValueError("请先配置" + ("分析模型" if role == "analysis_provider" else "审查模型"))
        provider = self.context.get_provider_by_id(provider_id)
        # Explicit ordinary-chat model binding; no decision-provider or silent fallback.
        if (
            provider is None
            or not callable(getattr(provider, "text_chat", None))
            or getattr(provider, "is_native_decision_provider", False)
        ):
            raise ValueError("所选模型不可用或不是普通聊天模型")
        timeout = max(5, min(180, int(self.config.get("model_timeout", 60))))
        payload = encode(data)
        if len(payload) > MODEL_INPUT_LIMIT:
            raise ValueError("证据超出单次模型输入预算，请缩小提案引用范围")
        model = provider_model(provider)
        if review_dispatch and (
            review_dispatch.provider_id != provider_id
            or review_dispatch.provider_instance is not provider
            or review_dispatch.model != model
            or not self.engine.allowed(review_dispatch.scope)
        ):
            raise ReviewUnavailable("模型绑定或会话权限已改变")
        attributed_call = getattr(provider, "_series_attributed_call_v1", None)
        source = data.get("proposal") or next(iter(data.get("events", [])), {}) or data
        scope = review_dispatch.scope if review_dispatch else source.get("scope", "")
        operation_id = review_dispatch.operation_id if review_dispatch else ""
        attribution = (
            attributed_call(
                f"dynamics_learning.{role.removesuffix('_provider')}", scope=scope, operation_id=operation_id
            )
            if callable(attributed_call)
            else nullcontext()
        )
        call_kwargs = dict(_text_chat_extra_kwargs(provider, self.config))
        if _accepts_kwarg(provider.text_chat, "contexts"):
            call_kwargs["contexts"] = _cache_anchor_contexts(instruction)
        try:
            async with asyncio.timeout(timeout):
                with attribution:
                    if review_dispatch:
                        self.engine.store.automatic_reviews.record_dispatch(review_dispatch)
                    response = await series_llm_call(
                        self.context,
                        provider_id,
                        PLUGIN,
                        role.removesuffix("_provider"),
                        lambda: provider.text_chat(prompt=payload, system_prompt=instruction, **call_kwargs),
                    )
        except Exception as exc:
            if is_provider_budget_busy_error(exc):
                raise ReviewUnavailable(
                    f"共享模型并发池繁忙 ({getattr(exc, 'reason', 'busy')})"
                ) from exc
            raise
        if (
            self.closed
            or self.context.get_provider_by_id(provider_id) is not provider
            or self._effective_provider(role) != provider_id
            or provider_model(provider) != model
        ):
            raise ReviewUnavailable("模型绑定已改变，本次结果失效")
        content = getattr(response, "completion_text", None)
        try:
            return decode_model_object(content)
        except ValueError as exc:
            raise MalformedModelResponse(str(exc)) from exc

    async def _authorized(self):
        from astrbot.dashboard.api.auth import require_dashboard_user

        raw = getattr(request, "_request", None)
        if raw is None:
            raise PermissionError("需要 Dashboard 管理员登录")
        username = await require_dashboard_user(raw)
        expected = self.context.get_config().get("dashboard", {}).get("username")
        if not username or (expected and username != expected):
            raise PermissionError("需要 Dashboard 管理员登录")
        return "human:" + str(username)

    async def _body(self, allowed):
        raw = await request._request.body()
        if len(raw) > 16_384:
            raise ValueError("body too large")
        body = json.loads(raw)
        if not isinstance(body, dict) or set(body) - set(allowed):
            raise ValueError("invalid request fields")
        return body

    async def _serve(self, operation, fields=()):
        try:
            actor = await self._authorized()
            if self.closed or self.engine is None:
                raise ValueError("插件未就绪")
            body = await self._body(fields) if fields else {}
            data = await operation(body, actor)
            return json_response({"status": "ok", "data": data})
        except PermissionError:
            return json_response({"status": "error", "message": "需要 Dashboard 管理员登录"}, status_code=403)
        except (ValueError, KeyError) as exc:
            return json_response({"status": "error", "message": str(exc)[:300]}, status_code=400)
        except Exception as exc:
            logger.warning("[DynamicsLearning] API failed type=%s", type(exc).__name__)
            return json_response({"status": "error", "message": "请求失败：" + type(exc).__name__}, status_code=400)

    def _enqueue_job(self, key, call):
        if key in self.jobs or len(self.jobs) >= 2:
            raise ValueError("已有任务运行，请等待完成")
        self.jobs.add(key)

        async def run():
            try:
                await call()
                self.engine.last_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.engine.last_error = str(exc)[:200] if isinstance(exc, ValueError) else type(exc).__name__
            finally:
                self.jobs.discard(key)

        task = asyncio.create_task(run())
        self.tasks.append(task)
        self.tasks[:] = [t for t in self.tasks if not t.done()]
        return {"queued": True}

    async def overview(self):
        async def operation(body, actor):
            return {
                **await self.engine.overview(),
                "jobs": list(self.jobs),
                "social_projection_error": self.projection_error,
                "social_queue": self.engine.social.queue.qsize(),
                "social_dropped": self.engine.social.dropped,
                "social_last_error": self.engine.social.last_error,
                "provider_readiness": self._provider_status(),
                "legacy_kv_v1": self.legacy_kv_v1,
            }

        return await self._serve(operation)

    async def social(self):
        async def operation(body, actor):
            return await self.engine.social.overview(
                text(body.get("scope"), "scope", 512),
                body.get("limit", 40),
                text(body.get("query", ""), "query", 128, empty=True),
            )

        return await self._serve(operation, ("scope", "limit", "query"))

    async def members(self):
        async def operation(body, actor):
            return self.engine.social.members(
                text(body.get("scope"), "scope", 512),
                text(body.get("query", ""), "query", 128, empty=True),
                body.get("page", 1),
            )

        return await self._serve(operation, ("scope", "query", "page"))

    async def member_detail(self):
        async def operation(body, actor):
            return self.engine.social.member_detail(
                text(body.get("scope"), "scope", 512), text(body.get("member"), "member", 256)
            )

        return await self._serve(operation, ("scope", "member"))

    async def entries(self):
        async def operation(body, actor):
            scope = text(body.get("scope", ""), "scope", 512, empty=True)
            scopes = tuple(self._effective_sessions())
            if scope and scope not in scopes:
                raise ValueError("scope not enabled")
            return await asyncio.to_thread(
                self.engine.store.console.entries,
                scope,
                text(body.get("category", ""), "category", 80, empty=True),
                text(body.get("state", "active"), "state", 20),
                text(body.get("query", ""), "query", 128, empty=True),
                body.get("page", 1),
                affection_view=body.get("affection_view"),
                current_personas=dict(self.engine.current_personas),
                allowed_scopes=scopes,
            )

        return await self._serve(operation, ("scope", "category", "state", "query", "page", "affection_view"))

    async def proposal_detail(self):
        async def operation(body, actor):
            p = await asyncio.to_thread(self.engine.store.get, text(body.get("id"), "id", 128))
            if p["scope"] not in self._effective_sessions():
                raise ValueError("scope not enabled")
            if p["state"] in {"applied", "rolled_back"}:
                p["observations"] = await asyncio.to_thread(self.engine.store.observations, p)
            return p

        return await self._serve(operation, ("id",))

    async def usage(self):
        async def operation(body, actor):
            scope = text(body.get("scope"), "scope", 512)
            if scope not in self._effective_sessions():
                raise ValueError("scope not enabled")
            return await asyncio.to_thread(self.engine.store.console.usage, scope, body.get("page", 1))

        return await self._serve(operation, ("scope", "page"))

    async def profile(self):
        async def operation(body, actor):
            return await self.engine.profile_preview(
                text(body.get("scope"), "scope", 512), text(body.get("member"), "member", 128)
            )

        return await self._serve(operation, ("scope", "member"))

    async def history(self):
        async def operation(body, actor):
            scope = text(body.get("scope"), "scope", 512)
            if scope not in self._effective_sessions():
                raise ValueError("scope not enabled")
            return await asyncio.to_thread(
                self.engine.store.native.history,
                scope,
                text(body.get("category", ""), "category", 80, empty=True),
                body.get("page", 1),
            )

        return await self._serve(operation, ("scope", "category", "page"))

    async def review_group(self):
        async def operation(body, actor):
            scope = text(body.get("scope"), "scope", 512)
            if scope not in self._effective_sessions():
                raise ValueError("scope not enabled")
            result = await asyncio.to_thread(
                self.engine.store.review_group,
                scope,
                text(body.get("persona"), "persona", 512),
                text(body.get("member", ""), "member", 256, empty=True),
                body.get("page", 1),
            )
            result["proposals"] = [self.engine.store.console.summary(p) for p in result["proposals"]]
            return result

        return await self._serve(operation, ("scope", "persona", "member", "page"))

    async def review_queue(self):
        async def operation(body, actor):
            return await self.engine.review_queue(
                scope=text(body.get("scope", ""), "scope", 512, empty=True),
                query=text(body.get("query", ""), "query", 160, empty=True),
                category=text(body.get("category", ""), "category", 80, empty=True),
                state=text(body.get("state", ""), "state", 32, empty=True),
                reason=text(body.get("reason", ""), "reason", 40, empty=True),
                view=text(body.get("view", "ready"), "view", 16),
                page=body.get("page", 1),
            )

        return await self._serve(operation, ("scope", "query", "category", "state", "view", "reason", "page"))

    async def governance_action(self):
        from .core.workbench import govern

        async def operation(body, actor):
            scope = text(body.get("scope"), "scope", 512)
            action = text(body.get("action"), "action", 16)
            async with self.engine.lock:
                rows = await self.engine.batch_snapshot(scope, body.get("items"), actor)
                result = await self.engine.review_write(lambda: govern(self.engine, scope, rows, action, actor))
            self.engine.automatic_reviews.notify_context_change()
            return result

        return await self._serve(operation, ("scope", "action", "items"))

    async def library(self):
        from .core.workbench import library

        async def operation(body, actor):
            return await asyncio.to_thread(
                library,
                self.engine.store,
                self._effective_sessions(),
                text(body.get("scope", ""), "scope", 512, empty=True),
                text(body.get("state", ""), "state", 32, empty=True),
                text(body.get("category", ""), "category", 80, empty=True),
                text(body.get("query", ""), "query", 160, empty=True),
                body.get("page", 1),
            )

        return await self._serve(operation, ("scope", "state", "category", "query", "page"))

    async def material_detail(self):
        from .core.workbench import detail

        async def operation(body, actor):
            return await asyncio.to_thread(
                detail,
                self.engine.store,
                self._effective_sessions(),
                text(body.get("proposal_id", ""), "proposal_id", 128, empty=True),
                text(body.get("entry_id", ""), "entry_id", 256, empty=True),
            )

        return await self._serve(operation, ("proposal_id", "entry_id"))

    async def reply_preview(self):
        import inspect

        from .core.native import instances

        async def operation(body, actor):
            scope = text(body.get("scope"), "scope", 512)
            member = text(body.get("member"), "member", 128)
            query = text(body.get("query", ""), "query", 2000, empty=True)
            if not self.engine.allowed(scope) or ":GroupMessage:" not in scope:
                raise ValueError("请选择已启用群聊")
            persona = self.engine.current_personas.get(scope)
            if not persona:
                return {
                    "available": False,
                    "scope": scope,
                    "member": member,
                    "persona": "",
                    "items": [],
                    "offered": [],
                    "candidates": [],
                    "reason": "尚未取得该群当前角色快照，请等待真实消息",
                    "note": "历史成员名单不等于当前角色身份已确认。",
                    "model_called": False,
                    "message_sent": False,
                }
            hosts = [p for _, p in instances(self.context) if callable(getattr(p, "preview_learning_material", None))]
            if len(hosts) != 1:
                raise ValueError("Chat Dynamics 预览接口未就绪")
            result = hosts[0].preview_learning_material(scope, persona, member, query)
            return await asyncio.wait_for(result, 12) if inspect.isawaitable(result) else result

        return await self._serve(operation, ("scope", "member", "query"))

    async def group_batch_action(self):
        async def operation(body, actor):
            return await self.engine.group_batch_action(text(body.get("scope"), "scope", 512), body.get("items"), actor)

        return await self._serve(operation, ("scope", "items"))

    async def group_batch_reject(self):
        async def operation(body, actor):
            return await self.engine.group_batch_reject(
                text(body.get("scope"), "scope", 512),
                body.get("items"),
                actor,
                text(body.get("reason", ""), "reason", 1000, empty=True),
            )

        return await self._serve(operation, ("scope", "items", "reason"))

    async def group_batch_review(self):
        async def operation(body, actor):
            scope = text(body.get("scope"), "scope", 512)
            key = "batch-review:" + scope
            if key in self.jobs or len(self.jobs) >= 2:
                raise ValueError("已有任务运行，请等待完成")
            if not self._effective_provider("review_provider"):
                raise ValueError("请先配置审查模型")
            job = await self.engine.review_batches.prepare(scope, body.get("items"), actor)
            try:
                self._enqueue_job(key, lambda: self.engine.review_batches.run(job))
            except Exception:
                self.engine.review_batches.interrupt(job)
                raise
            return {"queued": True, "id": job["id"], "count": len(job["items"])}

        return await self._serve(operation, ("scope", "items"))

    async def review_batch_jobs(self):
        async def operation(body, actor):
            return {
                "jobs": await asyncio.to_thread(
                    self.engine.review_batches.recent, text(body.get("scope", ""), "scope", 512, empty=True)
                )
            }

        return await self._serve(operation, ("scope",))

    async def review_batch_control(self):
        async def operation(body, actor):
            identity = text(body.get("id"), "id", 128)
            action = text(body.get("action"), "action", 16)
            manager = self.engine.review_batches
            job = manager.get(identity)
            if not job:
                raise ValueError("任务不存在")
            if action in {"resume", "retry"}:
                if "batch-review:" + job["scope"] in self.jobs or len(self.jobs) >= 2:
                    raise ValueError("当前任务仍在结束或已有任务运行，请稍后重试")
            result = await manager.control(identity, action, actor, require_provider=True)
            if action in {"resume", "retry"}:
                try:
                    self._enqueue_job("batch-review:" + result["scope"], lambda: manager.run(result))
                except Exception:
                    manager.interrupt(result)
                    raise
            return {
                "id": result["id"],
                "status": result["status"],
                "count": len(result["items"]),
                "retry_skipped": len(result.get("retry_skipped", [])),
            }

        return await self._serve(operation, ("id", "action"))

    async def batch_action(self):
        async def operation(body, actor):
            return await self.engine.batch_action(
                text(body.get("scope"), "scope", 512),
                text(body.get("persona"), "persona", 512),
                text(body.get("member", ""), "member", 256, empty=True),
                body.get("items"),
                actor,
            )

        return await self._serve(operation, ("scope", "persona", "member", "items"))

    async def jargon_propose(self):
        async def operation(body, actor):
            p = await self.engine.correct_jargon(
                text(body.get("id"), "id", 128),
                body.get("revision"),
                text(body.get("value"), "meaning", 1000),
                text(body.get("reason"), "reason", 1000),
                actor,
            )
            queued = False
            if p["mode"] != "human" and p["state"] == "pending":
                try:
                    self._enqueue_job("review:" + p["id"], lambda: self.engine.review(p["id"]))
                    queued = True
                except ValueError:
                    queued = False
            return {"id": p["id"], "state": p["state"], "queued": queued}

        return await self._serve(operation, ("id", "revision", "value", "reason"))

    async def analyze(self):
        async def operation(body, actor):
            scope, persona = text(body.get("scope"), "scope", 512), text(body.get("persona"), "persona", 512)
            return self._enqueue_job("analyze:" + scope + ":" + persona, lambda: self.engine.analyze(scope, persona))

        return await self._serve(operation, ("scope", "persona"))

    async def review(self):
        async def operation(body, actor):
            pid = text(body.get("id"), "id", 64)
            return self._enqueue_job("review:" + pid, lambda: self.engine.review(pid))

        return await self._serve(operation, ("id",))

    async def action(self):
        async def operation(body, actor):
            return await self.engine.action(
                text(body.get("id"), "id", 64),
                body.get("action"),
                actor,
                body.get("updated"),
                text(body.get("reason", ""), "reason", 1000, empty=True),
            )

        return await self._serve(operation, ("id", "action", "updated", "reason"))

    async def amend(self):
        async def operation(body, actor):
            return await self.engine.amend(
                text(body.get("id"), "id", 64),
                body.get("value"),
                text(body.get("reason"), "reason", 1500),
                actor,
                expected_updated=body.get("updated"),
            )

        return await self._serve(operation, ("id", "value", "reason", "updated"))

    async def purge(self):
        async def operation(body, actor):
            scope = text(body.get("scope"), "scope", 512)
            if body.get("confirm_scope") != scope:
                raise ValueError("请输入完整会话标识确认删除")
            await self.engine.purge(scope)
            return {"deleted": True}

        return await self._serve(operation, ("scope", "confirm_scope"))

    async def terminate(self):
        panel = getattr(self, "_series_panels", None)
        if panel:
            panel.close()
        self.ready = False
        self.closed = True
        if self.engine:
            self.engine.closed = True
            self.engine.generation += 1
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.engine and self.engine.social:
            await self.engine.social.drain()
        # Only remove this instance's routes; never unregister a replacement instance.
        registered = getattr(self.context, "registered_web_apis", None)
        if isinstance(registered, list):
            registered[:] = [r for r in registered if getattr(r[1], "__self__", None) is not self]
        if self.engine:
            async with self.engine.lock:
                await asyncio.to_thread(self.engine.store.close)
