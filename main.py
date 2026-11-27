# ruff: noqa: E402 -- owned module refresh must precede imports on hot reload.
from __future__ import annotations

# Refresh owned modules before importing domain code on a scoped hot reload.
from .reload_bootstrap import ensure_current

ensure_current(
    __package__,
    __name__,
    __file__,
    version="2.0.0",
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
    jargon_matches,
    text,
)
from . import series_panels
from .core.engine import Engine
from .core.model_projection import encode_prompt_payload
from .core.native import (
    affection_query_allowed,
    current_persona,
    group_projection,
    private_projection,
    replacement_allowed,
    role_provider,
)
from .core.profiles import native_message
from .core.review_batches import ReviewDeferred, ReviewUnavailable
from .core.review_evidence import provider_model
from .core.series_bridge import (
    is_member_privacy_muted,
    normalize_str_list,
    read_extended_shared_config,
    resolve_shared_persona_id,
)
from .core.social import Social, character_key, observed_event
from .core.source_withdrawals import recall_source
from .core.store import Store
from .series_panels import (
    find_provider_silently,
    inherit_umos,
    is_provider_budget_busy_error,
    is_series_group_excluded,
    require_series_admin,
    resolve_series_timezone,
    series_llm_call,
)

PLUGIN = "astrbot_plugin_dynamics_learning"
DEFAULT_MODEL_REASONING_EFFORT = "low"
SINGLE_REVIEW_MAX_TOKENS = 768
MIN_SAFE_SINGLE_REVIEW_MAX_TOKENS = 512
BATCH_OR_ANALYZE_MAX_TOKENS = 1536
MIN_SAFE_MODEL_MAX_TOKENS = 1024
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


def _text_chat_extra_kwargs(provider, config, *, role: str = "", is_single_review: bool = False):
    """Build optional max_tokens / reasoning_effort kwargs accepted by provider.text_chat.

    Defaults reasoning_effort to 'low' (when unset in config) and caps output tokens
    per task class (768 for single-item review, 1536 for batch review / analyze) to
    prevent runaway reasoning outputs on structured JSON classification tasks.
    """
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
            floor = MIN_SAFE_SINGLE_REVIEW_MAX_TOKENS if is_single_review else MIN_SAFE_MODEL_MAX_TOKENS
            candidates["max_tokens"] = max(floor, parsed_max)
        elif parsed_max == 0:
            candidates["max_tokens"] = SINGLE_REVIEW_MAX_TOKENS if is_single_review else BATCH_OR_ANALYZE_MAX_TOKENS
    if "model_reasoning_effort" in config:
        raw_val = config.get("model_reasoning_effort")
        raw_effort = (DEFAULT_MODEL_REASONING_EFFORT if raw_val is None else str(raw_val)).strip().lower()
    else:
        raw_effort = DEFAULT_MODEL_REASONING_EFFORT
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
    out = {}
    for key, val in candidates.items():
        if has_varkw or (key in params and params[key].kind in allowed_kinds):
            out[key] = val
    if (
        "reasoning_effort" in candidates
        and "reasoning_effort" not in out
        and "extra_body" in params
        and params["extra_body"].kind in allowed_kinds
    ):
        out["extra_body"] = {"reasoning_effort": candidates["reasoning_effort"]}
    return out


def _command_decorator(*args, **kwargs):
    cmd = getattr(filter, "command", None)
    return cmd(*args, **kwargs) if callable(cmd) else (lambda fn: fn)


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
        self._scope_config_version = 0

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
        explicit = normalize_str_list(self.config.get("sessions"))
        excluded = set(normalize_str_list(self.config.get("excluded_sessions")))
        if explicit:
            return [s for s in explicit if s not in excluded]
        inherited = inherit_umos([], self.context) if self.context is not None else []
        return [s for s in inherited if s not in excluded]

    def _sessions_source(self) -> str:
        fn = getattr(self.engine, "sessions_source", None)
        if callable(fn):
            return fn()
        if normalize_str_list(self.config.get("sessions")):
            return "explicit"
        if self.context is not None and inherit_umos([], self.context):
            return "inherited"
        return "empty"

    def _save_config_safely(self) -> None:
        saver = getattr(self.config, "save_config", None)
        if callable(saver):
            try:
                saver()
            except Exception as exc:
                logger.warning("[DynamicsLearning] Failed to save config: %s", type(exc).__name__)

    def _current_scope_version(self, scope: str) -> str:
        proj_gen = int(self.engine.scope_projection_generation(scope)) if (self.engine and scope) else 0
        return f"{int(getattr(self, '_scope_config_version', 0))}:{proj_gen}"

    def _is_privacy_muted(self, scope: str, member_id: str) -> bool:
        if self.engine is not None and hasattr(self.engine, "is_privacy_muted"):
            return bool(self.engine.is_privacy_muted(scope, member_id))
        return is_member_privacy_muted(self.context, scope, member_id, "learning")

    def _effective_provider_with_source(self, role: str) -> tuple[str, str]:
        if self.engine is not None and hasattr(self.engine, "provider_with_source"):
            pid, source = self.engine.provider_with_source(role)
            return str(pid or "").strip(), str(source or "none")
        explicit = str(self.config.get(role, "") or "").strip()
        if explicit:
            return explicit, "explicit"
        if self.context is None:
            return "", "none"
        shared = read_extended_shared_config(self.context)
        if role == "analysis_provider":
            light = str(shared.get("light_analysis_provider") or "").strip()
            if light and not Engine._is_native_decision_provider(self.context, light):
                return light, "shared_light"
            reply_p = str(shared.get("reply_provider") or "").strip()
            if reply_p:
                return reply_p, "shared_reply"
            host_p = Engine._host_default_provider_id(self.context)
            return (host_p, "host_default") if host_p else ("", "none")
        if role == "review_provider":
            light = str(shared.get("light_analysis_provider") or "").strip()
            if light and not Engine._is_native_decision_provider(self.context, light):
                return light, "shared_light"
            candidate = str(shared.get("decision_provider") or "").strip()
            if candidate and not Engine._is_native_decision_provider(self.context, candidate):
                return candidate, "shared_decision"
            reply_p = str(shared.get("reply_provider") or "").strip()
            if reply_p:
                return reply_p, "shared_reply"
            host_p = Engine._host_default_provider_id(self.context)
            return (host_p, "host_default") if host_p else ("", "none")
        return "", "none"

    def _effective_provider(self, role: str) -> str:
        pid, _source = self._effective_provider_with_source(role)
        return pid

    def _provider_status(self):
        status = {}
        for role in ("analysis_provider", "review_provider"):
            pid, source = self._effective_provider_with_source(role)
            if not pid:
                status[role] = {"configured": False, "ready": False, "provider_id": "", "source": source}
                continue
            provider = find_provider_silently(self.context, pid)
            ready = bool(
                provider is not None
                and callable(getattr(provider, "text_chat", None))
                and not getattr(provider, "is_native_decision_provider", False)
            )
            status[role] = {"configured": True, "ready": ready, "provider_id": pid, "source": source}
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
            ("archive_stale", self.archive_stale, ["POST"]),
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
                "ready_count": 0,
                "stale_count": 0,
                "waiting_count": 0,
                "pending_total": 0,
            }
        try:
            counts = self.engine.store.proposal_counts(readonly=True)
            q = self.engine.queue_counts(readonly=True)
            ready_count = int(q.get("ready_count", 0))
            stale_count = int(q.get("stale_count", 0))
            waiting_count = int(q.get("waiting_count", 0)) + int(q.get("held_count", 0))
            pending_total = int(q.get("pending_total", 0))
            applied = int(counts.get("applied", 0))
            master_enabled = bool(self.config.get("enabled", True))
            job_note = f"处理中 {len(self.jobs)} 项" if self.jobs else "队列空闲"
            headline_parts = [f"可审 {ready_count} 项"]
            if waiting_count > 0:
                headline_parts.append(f"待初审 {waiting_count} 项")
            if stale_count > 0:
                headline_parts.append(f"失效 {stale_count} 项")
            headline_parts.append(f"已应用 {applied} 条")
            if not master_enabled:
                if ready_count > 0:
                    label = (
                        f"总开关已关闭 · {ready_count} 可审 · {stale_count} 失效"
                        if stale_count > 0
                        else f"总开关已关闭 · {ready_count} 项可审"
                    )
                elif stale_count > 0:
                    label = f"总开关已关闭 · {stale_count} 失效"
                else:
                    label = "总开关已关闭"
                headline = "学习总开关已关闭 · " + " · ".join(headline_parts)
                return {
                    "badge": {"count": ready_count, "tone": "muted", "label": label},
                    "pulse": {"headline": headline, "updated_at": updated_at},
                    "ready_count": ready_count,
                    "stale_count": stale_count,
                    "waiting_count": waiting_count,
                    "pending_total": pending_total,
                }
            if ready_count > 0:
                tone = "warm"
                label = (
                    f"{ready_count} 项待审 · {stale_count} 失效"
                    if stale_count > 0
                    else f"{ready_count} 项待审"
                )
            elif waiting_count > 0:
                tone = "info"
                label = (
                    f"{waiting_count} 待初审 · {stale_count} 失效"
                    if stale_count > 0
                    else f"{waiting_count} 项待初审"
                )
            elif stale_count > 0:
                tone = "muted"
                label = f"{stale_count} 项已失效"
            else:
                tone = "accent"
                label = "队列空闲"
            headline_parts.append(job_note)
            headline = " · ".join(headline_parts)
            return {
                "badge": {"count": ready_count, "tone": tone, "label": label},
                "pulse": {"headline": headline, "updated_at": updated_at},
                "ready_count": ready_count,
                "stale_count": stale_count,
                "waiting_count": waiting_count,
                "pending_total": pending_total,
            }
        except Exception:
            return {
                "badge": {"count": 0, "tone": "muted", "label": "暂不可用"},
                "pulse": {"headline": "状态读取暂不可用", "updated_at": updated_at},
                "ready_count": 0,
                "stale_count": 0,
                "waiting_count": 0,
                "pending_total": 0,
            }

    def get_series_health(self) -> dict:
        """Synchronous, read-only health & provider binding diagnostics for series_panels.read_plugin_health."""
        providers: list[dict[str, str]] = []
        issues: list[str] = []
        for role, slot_name in (("analysis_provider", "analysis"), ("review_provider", "review")):
            pid = self._effective_provider(role)
            if not pid:
                providers.append(
                    {
                        "role": slot_name,
                        "provider_id": "",
                        "status": "unconfigured",
                        "detail": "未配置分析模型" if slot_name == "analysis" else "未配置审查模型",
                    }
                )
                issues.append(f"provider_unconfigured:{slot_name}")
                continue
            found = find_provider_silently(self.context, pid)
            if found is None:
                providers.append(
                    {
                        "role": slot_name,
                        "provider_id": pid,
                        "status": "missing",
                        "detail": f"Provider {pid} 未找到",
                    }
                )
                issues.append(f"provider_missing:{slot_name}:{pid}")
            elif getattr(found, "is_native_decision_provider", False) or not callable(
                getattr(found, "text_chat", None)
            ):
                providers.append(
                    {
                        "role": slot_name,
                        "provider_id": pid,
                        "status": "incompatible",
                        "detail": f"Provider {pid} 不是可用的普通聊天模型",
                    }
                )
                issues.append(f"provider_incompatible:{slot_name}:{pid}")
            else:
                providers.append(
                    {
                        "role": slot_name,
                        "provider_id": pid,
                        "status": "ok",
                        "detail": "ok",
                    }
                )
        ready = bool(self.ready and not self.closed and self.engine is not None)
        if not ready:
            issues.append("plugin_not_ready")
        last_error = str(getattr(self.engine, "last_error", "") or "") if self.engine is not None else ""
        if last_error:
            issues.append(f"last_error:{last_error[:120]}")
        sessions = self._effective_sessions()
        queue_size = 0
        dropped = 0
        ready_count = 0
        stale_count = 0
        if self.engine is not None:
            try:
                queue_size = int(self.engine.queue.qsize())
            except Exception:
                queue_size = 0
            dropped = int(getattr(self.engine, "dropped", 0) or 0)
            try:
                q_counts = self.engine.queue_counts(readonly=True)
                ready_count = int(q_counts.get("ready_count", 0))
                stale_count = int(q_counts.get("stale_count", 0))
            except Exception:
                pass
        return {
            "ready": ready,
            "degraded": bool(issues) or not ready,
            "providers": providers,
            "issues": issues,
            "metrics": {
                "enabled": bool(self.config.get("enabled", True)),
                "sessions": len(sessions),
                "queue": queue_size,
                "dropped": dropped,
                "active_jobs": len(self.jobs),
                "ready_count": ready_count,
                "stale_count": stale_count,
            },
        }

    def get_series_scope_status(self, umo: str) -> dict:
        """Read-only per-scope status for the cross-plugin group control matrix."""
        scope = str(umo or "").strip()[:512]
        master_enabled = bool(self.config.get("enabled", True))
        mode = self._sessions_source()
        if self.closed or not self.ready or self.engine is None:
            return {
                "umo": scope,
                "enabled": False,
                "writable": False,
                "mode": mode,
                "configured": False,
                "master_enabled": master_enabled,
                "supported": False,
                "persona_id": "",
                "activity_today": None,
                "summary": "群间学习尚未就绪",
                "detail": "群间学习尚未就绪",
                "version": self._current_scope_version(scope),
            }
        try:
            sessions = self._effective_sessions()
            configured = bool(scope and scope in sessions)
            excluded_by_core = bool(scope and is_series_group_excluded(self.context, umo=scope))
            active = bool(scope and not excluded_by_core and self.engine.allowed(scope))
            persona_id = str(self.engine.current_personas.get(scope, "") or "") or (
                resolve_shared_persona_id(self.context) if scope else ""
            )
            version = self._current_scope_version(scope)
            if not scope:
                summary = (
                    f"已配置 {len(sessions)} 个会话"
                    if master_enabled
                    else f"学习总开关已关闭（已配置 {len(sessions)} 个会话）"
                )
                return {
                    "umo": "",
                    "enabled": False,
                    "writable": True,
                    "mode": mode,
                    "configured": False,
                    "excluded_by_core": False,
                    "master_enabled": master_enabled,
                    "supported": True,
                    "persona_id": "",
                    "activity_today": 0,
                    "summary": summary,
                    "detail": summary,
                    "version": version,
                }
            now = time.time()
            tz = resolve_series_timezone(self.context)
            today_bounds = _date_window("today", tz=tz)
            day_start, day_end = today_bounds if today_bounds is not None else (0.0, now + 1.0)
            store = self.engine.store
            with store.readonly_connection() as db:
                pending_row = db.execute(
                    "SELECT count(*) FROM proposals WHERE scope=? AND state IN ('pending','reviewed')",
                    (scope,),
                ).fetchone()
                active_row = db.execute(
                    "SELECT count(*) FROM entries WHERE json_extract(body,'$.scope')=? "
                    "AND json_extract(body,'$.active')=1 "
                    "AND (json_extract(body,'$.category')='affection' OR json_extract(body,'$.expires')>?)",
                    (scope, now),
                ).fetchone()
                today_row = db.execute(
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
                "umo": scope,
                "enabled": active,
                "writable": True,
                "mode": mode,
                "configured": configured,
                "excluded_by_core": excluded_by_core,
                "master_enabled": master_enabled,
                "supported": True,
                "persona_id": persona_id,
                "activity_today": activity_today,
                "summary": summary,
                "detail": summary,
                "version": version,
            }
        except Exception:
            return {
                "umo": scope,
                "enabled": None,
                "writable": False,
                "mode": mode,
                "configured": False,
                "master_enabled": master_enabled,
                "supported": True,
                "persona_id": "",
                "activity_today": None,
                "summary": "状态读取失败",
                "detail": "状态读取失败",
                "version": self._current_scope_version(scope),
            }

    async def set_series_scope_enabled(
        self, umo: str, enabled: bool, expected_version: str | int | None = None
    ) -> dict:
        """Optimistically toggle whether a UMO is enabled in Dynamics Learning."""
        scope = str(umo or "").strip()[:512]
        if not scope:
            return {
                "ok": False,
                "umo": "",
                "enabled": False,
                "version": self._current_scope_version(""),
                "reason": "invalid_umo",
            }
        if self.closed or not self.ready or self.engine is None:
            return {
                "ok": False,
                "umo": scope,
                "enabled": False,
                "version": self._current_scope_version(scope),
                "reason": "plugin_not_ready",
            }
        target_enabled = bool(enabled)
        async with self.engine.lock:
            current_version = self._current_scope_version(scope)
            if expected_version is not None and str(expected_version).strip() != str(current_version):
                return {
                    "ok": False,
                    "conflict": True,
                    "umo": scope,
                    "enabled": bool(self.engine.allowed(scope)),
                    "version": str(current_version),
                    "reason": "version_conflict",
                    "summary": "状态已变更（版本冲突），请刷新后重试",
                }

            explicit = normalize_str_list(self.config.get("sessions"))
            excluded = normalize_str_list(self.config.get("excluded_sessions"))
            inherited = inherit_umos([], self.context) if self.context is not None else []

            if target_enabled:
                if scope in excluded:
                    excluded = [s for s in excluded if s != scope]
                if explicit:
                    if scope not in explicit:
                        explicit.append(scope)
                elif scope not in inherited:
                    merged = list(dict.fromkeys([*inherited, scope]))
                    explicit = [s for s in merged if s not in excluded]
                if not bool(self.config.get("enabled", True)):
                    self.config["enabled"] = True
            else:
                if explicit:
                    explicit = [s for s in explicit if s != scope]
                    if not explicit and scope in inherited and scope not in excluded:
                        excluded.append(scope)
                elif scope in inherited:
                    if scope not in excluded:
                        excluded.append(scope)

            self.config["sessions"] = explicit
            if excluded or "excluded_sessions" in self.config:
                self.config["excluded_sessions"] = excluded

            self._scope_config_version += 1
            self.engine.bump_projection_generation(scope)
            self._save_config_safely()
            series_panels.invalidate_presence_cache(PLUGIN, context=self.context)
            if target_enabled and self.engine.social is not None and ":GroupMessage:" in scope:
                self.engine.social.request_refresh(scope)

            active_now = bool(self.engine.allowed(scope))
            excluded_by_core = bool(is_series_group_excluded(self.context, umo=scope))
            new_version = self._current_scope_version(scope)
            reason = "excluded_by_core" if (target_enabled and excluded_by_core) else "ok"
            summary = "开启" if active_now else ("主控排除" if excluded_by_core else "已暂停")
            return {
                "ok": True,
                "umo": scope,
                "enabled": active_now,
                "version": str(new_version),
                "reason": reason,
                "summary": summary,
            }

    def activate_series_runtime(self) -> dict:
        """Uniform runtime activation hook for Series Setup Wizard step 4."""
        if self.closed or self.engine is None:
            return {
                "ok": False,
                "enabled": False,
                "ready": False,
                "reason": "plugin_not_ready",
            }
        changed = False
        if not bool(self.config.get("enabled", True)):
            self.config["enabled"] = True
            changed = True
        if changed:
            self._scope_config_version += 1
            self._save_config_safely()
            self.engine.bump_projection_generation()
            series_panels.invalidate_presence_cache(PLUGIN, context=self.context)
        return {
            "ok": True,
            "enabled": bool(self.config.get("enabled", True)),
            "ready": bool(self.ready and not self.closed),
            "changed": changed,
            "sessions": self._effective_sessions(),
            "sessions_source": self._sessions_source(),
            "reason": "ok",
        }

    def get_horizons_social_hints(self, umo: str, member_id: str = "") -> dict:
        """Read-only, sanitized group interests & member dislikes/boundaries for Dynamics Horizons."""
        if self.closed or not self.ready or self.engine is None:
            return {
                "group_interests": [],
                "member_dislikes": {},
                "member_boundaries": {},
            }
        return self.engine.horizons_social_hints(umo, member_id=member_id)

    def ingest_horizons_reflection(
        self,
        umo: str,
        member_id: str,
        topic: str,
        summary: str,
        sentiment: str = "positive",
    ) -> dict:
        """Record an interest/affection clue coming from a shared Dynamics Horizons discussion."""
        if self.closed or not self.ready or self.engine is None:
            return {"ok": False, "accepted": False, "reason": "plugin_not_ready"}
        return self.engine.ingest_horizons_reflection(
            umo, member_id, topic, summary, sentiment=sentiment
        )

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
            with store.readonly_connection() as db:
                prop_rows = db.execute(
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
                    entry_rows = db.execute(
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
            with store.readonly_connection() as db:
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
                rows = db.execute(sql, params).fetchall()
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
        filtered = []
        for ev in events:
            if isinstance(ev, dict):
                ev_scope = ev.get("scope")
                ev_umo = str(ev_scope.get("umo", "") or "") if isinstance(ev_scope, dict) else ""
                ev_member = str(ev.get("actor_id") or ev.get("member") or "").strip()
                if ev_umo and ev_member and self._is_privacy_muted(ev_umo, ev_member):
                    continue
            filtered.append(ev)
        async with self.engine.lock:
            accepted = await asyncio.to_thread(
                self.engine.store.native.observe, filtered, self._effective_sessions()
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
            if query and not jargon_matches(entry["key"], query):
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
        if self._is_privacy_muted(str(scope or ""), str(member or "")):
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
                if self._is_privacy_muted(str(scope.get("umo", "") or ""), str(member)):
                    continue
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
            if self._is_privacy_muted(str(scope.get("umo", "") or ""), member):
                return
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
        if not self.ready or self.engine is None or not affection_query_allowed(self):
            return encode({"available": False})
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        if is_series_group_excluded(self.context, umo=umo):
            return encode({"available": False, "reason": "当前群已被主控排除"})
        sender = str(event.get_sender_id())
        member = str(user_id or sender)
        check = getattr(event, "is_admin", None)
        admin = bool(check()) if callable(check) else False
        if (member != sender or mode == "ranking") and not admin:
            return encode({"available": False, "reason": "查询其他成员或排名需要管理员权限"})
        scope = await self._character_for_event(event)
        fallback_persona = ""
        if not scope:
            if ":GroupMessage:" in umo and self.engine.allowed(umo):
                known_persona = self.engine.current_personas.get(umo)
                if known_persona:
                    fallback_persona = known_persona
                    scope = {
                        "platform": umo.rsplit(":", 2)[0],
                        "umo": umo,
                        "persona_id": known_persona,
                        "instance_id": "",
                    }
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
        persona = fallback_persona or current_persona(self.engine, scope)
        if not persona:
            return encode({"available": False})
        if mode == "user":
            if self._is_privacy_muted(scope["umo"], member):
                return encode({"available": False, "reason": "该成员已启用隐私静音"})
            return encode(self.engine.social.query(scope["umo"], member, "affection"))
        if mode != "ranking":
            return encode({"available": False, "reason": "未知查询模式"})
        entries = [
            e
            for e in self.engine.cache
            if (e["scope"], e["persona"], e["category"]) == (scope["umo"], persona, "affection")
            and e["active"]
            and not self._is_privacy_muted(scope["umo"], str(e.get("member") or ""))
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
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        if is_series_group_excluded(self.context, umo=umo):
            return
        sender_id = str(event.get_sender_id() or "").strip() if callable(getattr(event, "get_sender_id", None)) else ""
        if sender_id and self._is_privacy_muted(umo, sender_id):
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
            kind(str): profile 为群友表达方式、打字习惯与喜恶，overview 为全群社交分布（仅管理员），affection 为角色对成员的好感，relations 为成员往来，jargon 为具体黑话。
            member_id(str): 当前群成员的账号，留空使用当前发言者；指定其他成员仅管理员。不能查询其他群。
            term(str): kind 为 jargon 时填写需要解释的完整词语。
        """
        if self.closed or not self.ready or not self.engine or not self.engine.social:
            return encode({"available": False, "reason": "群间学习尚未就绪"})
        scope = str(getattr(event, "unified_msg_origin", "") or "")
        sender = str(event.get_sender_id() or "").strip()
        requested_member = str(member_id or "").strip()
        member = requested_member or sender
        kind_clean = str(kind or "overview").strip()
        check = getattr(event, "is_admin", None)
        admin = bool(check()) if callable(check) else False
        if kind_clean not in {"overview", "affection", "profile", "relations", "jargon"}:
            return encode({"available": False, "reason": "未知查询类型"})
        if kind_clean == "overview" and not admin:
            return encode({"available": False, "reason": "查询全群社交状态需要管理员权限"})
        if kind_clean in {"affection", "profile", "relations"}:
            if not member or len(member) > 128:
                return encode({"available": False, "reason": "仅支持当前群成员"})
            if (not sender or member != sender) and not admin:
                return encode({"available": False, "reason": "查询其他成员的社交状态、好感度或画像需要管理员权限"})
        if kind_clean == "profile":
            if ":GroupMessage:" not in scope:
                return encode({"available": False, "reason": "仅支持当前群成员"})
            return encode(self.engine.member_profile(scope, member))
        try:
            return encode(self.engine.social.query(scope, member, kind_clean, str(term or "").strip()))
        except ValueError as exc:
            return encode({"available": False, "reason": str(exc)})

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
        payload = encode_prompt_payload(data)
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
        circuit_probe = getattr(series_panels, "provider_circuit_remaining", lambda _c, _p: 0.0)
        circuit_remaining = float(circuit_probe(self.context, provider_id) or 0.0)
        if circuit_remaining > 0.0:
            raise ReviewDeferred(
                f"共享模型熔断冷却中 (circuit_open, retry_after={circuit_remaining:.1f}s)",
                retry_after=circuit_remaining,
                reason="circuit_open",
            )
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
        is_single_review = (
            role == "review_provider"
            and isinstance(data, dict)
            and "proposal" in data
            and "proposals" not in data
        )
        call_kwargs = dict(
            _text_chat_extra_kwargs(provider, self.config, role=role, is_single_review=is_single_review)
        )
        if _accepts_kwarg(provider.text_chat, "contexts"):
            call_kwargs["contexts"] = _cache_anchor_contexts(instruction)

        async def _invoke_provider():
            if review_dispatch and self.engine is not None:
                await asyncio.to_thread(self.engine.store.automatic_reviews.record_dispatch, review_dispatch)
            return await provider.text_chat(prompt=payload, system_prompt=instruction, **call_kwargs)

        try:
            async with asyncio.timeout(timeout):
                with attribution:
                    response = await series_llm_call(
                        self.context,
                        provider_id,
                        PLUGIN,
                        role.removesuffix("_provider"),
                        _invoke_provider,
                        model=model,
                        group_id=scope,
                    )
        except ReviewDeferred:
            raise
        except Exception as exc:
            if is_provider_budget_busy_error(exc):
                retry_after = float(getattr(exc, "retry_after", 0.0) or 0.0)
                reason = str(getattr(exc, "reason", "busy") or "busy")
                raise ReviewDeferred(
                    f"共享模型并发池繁忙 ({reason})",
                    retry_after=retry_after,
                    reason=reason,
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
        if self.engine is not None:
            raw_comp = getattr(response, "raw_completion", None)
            usage = getattr(raw_comp, "usage", None) if raw_comp is not None else getattr(response, "usage", None)
            if usage is not None:
                pt = getattr(usage, "prompt_tokens", None) if not isinstance(usage, dict) else usage.get("prompt_tokens")
                ct = getattr(usage, "completion_tokens", None) if not isinstance(usage, dict) else usage.get("completion_tokens")
                if isinstance(pt, int) and isinstance(ct, int) and (pt > 0 or ct > 0):
                    self.engine._last_provider_tokens = (pt, ct)
        try:
            return decode_model_object(content)
        except ValueError as exc:
            raise MalformedModelResponse(str(exc)) from exc

    async def _authorized(self, *, check_csrf: bool = True):
        username = await require_series_admin(self.context, request, check_csrf=check_csrf)
        return "human:" + username

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
        except PermissionError as exc:
            return json_response(
                {"status": "error", "message": str(exc) or "需要 Dashboard 管理员登录"},
                status_code=403,
            )
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

    @_command_decorator("我的印象", alias={"my_impression"})
    async def cmd_my_impression(self, event: AstrMessageEvent):
        """查看当前角色对本人在本群已批准的脱敏印象卡片（仅本人可见本群资料）。"""
        from .core.member_commands import build_my_impression_payload

        if self.closed or not self.ready or self.engine is None:
            yield event.plain_result("群间学习尚未就绪，请稍后再试。")
            return
        umo = str(getattr(event, "unified_msg_origin", "") or "")
        sender_id = str(event.get_sender_id() or "").strip()
        name_getter = getattr(event, "get_sender_name", None)
        sender_name = str(name_getter() or "").strip() if callable(name_getter) else ""
        payload = await asyncio.to_thread(
            build_my_impression_payload,
            self.engine,
            self.context,
            scope=umo,
            member=sender_id,
            member_name=sender_name,
        )
        if not payload.get("ok"):
            yield event.plain_result(str(payload.get("reason") or "暂时无法生成印象卡片。"))
            return
        image_path = str(payload.get("image_path") or "")
        image_maker = getattr(event, "image_result", None)
        if image_path and callable(image_maker):
            try:
                yield image_maker(image_path)
                return
            except Exception:
                pass
        yield event.plain_result(str(payload.get("text") or ""))

    @_command_decorator("群黑话", alias={"group_jargon"})
    async def cmd_group_jargon(self, event: AstrMessageEvent, keyword: str = ""):
        """检索本群已批准的黑话词库。"""
        from .core.member_commands import query_group_jargon_list

        umo = str(getattr(event, "unified_msg_origin", "") or "")
        raw_msg = str(event.get_message_str() or "").strip()
        kw = str(keyword or "").strip()
        if not kw and " " in raw_msg:
            kw = raw_msg.split(None, 1)[1].strip()
        res = query_group_jargon_list(self.engine, scope=umo, keyword=kw)
        yield event.plain_result(str(res.get("text") or ""))

    @_command_decorator("纠正黑话", alias={"correct_jargon"})
    async def cmd_correct_jargon(self, event: AstrMessageEvent):
        """群友自助提交黑话词条补充或释义纠偏，进入审查队列。"""
        from .core.member_commands import submit_member_jargon_correction

        umo = str(getattr(event, "unified_msg_origin", "") or "")
        sender_id = str(event.get_sender_id() or "").strip()
        raw_msg = str(event.get_message_str() or "").strip()
        args = raw_msg.split(None, 1)[1].strip() if " " in raw_msg else ""
        res = await asyncio.to_thread(
            submit_member_jargon_correction,
            self.engine,
            scope=umo,
            member=sender_id,
            raw_args=args,
        )
        if res.get("ok") and isinstance(res.get("proposal"), dict):
            prop = res["proposal"]
            if prop.get("mode") != "human" and prop.get("state") == "pending":
                try:
                    await self.engine.automatic_reviews.enqueue([prop])
                except Exception:
                    pass
            series_panels.invalidate_presence_cache(PLUGIN, context=self.context)
        yield event.plain_result(str(res.get("text") or ""))

    @_command_decorator("习惯纠偏", alias={"correct_habit"})
    async def cmd_correct_habit(self, event: AstrMessageEvent):
        """群友自助提交表达/习惯/喜恶边界纠偏，进入审查队列。"""
        from .core.member_commands import submit_member_habit_correction

        umo = str(getattr(event, "unified_msg_origin", "") or "")
        sender_id = str(event.get_sender_id() or "").strip()
        raw_msg = str(event.get_message_str() or "").strip()
        note = raw_msg.split(None, 1)[1].strip() if " " in raw_msg else ""
        res = await asyncio.to_thread(
            submit_member_habit_correction,
            self.engine,
            scope=umo,
            member=sender_id,
            note=note,
        )
        if res.get("ok") and isinstance(res.get("proposal"), dict):
            prop = res["proposal"]
            if prop.get("mode") != "human" and prop.get("state") == "pending":
                try:
                    await self.engine.automatic_reviews.enqueue([prop])
                except Exception:
                    pass
            series_panels.invalidate_presence_cache(PLUGIN, context=self.context)
        yield event.plain_result(str(res.get("text") or ""))

    @_command_decorator("学习静音", alias={"learning_mute"})
    async def cmd_learning_mute(self, event: AstrMessageEvent, state: str = ""):
        """成员级退出/恢复本群学习采集与投影（兼容系列隐私静音）。"""
        from .core.member_commands import handle_learning_mute_command

        umo = str(getattr(event, "unified_msg_origin", "") or "")
        sender_id = str(event.get_sender_id() or "").strip()
        raw_msg = str(event.get_message_str() or "").strip()
        arg = str(state or "").strip()
        if not arg and " " in raw_msg:
            arg = raw_msg.split(None, 1)[1].strip()
        res = await handle_learning_mute_command(
            self.engine,
            self.context,
            scope=umo,
            member=sender_id,
            arg=arg,
        )
        yield event.plain_result(str(res.get("text") or ""))

    async def governance_action(self):
        from .core.contract import CATEGORIES
        from .core.presets import OPERATION_PRESETS, detect_active_preset, validate_review_modes
        from .core.workbench import govern

        async def operation(body, actor):
            action = text(body.get("action"), "action", 32)
            if action == "apply_preset":
                preset_id = text(body.get("preset"), "preset", 48)
                spec = OPERATION_PRESETS.get(preset_id)
                if spec is None:
                    raise ValueError(f"未知运营预设：{preset_id}")
                async with self.engine.lock:
                    for k, v in spec["config"].items():
                        self.config[k] = v
                    self._save_config_safely()
                    series_panels.invalidate_presence_cache(PLUGIN, context=self.context)
                return {
                    "action": "apply_preset",
                    "preset": preset_id,
                    "label": spec["label"],
                    "active_preset": detect_active_preset(self.config),
                    "review_modes": {c: self.engine.mode(c) for c in sorted(CATEGORIES)},
                }
            if action == "update_review_modes":
                cleaned = validate_review_modes(body.get("review_modes"))
                async with self.engine.lock:
                    for cat, mode in cleaned.items():
                        self.config[f"{cat}_review"] = mode
                    self._save_config_safely()
                    series_panels.invalidate_presence_cache(PLUGIN, context=self.context)
                return {
                    "action": "update_review_modes",
                    "updated": cleaned,
                    "active_preset": detect_active_preset(self.config),
                    "review_modes": {c: self.engine.mode(c) for c in sorted(CATEGORIES)},
                }
            scope = text(body.get("scope"), "scope", 512)
            if action == "archive_all_stale":
                return await self.engine.archive_group_stale(scope, actor)
            async with self.engine.lock:
                rows = await self.engine.batch_snapshot(scope, body.get("items"), actor)
                result = await self.engine.review_write(lambda: govern(self.engine, scope, rows, action, actor))
            self.engine.automatic_reviews.notify_context_change()
            return result

        return await self._serve(operation, ("scope", "action", "items", "preset", "review_modes"))

    async def archive_stale(self):
        async def operation(body, actor):
            return await self.engine.archive_group_stale(text(body.get("scope"), "scope", 512), actor)

        return await self._serve(operation, ("scope",))

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
            partial = body.get("partial", False)
            if not isinstance(partial, bool):
                raise ValueError("invalid partial flag")
            return await self.engine.group_batch_action(
                text(body.get("scope"), "scope", 512),
                body.get("items"),
                actor,
                partial=partial,
            )

        return await self._serve(operation, ("scope", "items", "partial"))

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
