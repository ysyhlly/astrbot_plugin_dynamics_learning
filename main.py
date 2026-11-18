# ruff: noqa: E402 -- owned module refresh must precede imports on hot reload.
from __future__ import annotations

# Refresh owned modules before importing domain code on a scoped hot reload.
from .reload_bootstrap import ensure_current

ensure_current(
    __package__,
    __name__,
    __file__,
    version="2.0.0-rc.3",
    checks=[("core.engine", "Engine", "_review_contract_v1")],
)

import asyncio
import json

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.api.web import json_response, request

from .core.contract import PERSONALITY_POLICY, decode_model_object, encode, text
from .core.engine import Engine
from .core.native import current_persona, group_projection, private_projection, replacement_allowed, role_provider
from .core.profiles import native_message
from .core.social import Social, character_key, observed_event
from .core.store import Store
from .series_panels import get_series_defaults, inherit_provider, inherit_umos

PLUGIN = "astrbot_plugin_dynamics_learning"


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
            return inherit_provider("", self.context, role="reply")
        if role == "review_provider":
            defaults = get_series_defaults(self.context)
            candidate = str(defaults.get("decision_provider") or "").strip()
            if candidate:
                getter = getattr(self.context, "get_provider_by_id", None)
                provider = getter(candidate) if callable(getter) else None
                if provider is None or not getattr(provider, "is_native_decision_provider", False):
                    return candidate
            return str(defaults.get("reply_provider") or "").strip()
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
        self.engine = Engine(store, self.config, self._model, host_context=self.context)
        await self.engine.refresh()
        self.engine.social = await asyncio.to_thread(Social, self.engine, self.context, logger)
        await self.engine.social.initialize()
        await self.engine.apply_pending_affection()
        await self.engine.apply_approved_profiles()
        self.tasks = [
            asyncio.create_task(self.engine.consume()),
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

                req.extra_user_content_parts.append(TextPart(text=content))
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

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE, priority=200)
    async def observe_group_message(self, event: AstrMessageEvent):
        if self.closed or not self.engine or not self.engine.social:
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

    async def _model(self, role, instruction, data):
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
        if len(payload) > 140_000:
            raise ValueError("证据超出单次模型输入预算，请缩小提案引用范围")
        async with asyncio.timeout(timeout):
            response = await provider.text_chat(prompt=payload, system_prompt=instruction, contexts=[])
        if (
            self.closed
            or self.context.get_provider_by_id(provider_id) is not provider
            or self._effective_provider(role) != provider_id
        ):
            raise ValueError("模型绑定已改变，本次结果失效")
        content = getattr(response, "completion_text", None)
        return decode_model_object(content)

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
                return await self.engine.review_write(lambda: govern(self.engine, scope, rows, action, actor))

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
