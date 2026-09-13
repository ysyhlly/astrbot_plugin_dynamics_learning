"""Plugin-page Web API handlers for Dynamics Learning.

Every response uses the same envelope as ChatDynamics (status/ok/data/error) so
the existing page bridge works unchanged. No endpoint can change a ChatDynamics
setting: the only mutations are this plugin's own sample store and its own
policy records.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from .web_compat import error_response, json_response, query_value, request_json

logger = logging.getLogger("astrbot_plugin_dynamics_learning.web_api")

PLUGIN_NAME = "astrbot_plugin_dynamics_learning"
_MAX_BODY_BYTES = 4 * 1024 * 1024
_MAX_PAGE_SIZE = 200
_ALLOWED_ACTIONS = ("accept", "ignore", "reopen", "rollback")


def _json_ok(data: Any):
    return json_response(
        {"status": "ok", "ok": True, "data": data, "error": None, "message": None},
        status_code=200,
    )


def _json_err(message: str, status_code: int = 400, headers: Any = None):
    try:
        return error_response(message, status_code=status_code, data=None, headers=headers)
    except TypeError:  # pragma: no cover - older SDK signature
        return error_response(message, status_code=status_code, data=None)


def _query_param(name: str, default: str = "") -> str:
    try:
        return query_value(name) or default
    except Exception:
        return default


def _query_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(_query_param(name, str(default)))))
    except (TypeError, ValueError):
        return default


async def _json_body() -> Dict[str, Any]:
    try:
        payload = await request_json({})
    except Exception:
        return {"__invalid_body__": "invalid JSON body"}
    if not isinstance(payload, dict):
        return {"__invalid_body__": "JSON body must be an object"}
    try:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return {"__invalid_body__": "invalid JSON body"}
    if len(encoded) > _MAX_BODY_BYTES:
        return {"__invalid_body__": "body too large"}
    return payload


class LearningWebAPI:
    """Register the learning endpoints behind the host Dashboard auth scope."""

    def __init__(self, plugin: Any) -> None:
        self.plugin = plugin
        self.registered = False
        self._registered_endpoints: set[str] = set()

    @staticmethod
    def _endpoint_key(route: str, methods: list[str]) -> str:
        # AstrBot allows the same path with different methods, so dedupe must
        # include the method set.
        return f"{route}@" + "|".join(sorted(str(method).upper() for method in (methods or [])))

    def register(self) -> None:
        if self.registered:
            return
        context = getattr(self.plugin, "context", None)
        if context is None or not hasattr(context, "register_web_api"):
            return
        routes = [
            ("overview", self.overview, ["GET"], "学习层状态、数据面与最近一次分析"),
            ("samples", self.samples, ["GET"], "分页查看学习样本（不含正文）"),
            ("ingest", self.ingest, ["POST"], "从 ChatDynamics 只读导入标注，或导入导出文件"),
            ("analyze", self.analyze, ["POST"], "运行学习与离线评测并保存结果"),
            ("report", self.report, ["GET"], "最近一次分析结果"),
            ("policies", self.policies, ["GET"], "策略候选与版本记录"),
            ("policy", self.policy, ["POST"], "标记采纳/忽略/回滚策略记录"),
            ("export", self.export, ["GET"], "导出样本、报告与策略记录 JSON"),
            ("reset", self.reset, ["POST"], "清空本插件的样本与报告（不可逆）"),
        ]
        for endpoint, handler, methods, desc in routes:
            route = f"/{PLUGIN_NAME}/{endpoint}"
            key = self._endpoint_key(route, methods)
            if key in self._registered_endpoints:
                continue
            try:
                context.register_web_api(route, handler, methods, desc)
            except Exception as exc:
                logger.error(
                    "[DynamicsLearning] Web API registration failed endpoint=%s methods=%s type=%s",
                    route, ",".join(methods), type(exc).__name__,
                )
                continue
            self._registered_endpoints.add(key)
        self.registered = all(
            self._endpoint_key(f"/{PLUGIN_NAME}/{endpoint}", methods) in self._registered_endpoints
            for endpoint, _handler, methods, _desc in routes
        )

    # ---- handlers ------------------------------------------------------

    async def overview(self):
        try:
            return _json_ok(await self.plugin.overview_payload())
        except Exception as exc:
            logger.error("[DynamicsLearning] overview failed type=%s", type(exc).__name__)
            return _json_err("读取学习层状态失败", 500)

    async def samples(self):
        try:
            return _json_ok(await self.plugin.samples_payload(
                page=_query_int("page", 1, 1, 100_000),
                page_size=_query_int("page_size", 50, 1, _MAX_PAGE_SIZE),
                task=_query_param("task"),
                session=_query_param("session"),
            ))
        except Exception as exc:
            logger.error("[DynamicsLearning] samples failed type=%s", type(exc).__name__)
            return _json_err("读取学习样本失败", 500)

    async def ingest(self):
        body = await _json_body()
        if "__invalid_body__" in body:
            return _json_err(body["__invalid_body__"])
        source = body.get("source") or "host"
        if source not in ("host", "export"):
            return _json_err("source 只支持 host 或 export")
        payload = body.get("payload")
        if source == "export" and payload is None:
            return _json_err("source=export 需要 payload")
        try:
            return _json_ok(await self.plugin.ingest(source=source, payload=payload))
        except Exception as exc:
            logger.error("[DynamicsLearning] ingest failed type=%s", type(exc).__name__)
            return _json_err("导入失败", 500)

    async def analyze(self):
        body = await _json_body()
        if "__invalid_body__" in body:
            return _json_err(body["__invalid_body__"])
        with_evaluation = body.get("with_evaluation", True)
        if not isinstance(with_evaluation, bool):
            return _json_err("with_evaluation 必须是布尔值")
        with_tuning = body.get("with_tuning", True)
        if not isinstance(with_tuning, bool):
            return _json_err("with_tuning 必须是布尔值")
        try:
            return _json_ok(await self.plugin.run_analysis(with_evaluation=with_evaluation,
                                                           with_tuning=with_tuning))
        except Exception as exc:
            logger.error("[DynamicsLearning] analyze failed type=%s", type(exc).__name__)
            return _json_err("分析失败", 500)

    async def report(self):
        try:
            return _json_ok(await self.plugin.report_payload())
        except Exception as exc:
            logger.error("[DynamicsLearning] report failed type=%s", type(exc).__name__)
            return _json_err("读取分析结果失败", 500)

    async def policies(self):
        try:
            return _json_ok(await self.plugin.policies_payload())
        except Exception as exc:
            logger.error("[DynamicsLearning] policies failed type=%s", type(exc).__name__)
            return _json_err("读取策略记录失败", 500)

    async def policy(self):
        body = await _json_body()
        if "__invalid_body__" in body:
            return _json_err(body["__invalid_body__"])
        version = body.get("version")
        action = body.get("action")
        if not isinstance(version, str) or not version.strip():
            return _json_err("需要 version")
        if not isinstance(action, str) or action not in _ALLOWED_ACTIONS:
            return _json_err("action 只支持 " + "/".join(_ALLOWED_ACTIONS))
        try:
            return _json_ok(await self.plugin.update_policy(version.strip(), action))
        except ValueError as exc:
            return _json_err(str(exc), 404)
        except Exception as exc:
            logger.error("[DynamicsLearning] policy failed type=%s", type(exc).__name__)
            return _json_err("更新策略记录失败", 500)

    async def export(self):
        try:
            payload = await self.plugin.export_payload()
            return json_response(
                payload, status_code=200,
                headers={"Content-Disposition": 'attachment; filename="dynamics_learning.json"'})
        except Exception as exc:
            logger.error("[DynamicsLearning] export failed type=%s", type(exc).__name__)
            return _json_err("导出失败", 500)

    async def reset(self):
        body = await _json_body()
        if "__invalid_body__" in body:
            return _json_err(body["__invalid_body__"])
        if body.get("confirm") != "reset":
            return _json_err('需要 confirm="reset" 才会清空')
        try:
            return _json_ok(await self.plugin.reset_storage())
        except Exception as exc:
            logger.error("[DynamicsLearning] reset failed type=%s", type(exc).__name__)
            return _json_err("清空失败", 500)


__all__ = ["PLUGIN_NAME", "LearningWebAPI"]
