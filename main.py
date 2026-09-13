"""AstrBot Dynamics Learning plugin orchestrator.

Dynamics Learning is a read-only companion to ChatDynamics. It imports the
host's human annotations, turns them into replayable learning samples, learns
error patterns per task, and evaluates candidate parameters offline. It has no
path that writes to the host plugin: no configuration change, no KV write, no
runtime override. "Accepting" a policy only records the decision here.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Optional

try:
    from astrbot.api import logger
    from astrbot.api.star import Context, Star, register
except ImportError as exc:  # pragma: no cover - the host runtime always has it
    raise ImportError(
        "Dynamics Learning 需要在 AstrBot (>=4.16,<5) 运行时中加载，当前环境无法导入 astrbot.api。"
    ) from exc

from .core.config import LearningConfig, parse_learning_config
from .core.ingest import IngestResult, collect_from_host, parse_export
from .core.policy import BASE_POLICY
from .core.report import analyze, policy_rows
from .core.samples import LearningSample, build_dataset
from .core.store import LearningStore
from .core.web_api import LearningWebAPI, PLUGIN_NAME

AUTO_ANALYZE_MIN_INTERVAL = 900.0
_MAX_SAMPLE_PAGE = 200


@register(
    PLUGIN_NAME,
    "ysyhlly",
    "群间 · Dynamics Learning",
    "v0.6.0",
    "",
)
class DynamicsLearningPlugin(Star):
    """Shadow-learning companion: learn, analyse, recommend. Never auto-apply."""

    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        self.config: Any = config
        self.store = LearningStore(self)
        self.web = LearningWebAPI(self)
        self._samples: Optional[list[LearningSample]] = None
        self._last_report: Optional[dict[str, Any]] = None
        self._last_diagnostics: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._analysis_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._shutting_down = False

    # ---- lifecycle -----------------------------------------------------

    async def initialize(self) -> None:
        self.web.register()
        state = await self.store.load_state()
        report = state.get("last_report")
        self._last_report = report if isinstance(report, dict) else None
        diagnostics = state.get("last_diagnostics")
        self._last_diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        self._samples = await self.store.load_samples(config=self.runtime_config())
        self._stop.clear()
        if self.runtime_config().auto_analyze:
            self._analysis_task = asyncio.create_task(self._auto_analysis_loop())
        logger.info(
            "[DynamicsLearning] Ready. samples=%d source=%s auto_analyze=%s",
            len(self._samples or []),
            self.runtime_config().source_plugin_id,
            self.runtime_config().auto_analyze,
        )

    async def terminate(self) -> None:
        self._shutting_down = True
        self._stop.set()
        task = self._analysis_task
        self._analysis_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _auto_analysis_loop(self) -> None:
        """Periodic import + analysis. Off by default; never touches the host."""
        while not self._stop.is_set():
            interval = max(AUTO_ANALYZE_MIN_INTERVAL,
                           self.runtime_config().auto_analyze_interval_minutes * 60.0)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            if self._shutting_down:
                return
            try:
                await self.ingest(source="host")
                await self.run_analysis(with_evaluation=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[DynamicsLearning] auto analysis failed type=%s", type(exc).__name__)

    # ---- configuration -------------------------------------------------

    def runtime_config(self) -> LearningConfig:
        return parse_learning_config(getattr(self, "config", None))

    # ---- storage helpers -----------------------------------------------

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        return await value if inspect.isawaitable(value) else value

    async def load_samples(self, *, refresh: bool = False) -> list[LearningSample]:
        if self._samples is None or refresh:
            self._samples = await self.store.load_samples(config=self.runtime_config())
        return self._samples

    # ---- ingest --------------------------------------------------------

    async def ingest(self, *, source: str = "host", payload: Any = None) -> dict[str, Any]:
        config = self.runtime_config()
        async with self._lock:
            if source == "export":
                result = parse_export(payload)
            else:
                # A missing or broken host must not surface as a plugin error:
                # this plugin observes ChatDynamics and has no business taking
                # the bot down with it.
                try:
                    result = await collect_from_host(config.source_plugin_id)
                except Exception as exc:
                    logger.warning("[DynamicsLearning] host read failed type=%s", type(exc).__name__)
                    result = IngestResult(diagnostics={
                        "source": "shared_preferences", "available": False,
                        "error": type(exc).__name__, "records": 0})
            samples = build_dataset(result.annotations, config=config)
            by_session: dict[str, list[LearningSample]] = {}
            for sample in samples:
                by_session.setdefault(sample.session_key, []).append(sample)
            if by_session:
                for session_key, rows in by_session.items():
                    await self.store.replace_session(session_key, rows, config=config)
                self._samples = None
            await self.load_samples(refresh=True)
            self._last_diagnostics = dict(result.diagnostics)
            await self.store.patch_state(last_diagnostics=self._last_diagnostics,
                                         last_ingest_at=time.time())
            stored = len(self._samples or [])
        available = result.diagnostics.get("available")
        return {
            "source": source,
            "ok": available is not False,
            # One host annotation yields up to three samples, so the two counts
            # are reported separately rather than as one ambiguous "imported".
            "annotations": len(result.annotations),
            "imported_samples": len(samples),
            "sessions": len(by_session),
            "stored_samples": stored,
            "diagnostics": dict(result.diagnostics),
            "note": "只读取 ChatDynamics 的共享首选项，不写入本体任何数据。",
        }

    # ---- analysis ------------------------------------------------------

    async def run_analysis(self, *, with_evaluation: bool = True,
                           with_tuning: bool = True) -> dict[str, Any]:
        async with self._lock:
            samples = await self.load_samples(refresh=True)
            policies = await self.store.load_policies()
            result = analyze(
                samples,
                config=self.runtime_config(),
                existing_versions=[row.version for row in policies],
                baseline_policy=BASE_POLICY,
                with_evaluation=with_evaluation,
                with_tuning=with_tuning,
            )
            payload = _trim(result.as_dict())
            self._last_report = payload
            await self.store.patch_state(last_report=payload, last_analysis_at=time.time())
            # The iterative runs are what carry a promotion verdict, so they own
            # the policy records whenever they ran. The single-shot candidate is
            # only persisted as a fallback, never alongside them, so the same
            # adjustment is not recorded twice under two versions.
            recorded = [row.candidate for row in result.tuning if row.candidate is not None]
            if not recorded and result.evaluation is not None and result.evaluation.candidate:
                recorded = [result.evaluation.candidate]
            for candidate in recorded:
                await self.store.append_policy(candidate)
        return self.report_payload_sync()

    def report_payload_sync(self) -> dict[str, Any]:
        return {
            "report": self._last_report,
            "has_report": self._last_report is not None,
            "diagnostics": dict(self._last_diagnostics),
            "limits": {
                "note": "本插件不会自动修改 ChatDynamics 配置；采纳只写入本插件的策略记录。",
                "shadow_only": True,
            },
        }

    # ---- web payloads --------------------------------------------------

    async def overview_payload(self) -> dict[str, Any]:
        config = self.runtime_config()
        samples = await self.load_samples()
        state = await self.store.load_state()
        policies = await self.store.load_policies()
        return {
            "plugin": PLUGIN_NAME,
            "version": "v0.6.0",
            "config": config.as_dict(),
            "dataset": {
                "samples": len(samples),
                "sessions": len({sample.session_hash for sample in samples}),
                "tasks": {
                    task: sum(1 for sample in samples if sample.task == task)
                    for task in ("recipient", "topic", "reply")
                },
            },
            "diagnostics": dict(self._last_diagnostics),
            "has_report": self._last_report is not None,
            "last_analysis_at": state.get("last_analysis_at"),
            "last_ingest_at": state.get("last_ingest_at"),
            "policies": len(policies),
            "contract": {
                "reads": ["panel_runtime_v1", "topic_annotations_v1_<sha256(session)>"],
                "writes_to_host": False,
                "shadow_only": True,
            },
        }

    async def samples_payload(self, *, page: int = 1, page_size: int = 50,
                              task: str = "", session: str = "") -> dict[str, Any]:
        samples = await self.load_samples()
        rows = samples
        if task:
            rows = [row for row in rows if row.task == task]
        if session:
            digest = session
            if len(digest) != 64:
                from .core.samples import session_hash
                digest = session_hash(session)
            rows = [row for row in rows if row.session_hash == digest]
        page_size = max(1, min(_MAX_SAMPLE_PAGE, int(page_size or 50)))
        page = max(1, int(page or 1))
        start = (page - 1) * page_size
        window = rows[start:start + page_size]
        return {
            "total": len(rows), "page": page, "page_size": page_size,
            "rows": [{
                "sample_id": row.sample_id,
                "session": _redact(row.session_key),
                "session_hash": row.session_hash[:12],
                "msg_id": _redact(row.msg_id),
                "task": row.task,
                "predicted": _redact(row.predicted),
                "expected": _redact(row.expected),
                "correct": row.correct,
                "confidence": round(float(row.confidence), 4),
                "error_type": row.error_type,
                "timestamp": row.timestamp,
                "codes": list(row.trace.get("evidence_summary", {}).get("codes", ()))[:12]
                if isinstance(row.trace.get("evidence_summary"), dict) else [],
            } for row in window],
            "note": "样本不含消息正文；身份字段按首尾保留脱敏。",
        }

    async def report_payload(self) -> dict[str, Any]:
        return self.report_payload_sync()

    async def policies_payload(self) -> dict[str, Any]:
        policies = await self.store.load_policies()
        return {"total": len(policies), "rows": policy_rows(policies),
                "note": "策略记录只是建议与结论，不会改动 ChatDynamics 配置。"}

    async def update_policy(self, version: str, action: str) -> dict[str, Any]:
        status = {"accept": "accepted", "ignore": "rejected",
                  "reopen": "candidate", "rollback": "rolled_back"}[action]
        updated = await self.store.update_policy_status(version, status)
        if updated is None:
            raise ValueError(f"找不到策略版本 {version}")
        return {"version": updated.version, "status": updated.status,
                "note": "已更新本插件的策略记录；ChatDynamics 配置未发生任何变化。"}

    async def export_payload(self) -> dict[str, Any]:
        samples = await self.load_samples()
        policies = await self.store.load_policies()
        return {
            "schema": "dynamics_learning_export_v1",
            "exported_at": time.time(),
            "samples": [row.as_dict(include_trace=False) for row in samples],
            "report": self._last_report,
            "policies": policy_rows(policies),
            "diagnostics": dict(self._last_diagnostics),
        }

    async def reset_storage(self) -> dict[str, Any]:
        async with self._lock:
            removed = await self.store.clear_samples()
            await self.store.save_policies([])
            await self.store.save_state({})
            self._samples = []
            self._last_report = None
            self._last_diagnostics = {}
        return {"removed_samples": removed, "note": "已清空本插件的样本与报告。"}


def _redact(value: Any) -> str:
    """Keep identity fields out of the page while staying recognisable.

    Matches the host console's convention: short values are masked entirely and
    longer ones keep only their head and tail.
    """
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 6:
        return "*" * len(text)
    return f"{text[:3]}…{text[-2:]}"


def _trim(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop sweep curves before persisting a report to shared preferences."""
    def walk(value: Any, depth: int = 0) -> Any:
        if depth > 8:
            return None
        if isinstance(value, dict):
            return {key: walk(item, depth + 1) for key, item in value.items() if key != "curve"}
        if isinstance(value, list):
            return [walk(item, depth + 1) for item in value[:256]]
        return value
    return walk(payload)


__all__ = ["PLUGIN_NAME", "DynamicsLearningPlugin"]
