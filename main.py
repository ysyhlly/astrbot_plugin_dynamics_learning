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
import math
from dataclasses import replace
import re
import time
from typing import Any, Optional

try:
    from astrbot.api import logger
    from astrbot.api.star import Context, Star, register
except ImportError as exc:  # pragma: no cover - the host runtime always has it
    raise ImportError(
        "Dynamics Learning 需要在 AstrBot (>=4.16,<5) 运行时中加载，当前环境无法导入 astrbot.api。"
    ) from exc

from .core.attribution import attribution_report
from .core.config import LearningConfig, parse_learning_config
from .core.ingest import IngestResult, collect_from_host, parse_export, parse_preferences
from .core.policy import (
    ACTION_STATUS, ACTIONS, BASE_POLICY, CONSOLE_ACTIONS, available_actions, candidate_payload,
    normalize_status, published_payload, PARAM_SPECS, validate_candidate_evidence,
)
from .core.quality import dataset_gate, quality_report
from .core.reply_review import (  # noqa: F401 - review mixin compatibility exports
    REPLY_REVIEW_SCHEMA_VERSION, build_digest as build_reply_digest,
    build_prompt as build_reply_prompt, digest_fingerprint as reply_digest_fingerprint,
    parse_reply_review, select_messages,
)
from .core.review import (  # noqa: F401 - review mixin compatibility exports
    REVIEW_SCHEMA_VERSION, build_digest, build_prompt, digest_fingerprint, parse_review,
)
from .core.report import analyze, policy_rows
from .core.shadow import evaluate_shadow, rules_from_config
from .core.shadow_coverage import evaluate_shadow_coverage
from .core import scope_profile
from .core.samples import LearningSample, build_dataset, session_hash
from .core.store import LearningStore
from .core.query_cache import QueryCache
from .core.review_runtime import ReviewRuntimeMixin
from .core.evaluator import dataset_fingerprint
from .core.window import session_digest, window_payload
from .core.web_api import LearningWebAPI, PLUGIN_NAME

AUTO_ANALYZE_MIN_INTERVAL = 900.0
_MAX_SAMPLE_PAGE = 200
# A provider that just failed is not asked again for this long, so a page left
# open cannot turn one broken model call into one per refresh.
REVIEW_RETRY_SECONDS = 90.0


def _completion_text(response: Any) -> str:
    """Best-effort text out of whatever the host's llm_generate returned."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    for name in ("completion_text", "text"):
        value = getattr(response, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


@register(
    PLUGIN_NAME,
    "ysyhlly",
    "群间 · Dynamics Learning",
    "v1.4.0",
    "",
)
class DynamicsLearningPlugin(ReviewRuntimeMixin, Star):
    """Shadow-learning companion: learn, analyse, recommend. Never auto-apply."""

    def __init__(self, context: Context, config: Any = None):
        super().__init__(context)
        self.config: Any = config
        self.store = LearningStore(self)
        self.web = LearningWebAPI(self)
        self._samples: Optional[list[LearningSample]] = None
        self._last_report: Optional[dict[str, Any]] = None
        self._last_diagnostics: dict[str, Any] = {}
        self._last_contract: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._query_cache = QueryCache()
        self._init_review_runtime()
        self._analysis_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._shutting_down = False
        self._review_failed_at = 0.0
        self._review_failure_reason = ""
        # The post-mortem carries message text, so its result lives in memory for
        # the life of the process and is never written to the KV store.
        self._reply_review_cache: dict[str, Any] = {}
        self._reply_review_failed_at = 0.0
        self._reply_review_failure_reason = ""

    # ---- lifecycle -----------------------------------------------------

    async def initialize(self) -> None:
        self.web.register()
        state = await self.store.load_state()
        report = state.get("last_report")
        self._last_report = report if isinstance(report, dict) else None
        diagnostics = state.get("last_diagnostics")
        self._last_diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        contract = state.get("last_contract_stats")
        self._last_contract = contract if isinstance(contract, dict) else {}
        self._samples = await self.store.load_samples(config=self.runtime_config())
        fingerprint = await asyncio.to_thread(dataset_fingerprint, self._samples)
        policies = await self.store.load_policies()
        for index, candidate in enumerate(policies):
            if (candidate.training_dataset.get("fingerprint") != fingerprint
                    or (candidate.status in {"validated", "shadow", "promoted"}
                        and (validate_candidate_evidence(candidate)
                             or candidate.evidence.get("final_validation", {}).get("verdict") != "accepted"))):
                policies[index] = replace(candidate, evidence={**candidate.evidence,
                    "stale": True, "stale_reason": "dataset_or_validation_changed"})
        await self.store.save_policies(policies)
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
        await self.invalidate_reviews()
        await self._query_cache.close()
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
            self._query_cache.invalidate()
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
            if result.diagnostics.get("available") is False:
                return {"source": source, "ok": False, "annotations": 0,
                        "imported_samples": 0, "sessions": 0,
                        "stored_samples": len(await self.load_samples()),
                        "diagnostics": dict(result.diagnostics),
                        "contract": dict(self._last_contract)}
            # The runtime snapshot decides nothing about identity; it records
            # which host facts (umo / group_id / bot_id) were available, so the
            # contract health report can tell a confirmed session scope from a
            # fallback.
            samples = await asyncio.to_thread(build_dataset, result.annotations,
                                              session_meta=result.sessions, config=config)
            samples = [row for row in samples if row.session_key in
                       (result.confirmed_sessions | result.partial_sessions)]
            before_fingerprint = await asyncio.to_thread(dataset_fingerprint, await self.load_samples())
            by_session: dict[str, list[LearningSample]] = {
                key: [] for key in result.confirmed_sessions}
            partial = result.partial_sessions - result.confirmed_sessions
            revised_messages = {(session, str(record.get("msg_id") or ""))
                                for session, record in result.annotations if session in partial}
            # Bare annotation records are an incremental import, never a full snapshot.
            for session in partial:
                by_session.setdefault(session, [])
            for old in await self.load_samples():
                if (old.session_key in partial
                        and (old.session_key, old.msg_id) not in revised_messages):
                    by_session.setdefault(old.session_key, []).append(old)
            for sample in samples:
                if sample.session_key in result.confirmed_sessions:
                    by_session[sample.session_key].append(sample)
                elif sample.session_key in partial:
                    by_session[sample.session_key].append(sample)
            if by_session:
                try:
                    await self.store.replace_sessions(by_session, config=config)
                finally:
                    self._samples = None
                    self._query_cache.invalidate()
            await self.load_samples(refresh=True)
            fingerprint = await asyncio.to_thread(dataset_fingerprint, self._samples or [])
            if fingerprint != before_fingerprint:
                await self.invalidate_reviews()
                await self.store.clear_review()
                if self._last_report is not None:
                    self._last_report = {**self._last_report, "stale": True,
                                         "stale_reason": "dataset_changed"}
                policies = await self.store.load_policies()
                policies = [replace(row, evidence={**row.evidence, "stale": True,
                            "stale_reason": "dataset_changed"}) for row in policies]
                await self.store.save_policies(policies)
            self._last_diagnostics = dict(result.diagnostics)
            # The contract plane is a snapshot: it describes the raw records as
            # they were at this moment, and nothing downstream can reconstruct it
            # later. `/quality` reads it back with its own timestamp.
            self._last_contract = result.contract.as_dict()
            await self.store.patch_state(last_diagnostics=self._last_diagnostics,
                                         last_contract_stats=self._last_contract,
                                         last_ingest_at=time.time(),
                                         dataset_fingerprint=fingerprint,
                                         last_report=self._last_report)
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
            "contract": dict(self._last_contract or {}),
            "note": "只读取 ChatDynamics 的共享首选项，不写入本体任何数据。",
        }

    # ---- analysis ------------------------------------------------------

    async def _analysis_baseline(self) -> tuple[dict[str, float], str, str | None]:
        """Read the loaded host's effective policy inputs, never its config file."""
        version = str(self._last_diagnostics.get("host_version") or "") or None
        getter = getattr(self.context, "get_all_stars", None)
        try:
            stars = await self._maybe_await(getter()) if callable(getter) else []
            for metadata in stars:
                identity = f"{getattr(metadata, 'author', '')}/{getattr(metadata, 'name', '')}"
                if identity != self.runtime_config().source_plugin_id:
                    continue
                host = getattr(metadata, "star_cls", None)
                if host is None or not getattr(metadata, "activated", True):
                    continue
                reader = getattr(host, "_learning_policy_effective_config", None)
                if not callable(reader):
                    continue
                values = await self._maybe_await(reader())
                if not isinstance(values, dict):
                    continue
                if not all(isinstance(values.get(key), (float, int))
                           and not isinstance(values[key], bool)
                           and math.isfinite(values[key])
                           and spec["min"] <= values[key] <= spec["max"]
                           for key, spec in PARAM_SPECS.items()):
                    continue
                return ({key: float(values[key]) for key in PARAM_SPECS}, "host_effective",
                        str(getattr(metadata, "version", "") or "") or version)
        except Exception as exc:
            logger.warning("[DynamicsLearning] baseline read failed type=%s", type(exc).__name__)
        return dict(BASE_POLICY), "default_reference", version

    async def run_analysis(self, *, with_evaluation: bool = True,
                           with_tuning: bool = True) -> dict[str, Any]:
        async with self._lock:
            samples = await self.load_samples(refresh=True)
            policies = await self.store.load_policies()
            baseline, baseline_source, host_version = await self._analysis_baseline()
            result = await asyncio.to_thread(
                analyze,
                samples,
                config=self.runtime_config(),
                existing_versions=[row.version for row in policies],
                baseline_policy=baseline,
                baseline_source=baseline_source,
                with_evaluation=with_evaluation,
                with_tuning=with_tuning,
                # The host's self-reported version, when it reports one. It
                # travels into the policy record's `target` block and from there
                # into /published: a consumer that cannot see which version a
                # policy was validated against has no basis for `active`.
                host_version=host_version,
            )
            if self._shutting_down:
                return self.report_payload_sync()
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
            # A corpus that failed the dataset gate produces no policy record at
            # all. Recording one as "proposed" would leave a promotion candidate
            # sitting in the console with a version number, waiting for someone
            # to click it — which is exactly what the gate exists to prevent.
            if result.dataset_gate.get("ok", True):
                for candidate in recorded:
                    if candidate.status == "validated" and validate_candidate_evidence(candidate):
                        raise ValueError("最终候选与验证证据身份不一致，拒绝保存")
                    await self.store.append_policy(candidate)
            await self._save_policy_offers()
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
            "version": "v1.4.0",
            "config": config.as_dict(),
            "dataset": {
                "samples": len(samples),
                "sessions": len({sample.session_hash for sample in samples}),
                "tasks": {
                    task: sum(1 for sample in samples if sample.task == task)
                    for task in ("recipient", "topic", "reply_admission", "reply_outcome")
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
                              task: str = "", session: str = "", scope: str = "") -> dict[str, Any]:
        samples = await self.load_samples()
        rows = samples
        if task:
            rows = [row for row in rows if row.task == task]
        if session:
            digest = resolve_session_digest(session)
            rows = [row for row in rows if row.session_hash == digest]
        if scope:
            digest = resolve_session_digest(scope)
            rows = [row for row in rows if row.scope_hash == digest]
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
                # The label columns travel **unredacted**, which is the point of
                # them: predicted/expected are drawn from a closed vocabulary
                # (bot / other / reply / silent / a topic id), never from message
                # text. Running them through _redact masks every value to
                # asterisks -- "bot" and "other" both render as five stars -- so
                # the page ends up with a table nobody can read. Redaction
                # belongs on the identity columns above, and nowhere else.
                "predicted": row.predicted,
                "expected": row.expected,
                "correct": row.correct,
                "confidence": round(float(row.confidence), 4),
                "error_type": row.error_type,
                "timestamp": row.timestamp,
                "codes": list(row.trace.get("evidence_summary", {}).get("codes", ()))[:12]
                if isinstance(row.trace.get("evidence_summary"), dict) else [],
            } for row in window],
            "note": "样本不含消息正文；会话与消息 id 按首尾保留脱敏，预测与标注是标签本身。",
            "label_vocabulary": {"recipient": ["bot", "other"],
                                 "reply_admission": ["reply", "silent"],
                                 "reply_outcome": ["reply", "silent"],
                                 "topic": ["话题 id，或 NEW:<msg_id>"]},
        }

    async def quality_payload(self) -> dict[str, Any]:
        """Contract health: what the recorded data can and cannot answer.

        The sample plane is recomputed here on every request; the contract plane
        is the snapshot the last import took, so it can only ever be as fresh as
        that import — which is why it travels with its own timestamp.
        """
        config = self.runtime_config()
        samples = await self.load_samples()
        state = await self.store.load_state()
        contract = self._last_contract or state.get("last_contract_stats")
        payload = await self._query_cache.compute(
            ("quality", config, repr(contract), state.get("last_ingest_at")), quality_report,
            samples,
            contract=contract if isinstance(contract, dict) else None,
            reader_version=int(self._last_diagnostics.get("reader_version") or 0) or None,
            ingest_at=_as_float(state.get("last_ingest_at")),
            min_samples=config.min_samples_for_evaluation,
        )
        # The gate travels with quality, not with the report: it is a statement
        # about the corpus, and a reader has to be able to see it before running
        # an analysis that would produce nothing.
        payload["dataset_gate"] = await self._query_cache.compute(
            ("dataset_gate", config, repr(contract)), dataset_gate,
            samples, config=config,
            contract=contract if isinstance(contract, dict) else None)
        return payload

    async def annotation_window_payload(self, *, include_messages: bool = False,
                                        sp_module: Any = None) -> dict[str, Any]:
        """Which messages can still be labelled, and for how much longer.

        Two reads of the host shared preferences — its runtime snapshot and the
        annotation keys — and no model call: this is a fact about the window, and
        the reader is about to open the reply page with it. Nothing is written:
        the export is assembled here and downloaded by the page.
        """
        config = self.runtime_config()
        runtime: Any = None
        annotated: dict[str, set[str]] = {}
        status = "unavailable"
        try:
            if sp_module is None:
                from astrbot.core import sp as sp_module
            runtime = await sp_module.get_async(
                scope="plugin", scope_id=config.source_plugin_id,
                key="panel_runtime_v1", default=None)
            status = "ok" if runtime is not None else "missing"
            rows = await sp_module.range_get_async("plugin", config.source_plugin_id, None)
            for session_key, record in parse_preferences(rows).annotations:
                msg_id = record.get("msg_id")
                if isinstance(msg_id, str) and msg_id:
                    annotated.setdefault(session_digest(session_key), set()).add(msg_id)
        except Exception as exc:
            logger.warning("[DynamicsLearning] annotation window read failed type=%s",
                           type(exc).__name__)
            status = "unavailable"
        payload = window_payload(runtime, annotated, now_wall=time.time(),
                                 include_messages=include_messages)
        payload["source_status"] = status
        return payload
    async def attribution_payload(self, *, examples: int = 8) -> dict[str, Any]:
        """The error attribution chain, computed from the stored samples.

        Recomputed on every request rather than read out of the last analysis:
        the chain is a pure function of the samples, and a page that showed a
        stale table beside live sample counts would be the one place a reader
        could not tell which batch it described.
        """
        config = self.runtime_config()
        return await self._query_cache.compute(
            ("attribution", config, examples), attribution_report,
            await self.load_samples(),
            min_messages=config.min_samples_for_evaluation,
            min_samples=config.min_samples_for_evaluation,
            examples=max(1, min(50, examples)),
        )

    async def shadow_payload(self) -> dict[str, Any]:
        """The shadow A/B result, recomputed live from the stored samples.

        Live rather than read out of the last analysis for the same reason the
        attribution table is: it is a pure function of the samples, and a page
        showing a stale verdict beside live counts is the one place a reader
        cannot tell which batch it describes.
        """
        config = self.runtime_config()
        result = await self._query_cache.compute(("shadow", config), evaluate_shadow,
                                 await self.load_samples(),
                                 rules=rules_from_config(config),
                                 min_samples=config.gate_min_samples)
        result["operational"] = await self.operational_shadow_payload()
        return result

    async def operational_shadow_payload(self, *, sp_module: Any = None) -> dict[str, Any]:
        """Read the host's bounded comparison log, without creating labelled samples."""
        payload = None
        status = "unavailable"
        try:
            if sp_module is None:
                from astrbot.core import sp as sp_module
            payload = await sp_module.get_async(
                scope="plugin", scope_id=self.runtime_config().source_plugin_id,
                key="shadow_telemetry_v1", default=None)
            status = "ok" if payload is not None else "missing"
        except Exception as exc:
            logger.warning("[DynamicsLearning] telemetry read failed type=%s", type(exc).__name__)
        result = await asyncio.to_thread(evaluate_shadow_coverage, payload)
        result["source_status"] = status
        return result

    async def scopes_payload(self) -> dict[str, Any]:
        """Every reviewed scope, one row each, already ranked for the list view."""
        samples = await self.load_samples()
        return await self._query_cache.compute(("scopes", self.runtime_config()),
                                               scope_profile.scopes_payload, samples)

    async def scope_payload(self, identifier: str) -> dict[str, Any]:
        """One scope's review profile against the leave-one-out baseline.

        The comparison is computed here rather than in the page: a server-side
        number can be tested, and a page that recomputed the baseline could
        disagree with the report it is sitting next to.
        """
        digest = resolve_session_digest(identifier)
        samples = await self.load_samples()
        payload = await self._query_cache.compute(("scope", self.runtime_config(), digest),
                                                  scope_profile.scope_payload, samples, digest)
        if payload is None:
            raise ValueError(f"数据集里没有作用域 {digest[:12]} 的样本")
        return payload

    async def report_payload(self) -> dict[str, Any]:
        return self.report_payload_sync()

    async def policies_payload(self) -> dict[str, Any]:
        async with self._lock:
            return await self._policies_payload_locked()

    async def _policies_payload_locked(self) -> dict[str, Any]:
        policies = await self.store.load_policies()
        counts: dict[str, int] = {}
        for row in policies:
            key = normalize_status(row.status)
            counts[key] = counts.get(key, 0) + 1
        published = published_payload(policies)
        rows = policy_rows(policies)
        revision = await self.store.policy_revision()
        # Which buttons a row may offer is a property of its *current* status,
        # and the store is the only thing that knows it. Deriving it here means
        # the page never renders an arrow the state machine would refuse — the
        # alternative is a 400 the reader has to read as a UI bug.
        for policy_row in rows:
            policy_row["revision"] = revision
            policy_row["available_actions"] = available_actions(policy_row.get("status") or "")
            if policy_row.get("evidence", {}).get("stale"):
                policy_row["available_actions"] = [action for action in policy_row["available_actions"]
                                                   if action in {"ignore", "rollback", "supersede"}]
        return {"total": len(policies), "rows": rows,
                "actions": [{"action": action, "label": label}
                            for action, label in CONSOLE_ACTIONS],
                "status_counts": counts,
                # The published face travels with the records so the page can
                # show what was offered to the host next to what this plugin
                # decided, instead of leaving a reader to join two requests.
                "published": published["policies"],
                "note": "策略记录只是建议与结论，不会改动 ChatDynamics 配置；"
                        "只有 promoted 的记录会出现在 /published。"}

    async def update_policy(self, version: str, action: str, *,
                            expected_revision: str | None = None) -> dict[str, Any]:
        async with self._lock:
            return await self._update_policy_locked(version, action, expected_revision=expected_revision)

    async def _update_policy_locked(self, version: str, action: str, *,
                                    expected_revision: str | None = None) -> dict[str, Any]:
        """Move one policy record through the state machine.

        This writes a record inside this plugin and nothing else. The published
        file is a *description* of what was decided here; adopting it remains
        ChatDynamics' decision.
        """
        status = ACTION_STATUS.get(action)
        if status is None:
            raise ValueError(f"未知操作 {action}；支持：{'/'.join(ACTIONS)}")
        updated = await self.store.update_policy_status(
            version, status, reason=f"控制台操作 {action}",
            expected_revision=expected_revision)
        if updated is None:
            raise ValueError(f"找不到策略版本 {version}")
        # The publish contract is materialised on every state change, so the
        # consumer always finds the current answer in one key rather than a
        # snapshot from whenever an endpoint last happened to be called.
        await self._save_policy_offers()
        return {"version": updated.version, "status": updated.status,
                "status_label": updated.label,
                "note": "已更新本插件的策略记录；ChatDynamics 配置未发生任何变化。"}

    async def _save_policy_offers(self) -> None:
        policies = await self.store.load_policies()
        await self.store.save_policies(policies)

    async def candidate_payload(self) -> dict[str, Any]:
        """Read-only validated/shadow/promoted offers for shadow consumers."""
        async with self._lock:
            return {**candidate_payload(await self.store.load_policies()),
                    "revision": await self.store.policy_revision()}

    async def published_payload(self) -> dict[str, Any]:
        """The read-only offer for ChatDynamics: promoted policies only.

        Computed live from the policy records — the stored copy is written by
        the callers that change those records, never read back here, so a stale
        key can never be served as if it were current.
        """
        async with self._lock:
            return {**published_payload(await self.store.load_policies()),
                    "revision": await self.store.policy_revision()}

    async def export_payload(self) -> dict[str, Any]:
        async with self._lock:
            samples = await self.load_samples()
            policies = await self.store.load_policies()
        return {
            "schema": "dynamics_learning_export_v1",
            "purpose": "diagnostic",
            "restorable": False,
            "exported_at": time.time(),
            "samples": [row.as_dict(include_trace=False) for row in samples],
            "published": published_payload(policies)["policies"],
            "shadow": await self.shadow_payload(),
            "report": self._last_report,
            "policies": policy_rows(policies),
            "diagnostics": dict(self._last_diagnostics),
        }

    async def reset_storage(self) -> dict[str, Any]:
        async with self._lock:
            await self.invalidate_reviews()
            self._query_cache.invalidate()
            removed = await self.store.clear_samples()
            await self.store.save_policies([])
            await self.store.clear_published()
            await self.store.clear_candidate()
            await self.store.clear_review()
            await self.store.save_state({})
            self._samples = []
            self._last_report = None
            self._last_diagnostics = {}
            self._last_contract = {}
            self._reply_review_cache = {}
            self._review_failed_at = 0.0
            self._review_failure_reason = ""
            self._reply_review_failed_at = 0.0
            self._reply_review_failure_reason = ""
        return {"removed_samples": removed, "note": "已清空本插件的样本与报告。"}


_SESSION_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SCOPE_LABEL_RE = re.compile(r"^[0-9a-f]{12}$")


def resolve_session_digest(value: str) -> str:
    """A full digest passes through; a raw session key is hashed.

    A 12-character hex string is rejected rather than hashed. That is exactly the
    shape of the display label, and hashing one would answer "no samples" with
    HTTP 200 — a wrong answer that is indistinguishable from an empty result.
    """
    text = str(value).strip()
    if _SESSION_DIGEST_RE.match(text):
        return text
    if _SCOPE_LABEL_RE.match(text):
        raise ValueError("session 需要完整 64 位 scope_hash；"
                         "12 位十六进制是页面展示用的 scope_label，不能当查询键")
    return session_hash(text)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _redact(value: Any) -> str:
    """Keep **identity** fields out of the page while staying recognisable.

    Matches the host console's convention: short values are masked entirely and
    longer ones keep only their head and tail.

    Identity only. This is deliberately not applied to the label columns
    (predicted / expected): they hold vocabulary tokens and topic ids, and the
    masking rule turns every one of them into asterisks — "bot" and "other"
    become indistinguishable — which is a table that cannot be read.
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


__all__ = ["PLUGIN_NAME", "DynamicsLearningPlugin", "resolve_session_digest", "attribution_report"]
