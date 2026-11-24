"""Tests for shared ProviderBudget, telemetry, timezone, provider fallback, exclude_groups, and quiet_hours in Dynamics Learning."""

from __future__ import annotations

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.review_batches import ReviewDeferred
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.main import DynamicsLearning, _date_window
from astrbot_plugin_dynamics_learning.series_panels import (
    ProviderBudgetBusyError,
    is_series_group_excluded,
    is_series_in_quiet_hours,
    resolve_series_provider,
    resolve_series_timezone,
    resolve_series_timezone_name,
    series_llm_call,
)


def _chat_ctx(
    *,
    activated: bool = True,
    rhythm_timezone: str = "America/New_York",
    reply_provider: str = "shared-reply-p",
    decision_provider: str = "shared-decision-p",
    exclude_groups: list[str] | None = None,
    in_quiet_hours: bool = False,
    budget=None,
    providers: dict | None = None,
    using_provider=None,
):
    class ChatOwner:
        def get_series_shared_config(self):
            return {
                "reply_provider": reply_provider,
                "decision_provider": decision_provider,
                "rhythm_timezone": rhythm_timezone,
                "exclude_groups": list(exclude_groups or []),
                "takeover_groups": ["qq:GroupMessage:1001", "qq:GroupMessage:9001"],
                "takeover_all_groups": False,
                "persona_id": "p-main",
                "ui_language": "zh-CN",
                "quiet_hours": {
                    "daily_rhythm_enabled": True,
                    "force_sleep": False,
                    "in_quiet_hours": in_quiet_hours,
                },
            }

    provider_map = dict(providers or {})
    ctx = SimpleNamespace(
        get_all_stars=lambda: {
            "chat": SimpleNamespace(
                name="astrbot_plugin_chat_dynamics",
                activated=activated,
                star_cls=ChatOwner(),
            )
        },
        get_provider_by_id=lambda pid: provider_map.get(pid),
        get_using_provider=lambda *_a, **_kw: using_provider,
    )
    if budget is not None:
        ctx._chat_dynamics_provider_budget = budget
    return ctx


class _FakeBudget:
    supports_series_telemetry = True

    def __init__(self, *, busy: bool = False):
        self.busy = busy
        self.calls: list[dict] = []

    async def run(
        self,
        provider_id: str,
        purpose: str,
        invoke,
        *,
        plugin_id: str = "unknown",
        acquire_timeout: float | None = None,
    ):
        self.calls.append(
            {
                "provider_id": provider_id,
                "purpose": purpose,
                "plugin_id": plugin_id,
                "acquire_timeout": acquire_timeout,
            }
        )
        if self.busy:
            raise ProviderBudgetBusyError(
                "queue full", provider=provider_id, purpose=purpose, reason="queue_full"
            )
        res = invoke()
        if hasattr(res, "__await__"):
            res = await res
        return res


@pytest.mark.asyncio
async def test_learning_model_delegates_to_shared_budget_and_degrades_on_busy():
    provider = SimpleNamespace(
        provider_config={"id": "shared-reply-p", "model": "m1"},
        text_chat=AsyncMock(return_value=SimpleNamespace(completion_text='{"proposals":[]}')),
        is_native_decision_provider=False,
    )
    budget = _FakeBudget(busy=False)
    ctx = _chat_ctx(budget=budget, providers={"shared-reply-p": provider})
    plugin = DynamicsLearning(ctx, {"analysis_provider": "", "review_provider": ""})

    res = await plugin._model("analysis_provider", "sys", {"events": []})
    assert res == {"proposals": []}
    assert budget.calls[0]["plugin_id"] == "astrbot_plugin_dynamics_learning"
    assert budget.calls[0]["provider_id"] == "shared-reply-p"
    assert budget.calls[0]["purpose"] == "analysis"

    busy_budget = _FakeBudget(busy=True)
    ctx_busy = _chat_ctx(budget=busy_budget, providers={"shared-reply-p": provider})
    plugin_busy = DynamicsLearning(ctx_busy, {"analysis_provider": "", "review_provider": ""})
    with pytest.raises(ReviewDeferred, match="共享模型并发池繁忙") as exc_info:
        await plugin_busy._model("analysis_provider", "sys", {"events": []})
    assert exc_info.value.reason == "queue_full"


def test_learning_provider_fallback_chain_and_explicit_override(tmp_path):
    host_provider = SimpleNamespace(
        provider_config={"id": "host-default-p"},
        text_chat=AsyncMock(),
        is_native_decision_provider=False,
    )
    store = Store(tmp_path / "learning.sqlite3")
    try:
        # 1. Explicit wins over shared & host
        ctx_shared = _chat_ctx(using_provider=host_provider)
        eng_explicit = Engine(
            store,
            {"analysis_provider": "own-p", "review_provider": "own-rev"},
            lambda *a: None,
            host_context=ctx_shared,
        )
        assert eng_explicit.provider_for("analysis_provider") == "own-p"
        assert eng_explicit.provider_for("review_provider") == "own-rev"

        # 2. Blank inherits shared provider
        eng_shared = Engine(
            store,
            {"analysis_provider": "", "review_provider": ""},
            lambda *a: None,
            host_context=ctx_shared,
        )
        assert eng_shared.provider_for("analysis_provider") == "shared-reply-p"
        assert eng_shared.provider_for("review_provider") == "shared-decision-p"

        # 3. Standalone without Chat Dynamics falls back to host default provider
        standalone_ctx = SimpleNamespace(
            get_all_stars=lambda: {},
            get_using_provider=lambda: host_provider,
        )
        eng_host = Engine(
            store,
            {"analysis_provider": "", "review_provider": ""},
            lambda *a: None,
            host_context=standalone_ctx,
        )
        assert eng_host.provider_for("analysis_provider") == "host-default-p"
        assert eng_host.provider_for("review_provider") == "host-default-p"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_learning_standalone_fallback_without_chat_dynamics():
    standalone_ctx = SimpleNamespace(get_all_stars=lambda: {})
    assert resolve_series_timezone_name(standalone_ctx) == "Asia/Shanghai"
    assert resolve_series_provider(standalone_ctx, "") == ""
    assert is_series_group_excluded(standalone_ctx, umo="qq:GroupMessage:1001") is False
    assert is_series_in_quiet_hours(standalone_ctx) is False

    res = await series_llm_call(
        standalone_ctx,
        "p1",
        "astrbot_plugin_dynamics_learning",
        "analysis",
        lambda: SimpleNamespace(completion_text='{"proposals":[]}'),
    )
    assert res.completion_text == '{"proposals":[]}'


def test_learning_timezone_day_window_inheritance():
    ctx_ny = _chat_ctx(rhythm_timezone="America/New_York")
    tz_ny = resolve_series_timezone(ctx_ny)
    assert getattr(tz_ny, "key", "") == "America/New_York"

    ts = datetime.datetime(2026, 11, 22, 2, 0, tzinfo=datetime.timezone.utc).timestamp()
    start_ny, end_ny = _date_window("today", now_ts=ts, tz=tz_ny)
    start_sh, _ = _date_window("today", now_ts=ts, tz=resolve_series_timezone(None))
    assert start_ny != start_sh
    assert end_ny - start_ny == 86400.0


@pytest.mark.asyncio
async def test_learning_exclude_groups_blocks_engine_and_scope_status(tmp_path):
    excluded_umo = "qq:GroupMessage:9001"
    allowed_umo = "qq:GroupMessage:1001"
    ctx = _chat_ctx(exclude_groups=[excluded_umo], in_quiet_hours=True)
    assert is_series_in_quiet_hours(ctx) is True

    store = Store(tmp_path / "learning_excl.sqlite3")
    try:
        cfg = {"enabled": True, "sessions": [excluded_umo, allowed_umo]}
        eng = Engine(store, cfg, lambda *a: None, host_context=ctx)
        assert eng.allowed(excluded_umo) is False
        assert eng.allowed(allowed_umo) is True

        plugin = DynamicsLearning(ctx, cfg)
        plugin.engine = eng
        plugin.ready = True
        st_excl = plugin.get_series_scope_status(excluded_umo)
        assert st_excl["enabled"] is False
        assert st_excl["excluded_by_core"] is True
        assert "主控排除" in st_excl["summary"]

        st_ok = plugin.get_series_scope_status(allowed_umo)
        assert st_ok["enabled"] is True
        assert st_ok["excluded_by_core"] is False

        social_excl = await plugin.get_social_context({"umo": excluded_umo}, ["u1"])
        assert social_excl["available"] is False
        assert social_excl["items"] == []
    finally:
        store.close()
