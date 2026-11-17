"""Tests for read-only series shared defaults inheritance in Dynamics Learning."""

from __future__ import annotations

from types import SimpleNamespace

from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store


def _chat_context(
    *,
    activated: bool = True,
    reply_provider: str = "chat-reply-p",
    decision_provider: str = "chat-decision-p",
    takeover_groups: list[str] | None = None,
    decision_native: bool = False,
):
    class ChatOwner:
        def get_series_shared_config(self):
            return {
                "takeover_all_groups": False,
                "takeover_groups": takeover_groups
                if takeover_groups is not None
                else ["qq:GroupMessage:1001"],
                "exclude_groups": [],
                "reply_provider": reply_provider,
                "decision_provider": decision_provider,
                "persona_id": "p-main",
                "bot_names": ["AnAn"],
                "rhythm_timezone": "Asia/Shanghai",
                "quiet_hours": {"daily_rhythm_enabled": True},
                "ui_language": "zh-CN",
            }

    def get_provider(pid: str):
        if pid == decision_provider:
            return SimpleNamespace(is_native_decision_provider=decision_native)
        return SimpleNamespace(is_native_decision_provider=False)

    return SimpleNamespace(
        get_all_stars=lambda: {
            "chat": SimpleNamespace(
                name="astrbot_plugin_chat_dynamics",
                activated=activated,
                star_cls=ChatOwner(),
            )
        },
        get_provider_by_id=get_provider,
    )


def test_learning_engine_inherits_series_defaults_read_only(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    ctx = _chat_context()
    cfg = {"analysis_provider": "", "review_provider": "", "sessions": []}
    engine = Engine(store, cfg, lambda *a: None, host_context=ctx)

    assert engine.provider_for("analysis_provider") == "chat-reply-p"
    assert engine.provider_for("review_provider") == "chat-decision-p"
    assert engine.effective_sessions() == ["qq:GroupMessage:1001"]
    assert engine.allowed("qq:GroupMessage:1001") is True
    # Read-only: companion config itself must remain untouched.
    assert cfg == {"analysis_provider": "", "review_provider": "", "sessions": []}
    store.close()


def test_learning_engine_explicit_config_takes_priority(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    ctx = _chat_context()
    cfg = {
        "analysis_provider": "own-analysis",
        "review_provider": "own-review",
        "sessions": ["qq:GroupMessage:9999"],
    }
    engine = Engine(store, cfg, lambda *a: None, host_context=ctx)

    assert engine.provider_for("analysis_provider") == "own-analysis"
    assert engine.provider_for("review_provider") == "own-review"
    assert engine.effective_sessions() == ["qq:GroupMessage:9999"]
    assert engine.allowed("qq:GroupMessage:1001") is False
    assert engine.allowed("qq:GroupMessage:9999") is True
    store.close()


def test_learning_engine_safe_fallback_when_chat_dynamics_missing_or_disabled(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    cfg = {"analysis_provider": "", "review_provider": "", "sessions": []}

    missing_ctx = SimpleNamespace(get_all_stars=lambda: {})
    engine_missing = Engine(store, cfg, lambda *a: None, host_context=missing_ctx)
    assert engine_missing.provider_for("analysis_provider") == ""
    assert engine_missing.provider_for("review_provider") == ""
    assert engine_missing.effective_sessions() == []

    disabled_ctx = _chat_context(activated=False)
    engine_disabled = Engine(store, cfg, lambda *a: None, host_context=disabled_ctx)
    assert engine_disabled.provider_for("analysis_provider") == ""
    assert engine_disabled.provider_for("review_provider") == ""
    assert engine_disabled.effective_sessions() == []

    # Native decision providers from chat_dynamics must not bind to ordinary review_provider.
    native_ctx = _chat_context(decision_native=True)
    engine_native = Engine(store, cfg, lambda *a: None, host_context=native_ctx)
    assert engine_native.provider_for("review_provider") == "chat-reply-p"
    store.close()


def test_series_panels_presence_supports_custom_dir_and_defaults(tmp_path):
    from astrbot_plugin_dynamics_learning.series_panels import presence

    custom_world = tmp_path / "astrbot_plugin_dynamics_world-main"
    (custom_world / "pages" / "world").mkdir(parents=True)
    (custom_world / "pages" / "world" / "index.html").write_text("<h1>w</h1>", "utf-8")

    class ChatOwner:
        def get_series_shared_config(self):
            return {"reply_provider": "shared-r", "takeover_groups": ["qq:GroupMessage:7"]}

    ctx = SimpleNamespace(
        get_all_stars=lambda: {
            "chat": SimpleNamespace(
                name="astrbot_plugin_chat_dynamics",
                activated=True,
                star_cls=ChatOwner(),
            ),
            "world": SimpleNamespace(
                name="astrbot_plugin_dynamics_world",
                root_dir_name="astrbot_plugin_dynamics_world-main",
                activated=True,
                star_cls=SimpleNamespace(),
            ),
        }
    )
    data = presence(ctx, "astrbot_plugin_dynamics_learning", root=tmp_path)
    assert data["schema_version"] == 1
    world_entry = next(
        row for row in data["plugins"] if row["id"] == "astrbot_plugin_dynamics_world"
    )
    assert world_entry["can_open"] is True
    assert data["series_defaults"]["reply_provider"] == "shared-r"
    assert data["series_defaults"]["takeover_groups"] == ["qq:GroupMessage:7"]


def test_learning_review_and_overview_work_with_inherited_config(tmp_path):
    import asyncio
    from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record

    store = Store(tmp_path / "learning-review.sqlite3")
    ctx = _chat_context()
    cfg = {"analysis_provider": "", "review_provider": "", "sessions": [], "style_review": "llm"}

    async def model(role, instruction, data):
        assert role == "review_provider"
        return {"verdict": "approve", "reason": "核对通过", "risks": []}

    engine = Engine(store, cfg, model, host_context=ctx)
    ev = event_record(
        dict(
            schema_version=1,
            kind="decision",
            scope="qq:GroupMessage:1001",
            persona="p-main",
            member="alice",
            turn_id="0:m1",
            message_id="m1",
            host_version="host1",
            baseline="config1",
            baseline_threshold=70,
            text="请简短一些",
            mandatory=False,
            backend="jev",
        )
    )
    prop = store.propose(
        candidate(
            dict(
                category="style",
                key="length",
                value="回答偏好简短",
                member="alice",
                reason="明确要求",
                evidence_ids=[ev["id"]],
                days=30,
            ),
            [ev],
            None,
        ),
        "llm",
        "test",
    )

    async def run():
        queue_before = await engine.review_queue(view="all")
        assert queue_before["total"] == 1
        reviewed = await engine.review(prop["id"])
        assert reviewed["state"] == "applied"
        overview = await engine.overview()
        assert overview["sessions"] == ["qq:GroupMessage:1001"]

    asyncio.run(run())
    store.close()
