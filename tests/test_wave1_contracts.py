"""Wave 1 tests: schema hints, query_affection gate, review risks/guard, privacy mute, scope toggle & Horizons contracts."""

from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from astrbot_plugin_dynamics_learning.core.console import Console
from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.native import (
    affection_query_allowed,
    group_projection,
    private_projection,
)
from astrbot_plugin_dynamics_learning.core.reply_projection import private_material
from astrbot_plugin_dynamics_learning.core.series_bridge import (
    is_member_privacy_muted,
    normalize_str_list,
    read_extended_shared_config,
)
from astrbot_plugin_dynamics_learning.core.social import Social
from astrbot_plugin_dynamics_learning.core.store import Store

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "aiocqhttp:GroupMessage:1001"
SCOPE_2 = "aiocqhttp:GroupMessage:2002"
CHARACTER = {
    "platform": "aiocqhttp",
    "umo": SCOPE,
    "persona_id": "anan",
    "instance_id": "inst-1",
}


class _SavableConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_calls = 0

    def save_config(self):
        self.save_calls += 1


def _load_main_module():
    """Import astrbot_plugin_dynamics_learning.main even without AstrBot SDK installed."""
    try:
        return importlib.import_module("astrbot_plugin_dynamics_learning.main"), None
    except ModuleNotFoundError:
        pass

    saved: dict[str, types.ModuleType | None] = {}
    stub_names = (
        "astrbot",
        "astrbot.api",
        "astrbot.api.event",
        "astrbot.api.star",
        "astrbot.api.web",
        "astrbot.core",
        "astrbot.core.agent",
        "astrbot.core.agent.message",
    )
    for name in stub_names:
        saved[name] = sys.modules.get(name)

    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    api_mod.AstrBotConfig = dict
    api_mod.logger = NS(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)

    event_mod = types.ModuleType("astrbot.api.event")
    event_mod.AstrMessageEvent = object

    class _Filter:
        class PlatformAdapterType:
            AIOCQHTTP = "aiocqhttp"

        class EventMessageType:
            ALL = "all"
            GROUP_MESSAGE = "group"

        @staticmethod
        def on_llm_request(**_kw):
            return lambda fn: fn

        @staticmethod
        def llm_tool(**_kw):
            return lambda fn: fn

        @staticmethod
        def command(*_a, **_kw):
            return lambda fn: fn

        @staticmethod
        def platform_adapter_type(*_a, **_kw):
            return lambda fn: fn

        @staticmethod
        def event_message_type(*_a, **_kw):
            return lambda fn: fn

    event_mod.filter = _Filter

    star_mod = types.ModuleType("astrbot.api.star")

    class _Star:
        def __init__(self, context=None):
            self.context = context

    star_mod.Context = object
    star_mod.Star = _Star
    star_mod.StarTools = NS(get_data_dir=lambda *a, **k: Path("."))

    web_mod = types.ModuleType("astrbot.api.web")
    web_mod.json_response = lambda data, status_code=200: NS(body=json.dumps(data), status_code=status_code)
    web_mod.request = None

    core_mod = types.ModuleType("astrbot.core")
    agent_mod = types.ModuleType("astrbot.core.agent")
    msg_mod = types.ModuleType("astrbot.core.agent.message")

    class _TextPart:
        def __init__(self, text=""):
            self.text = text

        def mark_as_temp(self):
            return self

    msg_mod.TextPart = _TextPart

    sys.modules["astrbot"] = astrbot_mod
    sys.modules["astrbot.api"] = api_mod
    sys.modules["astrbot.api.event"] = event_mod
    sys.modules["astrbot.api.star"] = star_mod
    sys.modules["astrbot.api.web"] = web_mod
    sys.modules["astrbot.core"] = core_mod
    sys.modules["astrbot.core.agent"] = agent_mod
    sys.modules["astrbot.core.agent.message"] = msg_mod

    mod = importlib.import_module("astrbot_plugin_dynamics_learning.main")
    return mod, saved


@pytest.fixture
def main_mod():
    mod, saved = _load_main_module()
    yield mod
    if saved is not None:
        sys.modules.pop("astrbot_plugin_dynamics_learning.main", None)
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _chat_ctx(
    *,
    takeover_groups=None,
    exclude_groups=None,
    reply_provider="shared-reply",
    decision_provider="shared-decision",
    light_analysis_provider="",
    persona_id="anan-shared",
    private_shared_group_sources=None,
    muted_pairs=None,
    extra_stars=None,
    providers=None,
    using_provider=None,
):
    muted = set(muted_pairs or ())

    class ChatOwner:
        def get_series_shared_config(self):
            return {
                "takeover_all_groups": False,
                "takeover_groups": list(takeover_groups) if takeover_groups is not None else [SCOPE],
                "exclude_groups": list(exclude_groups or []),
                "reply_provider": reply_provider,
                "decision_provider": decision_provider,
                "light_analysis_provider": light_analysis_provider,
                "persona_id": persona_id,
                "private_shared_group_sources": private_shared_group_sources
                if private_shared_group_sources is not None
                else [],
                "rhythm_timezone": "Asia/Shanghai",
                "ui_language": "zh-CN",
            }

        def get_series_privacy_muted(self, umo: str, member_id: str, scope: str = ""):
            is_muted = (umo, member_id, scope) in muted or (umo, member_id, "global") in muted
            return {
                "muted": is_muted,
                "scopes": ["learning"] if is_muted else [],
                "updated_at": "2026-11-27T00:00:00+08:00",
                "by": "admin" if is_muted else "",
            }

    stars = [
        NS(name="astrbot_plugin_chat_dynamics", activated=True, star_cls=ChatOwner()),
        *(extra_stars or ()),
    ]
    provider_map = dict(providers or {})
    return NS(
        get_all_stars=lambda: stars,
        get_provider_by_id=lambda pid: provider_map.get(pid),
        get_using_provider=lambda *_a, **_kw: using_provider,
    )


def _make_raw_event(
    *,
    scope=SCOPE,
    persona="anan",
    member="alice",
    text="我特别喜欢抹茶蛋糕，但不喜欢剧透",
    msg_id="m1",
    turn_id="t1",
):
    return {
        "schema_version": 1,
        "kind": "observed",
        "scope": scope,
        "persona": persona,
        "member": member,
        "turn_id": turn_id,
        "message_id": msg_id,
        "text": text,
        "host_version": "v4",
        "baseline": "b1",
        "baseline_threshold": 60,
        "mandatory": False,
        "timestamp": time.time(),
        "context": {
            "source_verified": True,
            "character_scope": {**CHARACTER, "umo": scope, "persona_id": persona},
        },
    }


def _make_source_event(**kwargs):
    return event_record(_make_raw_event(**kwargs))


def _seed_preference(store: Store, *, scope=SCOPE, persona="anan", member="alice", topic="抹茶", stance="like", quote="我喜欢抹茶", msg_id="m-pref"):
    ev = _make_source_event(scope=scope, persona=persona, member=member, text=quote, msg_id=msg_id, turn_id="t-" + msg_id)
    data = candidate(
        {
            "category": "preference",
            "key": topic,
            "member": member,
            "value": {
                "topic": topic,
                "stance": stance,
                "condition": "",
                "quote": quote,
                "source_id": ev["id"],
            },
            "reason": "成员明确自述",
            "evidence_ids": [ev["id"]],
            "days": 30,
        },
        [ev],
        None,
    )
    prop = store.propose(data, "human", "test")
    store.apply(prop["id"], "human:admin", prop["updated"])
    return prop


def _seed_affection(store: Store, *, scope=SCOPE, persona="anan", member="alice", delta="8", msg_id="m-aff"):
    ev = _make_source_event(scope=scope, persona=persona, member=member, text="谢谢你帮忙", msg_id=msg_id, turn_id="t-" + msg_id)
    data = candidate(
        {
            "category": "affection",
            "key": "bot_affection",
            "member": member,
            "value": {"experience_delta": delta, "tone": 0.4},
            "reason": "友好互动",
            "evidence_ids": [ev["id"]],
            "days": 30,
        },
        [ev],
        None,
    )
    prop = store.propose(data, "auto", "test")
    store.apply(prop["id"], "system:auto-affection", prop["updated"], automatic_affection=True)
    return prop


# ---------------------------------------------------------------------------
# 1. Schema hints, HTML copy & Review UI risk/guard helpers
# ---------------------------------------------------------------------------


def test_schema_and_html_hints_reflect_auto_inheritance():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    assert "自动继承" in schema["sessions"]["hint"]
    assert "显式白名单" in schema["sessions"]["hint"]
    assert "自动回退" in schema["analysis_provider"]["hint"]
    assert "轻量分析模型" in schema["analysis_provider"]["hint"]
    assert "自动回退" in schema["review_provider"]["hint"]
    assert "逗号" in schema["private_shared_group_sources"]["hint"]

    html = (ROOT / "pages" / "review" / "index.html").read_text(encoding="utf-8")
    assert "自动继承群间核心" in html
    assert "显式白名单" in html


def test_review_app_js_surfaces_risks_auto_guard_and_session_source():
    js_path = ROOT / "pages" / "review" / "app.js"
    js_src = js_path.read_text(encoding="utf-8")
    css_src = (ROOT / "pages" / "review" / "style.css").read_text(encoding="utf-8")

    for token in (
        "formatAutoGuardReason",
        "collectProposalRisks",
        "explainRiskRedline",
        "renderRiskBanner",
        "review-risk-banner",
        "sessionSourceMeta",
        "继承自群间核心",
        "显式白名单",
    ):
        assert token in js_src
    assert ".review-risk-banner" in css_src
    assert ".scope-source-banner" in css_src

    node = shutil.which("node")
    if node:
        res = subprocess.run([node, "--check", str(js_path)], capture_output=True, text=True, check=False)
        assert res.returncode == 0, res.stderr


def test_console_summary_surfaces_deduplicated_risks_and_auto_guard_reason():
    proposal = {
        "id": "p1",
        "scope": SCOPE,
        "persona": "anan",
        "member": "alice",
        "category": "affection",
        "key": "bot_affection",
        "value": {"affection_level": 12.0, "experience_delta": "15"},
        "state": "pending",
        "mode": "llm_then_human",
        "created": 1.0,
        "updated": 2.0,
        "days": 30,
        "reason": "单次互动波动大",
        "evidence": [],
        "auto_guard_reason": "spike:15>=10",
        "reviews": [
            {"actor": "llm:r1", "verdict": "defer", "reason": "单次跨度偏大", "risks": ["可能固化短期情绪", "归属存疑"]},
            {"actor": "llm:r2", "verdict": "approve", "reason": "仍有风险", "risks": ["可能固化短期情绪", "需复核隐私"]},
        ],
    }
    summary = Console.summary(proposal)
    assert summary["auto_guard_reason"] == "spike:15>=10"
    assert summary["risks"] == ["可能固化短期情绪", "需复核隐私", "归属存疑"]


# ---------------------------------------------------------------------------
# 2. Extended shared config inheritance (light_analysis_provider, persona_id, private sources)
# ---------------------------------------------------------------------------


def test_extended_shared_config_inheritance_and_comma_list_normalization(tmp_path):
    assert normalize_str_list("a, b，c\nd ; a") == ["a", "b", "c", "d"]
    assert normalize_str_list(["a", " b ", "a", "", None]) == ["a", "b"]

    store = Store(tmp_path / "shared.sqlite3")
    try:
        ctx = _chat_ctx(
            light_analysis_provider="shared-light-p",
            decision_provider="shared-decision-p",
            reply_provider="shared-reply-p",
            persona_id="anan-v4",
            private_shared_group_sources=f"{SCOPE}, {SCOPE_2}",
        )
        shared = read_extended_shared_config(ctx)
        assert shared["light_analysis_provider"] == "shared-light-p"
        assert shared["persona_id"] == "anan-v4"
        assert shared["private_shared_group_sources"] == [SCOPE, SCOPE_2]

        eng = Engine(store, {"sessions": [], "analysis_provider": "", "review_provider": ""}, lambda *a: None, host_context=ctx)
        assert eng.provider_with_source("analysis_provider") == ("shared-light-p", "shared_light")
        assert eng.provider_with_source("review_provider") == ("shared-light-p", "shared_light")
        assert eng.shared_persona_id() == "anan-v4"
        assert eng.effective_private_shared_group_sources() == [SCOPE, SCOPE_2]
        assert eng.sessions_source() == "inherited"

        # Comma-separated explicit private_shared_group_sources overrides inherited
        eng_csv = Engine(
            store,
            {"sessions": [SCOPE], "private_shared_group_sources": f"{SCOPE_2}，{SCOPE}"},
            lambda *a: None,
            host_context=ctx,
        )
        assert eng_csv.effective_private_shared_group_sources() == [SCOPE_2, SCOPE]
        assert eng_csv.sessions_source() == "explicit"

        # Native decision provider in light_analysis_provider must be skipped
        native_light_ctx = _chat_ctx(
            light_analysis_provider="jev-native",
            decision_provider="shared-decision-p",
            reply_provider="shared-reply-p",
            providers={"jev-native": NS(is_native_decision_provider=True)},
        )
        eng_safe = Engine(store, {"analysis_provider": "", "review_provider": ""}, lambda *a: None, host_context=native_light_ctx)
        assert eng_safe.provider_for("analysis_provider") == "shared-reply-p"
        assert eng_safe.provider_for("review_provider") == "shared-decision-p"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 3. query_affection gate when replace_selflearning=False
# ---------------------------------------------------------------------------


def test_query_affection_allowed_when_legacy_selflearning_inactive(tmp_path, main_mod):
    async def run():
        store = Store(tmp_path / "aff_gate.sqlite3")
        try:
            ctx = _chat_ctx(takeover_groups=[SCOPE])
            cfg = _SavableConfig({"enabled": True, "sessions": [SCOPE], "replace_selflearning": False})
            plugin = main_mod.DynamicsLearning(ctx, cfg)
            plugin.engine = Engine(store, cfg, lambda *a: None, host_context=ctx)
            plugin.engine.social = Social(plugin.engine, ctx)
            plugin.ready = True

            _seed_affection(store, scope=SCOPE, persona="anan", member="alice", delta="6")
            plugin.engine.current_personas[SCOPE] = "anan"
            await plugin.engine.refresh()

            assert affection_query_allowed(plugin) is True
            event = NS(
                unified_msg_origin=SCOPE,
                get_sender_id=lambda: "alice",
                is_admin=lambda: False,
            )
            # Even with replace_selflearning=False and no worldbook, group read-only affection query succeeds
            res = json.loads(await plugin.query_affection(event))
            assert res["available"] is True
            assert res["member"] == "alice"
            assert res["relationship"]["affection_level"] > 0

            # Non-admin still blocked from querying other members
            res_other = json.loads(await plugin.query_affection(event, user_id="bob"))
            assert res_other["available"] is False

            # When legacy self_learning IS active, query_affection must stand down
            ctx_legacy = _chat_ctx(
                takeover_groups=[SCOPE],
                extra_stars=[NS(name="astrbot_plugin_self_learning", activated=True, star_cls=object())],
            )
            plugin.context = ctx_legacy
            assert affection_query_allowed(plugin) is False
            res_blocked = json.loads(await plugin.query_affection(event))
            assert res_blocked["available"] is False
        finally:
            store.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. set_series_scope_enabled (optimistic lock) & activate_series_runtime
# ---------------------------------------------------------------------------


def test_set_series_scope_enabled_optimistic_lock_and_activate_runtime(tmp_path, main_mod):
    async def run():
        store = Store(tmp_path / "scope_toggle.sqlite3")
        try:
            ctx = _chat_ctx(takeover_groups=[SCOPE])
            cfg = _SavableConfig({"enabled": False, "sessions": []})
            plugin = main_mod.DynamicsLearning(ctx, cfg)
            plugin.engine = Engine(store, cfg, lambda *a: None, host_context=ctx)
            plugin.engine.social = Social(plugin.engine, ctx)
            plugin.ready = True

            # Master switch off initially
            st0 = plugin.get_series_scope_status(SCOPE)
            assert st0["umo"] == SCOPE
            assert st0["writable"] is True
            assert st0["mode"] == "inherited"
            assert st0["enabled"] is False
            assert st0["persona_id"] == "anan-shared"

            # Activate runtime via Setup Wizard hook
            act = plugin.activate_series_runtime()
            assert act["ok"] is True
            assert act["enabled"] is True
            assert act["changed"] is True
            assert cfg["enabled"] is True
            assert cfg.save_calls == 1

            # Idempotent second activation
            act2 = plugin.activate_series_runtime()
            assert act2["ok"] is True and act2["changed"] is False

            st1 = plugin.get_series_scope_status(SCOPE)
            assert st1["enabled"] is True
            v1 = st1["version"]

            # Wrong version -> optimistic lock conflict
            conflict = await plugin.set_series_scope_enabled(SCOPE, False, expected_version="stale-version")
            assert conflict["ok"] is False
            assert conflict["conflict"] is True
            assert conflict["reason"] == "version_conflict"
            assert conflict["version"] == v1

            # Disable an inherited group -> stays disabled (not re-inherited!)
            off = await plugin.set_series_scope_enabled(SCOPE, False, expected_version=v1)
            assert off["ok"] is True
            assert off["enabled"] is False
            assert off["version"] != v1
            assert plugin.engine.allowed(SCOPE) is False
            assert SCOPE in cfg["excluded_sessions"]

            # Re-enable it using new version
            on = await plugin.set_series_scope_enabled(SCOPE, True, expected_version=off["version"])
            assert on["ok"] is True
            assert on["enabled"] is True
            assert plugin.engine.allowed(SCOPE) is True
            assert SCOPE not in cfg.get("excluded_sessions", [])

            # Enable a brand-new group not in inherited list
            v_new = plugin.get_series_scope_status(SCOPE_2)["version"]
            add2 = await plugin.set_series_scope_enabled(SCOPE_2, True, expected_version=v_new)
            assert add2["ok"] is True
            assert add2["enabled"] is True
            assert SCOPE in plugin._effective_sessions()
            assert SCOPE_2 in plugin._effective_sessions()
        finally:
            store.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. Cross-plugin privacy mute across observation, extraction & projection
# ---------------------------------------------------------------------------


def test_privacy_mute_blocks_observation_extraction_and_projection(tmp_path, main_mod):
    async def run():
        store = Store(tmp_path / "privacy.sqlite3")
        try:
            muted_pairs = {(SCOPE, "muted_u", "learning")}
            ctx = _chat_ctx(
                takeover_groups=[SCOPE],
                private_shared_group_sources=[SCOPE],
                muted_pairs=muted_pairs,
            )
            assert is_member_privacy_muted(ctx, SCOPE, "muted_u", "learning") is True
            assert is_member_privacy_muted(ctx, SCOPE, "alice", "learning") is False

            cfg = _SavableConfig(
                {
                    "enabled": True,
                    "sessions": [SCOPE],
                    "replace_selflearning": True,
                    "private_shared_group_sources": [SCOPE],
                }
            )
            plugin = main_mod.DynamicsLearning(ctx, cfg)
            model_calls = []

            async def fake_model(role, instruction, data):
                model_calls.append(data)
                return {
                    "proposals": [
                        {
                            "category": "style",
                            "key": "tone",
                            "member": "muted_u",
                            "value": "语气直接",
                            "reason": "观察到",
                            "evidence_ids": [ev_muted["id"]],
                            "days": 30,
                        }
                    ]
                }

            plugin.engine = Engine(store, cfg, fake_model, host_context=ctx)
            plugin.engine.social = Social(plugin.engine, ctx)
            plugin.ready = True

            # 1) Observation blocked for muted user, allowed for normal user
            raw_muted = _make_raw_event(scope=SCOPE, persona="anan", member="muted_u", text="不要学习我的消息", msg_id="m-muted", turn_id="t-muted")
            raw_ok = _make_raw_event(scope=SCOPE, persona="anan", member="alice", text="今天天气不错呀，聊点有趣的", msg_id="m-ok", turn_id="t-ok")
            ev_muted = event_record(raw_muted)
            assert plugin.engine.ingest(raw_muted) is False
            assert plugin.engine.capture_message({"scope": SCOPE, "member": "muted_u", "source_id": "m-muted", "text": "hi", "eligible": True, "created": time.time()}) is False
            assert plugin.engine.social.observe({"scope": SCOPE, "member": "muted_u", "source_id": "m-muted", "name": "M", "text": "hi", "directed": []}) is False
            assert plugin.engine.ingest(raw_ok) is True

            # 2) Projection & affection blocked for muted user even if pre-existing entries exist
            _seed_preference(store, scope=SCOPE, persona="anan", member="muted_u", topic="抹茶", stance="like", quote="我喜欢抹茶", msg_id="m-p1")
            _seed_preference(store, scope=SCOPE, persona="anan", member="alice", topic="咖啡", stance="like", quote="我喜欢咖啡", msg_id="m-p2")
            _seed_affection(store, scope=SCOPE, persona="anan", member="muted_u", delta="5", msg_id="m-a1")
            _seed_affection(store, scope=SCOPE, persona="anan", member="alice", delta="5", msg_id="m-a2")
            plugin.engine.current_personas[SCOPE] = "anan"
            plugin.engine.character_scopes[(SCOPE, "anan")] = dict(CHARACTER)
            await plugin.engine.refresh()

            ctx_muted = plugin.engine.context(scope=SCOPE, persona="anan", member="muted_u", baseline="b1", host_version="v4", query="抹茶")
            assert "抹茶" not in ctx_muted["context"]
            assert "bot_to_current_member" not in ctx_muted["context"]
            assert not any(row.get("member") == "muted_u" and row.get("category") == "preference" for row in ctx_muted["reply_projection"]["items"])

            ctx_alice = plugin.engine.context(scope=SCOPE, persona="anan", member="alice", baseline="b1", host_version="v4", query="咖啡")
            assert "咖啡" in ctx_alice["context"]
            assert "bot_to_current_member" in ctx_alice["context"]

            assert plugin.get_affection_for_decision(scope=SCOPE, persona="anan", member="muted_u") is None
            assert plugin.get_affection_for_decision(scope=SCOPE, persona="anan", member="alice") is not None

            gp = group_projection(plugin.engine, CHARACTER, ["muted_u", "alice"])
            assert [row["subject_id"] for row in gp] == ["alice"]

            priv_scope = {**CHARACTER, "umo": "aiocqhttp:FriendMessage:muted_u", "instance_id": "priv"}
            assert private_projection(plugin.engine, priv_scope, "muted_u") == []
            assert private_material(plugin.engine, priv_scope["umo"], "anan", "muted_u", priv_scope)["items"] == []

            priv_alice = {**CHARACTER, "umo": "aiocqhttp:FriendMessage:alice", "instance_id": "priv"}
            assert len(private_projection(plugin.engine, priv_alice, "alice")) == 1
            assert len(private_material(plugin.engine, priv_alice["umo"], "anan", "alice", priv_alice)["items"]) >= 1

            # 3) Extraction skips muted member if pre-existing event was in store
            store.ingest(ev_muted)
            analyze_res = await plugin.engine.analyze(SCOPE, "anan")
            assert analyze_res["proposals"] == 0
            assert model_calls == []
        finally:
            store.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# 6. Horizons bidirectional contracts
# ---------------------------------------------------------------------------


def test_horizons_social_hints_and_reflection_contract(tmp_path, main_mod):
    async def run():
        store = Store(tmp_path / "horizons.sqlite3")
        try:
            ctx = _chat_ctx(takeover_groups=[SCOPE], muted_pairs={(SCOPE, "muted_u", "learning")})
            cfg = _SavableConfig({"enabled": True, "sessions": [SCOPE]})
            plugin = main_mod.DynamicsLearning(ctx, cfg)
            plugin.engine = Engine(store, cfg, lambda *a: None, host_context=ctx)
            plugin.ready = True
            plugin.engine.current_personas[SCOPE] = "anan"

            _seed_preference(store, scope=SCOPE, persona="anan", member="alice", topic="胶片摄影", stance="like", quote="私密原话：我超爱拍胶片摄影", msg_id="h1")
            _seed_preference(store, scope=SCOPE, persona="anan", member="alice", topic="剧透", stance="boundary", quote="千万别给我剧透结局", msg_id="h2")
            _seed_preference(store, scope=SCOPE, persona="anan", member="bob", topic="恐怖片", stance="dislike", quote="我挺讨厌恐怖片的", msg_id="h3")
            _seed_preference(store, scope=SCOPE, persona="anan", member="muted_u", topic="隐藏话题", stance="boundary", quote="不要透露这个", msg_id="h4")
            await plugin.engine.refresh()

            hints = plugin.get_horizons_social_hints(SCOPE)
            assert hints["group_interests"] == ["胶片摄影"]
            assert hints["member_dislikes"] == {"bob": ["恐怖片"]}
            assert hints["member_boundaries"] == {"alice": ["剧透"]}
            # Sanitized: raw quotes and muted user must never leak
            serialized = json.dumps(hints, ensure_ascii=False)
            assert "私密原话" not in serialized
            assert "muted_u" not in serialized
            assert "隐藏话题" not in serialized

            # Member-filtered query
            alice_hints = plugin.get_horizons_social_hints(SCOPE, member_id="alice")
            assert alice_hints["group_interests"] == ["胶片摄影"]
            assert alice_hints["member_dislikes"] == {}
            assert alice_hints["member_boundaries"] == {"alice": ["剧透"]}

            # Ingest positive reflection -> recorded and surfaced in group_interests
            ing = plugin.ingest_horizons_reflection(
                SCOPE, "alice", "手冲咖啡", "一起聊了耶加雪菲的冲煮温度，气氛很好", sentiment="positive"
            )
            assert ing["ok"] is True
            assert ing["accepted"] is True
            assert ing["topic"] == "手冲咖啡"
            assert ing["sentiment"] == "positive"

            hints_after = plugin.get_horizons_social_hints(SCOPE)
            assert "手冲咖啡" in hints_after["group_interests"]

            # Privacy-muted member reflection is rejected
            muted_ing = plugin.ingest_horizons_reflection(SCOPE, "muted_u", "秘密主题", "总结", sentiment="positive")
            assert muted_ing["ok"] is False
            assert muted_ing["reason"] == "privacy_muted"

            # Unenabled scope and invalid sentiment rejected
            bad_scope = plugin.ingest_horizons_reflection(SCOPE_2, "alice", "主题", "总结")
            assert bad_scope["ok"] is False and bad_scope["reason"] == "scope_not_enabled"
            bad_sent = plugin.ingest_horizons_reflection(SCOPE, "alice", "主题", "总结", sentiment="weird")
            assert bad_sent["ok"] is False and bad_sent["reason"] == "invalid_sentiment"

            # Purge cleans horizons_reflections too
            await plugin.engine.purge(SCOPE)
            assert store.recent_horizons_reflections(SCOPE) == []
        finally:
            store.close()

    asyncio.run(run())
