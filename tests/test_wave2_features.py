"""Wave 2 tests: /我的印象 card, /群黑话, /纠正黑话, /习惯纠偏, /学习静音, Horizons proposals, persona labels, arbitrary version restore, LLM token stats, presets & schema grouping."""

from __future__ import annotations

import asyncio
import hashlib
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

from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.member_commands import (
    build_my_impression_payload,
    handle_learning_mute_command,
    query_group_jargon_list,
    submit_member_habit_correction,
    submit_member_jargon_correction,
)
from astrbot_plugin_dynamics_learning.core.presets import (
    OPERATION_PRESETS,
    detect_active_preset,
    validate_review_modes,
)
from astrbot_plugin_dynamics_learning.core.series_bridge import resolve_persona_labels
from astrbot_plugin_dynamics_learning.core.social import Social
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.core.workbench import detail as workbench_detail

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "aiocqhttp:GroupMessage:1001"
SCOPE_2 = "aiocqhttp:GroupMessage:2002"
CHARACTER = {
    "platform": "aiocqhttp",
    "umo": SCOPE,
    "persona_id": "安安",
    "instance_id": "inst-1",
}


class _SavableConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_calls = 0

    def save_config(self):
        self.save_calls += 1


def _load_main_module():
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
    msg_mod.TextPart = lambda text="": NS(text=text, mark_as_temp=lambda: None)

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


def _ev(scope=SCOPE, persona="p-hash", member="alice", text="我喜欢手冲咖啡", msg_id="m1"):
    return event_record(
        {
            "schema_version": 1,
            "kind": "observed",
            "scope": scope,
            "persona": persona,
            "member": member,
            "turn_id": "t-" + msg_id,
            "message_id": msg_id,
            "text": text,
            "host_version": "v4",
            "baseline": "b1",
            "baseline_threshold": 60,
            "mandatory": False,
            "timestamp": time.time(),
            "context": {
                "source_verified": True,
                "character_scope": {**CHARACTER, "umo": scope},
            },
        }
    )


def _seed_entry(store: Store, *, category, key, value, scope=SCOPE, persona="p-hash", member="", msg_id="m1", apply=True):
    ev = _ev(scope=scope, persona=persona, member=member or "alice", text=str(value.get("quote") if isinstance(value, dict) else value), msg_id=msg_id)
    if category == "preference":
        value = {**value, "source_id": ev["id"]}
    raw = {
        "category": category,
        "key": key,
        "member": member,
        "value": value,
        "reason": "test seed",
        "evidence_ids": [ev["id"]],
        "days": 30,
    }
    prev = next((e for e in store.entries() if e["scope"] == scope and e["persona"] == persona and e["category"] == category and e["key"] == key and e["member"] == member and e["active"]), None)
    data = candidate(raw, [ev], prev)
    mode = "auto" if category == "affection" else "human"
    prop = store.propose(data, mode, "test")
    if apply:
        prop = store.apply(prop["id"], "system:auto-affection" if category == "affection" else "human:admin", prop["updated"], automatic_affection=(category == "affection"))
    return prop


# ---------------------------------------------------------------------------
# 1. C-end /我的印象 card + /群黑话 + /纠正黑话 + /习惯纠偏 + /学习静音
# ---------------------------------------------------------------------------


def test_my_impression_card_rendering_isolation_and_mute(tmp_path):
    store = Store(tmp_path / "impression.sqlite3")
    try:
        prompt_text = "你是温暖细心的群聊伙伴安安"
        persona_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        ctx = NS(
            get_all_stars=lambda: [],
            persona_manager=NS(personas=[{"persona_id": "安安", "name": "安安", "prompt": prompt_text}]),
        )
        cfg = {"enabled": True, "sessions": [SCOPE, SCOPE_2]}
        engine = Engine(store, cfg, lambda *a: None, host_context=ctx)
        engine.social = Social(engine, ctx)
        engine.current_personas[SCOPE] = persona_hash
        engine.current_personas[SCOPE_2] = persona_hash
        engine.character_scopes[(SCOPE, persona_hash)] = dict(CHARACTER)

        _seed_entry(store, category="preference", key="咖啡", value={"topic": "手冲咖啡", "stance": "like", "condition": "", "quote": "我超喜欢手冲咖啡"}, persona=persona_hash, member="alice", msg_id="m-like")
        _seed_entry(store, category="preference", key="剧透", value={"topic": "番剧剧透", "stance": "boundary", "condition": "新番", "quote": "千万别给我剧透新番"}, persona=persona_hash, member="alice", msg_id="m-bd")
        _seed_entry(store, category="preference", key="他人秘密", value={"topic": "他人喜欢恐怖片", "stance": "like", "condition": "", "quote": "我喜欢恐怖片"}, persona=persona_hash, member="bob", msg_id="m-bob")
        _seed_entry(store, category="preference", key="跨群爱好", value={"topic": "跨群桌游", "stance": "like", "condition": "", "quote": "我喜欢跨群桌游"}, scope=SCOPE_2, persona=persona_hash, member="alice", msg_id="m-other-group")
        _seed_entry(store, category="affection", key="bot_affection", value={"experience_delta": "8", "tone": 0.3}, persona=persona_hash, member="alice", msg_id="m-aff")
        asyncio.run(engine.refresh())

        payload = build_my_impression_payload(engine, ctx, scope=SCOPE, member="alice", member_name="爱丽丝")
        assert payload["ok"] is True
        assert payload["persona_name"] == "安安"
        assert "手冲咖啡" in payload["likes"]
        assert any("番剧剧透" in b for b in payload["boundaries"])
        assert payload["affection"]["level"] > 0
        # Strict isolation: neither Bob's data nor SCOPE_2 data leaks
        assert "恐怖片" not in payload["text"]
        assert "跨群桌游" not in payload["text"]
        assert "手冲咖啡" in payload["text"]

        # PNG card rendered via Pillow
        img_path = Path(payload["image_path"])
        assert img_path.is_file()
        assert img_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

        # Enable learning mute -> impression blocked & privacy mute enforced
        mute_calls = []

        class ChatOwner:
            def get_series_privacy_muted(self, umo, member_id, scope=""):
                return {"muted": (umo, member_id, True) in mute_calls, "scopes": ["learning"]}

            async def set_series_privacy_muted(self, umo, member_id, muted, *, scopes=None, actor="", reason=""):
                mute_calls.append((umo, member_id, muted))
                return {"ok": True, "muted": muted}

        ctx_with_chat = NS(
            get_all_stars=lambda: [NS(name="astrbot_plugin_chat_dynamics", activated=True, star_cls=ChatOwner())],
            persona_manager=ctx.persona_manager,
        )
        engine._host_context = ctx_with_chat
        res_on = asyncio.run(handle_learning_mute_command(engine, ctx_with_chat, scope=SCOPE, member="alice", arg="开启"))
        assert res_on["ok"] is True and res_on["muted"] is True and res_on["synced_core"] is True
        assert engine.is_privacy_muted(SCOPE, "alice") is True
        muted_payload = build_my_impression_payload(engine, ctx_with_chat, scope=SCOPE, member="alice")
        assert muted_payload["ok"] is False and muted_payload["muted"] is True

        # Unmute restores access
        mute_calls.clear()
        res_off = asyncio.run(handle_learning_mute_command(engine, ctx_with_chat, scope=SCOPE, member="alice", arg="关闭"))
        assert res_off["ok"] is True and res_off["muted"] is False
        assert engine.is_privacy_muted(SCOPE, "alice") is False
    finally:
        store.close()


def test_jargon_lookup_and_member_correction_commands_enter_review_queue(tmp_path):
    store = Store(tmp_path / "member_cmds.sqlite3")
    try:
        cfg = {"enabled": True, "sessions": [SCOPE], "member_correction_daily_limit": 2}
        engine = Engine(store, cfg, lambda *a: None)
        engine.current_personas[SCOPE] = "anan"

        _seed_entry(store, category="jargon", key="赛博浇花", value="每天打开3D小屋看角色浇水", persona="anan", member="", msg_id="j1")
        asyncio.run(engine.refresh())

        # /群黑话 lookup
        found = query_group_jargon_list(engine, scope=SCOPE, keyword="浇花")
        assert found["ok"] is True and len(found["terms"]) == 1
        assert found["terms"][0]["term"] == "赛博浇花"

        # /纠正黑话 -> creates pending member_correction proposal
        c1 = submit_member_jargon_correction(engine, scope=SCOPE, member="alice", raw_args="赛博浇花：去小屋给花园浇水顺便逛一圈")
        assert c1["ok"] is True
        p1 = c1["proposal"]
        assert p1["state"] == "pending"
        assert p1["source_kind"] == "member_correction"
        assert p1["submitted_by"] == "member:alice"
        assert p1["category"] == "jargon"

        # /习惯纠偏 with boundary prefix -> creates structured preference proposal
        c2 = submit_member_habit_correction(engine, scope=SCOPE, member="alice", note="边界：千万别给我剧透推理小说结局")
        assert c2["ok"] is True
        p2 = c2["proposal"]
        assert p2["category"] == "preference"
        assert p2["value"]["stance"] == "boundary"
        assert p2["mode"] == "llm_then_human"

        # Daily limit reached (configured to 2 in test)
        c3 = submit_member_habit_correction(engine, scope=SCOPE, member="alice", note="回复短一点")
        assert c3["ok"] is False and "上限" in c3["text"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 2. Horizons reflection -> review queue candidates + ranked hints
# ---------------------------------------------------------------------------


def test_horizons_reflection_creates_review_candidates_under_personality_policy(tmp_path):
    store = Store(tmp_path / "horizons_w2.sqlite3")
    try:
        cfg = {"enabled": True, "sessions": [SCOPE], "affection_enabled": True}
        engine = Engine(store, cfg, lambda *a: None)
        engine.social = Social(engine)
        engine.current_personas[SCOPE] = "anan"
        engine.character_scopes[(SCOPE, "anan")] = dict(CHARACTER)

        res = engine.ingest_horizons_reflection(
            SCOPE,
            "alice",
            "胶片摄影",
            "聊到胶片机和柯达金200时特别开心，还分享了样片",
            sentiment="positive",
        )
        assert res["ok"] is True
        assert len(res["proposal_ids"]) == 2
        props = [store.get(pid) for pid in res["proposal_ids"]]
        cats = {p["category"] for p in props}
        assert cats == {"preference", "affection"}
        for p in props:
            assert p["state"] == "pending"
            assert p["mode"] in {"llm_then_human", "human"}
            assert p["source_kind"] == "horizons_reflection"
        aff_p = next(p for p in props if p["category"] == "affection")
        assert aff_p["auto_guard_reason"] == "horizons_reflection"

        # Weighted group_interests ranking: repeated approved/reflected topics rank first
        engine.ingest_horizons_reflection(SCOPE, "bob", "胶片摄影", "也想试试胶片机", sentiment="positive")
        engine.ingest_horizons_reflection(SCOPE, "bob", "桌游", "周末约一局", sentiment="positive")
        hints = engine.horizons_social_hints(SCOPE)
        assert hints["group_interests"][0] == "胶片摄影"
        assert "桌游" in hints["group_interests"]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 3. Persona label resolution, arbitrary version restore, LLM token stats & presets
# ---------------------------------------------------------------------------


def test_persona_label_resolution_handles_sha256_hashes(tmp_path):
    store = Store(tmp_path / "labels.sqlite3")
    try:
        prompt = "你是一个温和的猫耳少女角色"
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        ctx = NS(
            get_all_stars=lambda: [],
            persona_manager=NS(personas=[{"persona_id": "小橘", "name": "小橘", "prompt": prompt}]),
        )
        engine = Engine(store, {"sessions": [SCOPE]}, lambda *a: None, host_context=ctx)
        engine.character_scopes[(SCOPE, "hash-from-world")] = {**CHARACTER, "persona_id": "安安"}
        labels = resolve_persona_labels(ctx, store, engine)
        assert labels[prompt_hash] == "小橘"
        assert labels[prompt_hash[:12]] == "小橘"
        assert labels["hash-from-world"] == "安安"
    finally:
        store.close()


def test_arbitrary_historical_version_restore_and_workbench_flags(tmp_path):
    async def run():
        store = Store(tmp_path / "restore.sqlite3")
        try:
            cfg = {"enabled": True, "sessions": [SCOPE], "jargon_review": "human"}
            engine = Engine(store, cfg, lambda *a: None)
            engine.current_personas[SCOPE] = "anan"

            p1 = _seed_entry(store, category="jargon", key="摸鱼", value="版本1：上班偷懒", persona="anan", msg_id="r1")
            await engine.refresh()
            p2 = _seed_entry(store, category="jargon", key="摸鱼", value="版本2：划水休息", persona="anan", msg_id="r2")
            await engine.refresh()
            p3 = _seed_entry(store, category="jargon", key="摸鱼", value="版本3：带薪发呆", persona="anan", msg_id="r3")
            await engine.refresh()

            # Inspect detail for p1 (earlier version) -> can_restore=True, can_rollback=False
            d1 = workbench_detail(store, [SCOPE], proposal_id=p1["id"])
            assert d1["can_rollback"] is False
            assert d1["can_restore"] is True
            assert any(v["id"] == p3["id"] and v["is_current"] is True for v in d1["versions"])

            # Restore directly to p1 (skipping p2)
            p1_latest = store.get(p1["id"])
            restored = await engine.action(p1["id"], "restore", "human:admin", p1_latest["updated"], "回到最初释义")
            assert restored["state"] == "applied"
            assert restored["applied_revision"] == 4
            assert store.get(p3["id"])["state"] == "rolled_back"

            active_entry = next(e for e in engine.cache if e["id"] == p1["entry_id"])
            assert active_entry["value"] == "版本1：上班偷懒"
            assert active_entry["revision"] == 4
            assert active_entry["active"] is True

            # Recent applied list exposed in overview
            ov = await engine.overview()
            assert any(row["id"] == p1["id"] for row in ov["recent_applied"])
        finally:
            store.close()

    asyncio.run(run())


def test_llm_usage_token_stats_and_presets_governance_api(tmp_path, main_mod):
    async def run():
        store = Store(tmp_path / "usage_presets.sqlite3")
        try:
            store.console.record_llm_usage(
                scope=SCOPE, phase="analysis", ok=True, prompt_chars=440, completion_chars=220, elapsed_ms=350
            )
            store.console.record_llm_usage(
                scope=SCOPE,
                phase="review",
                ok=True,
                prompt_chars=200,
                completion_chars=100,
                prompt_tokens=120,
                completion_tokens=45,
                elapsed_ms=210,
            )
            store.console.record_llm_usage(
                scope=SCOPE, phase="maintenance", ok=False, prompt_chars=300, completion_chars=0, elapsed_ms=500
            )
            summary = store.console.llm_usage_summary(24)
            assert summary["total_calls"] == 3
            assert summary["ok_calls"] == 2
            assert summary["failed_calls"] == 1
            assert summary["by_phase"]["analysis"]["prompt_tokens"] == 200
            assert summary["by_phase"]["review"]["prompt_tokens"] == 120
            assert summary["by_phase"]["review"]["completion_tokens"] == 45
            assert summary["by_phase"]["maintenance"]["failed"] == 1

            # Presets & interactive mode switching via governance_action
            cfg = _SavableConfig({"enabled": True, "sessions": [SCOPE]})
            plugin = main_mod.DynamicsLearning(NS(get_all_stars=lambda: []), cfg)
            plugin.engine = Engine(store, cfg, lambda *a: None)
            plugin.ready = True

            body_box: dict = {}

            async def fake_serve(operation, fields=()):
                return await operation(body_box["payload"], "human:admin")

            plugin._serve = fake_serve

            assert set(OPERATION_PRESETS) == {"conservative", "balanced", "deep_resonance"}
            body_box["payload"] = {"action": "apply_preset", "preset": "deep_resonance"}
            res_p = await plugin.governance_action()
            assert res_p["active_preset"] == "deep_resonance"
            assert cfg["scene_learning_enabled"] is True
            assert cfg["expression_style_enabled"] is True
            assert cfg.save_calls == 1

            body_box["payload"] = {
                "action": "update_review_modes",
                "review_modes": {"jargon": "human", "affection": "llm_then_human"},
            }
            res_m = await plugin.governance_action()
            assert res_m["review_modes"]["jargon"] == "human"
            assert res_m["review_modes"]["affection"] == "llm_then_human"
            assert detect_active_preset(cfg) == "custom"
            assert cfg.save_calls == 2

            with pytest.raises(ValueError):
                validate_review_modes({"jargon": "auto"})
        finally:
            store.close()

    asyncio.run(run())


def test_schema_groups_and_folds_13_relation_weights_and_ui_js_syntax():
    schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
    assert "show_advanced_relation_weights" in schema
    assert schema["show_advanced_relation_weights"]["default"] is False
    assert schema["member_correction_daily_limit"]["default"] == 10

    relation_keys = [k for k in schema if k.startswith("relation_")]
    assert len(relation_keys) >= 13
    for rk in relation_keys:
        assert schema[rk].get("condition") == {"show_advanced_relation_weights": True}, rk

    # Preserves previously required keys & defaults
    assert schema["reply_projection_budget"]["default"] == 6000
    assert schema["max_jargon_per_turn"]["default"] == 4
    assert schema["audit_retention_days"]["default"] == 90

    app_js = ROOT / "pages" / "review" / "app.js"
    wb_js = ROOT / "pages" / "review" / "workbench.js"
    html = (ROOT / "pages" / "review" / "index.html").read_text(encoding="utf-8")
    assert "edit-structured" in html and "preset-list" in html and "llm-usage-box" in html and "recent-applied-panel" in html
    assert "https://" not in html and "http://" not in html

    app_src = app_js.read_text(encoding="utf-8")
    wb_src = wb_js.read_text(encoding="utf-8")
    for token in ("personaLabel", "renderStructuredEditForm", "renderPresets", "renderModesEditor", "renderRecentApplied", "llm-usage-box"):
        assert token in app_src
    assert "恢复到此历史版本" in wb_src

    node = shutil.which("node")
    if node:
        for f in (app_js, wb_js):
            chk = subprocess.run([node, "--check", str(f)], capture_output=True, text=True, check=False)
            assert chk.returncode == 0, chk.stderr
