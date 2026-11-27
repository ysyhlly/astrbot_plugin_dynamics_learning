"""Wave 3 tests: Dynamics Event Bus subscription, jargon heat metrics & co-reading reflection hooks."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.event_bus import (
    EVENT_BUS_DAILY_PROPOSAL_CAP,
    SUBSCRIBED_EVENTS,
    DynamicsEventBridge,
)
from astrbot_plugin_dynamics_learning.core.store import Store

SCOPE = "aiocqhttp:GroupMessage:3001"
SCOPE_EXCLUDED = "aiocqhttp:GroupMessage:3999"
CHARACTER = {
    "platform": "aiocqhttp",
    "umo": SCOPE,
    "persona_id": "安安",
    "instance_id": "inst-wave3",
}


class _FakeChatDynamicsOwner:
    """Minimal chat_dynamics stub exposing Dynamics Event Bus and privacy mute APIs."""

    def __init__(self, *, muted_members: set[tuple[str, str]] | None = None, excluded_umos: set[str] | None = None):
        self.subscriptions: dict[str, tuple[str, object, str]] = {}
        self._seq = 0
        self.muted_members = set(muted_members or set())
        self.excluded_umos = set(excluded_umos or set())
        self.config = {"group_sessions": [SCOPE], "persona_id": "安安"}

    def subscribe_dynamics_event(self, event_type: str, handler, *, owner: str = "") -> str:
        self._seq += 1
        token = f"tok-{self._seq}"
        self.subscriptions[token] = (event_type, handler, owner)
        return token

    def unsubscribe_dynamics_event(self, token: str) -> bool:
        return self.subscriptions.pop(token, None) is not None

    def emit(self, event_type: str, payload: dict) -> list:
        results = []
        for ev_type, handler, _owner in list(self.subscriptions.values()):
            if ev_type == event_type:
                results.append(handler(payload))
        return results

    def get_series_privacy_muted(self, umo: str, member_id: str, scope_kind: str = "learning") -> bool:
        return (str(umo), str(member_id)) in self.muted_members

    def get_series_excluded_groups(self) -> list[str]:
        return sorted(self.excluded_umos)


def _make_context(chat_owner: _FakeChatDynamicsOwner | None):
    stars = []
    if chat_owner is not None:
        stars.append(
            NS(
                name="astrbot_plugin_chat_dynamics",
                star_cls=chat_owner,
                module_path="astrbot_plugin_chat_dynamics.main",
                activated=True,
            )
        )
    return NS(
        get_all_stars=lambda: list(stars),
        get_all_providers=lambda: [],
        get_using_provider=lambda: None,
        register_web_api=lambda *a, **k: None,
        get_config=lambda: {},
    )


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

    class _StarTools:
        _dir: Path | None = None

        @classmethod
        def get_data_dir(cls, _name=None):
            return cls._dir or Path(".")

    star_mod.Context = object
    star_mod.Star = _Star
    star_mod.StarTools = _StarTools
    star_mod.register = lambda *a, **k: (lambda cls: cls)

    web_mod = types.ModuleType("astrbot.api.web")
    web_mod.json_response = lambda data, status_code=200: NS(body=json.dumps(data), status_code=status_code)
    web_mod.request = None

    core_mod = types.ModuleType("astrbot.core")
    agent_mod = types.ModuleType("astrbot.core.agent")
    msg_mod = types.ModuleType("astrbot.core.agent.message")
    msg_mod.TextPart = lambda text="": NS(text=text, mark_as_temp=lambda: None)

    stubs = {
        "astrbot": astrbot_mod,
        "astrbot.api": api_mod,
        "astrbot.api.event": event_mod,
        "astrbot.api.star": star_mod,
        "astrbot.api.web": web_mod,
        "astrbot.core": core_mod,
        "astrbot.core.agent": agent_mod,
        "astrbot.core.agent.message": msg_mod,
    }
    sys.modules.update(stubs)
    sys.modules.pop("astrbot_plugin_dynamics_learning.main", None)
    try:
        main_mod = importlib.import_module("astrbot_plugin_dynamics_learning.main")
    except Exception:
        sys.modules.pop("astrbot_plugin_dynamics_learning.main", None)
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        raise
    return main_mod, saved


@pytest.fixture()
def main_mod():
    mod, saved = _load_main_module()
    try:
        yield mod
    finally:
        if saved is not None:
            sys.modules.pop("astrbot_plugin_dynamics_learning.main", None)
            for name, prev in saved.items():
                if prev is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = prev


def _make_engine(tmp_path: Path, chat_owner: _FakeChatDynamicsOwner | None = None, **cfg_extra):
    store = Store(tmp_path / "wave3.sqlite3")
    cfg = {
        "enabled": True,
        "sessions": [SCOPE],
        "affection_enabled": True,
        "affection_auto_review_threshold": 10,
        "affection_auto_daily_limit": 20,
        "affection_auto_daily_max_count": 6,
        **cfg_extra,
    }
    ctx = _make_context(chat_owner)
    engine = Engine(store, cfg, lambda *a, **k: None, host_context=ctx)
    engine.current_personas[SCOPE] = "安安"
    engine.character_scopes[(SCOPE, "安安")] = dict(CHARACTER)
    return engine, store, ctx


def _approve_jargon(store: Store, *, scope: str, persona: str, term: str, meaning: str, created_ts: float | None = None):
    ev = event_record(
        {
            "schema_version": 1,
            "scope": scope,
            "persona": persona,
            "member": "u1",
            "message_id": f"m-{term}",
            "turn_id": f"t-{term}",
            "kind": "observed",
            "baseline": "b",
            "host_version": "1",
            "baseline_threshold": 0,
            "mandatory": False,
            "text": f"大家最近都在说{term}",
            "timestamp": created_ts or time.time(),
            "context": {},
        }
    )
    data = candidate(
        {
            "category": "jargon",
            "key": term,
            "member": "",
            "value": meaning,
            "reason": "群内高频梗词",
            "days": 30,
            "evidence_ids": [ev["id"]],
        },
        [ev],
        None,
    )
    prop = store.propose(data, "human", "test:jargon")
    if created_ts is not None:
        with store.transaction():
            store.db.execute("UPDATE proposals SET created=? WHERE id=?", (created_ts, prop["id"]))
    applied = store.apply(prop["id"], "human:admin", prop["updated"])
    return applied


# ---------------------------------------------------------------------------
# 1. Dynamics Event Bus integration
# ---------------------------------------------------------------------------


def test_event_bus_attach_idempotent_detach_and_missing_owner(tmp_path):
    # Graceful no-op when chat_dynamics is absent
    engine_no_owner, store_no, ctx_no = _make_engine(tmp_path / "no_owner", chat_owner=None)
    bridge_no = DynamicsEventBridge(engine_no_owner, ctx_no)
    assert bridge_no.attach() == []
    assert bridge_no.status()["connected"] is False
    assert bridge_no.detach() == 0
    store_no.close()

    # Dynamic subscribe, idempotent re-attach, and clean detach
    chat_owner = _FakeChatDynamicsOwner()
    engine, store, ctx = _make_engine(tmp_path / "with_owner", chat_owner=chat_owner)
    bridge = DynamicsEventBridge(engine, ctx)
    engine.event_bus = bridge

    attached = bridge.attach()
    assert set(attached) == set(SUBSCRIBED_EVENTS)
    assert len(chat_owner.subscriptions) == 3
    assert bridge.status()["connected"] is True

    # Idempotent re-attach does not duplicate subscriptions
    attached_again = bridge.attach()
    assert set(attached_again) == set(SUBSCRIBED_EVENTS)
    assert len(chat_owner.subscriptions) == 3

    removed = bridge.detach()
    assert removed == 3
    assert len(chat_owner.subscriptions) == 0
    assert bridge.status()["connected"] is False
    store.close()


def test_event_bus_vibe_transition_and_overview(tmp_path):
    chat_owner = _FakeChatDynamicsOwner()
    engine, store, ctx = _make_engine(tmp_path, chat_owner=chat_owner)
    bridge = DynamicsEventBridge(engine, ctx)
    engine.event_bus = bridge
    bridge.attach()

    results = chat_owner.emit(
        "vibe_transition",
        {"umo": SCOPE, "old_vibe": "casual", "new_vibe": "serious", "serious_mode": True},
    )
    assert len(results) == 1
    assert results[0]["ok"] is True
    assert engine.scope_vibes[SCOPE]["new_vibe"] == "serious"
    assert engine.scope_vibes[SCOPE]["serious_mode"] is True

    ov = asyncio.run(engine.overview())
    assert ov["event_bus"]["connected"] is True
    assert ov["event_bus"]["received_counts"]["vibe_transition"] == 1
    assert ov["event_bus"]["recent_count"] == 1
    store.close()


def test_event_bus_topic_archived_privacy_mute_and_daily_cap(tmp_path):
    chat_owner = _FakeChatDynamicsOwner(
        muted_members={(SCOPE, "muted_user")},
        excluded_umos={SCOPE_EXCLUDED},
    )
    engine, store, ctx = _make_engine(tmp_path, chat_owner=chat_owner)
    bridge = DynamicsEventBridge(engine, ctx)
    engine.event_bus = bridge
    bridge.attach()

    # Excluded / disabled scope is ignored
    res_excl = chat_owner.emit(
        "topic_archived",
        {
            "umo": SCOPE_EXCLUDED,
            "topic_id": "t-excl",
            "summary": "讨论手冲咖啡参数",
            "participants": ["u1", "u2"],
            "message_count": 8,
        },
    )[0]
    assert res_excl["proposals"] == 0
    assert res_excl["reason"] == "scope_not_enabled"

    # Active scope: creates reviewed relationship proposals and skips muted_user
    res = chat_owner.emit(
        "topic_archived",
        {
            "umo": SCOPE,
            "topic_id": "topic-coffee",
            "summary": "群友热烈讨论浅烘耶加雪菲手冲水温与研磨度",
            "participants": ["alice", "muted_user", "bob"],
            "message_count": 12,
        },
    )[0]
    assert res["ok"] is True
    assert res["proposals"] == 2
    props = [store.get(pid) for pid in res["proposal_ids"]]
    assert {p["member"] for p in props} == {"alice", "bob"}
    assert all(p["category"] == "relationship" for p in props)
    assert all(p["state"] == "pending" for p in props)
    assert all(p["mode"] in {"llm_then_human", "human"} for p in props)
    assert all(p["source_kind"] == "event_bus:topic_archived" for p in props)

    # Enforce daily proposal cap (EVENT_BUS_DAILY_PROPOSAL_CAP = 6)
    res2 = chat_owner.emit(
        "topic_archived",
        {
            "umo": SCOPE,
            "topic_id": "topic-film",
            "summary": "讨论胶片摄影冲洗与扫底色彩",
            "participants": ["u3", "u4", "u5", "u6"],
            "message_count": 15,
        },
    )[0]
    assert res2["proposals"] == EVENT_BUS_DAILY_PROPOSAL_CAP - 2  # 4 more => hits 6

    res3 = chat_owner.emit(
        "topic_archived",
        {
            "umo": SCOPE,
            "topic_id": "topic-extra",
            "summary": "超出日上限的新话题",
            "participants": ["u7", "u8"],
            "message_count": 10,
        },
    )[0]
    assert res3["proposals"] == 0
    assert res3["reason"] == "daily_cap_reached"
    store.close()


def test_event_bus_game_state_changed_only_proposes_on_completion_and_respects_mute(tmp_path):
    chat_owner = _FakeChatDynamicsOwner(muted_members={(SCOPE, "muted_p")})
    engine, store, ctx = _make_engine(tmp_path, chat_owner=chat_owner)
    bridge = DynamicsEventBridge(engine, ctx)
    engine.event_bus = bridge
    bridge.attach()

    # In-progress game: records state but does not create proposals
    res_running = chat_owner.emit(
        "game_state_changed",
        {
            "umo": SCOPE,
            "active": True,
            "engine": "turtle_soup",
            "activity_id": "soup-1",
            "title": "午夜列车之谜",
            "phase": "playing",
            "participants": ["alice", "bob"],
        },
    )[0]
    assert res_running["ok"] is True
    assert res_running["proposals"] == 0
    assert res_running["reason"] == "in_progress"
    assert engine.active_games[SCOPE]["active"] is True
    assert engine.active_games[SCOPE]["title"] == "午夜列车之谜"

    # Completed game: creates reviewed relationship proposals excluding muted_p
    res_done = chat_owner.emit(
        "game_state_changed",
        {
            "umo": SCOPE,
            "active": False,
            "engine": "turtle_soup",
            "activity_id": "soup-1",
            "title": "午夜列车之谜",
            "phase": "completed",
            "participants": [{"user_id": "alice"}, {"user_id": "muted_p"}, {"user_id": "charlie"}],
        },
    )[0]
    assert res_done["ok"] is True
    assert res_done["proposals"] == 2
    props = [store.get(pid) for pid in res_done["proposal_ids"]]
    assert {p["member"] for p in props} == {"alice", "charlie"}
    assert all(p["source_kind"] == "event_bus:game_state_changed" for p in props)
    assert all(p["state"] == "pending" for p in props)
    store.close()


def test_plugin_terminate_detaches_event_bus(main_mod, tmp_path):
    chat_owner = _FakeChatDynamicsOwner()
    engine, store, ctx = _make_engine(tmp_path, chat_owner=chat_owner)
    plugin = main_mod.DynamicsLearning(ctx, engine.config)
    plugin.engine = engine
    plugin.event_bus = DynamicsEventBridge(engine, ctx)
    plugin.engine.event_bus = plugin.event_bus
    plugin.event_bus.attach()
    assert len(chat_owner.subscriptions) == 3

    asyncio.run(plugin.terminate())
    assert len(chat_owner.subscriptions) == 0
    assert plugin.closed is True


# ---------------------------------------------------------------------------
# 2. Culture jargon heat metrics (get_culture_jargon_v1)
# ---------------------------------------------------------------------------


def test_get_culture_jargon_v1_returns_heat_metrics_and_stays_backwards_compatible(main_mod, tmp_path):
    chat_owner = _FakeChatDynamicsOwner()
    engine, store, ctx = _make_engine(tmp_path, chat_owner=chat_owner)
    plugin = main_mod.DynamicsLearning(ctx, engine.config)
    plugin.engine = engine
    plugin.ready = True

    t_old = time.time() - 86400 * 5
    t_new = time.time() - 3600
    applied_hot = _approve_jargon(store, scope=SCOPE, persona="安安", term="早C晚A", meaning="早上咖啡晚上小酌", created_ts=t_old)
    _approve_jargon(store, scope=SCOPE, persona="安安", term="冷门老梗", meaning="很少提及的旧词", created_ts=t_new)
    asyncio.run(engine.refresh())
    engine.current_personas[SCOPE] = "安安"

    # Record recent mentions in events + reply usage refs for "早C晚A"
    now = time.time()
    for i in range(4):
        store.ingest(
            event_record(
                {
                    "schema_version": 1,
                    "scope": SCOPE,
                    "persona": "安安",
                    "member": f"u{i}",
                    "message_id": f"msg-hot-{i}",
                    "turn_id": f"turn-hot-{i}",
                    "kind": "observed",
                    "baseline": "b",
                    "host_version": "1",
                    "baseline_threshold": 0,
                    "mandatory": False,
                    "text": f"今天又是早C晚A的一天 #{i}",
                    "timestamp": now - i * 60,
                    "context": {},
                }
            )
        )
    for idx in range(2):
        store.console.record_usage(
            {
                "schema_version": 1,
                "scope": SCOPE,
                "member": f"u{idx}",
                "turn_id": f"turn-usage-{idx}",
                "material_hash": f"mh-{idx}",
                "stage": "provider_kwargs_after_boundaries",
                "learning": [{"source_id": applied_hot["id"], "category": "jargon", "revision": "1"}],
                "profiles": [],
                "memory_refs": [],
                "omitted": [],
                "rhythm": {},
            },
            {SCOPE},
        )

    out = plugin.get_culture_jargon_v1(scope=SCOPE, limit=10)
    assert out["schema_version"] == 1
    assert out["scope"] == SCOPE
    assert out["evidence"] is False
    terms = out["terms"]
    assert len(terms) == 2

    # Hottest term sorted first
    hot = terms[0]
    cold = terms[1]
    assert hot["name"] == "早C晚A"
    assert hot["meaning"] == "早上咖啡晚上小酌"
    assert hot["revision"] == 1
    assert hot["recent_use_count"] >= 6  # 4 events + 2 reply_usage_refs
    assert hot["heat_score"] > cold["heat_score"]
    assert 0.0 < hot["heat_score"] <= 100.0
    assert hot["first_seen_at"] is not None
    assert abs(hot["first_seen_at"] - t_old) < 2.0
    assert isinstance(hot["first_seen_iso"], str) and len(hot["first_seen_iso"]) > 10

    assert cold["name"] == "冷门老梗"
    assert cold["recent_use_count"] == 0
    assert cold["first_seen_at"] is not None
    store.close()


# ---------------------------------------------------------------------------
# 3. Co-reading multi-member reflection & affection guard hooks
# ---------------------------------------------------------------------------


def test_ingest_horizons_reflection_co_reading_multi_member_guards_and_mute(main_mod, tmp_path):
    chat_owner = _FakeChatDynamicsOwner(muted_members={(SCOPE, "muted_reader")})
    engine, store, ctx = _make_engine(tmp_path, chat_owner=chat_owner)
    plugin = main_mod.DynamicsLearning(ctx, engine.config)
    plugin.engine = engine
    plugin.ready = True

    # Spike delta (12 >= threshold 10) across multiple co-reading participants + 1 muted member
    res = plugin.ingest_horizons_reflection(
        SCOPE,
        "alice",
        "三体共读第三周",
        "大家一起拆解了黑暗森林法则与面壁者博弈",
        "positive",
        participants=["bob", "muted_reader"],
        activity_kind="reading",
        experience_delta="12",
    )
    assert res["ok"] is True
    assert res["accepted"] is True
    assert res["affected_members"] == ["alice", "bob"]
    assert res["skipped_muted"] == ["muted_reader"]
    assert set(res["per_member_proposals"].keys()) == {"alice", "bob"}

    # Each eligible member should get preference + affection + co-reading relationship proposals (3 each = 6 total)
    assert len(res["proposal_ids"]) == 6
    all_props = [store.get(pid) for pid in res["proposal_ids"]]
    assert all(p["state"] == "pending" for p in all_props)
    assert all(p["mode"] in {"llm_then_human", "human"} for p in all_props)
    assert all(p["source_kind"] == "co_reading_reflection" for p in all_props)

    # Verify categories per member
    for member in ("alice", "bob"):
        m_cats = {store.get(pid)["category"] for pid in res["per_member_proposals"][member]}
        assert m_cats == {"preference", "affection", "relationship"}
        assert res["guard_reasons"][member].startswith("spike:")

    aff_props = [p for p in all_props if p["category"] == "affection"]
    assert len(aff_props) == 2
    for ap in aff_props:
        assert ap["auto_guard_reason"].startswith("spike:")
        assert str(ap["value"]["experience_delta"]) == "12"

    # Backwards-compatible 5-arg call still works seamlessly
    res_legacy = plugin.ingest_horizons_reflection(
        SCOPE,
        "charlie",
        "海岛旅行攻略",
        "分享了冲绳浮潜经验",
        "positive",
    )
    assert res_legacy["ok"] is True
    assert res_legacy["affected_members"] == ["charlie"]
    assert res_legacy["skipped_muted"] == []
    legacy_cats = {store.get(pid)["category"] for pid in res_legacy["proposal_ids"]}
    assert legacy_cats == {"preference", "affection"}
    store.close()
