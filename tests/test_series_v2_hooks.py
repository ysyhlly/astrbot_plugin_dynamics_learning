from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace as NS

from astrbot_plugin_dynamics_learning import series_panels as panels
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.main import PLUGIN, DynamicsLearning


def _make_plugin(tmp_path: Path, sessions: list[str]) -> DynamicsLearning:
    store = Store(tmp_path / "learning.sqlite3")
    config = {"sessions": list(sessions), "replace_selflearning": True, "jargon_review": "human"}
    engine = Engine(store, config, lambda *_args, **_kwargs: None)
    plugin = DynamicsLearning(NS(), config)
    plugin.engine = engine
    plugin.ready = True
    plugin.closed = False
    return plugin


def _seed_proposal(
    store: Store,
    *,
    scope: str,
    persona: str = "p1",
    member: str = "u1",
    category: str = "jargon",
    key: str = "赛博浇花",
    value: object = "每天打开3D小屋看角色浇水",
    apply: bool = False,
) -> dict:
    ev = {
        "id": f"ev-{key}-{member}",
        "message_id": f"msg-{key}-{member}",
        "turn_id": f"turn-{key}-{member}",
        "scope": scope,
        "persona": persona,
        "member": member if member else "u1",
        "kind": "observed",
        "text": f"原话关于{key}",
        "received_at": time.time(),
    }
    mode = "auto" if category == "affection" else "human"
    data = {
        "scope": scope,
        "persona": persona,
        "member": member,
        "category": category,
        "key": key,
        "value": value,
        "reason": f"观察到 {key}",
        "evidence": [ev],
        "days": 30,
        "baseline": "",
        "host_version": "",
    }
    proposal = store.propose(data, mode, "test:seed")
    if apply:
        store.apply(proposal["id"], "human:tester", proposal["updated"], automatic_affection=(category == "affection"))
        proposal = store.get(proposal["id"])
    return proposal


def test_presence_collects_badge_and_pulse_with_ttl_cache_and_isolation(tmp_path):
    panels.invalidate_summary_cache()
    calls = []

    def summary():
        calls.append(1)
        return {
            "badge": {"count": 3, "tone": "warm", "label": "3 项待审"},
            "pulse": {"headline": "队列空闲", "updated_at": "2026-11-21T02:00:00+08:00"},
        }

    def broken():
        raise RuntimeError("secret-token")

    async def async_summary():
        calls.append("async")
        return {"badge": {"count": 99, "tone": "danger", "label": "bad"}}

    stars = [
        NS(name="astrbot_plugin_dynamics_learning", activated=True, star_cls=NS(get_series_summary=summary)),
        NS(name="astrbot_plugin_dynamics_horizons", activated=True, star_cls=NS(get_series_summary=broken)),
        NS(name="astrbot_plugin_dynamics_memory", activated=True, star_cls=NS(get_series_summary=async_summary)),
    ]
    context = NS(get_all_stars=lambda: stars)

    first = panels.presence(context, PLUGIN, tmp_path)
    second = panels.presence(context, PLUGIN, tmp_path)
    rows = {r["id"]: r for r in first["plugins"]}

    assert rows["astrbot_plugin_chat_dynamics"]["label"] == "对话主控"
    assert rows["astrbot_plugin_group_chronicle"]["label"] == "群文化沿革"
    assert rows["astrbot_plugin_dynamics_learning"]["badge"] == {"count": 3, "tone": "warm", "label": "3 项待审"}
    assert rows["astrbot_plugin_dynamics_learning"]["pulse"]["headline"] == "队列空闲"
    assert rows["astrbot_plugin_dynamics_horizons"]["badge"] is None
    assert rows["astrbot_plugin_dynamics_memory"]["badge"] is None
    assert rows["astrbot_plugin_dynamics_world"]["badge"] is None
    # Cached across immediate second call; async hook was never executed.
    assert calls == [1]
    assert {r["id"]: r for r in second["plugins"]}["astrbot_plugin_dynamics_learning"]["badge"]["count"] == 3
    panels.invalidate_summary_cache()


def test_learning_v2_hooks_not_ready_and_populated(tmp_path):
    scope = "aiocqhttp:GroupMessage:1001"
    other = "aiocqhttp:GroupMessage:2002"
    plugin = _make_plugin(tmp_path, [scope])
    try:
        # Not-ready standalone degradation
        plugin.ready = False
        not_ready_summary = plugin.get_series_summary()
        assert not_ready_summary["badge"]["tone"] == "muted"
        assert not_ready_summary["badge"]["count"] == 0
        assert plugin.get_series_scope_status(scope)["enabled"] is False
        assert plugin.series_search_items("赛博") == []
        assert plugin.get_journal_events() == []

        plugin.ready = True
        empty_summary = plugin.get_series_summary()
        assert empty_summary["badge"]["count"] == 0
        assert empty_summary["badge"]["tone"] == "accent"

        # Seed pending proposal + applied entry + unauthorized scope record
        pending = _seed_proposal(
            plugin.engine.store,
            scope=scope,
            persona="anan-v1",
            member="",
            category="jargon",
            key="赛博浇花",
            value="每天打开3D小屋看看角色有没有浇水",
            apply=False,
        )
        applied = _seed_proposal(
            plugin.engine.store,
            scope=scope,
            persona="anan-v1",
            member="u42",
            category="relationship",
            key="深夜配色",
            value="喜欢深夜聊像素配色，常以妙啊收尾",
            apply=True,
        )
        _seed_proposal(
            plugin.engine.store,
            scope=other,
            persona="anan-v1",
            member="",
            category="jargon",
            key="赛博秘密",
            value="未授权群的内容",
            apply=False,
        )
        plugin.engine.current_personas[scope] = "anan-v1"

        summary = plugin.get_series_summary()
        assert summary["badge"]["count"] == 1  # ready_count in authorized scope only
        assert summary["ready_count"] == 1
        assert summary["stale_count"] == 0
        assert summary["badge"]["tone"] == "warm"
        assert "待审" in summary["badge"]["label"]
        assert summary["pulse"]["updated_at"]

        status_ok = plugin.get_series_scope_status(scope)
        assert status_ok["enabled"] is True
        assert status_ok["supported"] is True
        assert status_ok["persona_id"] == "anan-v1"
        assert status_ok["activity_today"] >= 2
        assert "待审 1" in status_ok["summary"]
        assert "有效资料 1" in status_ok["summary"]

        status_off = plugin.get_series_scope_status(other)
        assert status_off["enabled"] is False
        assert status_off["supported"] is True

        # Federated search respects allowed_scopes and finds both pending & active entries
        hits_pending = plugin.series_search_items("赛博", scope="", limit=8)
        assert len(hits_pending) == 1
        assert hits_pending[0]["kind"] == "learning_proposal"
        assert hits_pending[0]["plugin_id"] == PLUGIN
        assert hits_pending[0]["page"] == "review"
        assert hits_pending[0]["deep_link"]["id"] == pending["id"]

        hits_entry = plugin.series_search_items("像素配色", scope=scope, limit=8)
        assert len(hits_entry) == 1
        assert hits_entry[0]["kind"] == "learning_entry"
        assert hits_entry[0]["deep_link"]["tab"] == "entries"

        assert plugin.series_search_items("赛博", scope=other) == []
        assert plugin.series_search_items("") == []

        # Today's Living Journal events
        events = plugin.get_journal_events(persona="anan-v1", scope=scope, date="today", limit=10)
        ids = {e["id"] for e in events}
        assert pending["id"] in ids
        assert applied["id"] in ids
        assert all(e["category"] == "learning" for e in events)
        assert all(e["source_label"] == "群间学习" for e in events)
        assert all(e["target_plugin"] == PLUGIN and e["target_route"] == "review" for e in events)
        assert plugin.get_journal_events(scope=other) == []
    finally:
        plugin.engine.store.close()


def test_review_page_v2_structure_and_offline_integrity():
    root = Path(__file__).resolve().parents[1] / "pages" / "review"
    html = (root / "index.html").read_text(encoding="utf-8")
    css = (root / "style.css").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")

    # 5 core tabs + 5 ops tabs inside #nav-more-menu
    assert 'id="nav-more-menu"' in html
    assert 'id="tab-pending-badge"' in html
    assert 'class="info-popover"' in html
    before_more, after_more = html.split('id="nav-more-menu"', 1)
    for core_tab in ("home", "proposals", "entries", "social", "profiles"):
        assert f'data-tab="{core_tab}"' in before_more
    for ops_tab in ("preview", "library", "usage", "history", "scopes"):
        assert f'data-tab="{ops_tab}"' in after_more

    # Side-by-side before/after journal cards & dual-segment affection meter
    assert "review-journal-diff" in js and "review-journal-diff" in css
    assert "affection-meter" in js and "affection-seg-base" in js and "affection-seg-tone" in js
    assert "affection-seg-base" in css and "affection-seg-tone" in css

    # Strictly offline (no external CDN URLs)
    assert "https://" not in html and "http://" not in html


def test_master_switch_off_matrix_status_matches_engine_allowed(tmp_path):
    scope = "aiocqhttp:GroupMessage:1001"
    other = "aiocqhttp:GroupMessage:2002"
    plugin = _make_plugin(tmp_path, [scope])
    try:
        plugin.config["enabled"] = False
        _seed_proposal(
            plugin.engine.store,
            scope=scope,
            persona="anan-v1",
            key="测试提案",
            value="待审内容",
            apply=False,
        )
        plugin.engine.current_personas[scope] = "anan-v1"

        assert plugin.engine.allowed(scope) is False

        # Configured scope with master switch off: NOT enabled, but explicitly marked configured.
        configured_status = plugin.get_series_scope_status(scope)
        assert configured_status["enabled"] is False
        assert configured_status["configured"] is True
        assert configured_status["master_enabled"] is False
        assert configured_status["persona_id"] == "anan-v1"
        assert "总开关已关闭" in configured_status["summary"]
        assert "已配置该群" in configured_status["summary"]

        # Unconfigured scope with master switch off.
        other_status = plugin.get_series_scope_status(other)
        assert other_status["enabled"] is False
        assert other_status["configured"] is False
        assert other_status["master_enabled"] is False
        assert "总开关已关闭" in other_status["summary"]

        # Empty scope summary also reflects master switch off.
        root_status = plugin.get_series_scope_status("")
        assert root_status["enabled"] is False
        assert root_status["master_enabled"] is False
        assert "总开关已关闭" in root_status["summary"]

        # Badge & pulse summary also reflects master switch off.
        summary = plugin.get_series_summary()
        assert summary["badge"]["tone"] == "muted"
        assert "总开关已关闭" in summary["badge"]["label"]
        assert "总开关已关闭" in summary["pulse"]["headline"]

        # Re-enabling master switch restores active state consistent with engine.allowed.
        plugin.config["enabled"] = True
        assert plugin.engine.allowed(scope) is True
        restored = plugin.get_series_scope_status(scope)
        assert restored["enabled"] is True
        assert restored["configured"] is True
        assert restored["master_enabled"] is True
    finally:
        plugin.engine.store.close()


def test_journal_and_search_persona_isolation_and_unbound_scope(tmp_path):
    scope = "aiocqhttp:GroupMessage:3003"
    plugin = _make_plugin(tmp_path, [scope])
    try:
        p_a = _seed_proposal(
            plugin.engine.store,
            scope=scope,
            persona="persona_a",
            key="黑话A",
            value="角色A学到的黑话",
            apply=False,
        )
        p_b = _seed_proposal(
            plugin.engine.store,
            scope=scope,
            persona="persona_b",
            key="黑话B",
            value="角色B学到的黑话",
            apply=True,
        )

        # Unbound group (current_personas has no entry) with empty persona MUST return [] (no relaxation).
        assert plugin.get_journal_events(persona="", scope=scope, date="today") == []
        assert plugin.get_journal_events(persona="__none__", scope=scope, date="today") == []
        assert plugin.get_journal_events(persona="[%None]", scope=scope, date="today") == []

        # Explicit persona strictly isolates records in the same group after persona switch.
        ev_a = plugin.get_journal_events(persona="persona_a", scope=scope, date="today")
        assert [e["id"] for e in ev_a] == [p_a["id"]]
        ev_b = plugin.get_journal_events(persona="persona_b", scope=scope, date="today")
        assert [e["id"] for e in ev_b] == [p_b["id"]]

        # When group is bound to persona_b, empty persona resolves to persona_b only (never leaks persona_a).
        plugin.engine.current_personas[scope] = "persona_b"
        ev_bound = plugin.get_journal_events(persona="", scope=scope, date="today")
        assert [e["id"] for e in ev_bound] == [p_b["id"]]

        # Global query (no scope, no persona) returns allowed scopes across personas.
        ev_global = plugin.get_journal_events(persona="", scope="", date="today")
        assert {e["id"] for e in ev_global} == {p_a["id"], p_b["id"]}

        # Federated search respects optional persona filter across both proposals and active entries.
        hits_a = plugin.series_search_items("黑话", scope=scope, persona="persona_a")
        assert [h["deep_link"]["id"] for h in hits_a] == [p_a["id"]]
        hits_b = plugin.series_search_items("黑话", scope=scope, persona="persona_b")
        assert len(hits_b) == 1 and hits_b[0]["kind"] == "learning_entry"
        assert plugin.series_search_items("黑话", scope=scope, persona="__none__") == []
    finally:
        plugin.engine.store.close()


def test_journal_date_window_utc8_consistency_and_explicit_bounds(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from astrbot_plugin_dynamics_learning import main as main_mod

    # 2026-11-20 15:30 UTC == 2026-11-20 23:30+08:00 (previous Shanghai day)
    # 2026-11-20 16:30 UTC == 2026-11-21 00:30+08:00 (current Shanghai day)
    prev_shanghai_ts = datetime(2026, 11, 20, 15, 30, tzinfo=timezone.utc).timestamp()
    curr_shanghai_ts = datetime(2026, 11, 20, 16, 30, tzinfo=timezone.utc).timestamp()
    now_utc_ts = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc).timestamp()

    scope = "aiocqhttp:GroupMessage:4004"
    plugin = _make_plugin(tmp_path, [scope])
    try:
        p_prev = _seed_proposal(
            plugin.engine.store,
            scope=scope,
            persona="anan",
            key="昨日提案",
            value="上海昨日深夜",
            apply=False,
        )
        p_curr = _seed_proposal(
            plugin.engine.store,
            scope=scope,
            persona="anan",
            key="今日提案",
            value="上海今日凌晨",
            apply=False,
        )
        # Update timestamps directly to straddle UTC+8 midnight.
        with plugin.engine.store.lock, plugin.engine.store.db:
            plugin.engine.store.db.execute(
                "UPDATE proposals SET updated=? WHERE id=?", (prev_shanghai_ts, p_prev["id"])
            )
            plugin.engine.store.db.execute(
                "UPDATE proposals SET updated=? WHERE id=?", (curr_shanghai_ts, p_curr["id"])
            )

        monkeypatch.setattr(main_mod.time, "time", lambda: now_utc_ts)

        expected_start = datetime(2026, 11, 21, 0, 0, tzinfo=timezone(timedelta(hours=8))).timestamp()
        assert main_mod._date_window("today") == (expected_start, expected_start + 86400.0)
        assert main_mod._date_window("2026-11-21") == (expected_start, expected_start + 86400.0)

        ev_today = plugin.get_journal_events(persona="anan", scope=scope, date="today")
        assert [e["id"] for e in ev_today] == [p_curr["id"]]

        ev_date = plugin.get_journal_events(persona="anan", scope=scope, date="2026-11-21")
        assert [e["id"] for e in ev_date] == [p_curr["id"]]

        ev_prev = plugin.get_journal_events(persona="anan", scope=scope, date="2026-11-20")
        assert [e["id"] for e in ev_prev] == [p_prev["id"]]

        # Explicit start_ts/end_ts overrides date.
        ev_explicit = plugin.get_journal_events(
            persona="anan",
            scope=scope,
            date="2026-11-21",
            start_ts=prev_shanghai_ts - 60,
            end_ts=prev_shanghai_ts + 60,
        )
        assert [e["id"] for e in ev_explicit] == [p_prev["id"]]

        # Invalid date or bounds return [].
        assert plugin.get_journal_events(persona="anan", scope=scope, date="not-a-date") == []
        assert plugin.get_journal_events(persona="anan", scope=scope, start_ts=100.0, end_ts=100.0) == []
        assert plugin.get_journal_events(persona="anan", scope=scope, start_ts=100.0, end_ts=None) == []

        # activity_today in scope status also uses UTC+8 day boundary.
        status = plugin.get_series_scope_status(scope)
        assert status["activity_today"] == 1
    finally:
        plugin.engine.store.close()
