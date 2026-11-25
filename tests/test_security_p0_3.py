"""Batch 1 security and anti-pollution regression tests (P0-3a/b/c + empty dashboard username)."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.profiles import statistics
from astrbot_plugin_dynamics_learning.core.social import Social, character_key
from astrbot_plugin_dynamics_learning.core.store import Conflict, Store
from test_social import CHARACTER, SCOPE, evidence, legacy


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "security_p0_3.sqlite3")
    yield s
    s.close()


def _verified_sources(count=10, *, member="alice", scope=SCOPE, persona="fingerprint", verified=True):
    rows = []
    base = time.time() - 300
    for idx in range(count):
        ev = event_record(
            {
                "schema_version": 1,
                "kind": "observed",
                "scope": scope,
                "persona": persona,
                "member": member,
                "turn_id": f"turn:{member}:{idx}",
                "message_id": f"msg:{member}:{idx}",
                "host_version": "host1",
                "baseline": "base1",
                "baseline_threshold": 70,
                "text": f"短句测试{idx}！",
                "mandatory": False,
                "timestamp": base + idx * 5,
                "context": {"source_verified": verified, "character_scope": CHARACTER},
            }
        )
        rows.append(ev)
    return rows


def _affection_event(store, mid, *, member="alice"):
    ev = {**evidence(mid, member=member), "received_at": time.time(), "timestamp": time.time()}
    store.ingest(ev)
    return ev


def _affection_raw(ev, delta, *, tone=0.1, member="alice"):
    return {
        "category": "affection",
        "key": "bot_affection",
        "member": member,
        "value": {"experience_delta": delta, "tone": tone},
        "reason": "真实互动",
        "evidence_ids": [ev["id"]],
        "days": 30,
    }


# ---------------------------------------------------------------------------
# P0-3b: typing_habit string bypass hardening
# ---------------------------------------------------------------------------


def test_typing_habit_string_requires_ten_verified_native_messages():
    few = _verified_sources(9)
    raw = {
        "category": "typing_habit",
        "key": "punctuation",
        "member": "alice",
        "value": "喜欢使用感叹号",
        "reason": "观察总结",
        "evidence_ids": [e["id"] for e in few],
        "days": 30,
    }
    with pytest.raises(ValueError, match="ten distinct"):
        candidate(raw, few, None)

    unverified = _verified_sources(10, verified=False)
    with pytest.raises(ValueError, match="ten distinct"):
        candidate({**raw, "evidence_ids": [e["id"] for e in unverified]}, unverified, None)

    ten = _verified_sources(10, verified=True)
    with pytest.raises(ValueError, match="reserved"):
        candidate({**raw, "key": "observed_typing", "evidence_ids": [e["id"] for e in ten]}, ten, None)

    accepted = candidate({**raw, "evidence_ids": [e["id"] for e in ten]}, ten, None)
    assert accepted["value"] == "喜欢使用感叹号"
    assert accepted["profile_policy"] == 1


def test_typing_habit_string_cannot_auto_apply_under_llm_mode(store):
    async def run():
        ten = _verified_sources(10, verified=True)
        for ev in ten:
            store.ingest(ev)
        raw_str = {
            "category": "typing_habit",
            "key": "punctuation",
            "member": "alice",
            "value": "喜欢短句配感叹号",
            "reason": "10条独立原生消息统计",
            "evidence_ids": [e["id"] for e in ten],
            "days": 30,
        }
        data = candidate(raw_str, ten, None)

        async def model(role, instruction, payload):
            assert role == "review_provider"
            return {"verdict": "approve", "reason": "来源充分", "risks": []}

        engine = Engine(store, {"sessions": [SCOPE], "typing_habit_review": "llm", "review_provider": "test"}, model)
        await engine.refresh()
        engine.current_personas[SCOPE] = "fingerprint"

        # Effective mode for free-text typing_habit is escalated to llm_then_human.
        assert engine.proposal_mode(data) == "llm_then_human"
        p = store.propose(data, engine.proposal_mode(data), "llm:analysis")
        assert p["mode"] == "llm_then_human"

        # adopt_profile_llm_mode must not downgrade free-text typing_habit to llm auto-apply.
        adopted = store.adopt_profile_llm_mode(p["id"], p["updated"])
        assert adopted["mode"] == "llm_then_human"

        # LLM review approves, but MUST NOT auto-apply without human confirmation.
        reviewed = await engine.review(p["id"])
        assert reviewed["state"] == "reviewed"
        assert not store.entries()

        # apply_approved_profiles must skip free-text typing_habit.
        assert (await engine.apply_approved_profiles())["applied"] == 0
        assert not store.entries()

        # Store.apply enforces human actor directly as defense in depth.
        with pytest.raises(Conflict, match="human approval"):
            store.apply(reviewed["id"], "llm:test", reviewed["updated"])

        # Review queue surfaces it as ready for human confirmation, not automatic.
        queue = await engine.review_queue(scope=SCOPE, view="ready")
        assert queue["total"] == 1 and queue["proposals"][0]["id"] == reviewed["id"]

        # Human admin approval succeeds.
        applied = await engine.action(reviewed["id"], "approve", "human:admin", reviewed["updated"])
        assert applied["state"] == "applied"
        assert len(store.entries()) == 1

    asyncio.run(run())


def test_typing_habit_dict_statistics_still_auto_applies_under_llm_mode(store):
    async def run():
        ten = _verified_sources(10, verified=True)
        for ev in ten:
            store.ingest(ev)
        raw_dict = {
            "category": "typing_habit",
            "key": "observed_typing",
            "member": "alice",
            "value": {"kind": "observed_typing"},
            "reason": "原生统计",
            "evidence_ids": [e["id"] for e in ten],
            "days": 30,
        }
        data = candidate(raw_dict, ten, None)
        assert data["value"] == statistics(ten)

        async def model(role, instruction, payload):
            return {"verdict": "approve", "reason": "统计可靠", "risks": []}

        engine = Engine(store, {"sessions": [SCOPE], "typing_habit_review": "llm", "review_provider": "test"}, model)
        await engine.refresh()
        engine.current_personas[SCOPE] = "fingerprint"
        assert engine.proposal_mode(data) == "llm"
        p = store.propose(data, engine.proposal_mode(data), "system:profiles")
        applied = await engine.review(p["id"])
        assert applied["state"] == "applied"
        assert len(store.entries()) == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# P0-3c: affection_review="auto" anti-pollution guardrails
# ---------------------------------------------------------------------------


def test_auto_affection_spike_escalates_to_llm_then_human_review(store):
    async def run():
        legacy(store)
        store.bind_legacy_social(SCOPE, "fingerprint", character_key(CHARACTER))
        before = store.entries()[0]["value"]["affection_exp"]
        ev = _affection_event(store, "spike1")
        calls = []

        async def model(role, instruction, payload):
            calls.append(role)
            if role == "analysis_provider":
                return {"proposals": [_affection_raw(ev, 15)]}
            return {"verdict": "approve", "reason": "复审确认真实事件", "risks": ["幅度较大"]}

        engine = Engine(
            store,
            {"sessions": [SCOPE], "affection_review": "auto", "review_provider": "rev"},
            model,
        )
        await engine.refresh()
        await engine.analyze(SCOPE, "fingerprint")

        # Must NOT auto-apply; must run independent LLM review and wait for human approval.
        assert store.entries()[0]["value"]["affection_exp"] == before
        p = store.proposals()[0]
        assert p["state"] == "reviewed"
        assert p["mode"] == "llm_then_human"
        assert p["auto_guard_reason"].startswith("spike:")
        assert calls == ["analysis_provider", "review_provider"]

        # Store.apply blocks automatic_affection=True directly too.
        with pytest.raises(Conflict, match="requires review"):
            store.apply(p["id"], "system:auto-affection", p["updated"], automatic_affection=True)

        # Appears in human review queue as ready.
        q = await engine.review_queue(scope=SCOPE, view="ready")
        assert q["total"] == 1 and q["proposals"][0]["id"] == p["id"]

        # Human admin can approve it after LLM review.
        applied = await engine.action(p["id"], "approve", "human:admin", p["updated"])
        assert applied["state"] == "applied"
        assert float(store.entries()[0]["value"]["affection_exp"]) == pytest.approx(float(before) + 15.0)

    asyncio.run(run())


def test_auto_affection_daily_limit_stops_farming_and_rollback_restores_budget(store):
    async def run():
        legacy(store)
        store.bind_legacy_social(SCOPE, "fingerprint", character_key(CHARACTER))
        before = float(store.entries()[0]["value"]["affection_exp"])
        deltas = [8, 8, 8, 5]
        idx = 0

        async def model(role, instruction, payload):
            if role == "review_provider":
                return {"verdict": "reject", "reason": "单日频繁刷好感", "risks": ["疑似刷分"]}
            return {"proposals": [_affection_raw(payload["events"][-1], deltas[idx])]}

        engine = Engine(
            store,
            {"sessions": [SCOPE], "affection_review": "auto", "review_provider": "rev"},
            model,
        )
        await engine.refresh()
        engine.current_personas[SCOPE] = "fingerprint"

        for i in range(3):
            idx = i
            _affection_event(store, f"farm-{i}")
            await engine.analyze(SCOPE, "fingerprint")

        assert float(store.entries()[0]["value"]["affection_exp"]) == pytest.approx(before + 24.0)
        assert all(p["state"] == "applied" for p in store.proposals()[:3])

        # 4th delta (+5) would push daily cumulative abs delta to 29 > 25 -> escalated & rejected by review LLM.
        idx = 3
        _affection_event(store, "farm-3")
        await engine.analyze(SCOPE, "fingerprint")
        latest = store.proposals()[0]
        assert latest["auto_guard_reason"].startswith("daily_limit:")
        assert latest["state"] == "rejected"
        assert float(store.entries()[0]["value"]["affection_exp"]) == pytest.approx(before + 24.0)

        # Rolling back one applied +8 restores 8 of daily budget so a +5 can succeed.
        applied_third = next(p for p in store.proposals() if p["state"] == "applied")
        await engine.action(applied_third["id"], "rollback", "human:admin", applied_third["updated"])
        assert float(store.entries()[0]["value"]["affection_exp"]) == pytest.approx(before + 16.0)
        _affection_event(store, "farm-after-rollback")
        await engine.analyze(SCOPE, "fingerprint")
        assert float(store.entries()[0]["value"]["affection_exp"]) == pytest.approx(before + 21.0)

    asyncio.run(run())


def test_apply_pending_affection_escalates_large_legacy_pending_proposal(store):
    async def run():
        ev = _affection_event(store, "legacy-large")
        p = store.propose(candidate(_affection_raw(ev, 12), [ev], None), "auto", "legacy")
        engine = Engine(store, {"sessions": [SCOPE], "affection_review": "auto"}, None)
        await engine.refresh()
        engine.current_personas[SCOPE] = "fingerprint"

        res = await engine.apply_pending_affection()
        assert res == {"applied": 0, "stale": 0}
        updated = store.get(p["id"])
        assert updated["state"] == "pending"
        assert updated["mode"] == "llm_then_human"
        assert updated["auto_guard_reason"].startswith("spike:")
        assert not store.entries()

    asyncio.run(run())


def test_auto_affection_daily_count_cap_blocks_tone_spam(store):
    async def run():
        legacy(store)
        store.bind_legacy_social(SCOPE, "fingerprint", character_key(CHARACTER))
        step = 0

        async def model(role, instruction, payload):
            if role == "review_provider":
                return {"verdict": "reject", "reason": "高频刷语气", "risks": []}
            tone = 1.0 if step % 2 == 0 else -1.0
            return {"proposals": [_affection_raw(payload["events"][-1], 0, tone=tone)]}

        engine = Engine(
            store,
            {"sessions": [SCOPE], "affection_review": "auto", "review_provider": "rev"},
            model,
        )
        await engine.refresh()
        engine.current_personas[SCOPE] = "fingerprint"
        for i in range(6):
            step = i
            _affection_event(store, f"tone-{i}")
            await engine.analyze(SCOPE, "fingerprint")
        assert sum(1 for p in store.proposals() if p["state"] == "applied") == 6

        # 7th auto-affection within 24h hits daily_count cap and is escalated.
        step = 6
        _affection_event(store, "tone-6")
        await engine.analyze(SCOPE, "fingerprint")
        latest = store.proposals()[0]
        assert latest["auto_guard_reason"].startswith("daily_count:")
        assert latest["state"] == "rejected"

    asyncio.run(run())


# ---------------------------------------------------------------------------
# P0-3a: query_group_social authorization & empty dashboard username fix
# ---------------------------------------------------------------------------


def _load_plugin_class_offline(monkeypatch):
    """Import DynamicsLearning even when astrbot SDK is absent, for offline unit testing."""
    try:
        from astrbot_plugin_dynamics_learning import main as main_mod
        return main_mod
    except ImportError:
        pass

    astrbot_pkg = types.ModuleType("astrbot")
    api_pkg = types.ModuleType("astrbot.api")
    api_pkg.AstrBotConfig = dict
    api_pkg.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)

    event_pkg = types.ModuleType("astrbot.api.event")
    event_pkg.AstrMessageEvent = object

    class _Filter:
        class PlatformAdapterType:
            AIOCQHTTP = "aiocqhttp"

        class EventMessageType:
            ALL = "all"
            GROUP_MESSAGE = "group"

        @staticmethod
        def platform_adapter_type(*a, **k):
            return lambda fn: fn

        @staticmethod
        def event_message_type(*a, **k):
            return lambda fn: fn

        @staticmethod
        def on_llm_request(*a, **k):
            return lambda fn: fn

        @staticmethod
        def llm_tool(*a, **k):
            return lambda fn: fn

    event_pkg.filter = _Filter

    star_pkg = types.ModuleType("astrbot.api.star")

    class _Star:
        def __init__(self, context):
            self.context = context

    star_pkg.Context = object
    star_pkg.Star = _Star
    star_pkg.StarTools = SimpleNamespace(get_data_dir=lambda *a: Path("."))

    web_pkg = types.ModuleType("astrbot.api.web")
    web_pkg.request = SimpleNamespace(_request=None, method="GET", query={}, headers={})
    web_pkg.json_response = lambda data, status_code=200: SimpleNamespace(
        status_code=status_code, body=json.dumps(data).encode("utf-8")
    )

    dash_pkg = types.ModuleType("astrbot.dashboard")
    dash_api_pkg = types.ModuleType("astrbot.dashboard.api")
    dash_auth_pkg = types.ModuleType("astrbot.dashboard.api.auth")

    async def _default_auth(_raw):
        return None

    dash_auth_pkg.require_dashboard_user = _default_auth

    for name, mod in {
        "astrbot": astrbot_pkg,
        "astrbot.api": api_pkg,
        "astrbot.api.event": event_pkg,
        "astrbot.api.star": star_pkg,
        "astrbot.api.web": web_pkg,
        "astrbot.dashboard": dash_pkg,
        "astrbot.dashboard.api": dash_api_pkg,
        "astrbot.dashboard.api.auth": dash_auth_pkg,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    import importlib
    sys.modules.pop("astrbot_plugin_dynamics_learning.main", None)
    return importlib.import_module("astrbot_plugin_dynamics_learning.main")


def test_query_group_social_enforces_admin_for_others_and_overview(store, monkeypatch):
    main_mod = _load_plugin_class_offline(monkeypatch)

    async def run():
        legacy(store, member="alice")
        legacy(store, member="bob")
        legacy(store, category="jargon", member="", key="老六", value="出其不意")
        store.bind_legacy_social(SCOPE, "fingerprint", character_key(CHARACTER))
        engine = Engine(store, {"sessions": [SCOPE]}, None)
        await engine.refresh()
        engine.current_personas[SCOPE] = "fingerprint"
        social = Social(engine)
        engine.social = social
        await social.initialize()
        await social.graph.record_message(
            SCOPE,
            "alice",
            "Alice",
            "谢谢bob",
            window_seconds=300,
            max_recent=60,
            pair_weight=0.25,
            mention_weight=5,
            same_topic_weight=0.8,
            directed=[("bob", "reply", 8.0)],
            source_id="rel-1",
        )
        await social.refresh_group(SCOPE)

        # Add a verified profile entry for bob
        ten_bob = _verified_sources(10, member="bob", verified=True)
        for ev in ten_bob:
            store.ingest(ev)
        p_bob = store.propose(
            candidate(
                {
                    "category": "typing_habit",
                    "key": "observed_typing",
                    "member": "bob",
                    "value": {"kind": "observed_typing"},
                    "reason": "统计",
                    "evidence_ids": [e["id"] for e in ten_bob],
                    "days": 30,
                },
                ten_bob,
                None,
            ),
            "human",
            "test",
        )
        store.apply(p_bob["id"], "human:admin", p_bob["updated"])
        await engine.refresh()

        ctx = SimpleNamespace(get_config=lambda: {"dashboard": {"username": "admin"}})
        plugin = main_mod.DynamicsLearning(ctx, {"sessions": [SCOPE]})
        plugin.engine = engine
        plugin.ready = True

        user_event = SimpleNamespace(
            unified_msg_origin=SCOPE,
            get_sender_id=lambda: "alice",
            is_admin=lambda: False,
        )
        admin_event = SimpleNamespace(
            unified_msg_origin=SCOPE,
            get_sender_id=lambda: "alice",
            is_admin=lambda: True,
        )

        # Non-admin MUST be blocked from full-group overview and other members' data.
        res_overview = json.loads(await plugin.query_group_social(user_event, kind="overview"))
        assert not res_overview["available"] and "管理员" in res_overview["reason"]

        for kind in ("affection", "profile", "relations"):
            res_other = json.loads(await plugin.query_group_social(user_event, kind=kind, member_id="bob"))
            assert not res_other["available"] and "管理员" in res_other["reason"]

        # Non-admin CAN query self affection / relations and group jargon.
        res_self_aff = json.loads(await plugin.query_group_social(user_event, kind="affection"))
        assert res_self_aff["available"] and res_self_aff["member"] == "alice"

        res_self_rel = json.loads(await plugin.query_group_social(user_event, kind="relations"))
        assert res_self_rel["available"] and res_self_rel["member"] == "alice"

        res_jargon = json.loads(await plugin.query_group_social(user_event, kind="jargon", term="老六"))
        assert res_jargon["available"] and res_jargon["terms"][0]["term"] == "老六"

        # Unknown kind fails cleanly as JSON.
        res_unknown = json.loads(await plugin.query_group_social(user_event, kind="invalid_kind"))
        assert not res_unknown["available"]

        # Admin CAN query overview and bob's affection/profile/relations.
        assert json.loads(await plugin.query_group_social(admin_event, kind="overview"))["available"]
        res_admin_aff = json.loads(await plugin.query_group_social(admin_event, kind="affection", member_id="bob"))
        assert res_admin_aff["available"] and res_admin_aff["member"] == "bob"
        res_admin_prof = json.loads(await plugin.query_group_social(admin_event, kind="profile", member_id="bob"))
        assert res_admin_prof["available"] and res_admin_prof["member"] == "bob"

    asyncio.run(run())


def _ensure_dashboard_auth_module(monkeypatch):
    try:
        import importlib
        return importlib.import_module("astrbot.dashboard.api.auth")
    except ImportError:
        pass
    for name in ("astrbot", "astrbot.dashboard", "astrbot.dashboard.api"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    auth_mod = types.ModuleType("astrbot.dashboard.api.auth")

    async def _default_auth(_raw):
        return None

    auth_mod.require_dashboard_user = _default_auth
    monkeypatch.setitem(sys.modules, "astrbot.dashboard.api.auth", auth_mod)
    return auth_mod


@pytest.mark.parametrize("configured_username", ["", None, "   "])
def test_authorized_rejects_empty_dashboard_username(monkeypatch, configured_username):
    main_mod = _load_plugin_class_offline(monkeypatch)
    auth_mod = _ensure_dashboard_auth_module(monkeypatch)

    async def fake_require(_raw):
        return "attacker"

    monkeypatch.setattr(auth_mod, "require_dashboard_user", fake_require)
    monkeypatch.setattr(
        main_mod,
        "request",
        SimpleNamespace(
            _request=SimpleNamespace(method="GET", headers={}, cookies={}),
            method="GET",
            username="attacker",
            headers={},
            cookies={},
        ),
    )

    ctx = SimpleNamespace(get_config=lambda: {"dashboard": {"username": configured_username}})
    plugin = main_mod.DynamicsLearning(ctx, {})

    async def run():
        with pytest.raises(PermissionError):
            await plugin._authorized()

    asyncio.run(run())


def test_web_api_enforces_require_series_admin_and_csrf(store, monkeypatch):
    main_mod = _load_plugin_class_offline(monkeypatch)
    auth_mod = _ensure_dashboard_auth_module(monkeypatch)
    authed_user = "admin"

    async def fake_require(_raw):
        return authed_user

    monkeypatch.setattr(auth_mod, "require_dashboard_user", fake_require)
    ctx = SimpleNamespace(get_config=lambda: {"dashboard": {"username": "admin"}})
    plugin = main_mod.DynamicsLearning(ctx, {"sessions": [SCOPE]})
    plugin.engine = Engine(store, plugin.config, None)
    plugin.engine.social = Social(plugin.engine)
    plugin.ready = True

    def make_req(method="POST", headers=None, cookies=None, body=b"{}", username="admin"):
        async def _body():
            return body

        raw = SimpleNamespace(method=method, headers=headers or {}, cookies=cookies or {}, body=_body)
        return SimpleNamespace(
            _request=raw,
            method=method,
            username=username,
            headers=headers or {},
            cookies=cookies or {},
        )

    async def run():
        nonlocal authed_user

        # 1. GET overview passes for valid Dashboard admin without Origin/CSRF headers.
        monkeypatch.setattr(main_mod, "request", make_req(method="GET"))
        resp_get = await plugin.overview()
        assert resp_get.status_code == 200
        assert json.loads(resp_get.body)["status"] == "ok"

        # 2. POST write route without Origin/Referer/CSRF/Bearer proof is blocked (403).
        monkeypatch.setattr(
            main_mod,
            "request",
            make_req(method="POST", headers={"Host": "dashboard.local"}, body=b'{"scope":"x","confirm_scope":"x"}'),
        )
        resp_no_csrf = await plugin.purge()
        assert resp_no_csrf.status_code == 403

        # 3. Cross-site Origin on POST write route is blocked (403).
        monkeypatch.setattr(
            main_mod,
            "request",
            make_req(
                method="POST",
                headers={
                    "Host": "dashboard.local",
                    "Origin": "https://evil.example",
                    "Sec-Fetch-Site": "cross-site",
                },
                body=b'{"scope":"x","confirm_scope":"x"}',
            ),
        )
        resp_cross = await plugin.purge()
        assert resp_cross.status_code == 403

        # 4. api_key: principal is rejected even with valid same-origin headers (403).
        authed_user = "api_key:secret"
        monkeypatch.setattr(
            main_mod,
            "request",
            make_req(
                method="POST",
                username="api_key:secret",
                headers={
                    "Host": "dashboard.local",
                    "Origin": "https://dashboard.local",
                    "Sec-Fetch-Site": "same-origin",
                },
                body=b'{"scope":"x","confirm_scope":"x"}',
            ),
        )
        resp_apikey = await plugin.purge()
        assert resp_apikey.status_code == 403

        # 5. Mismatched CSRF cookie vs header is blocked (403).
        authed_user = "admin"
        monkeypatch.setattr(
            main_mod,
            "request",
            make_req(
                method="POST",
                headers={
                    "Host": "dashboard.local",
                    "Origin": "https://dashboard.local",
                    "Sec-Fetch-Site": "same-origin",
                    "X-CSRF-Token": "wrong",
                },
                cookies={"csrf_token": "expected"},
                body=b'{"scope":"x","confirm_scope":"x"}',
            ),
        )
        resp_bad_cookie = await plugin.purge()
        assert resp_bad_cookie.status_code == 403

        # 6. Legitimate same-origin Dashboard admin POST passes auth + CSRF and executes.
        monkeypatch.setattr(
            main_mod,
            "request",
            make_req(
                method="POST",
                headers={
                    "Host": "dashboard.local",
                    "Origin": "https://dashboard.local",
                    "Sec-Fetch-Site": "same-origin",
                    "X-CSRF-Token": "expected",
                },
                cookies={"csrf_token": "expected"},
                body=json.dumps({"scope": SCOPE, "confirm_scope": SCOPE}).encode("utf-8"),
            ),
        )
        resp_ok = await plugin.purge()
        assert resp_ok.status_code == 200
        assert json.loads(resp_ok.body)["data"] == {"deleted": True}

    asyncio.run(run())
