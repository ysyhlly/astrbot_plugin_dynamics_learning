from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace as NS

import pytest

from astrbot_plugin_dynamics_learning import series_panels
from astrbot_plugin_dynamics_learning.core.contract import (
    PERSONALITY_POLICY,
    candidate,
    encode,
    event_record,
    jargon_matches,
)
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.reply_projection import (
    UNIFIED_REPLY_BUDGET,
    prune_unified_channels,
)
from astrbot_plugin_dynamics_learning.core.social import Social
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.main import PLUGIN, DynamicsLearning
from test_group_review import OTHER, SCOPE, items, make
from test_learning import event, raw


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "p1_p2.sqlite3")
    yield s
    s.close()


def _make_participation_events(scope=SCOPE, persona="persona-a", baseline="base-v1", host_version="host-v1"):
    rows = []
    for i in range(10):
        e = event()
        e.update(
            id=f"ev-part-{baseline}-{i}",
            scope=scope,
            persona=persona,
            member="alice",
            turn_id=f"turn-{baseline}-{i}",
            message_id=f"msg-{baseline}-{i}",
            baseline=baseline,
            host_version=host_version,
            baseline_threshold=50.0,
        )
        rows.append(e)
    return rows


def test_store_prune_auto_archives_expired_persona_and_baseline_changed_with_audit(store):
    fresh = make(store, "fresh-key", member="alice", persona="persona-a")
    expired = make(store, "expired-key", member="bob", persona="persona-a")
    old_persona = make(store, "old-persona-key", member="carol", persona="persona-old")
    part_events = _make_participation_events(persona="persona-a", baseline="base-v1", host_version="host-v1")
    part_raw = {
        "category": "participation",
        "key": "reply_probability_threshold",
        "member": "",
        "value": 52.0,
        "reason": "十回合有效反馈",
        "evidence_ids": [e["id"] for e in part_events],
        "days": 30,
    }
    part_prop = store.propose(candidate(part_raw, part_events, None), "llm_then_human", "test")

    with store.transaction():
        expired["created"] = 1.0
        store.db.execute("UPDATE proposals SET created=1.0 WHERE id=?", (expired["id"],))
        store._save(expired)

    result = store.prune(
        30,
        current_personas={SCOPE: "persona-a"},
        latest_baselines={(SCOPE, "persona-a"): ("base-v2", "host-v1")},
    )
    assert result["archived"] == 3
    assert set(result["archived_ids"]) == {expired["id"], old_persona["id"], part_prop["id"]}

    assert store.get(fresh["id"])["state"] == "reviewed"
    for pid, expected_reason in (
        (expired["id"], "expired"),
        (old_persona["id"], "persona_changed"),
        (part_prop["id"], "baseline_changed"),
    ):
        archived = store.get(pid)
        assert archived["state"] == "archived"
        assert archived["archive_reason"] == expected_reason
        assert archived["archived_by"] == "system:auto-archive"
        assert archived["evidence"]
        audit_row = store.db.execute(
            "SELECT action, actor, body FROM audit WHERE proposal=? AND action='archive' ORDER BY id DESC LIMIT 1",
            (pid,),
        ).fetchone()
        assert audit_row is not None
        assert audit_row["actor"] == "system:auto-archive"
        assert expected_reason in audit_row["body"]


def test_store_prune_rolls_audit_table_by_retention_days(store):
    p = make(store, "audit-retention-key")
    now = time.time()
    with store.transaction():
        store.db.execute(
            "INSERT INTO audit(proposal,action,actor,created,body) VALUES(?,?,?,?,?)",
            (p["id"], "old_event", "test", now - 120 * 86400, "{}"),
        )
        store.db.execute(
            "INSERT INTO audit(proposal,action,actor,created,body) VALUES(?,?,?,?,?)",
            (p["id"], "mid_event", "test", now - 20 * 86400, "{}"),
        )
    # Default audit retention (90 days) drops 120d row, keeps 20d row.
    res1 = store.prune(30)
    assert res1["audit_pruned"] == 1
    actions = {r[0] for r in store.db.execute("SELECT action FROM audit").fetchall()}
    assert "old_event" not in actions
    assert "mid_event" in actions

    # Explicit audit_retention_days=7 drops 20d row too.
    res2 = store.prune(30, audit_retention_days=7)
    assert res2["audit_pruned"] >= 1
    actions_after = {r[0] for r in store.db.execute("SELECT action FROM audit").fetchall()}
    assert "mid_event" not in actions_after


def test_group_batch_action_partial_skips_drift_and_older_entry_conflict(store):
    async def run():
        ok_item = make(store, "ok-topic", member="alice")
        drifted = make(store, "drifted-topic", member="bob")
        older_same = make(store, "dup-topic", member="carol", value="旧版本描述")
        newer_same = make(store, "dup-topic", member="carol", value="新版本描述")
        with store.transaction():
            older_same["created"] = newer_same["created"] - 60
            store.db.execute("UPDATE proposals SET created=? WHERE id=?", (older_same["created"], older_same["id"]))
            store._save(older_same)
        older_same = store.get(older_same["id"])
        unapproved = make(store, "unapproved-topic", member="dave", approved=False)

        engine = Engine(store, {"sessions": [SCOPE]}, None)
        batch = items(ok_item, drifted, older_same, newer_same, unapproved)
        batch[1]["updated"] = 0  # simulate single-item drift

        # partial=False preserves legacy atomic rejection.
        with pytest.raises(ValueError):
            await engine.group_batch_action(SCOPE, batch, "human:admin", partial=False)
        assert not store.entries()

        # partial=True applies ok_item + newer_same, skips drifted + older_same + unapproved.
        res = await engine.group_batch_action(SCOPE, batch, "human:admin", partial=True)
        assert set(res["applied"]) == {ok_item["id"], newer_same["id"]}
        assert res["count"] == 2
        skipped_by_id = {row["id"]: row["reason"] for row in res["skipped"]}
        assert set(skipped_by_id) == {drifted["id"], older_same["id"], unapproved["id"]}
        assert res["skipped_count"] == 3
        assert "旧项" in skipped_by_id[older_same["id"]]
        assert store.get(newer_same["id"])["state"] == "applied"
        assert store.get(older_same["id"])["state"] == "reviewed"
        entries = {e["key"]: e["value"] for e in store.entries() if e["active"]}
        assert entries["dup-topic"] == "新版本描述"
        assert entries["ok-topic"]

    asyncio.run(run())


def test_archive_group_stale_archives_only_target_scope_stale_items(store):
    async def run():
        ready = make(store, "ready-item", member="alice", scope=SCOPE, persona="persona-a")
        expired = make(store, "expired-item", member="bob", scope=SCOPE, persona="persona-a")
        wrong_persona = make(store, "wrong-persona", member="carol", scope=SCOPE, persona="persona-old")
        other_stale = make(store, "other-stale", member="dave", scope=OTHER, persona="persona-old")
        with store.transaction():
            expired["created"] = 1.0
            store.db.execute("UPDATE proposals SET created=1.0 WHERE id=?", (expired["id"],))
            store._save(expired)

        engine = Engine(store, {"sessions": [SCOPE, OTHER]}, None)
        engine.current_personas[SCOPE] = "persona-a"
        engine.current_personas[OTHER] = "persona-a"

        res = await engine.archive_group_stale(SCOPE, "human:admin")
        assert res["scope"] == SCOPE
        assert res["count"] == 2
        assert set(res["archived"]) == {expired["id"], wrong_persona["id"]}
        assert store.get(ready["id"])["state"] == "reviewed"
        assert store.get(expired["id"])["state"] == "archived"
        assert store.get(wrong_persona["id"])["state"] == "archived"
        # Other group's stale proposal is untouched.
        assert store.get(other_stale["id"])["state"] == "reviewed"

    asyncio.run(run())


def test_series_summary_badge_prioritizes_ready_count_and_separates_stale(tmp_path):
    store = Store(tmp_path / "summary.sqlite3")
    try:
        ready = make(store, "ready-one", member="alice", scope=SCOPE, persona="persona-a", approved=True)
        stale1 = make(store, "stale-one", member="bob", scope=SCOPE, persona="persona-old", approved=True)
        stale2 = make(store, "stale-two", member="carol", scope=SCOPE, persona="persona-a", approved=True)
        with store.transaction():
            stale2["created"] = 1.0
            store.db.execute("UPDATE proposals SET created=1.0 WHERE id=?", (stale2["id"],))
            store._save(stale2)

        config = {"sessions": [SCOPE], "enabled": True}
        engine = Engine(store, config, None)
        engine.current_personas[SCOPE] = "persona-a"
        plugin = DynamicsLearning(NS(), config)
        plugin.engine = engine
        plugin.ready = True

        # 1 ready + 2 stale -> badge count is 1 (not 3), label/pulse separate stale.
        summary = plugin.get_series_summary()
        assert summary["badge"]["count"] == 1
        assert summary["badge"]["tone"] == "warm"
        assert summary["ready_count"] == 1
        assert summary["stale_count"] == 2
        assert "失效" in summary["badge"]["label"]
        assert "失效 2" in summary["pulse"]["headline"]

        # Approve the ready item -> 0 ready + 2 stale -> badge count drops to 0, tone muted.
        store.apply(ready["id"], "human:admin", ready["updated"])
        summary_stale_only = plugin.get_series_summary()
        assert summary_stale_only["badge"]["count"] == 0
        assert summary_stale_only["badge"]["tone"] == "muted"
        assert summary_stale_only["ready_count"] == 0
        assert summary_stale_only["stale_count"] == 2
        assert "2 项已失效" in summary_stale_only["badge"]["label"]
        assert stale1["id"]
    finally:
        store.close()


def test_jargon_ascii_min_length_and_cjk_compatibility():
    ev = event()
    # Single-char pure ASCII jargon must be rejected.
    for bad in ("a", "A", "6", " x "):
        with pytest.raises(ValueError, match="ASCII"):
            candidate(
                {**raw(), "category": "jargon", "member": "", "key": bad, "value": "单字母黑话", "evidence_ids": [ev["id"]]},
                [ev],
                None,
            )
    # 2+ char ASCII and 1-char CJK remain allowed.
    for good in ("gg", "PVP", "草", "A站"):
        data = candidate(
            {**raw(), "category": "jargon", "member": "", "key": good, "value": "群内用语", "evidence_ids": [ev["id"]]},
            [ev],
            None,
        )
        assert data["key"] == good


def test_jargon_word_boundary_casefold_matching_and_per_turn_cap(store):
    assert jargon_matches("gg", "GG well played")
    assert jargon_matches("gg", "好gg啊")
    assert not jargon_matches("gg", "egg")
    assert not jargon_matches("gg", "aggro")
    assert not jargon_matches("gg", "gg2")
    assert jargon_matches("A站", "我在a站看过")
    assert not jargon_matches("A站", "BA站不是")
    assert jargon_matches("草", "这也太草了吧")
    assert not jargon_matches("a", "a")

    async def run():
        ev = event()
        for idx, term in enumerate(["gg", "afk", "pvp", "dps", "tank", "boss"]):
            data = candidate(
                {
                    **raw(),
                    "category": "jargon",
                    "member": "",
                    "key": term,
                    "value": f"含义-{idx}",
                    "evidence_ids": [ev["id"]],
                },
                [ev],
                None,
            )
            p = store.propose(data, "human", "test")
            store.apply(p["id"], "human:admin", p["updated"])
        engine = Engine(store, {"sessions": [SCOPE]}, None)
        await engine.refresh()

        # Substring "egg" must NOT trigger "gg".
        miss = engine.context(
            scope=SCOPE, persona=ev["persona"], member="alice", baseline="base", host_version="host", query="I ate an egg"
        )
        assert not [r for r in miss["reply_projection"]["items"] if r["category"] == "jargon"]

        # Uppercase "GG AFK PVP DPS TANK BOSS" triggers casefold boundary match, capped to 4 per turn.
        hit = engine.context(
            scope=SCOPE,
            persona=ev["persona"],
            member="alice",
            baseline="base",
            host_version="host",
            query="GG AFK PVP DPS TANK BOSS!",
            trace=True,
        )
        matched_jargon = [r for r in hit["reply_projection"]["items"] if r["category"] == "jargon"]
        assert len(matched_jargon) == 4
        capped = [c for c in hit["candidates"] if c["category"] == "jargon" and not c["selected"]]
        assert len(capped) == 2
        assert all("数量预算" in c["reason"] for c in capped)

    asyncio.run(run())


def test_social_query_jargon_casefold_and_boundary_safety(store):
    async def run():
        ev = event()
        data = candidate(
            {**raw(), "category": "jargon", "member": "", "key": "gg", "value": "认输或结束", "evidence_ids": [ev["id"]]},
            [ev],
            None,
        )
        p = store.propose(data, "human", "test")
        store.apply(p["id"], "human:admin", p["updated"])
        engine = Engine(store, {"sessions": [SCOPE]}, None)
        engine.current_personas[SCOPE] = ev["persona"]
        await engine.refresh()
        social = Social(engine)

        with pytest.raises(ValueError):
            social.query(SCOPE, "alice", "jargon", "a")
        hit = social.query(SCOPE, "alice", "jargon", "GG")
        assert hit["available"] is True and hit["terms"][0]["term"] == "gg"
        miss = social.query(SCOPE, "alice", "jargon", "egg")
        assert miss["available"] is False and miss["terms"] == []

    asyncio.run(run())


def test_unified_reply_budget_enforces_priority_order_across_three_channels():
    boundary_item = {
        "source_id": "pref-boundary",
        "category": "preference",
        "key": "pref:b",
        "value": {"topic": "剧透", "stance": "boundary", "condition": "", "quote": "别剧透", "source_id": "e1"},
    }
    dislike_item = {
        "source_id": "pref-dislike",
        "category": "preference",
        "key": "pref:d",
        "value": {"topic": "长文", "stance": "dislike", "condition": "", "quote": "太长不看", "source_id": "e2"},
    }
    jargon_item = {
        "source_id": "jargon-1",
        "category": "jargon",
        "key": "gg",
        "value": "结束/认输 " + "x" * 180,
    }
    scene_item = {
        "source_id": "scene-1",
        "category": "scene_pattern",
        "value": {
            "scene": {"communication_intent": "banter"},
            "method": "轻接一句 " + "s" * 240,
            "examples": [{"trigger": "a" * 300, "response": "b" * 300}],
        },
    }
    expr_item = {
        "source_id": "expr-1",
        "category": "expression_style",
        "value": {
            "mode": "casual",
            "guidance": "短句自然收住 " + "e" * 360,
            "traits": [{"id": "t1", "kind": "length", "marker": ""}],
            "examples": [{"id": "ex1", "text": "笑死 " + "z" * 400}],
        },
        "selection": {"mode": "casual", "trait_ids": ["t1"], "example_ids": ["ex1"]},
    }
    other_item = {
        "source_id": "style-other",
        "category": "style",
        "key": "k",
        "value": "其他低优资料 " + "o" * 400,
    }

    # Tight budget: fits personality + boundary + dislike + jargon + trimmed scene, drops expression & other.
    budget = len(PERSONALITY_POLICY) + len(encode(boundary_item)) + len(encode(dislike_item)) + len(encode(jargon_item)) + 480
    pruned = prune_unified_channels(
        learning_items=[other_item, jargon_item, dislike_item, boundary_item],
        scene_items=[scene_item],
        expression_items=[expr_item],
        personality_policy=PERSONALITY_POLICY,
        budget=budget,
    )
    assert pruned["personality_policy"] == PERSONALITY_POLICY
    assert pruned["used_chars"] <= budget
    kept_learning = {r["source_id"] for r in pruned["learning_items"]}
    assert {"pref-boundary", "pref-dislike", "jargon-1"} <= kept_learning
    assert "style-other" not in kept_learning
    assert len(pruned["scene_items"]) == 1
    assert pruned["scene_items"][0]["value"]["examples"] == []  # trimmed examples to fit
    assert pruned["expression_items"] == []
    assert "expr-1" in pruned["omitted"]
    assert "style-other" in pruned["omitted"]


def test_sequential_three_channel_injection_respects_unified_budget(store):
    async def run():
        # Seed large approved channel 1 (preferences + jargon), channel 2 (scene), channel 3 (expression).
        base_ev = event_record(
            {
                "schema_version": 1,
                "kind": "observed",
                "scope": SCOPE,
                "persona": "persona",
                "member": "alice",
                "message_id": "m1",
                "turn_id": "0:m1",
                "baseline": "base",
                "host_version": "host",
                "baseline_threshold": 50,
                "mandatory": False,
                "text": "gg 我很讨厌剧透不要剧透",
                "timestamp": time.time(),
                "context": {"source_verified": True},
            }
        )
        j = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "gg",
                    "member": "",
                    "value": "结束 " + "文" * 700,
                    "reason": "明确黑话",
                    "evidence_ids": [base_ev["id"]],
                    "days": 30,
                },
                [base_ev],
                None,
            ),
            "human",
            "test",
        )
        store.apply(j["id"], "human:admin", j["updated"])
        for idx in range(3):
            s = store.propose(
                candidate(
                    {
                        "category": "style",
                        "key": f"style-{idx}",
                        "member": "alice",
                        "value": "该成员表达特点 " + "字" * 800,
                        "reason": "多条观察",
                        "evidence_ids": [base_ev["id"]],
                        "days": 30,
                    },
                    [base_ev],
                    None,
                ),
                "human",
                "test",
            )
            store.apply(s["id"], "human:admin", s["updated"])

        from test_expression_style import apply_style
        from test_scene_learning import SCENE, SCOPE as SCENE_SCOPE, approved as approve_scene

        # Use SCENE_SCOPE consistently by creating a dedicated engine on SCENE_SCOPE with small unified budget.
        store2 = Store(store.path.parent / "three_chan.sqlite3")
        try:
            approve_scene(store2)
            apply_style(store2)
            ev2 = event_record(
                {
                    "schema_version": 1,
                    "kind": "observed",
                    "scope": SCENE_SCOPE,
                    "persona": "persona",
                    "member": "alice",
                    "message_id": "m2",
                    "turn_id": "0:m2",
                    "baseline": "base",
                    "host_version": "host",
                    "baseline_threshold": 50,
                    "mandatory": False,
                    "text": "gg 又把大招交给空气了",
                    "timestamp": time.time(),
                    "context": {"source_verified": True},
                }
            )
            for idx in range(4):
                p = store2.propose(
                    candidate(
                        {
                            "category": "relationship",
                            "key": f"rel-{idx}",
                            "member": "alice",
                            "value": "长期互动说明 " + "甲" * 600,
                            "reason": "证据充分",
                            "evidence_ids": [ev2["id"]],
                            "days": 30,
                        },
                        [ev2],
                        None,
                    ),
                    "human",
                    "test",
                )
                store2.apply(p["id"], "human:admin", p["updated"])
            engine = Engine(
                store2,
                {
                    "sessions": [SCENE_SCOPE],
                    "scene_learning_enabled": True,
                    "expression_style_enabled": True,
                    "reply_projection_budget": 3000,
                },
                None,
            )
            engine.current_personas[SCENE_SCOPE] = "persona"
            await engine.refresh()

            c1 = engine.context(
                scope=SCENE_SCOPE,
                persona="persona",
                member="alice",
                baseline="base",
                host_version="host",
                query="gg 又把大招交给空气了",
            )
            c2 = engine.scenes.context(
                scope=SCENE_SCOPE,
                persona="persona",
                member="alice",
                scene=SCENE,
                query="gg 又把大招交给空气了",
            )
            c3 = engine.expressions.context(
                scope=SCENE_SCOPE,
                persona="persona",
                member="alice",
                scene={"communication_intent": "sharing", "expression_mode": "literal", "turn_update": "adds_detail"},
                action="reply",
                query="gg 又把大招交给空气了",
            )
            total_chars = (
                len(PERSONALITY_POLICY)
                + len(encode(c1["reply_projection"]["items"]))
                + sum(len(encode(r)) for r in c2["items"])
                + sum(len(encode(r)) for r in c3["items"])
            )
            assert total_chars <= 3000 <= UNIFIED_REPLY_BUDGET
        finally:
            store2.close()

    asyncio.run(run())


def test_provider_for_uses_silent_lookup_without_cold_start_warn(store):
    warn_calls = []

    def noisy_get_provider_by_id(pid):
        warn_calls.append(pid)
        return None

    chat_plugin = NS(
        get_series_shared_config=lambda: {"decision_provider": "dec-1", "reply_provider": "rep-1"},
    )
    decision_prov = NS(id="dec-1", is_native_decision_provider=True, text_chat=lambda **_kw: None)
    reply_prov = NS(id="rep-1", text_chat=lambda **_kw: None)
    host_ctx = NS(
        provider_manager=NS(inst_map={"dec-1": decision_prov, "rep-1": reply_prov}),
        get_provider_by_id=noisy_get_provider_by_id,
        get_all_stars=lambda: [
            NS(name="astrbot_plugin_chat_dynamics", activated=True, star_cls=chat_plugin),
        ],
    )
    engine = Engine(store, {"sessions": [SCOPE]}, None, host_context=host_ctx)
    assert engine.provider_for("review_provider") == "rep-1"
    assert warn_calls == []


def test_archive_stale_and_partial_batch_routes_work(store, monkeypatch):
    async def run():
        expired = make(store, "route-expired", member="alice", scope=SCOPE, persona="persona-a")
        good = make(store, "route-good", member="bob", scope=SCOPE, persona="persona-a")
        drifted = make(store, "route-drift", member="carol", scope=SCOPE, persona="persona-a")
        with store.transaction():
            expired["created"] = 1.0
            store.db.execute("UPDATE proposals SET created=1.0 WHERE id=?", (expired["id"],))
            store._save(expired)

        config = {"sessions": [SCOPE]}
        plugin = DynamicsLearning(NS(), config)
        plugin.engine = Engine(store, config, None)
        plugin.engine.current_personas[SCOPE] = "persona-a"
        plugin.ready = True

        body = {"scope": SCOPE}

        async def serve(operation, fields=()):
            assert not (set(body) - set(fields))
            return await operation(body, "human:admin")

        monkeypatch.setattr(plugin, "_serve", serve)
        archived = await plugin.archive_stale()
        assert archived["count"] == 1 and archived["archived"] == [expired["id"]]

        batch = items(good, drifted)
        batch[1]["updated"] = 0
        body = {"scope": SCOPE, "items": batch, "partial": True}
        partial_res = await plugin.group_batch_action()
        assert partial_res["applied"] == [good["id"]]
        assert partial_res["skipped_count"] == 1
        assert partial_res["skipped"][0]["id"] == drifted["id"]

    asyncio.run(run())


def test_get_series_health_contract_sync_readonly_and_silent(tmp_path):
    store = Store(tmp_path / "health.sqlite3")
    try:
        warn_calls = []
        ok_prov = NS(id="llm-ok", text_chat=lambda **_kw: None)
        ctx = NS(
            provider_manager=NS(inst_map={"llm-ok": ok_prov}),
            get_provider_by_id=lambda pid: warn_calls.append(pid) or None,
        )
        config = {
            "sessions": [SCOPE],
            "analysis_provider": "llm-ok",
            "review_provider": "missing-id",
        }
        engine = Engine(store, config, None, host_context=ctx)
        plugin = DynamicsLearning(ctx, config)
        plugin.engine = engine
        plugin.ready = True

        before_changes = store.db.total_changes
        health = plugin.get_series_health()
        assert store.db.total_changes == before_changes
        assert warn_calls == []
        assert health["ready"] is True
        assert health["degraded"] is True
        assert any(p["role"] == "analysis" and p["status"] == "ok" for p in health["providers"])
        assert any(p["role"] == "review" and p["status"] == "missing" for p in health["providers"])
        assert any("provider_missing:review:missing-id" in i for i in health["issues"])
        assert isinstance(health["metrics"], dict)

        # Contract compatibility with series_panels.read_plugin_health
        normalized = series_panels.read_plugin_health(PLUGIN, plugin)
        assert normalized is not None
        assert normalized["plugin_id"] == PLUGIN
        assert normalized["ready"] is True
        assert normalized["degraded"] is True
        assert len(normalized["providers"]) == 2
        assert normalized["metrics"]["sessions"] == 1

        # When both providers are valid -> healthy, non-degraded.
        config["review_provider"] = "llm-ok"
        clean_health = plugin.get_series_health()
        assert clean_health["ready"] is True
        assert clean_health["degraded"] is False
        assert clean_health["issues"] == []
        assert warn_calls == []
    finally:
        store.close()
