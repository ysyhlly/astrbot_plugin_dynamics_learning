import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_dynamics_learning.core.contract import candidate, encode, event_record
from astrbot_plugin_dynamics_learning.core.engine import (
    REVIEW,
    UNIFIED_REVIEW,
    Engine,
    parse_batch_review_verdicts,
)
from astrbot_plugin_dynamics_learning.core.expressions import REVIEW_STYLE
from astrbot_plugin_dynamics_learning.core.scenes import REVIEW_SCENES
from astrbot_plugin_dynamics_learning.core.store import Store
from test_expression_style import raw_style, samples
from test_group_review import SCOPE, items, make
from test_scene_learning import evidence as scene_evidence
from test_scene_learning import raw_pattern as scene_raw_pattern

PERSONA = "persona-a"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "cache_opt.sqlite3")
    yield s
    s.close()


def _make_event(
    *,
    scope=SCOPE,
    persona=PERSONA,
    member="alice",
    message_id="m1",
    turn_id="0:m1",
    text="今天一起刷副本",
    kind="observed",
    context=None,
):
    return event_record(
        {
            "schema_version": 1,
            "kind": kind,
            "scope": scope,
            "persona": persona,
            "member": member,
            "turn_id": turn_id,
            "message_id": message_id,
            "host_version": "host1",
            "baseline": "config1",
            "baseline_threshold": 70,
            "text": text,
            "mandatory": False,
            "backend": "jev",
            "context": context or {},
        }
    )


def _apply_entry(store, *, category, key, value, member="", scope=SCOPE, persona=PERSONA, evs=None, days=30):
    if evs is None:
        ev = _make_event(
            scope=scope,
            persona=persona,
            member=member or "alice",
            message_id=f"seed-{category}-{key}-{member}",
            turn_id=f"0:seed-{category}-{key}-{member}",
            text=f"seed {key}",
        )
        evs = [ev]
    raw_data = {
        "category": category,
        "key": key,
        "value": value,
        "member": member,
        "reason": "test seed",
        "evidence_ids": [e["id"] for e in evs],
        "days": days,
    }
    p = store.propose(candidate(raw_data, evs, None), "human", "seed")
    return store.apply(p["id"], "human:admin", p["updated"])


# ---------------------------------------------------------------------------
# DL-1: .mark_as_temp() on on_llm_request social context injection
# ---------------------------------------------------------------------------


def test_inject_social_context_marks_text_part_as_temp(store, monkeypatch):
    pytest.importorskip("astrbot.core.agent.message")
    from astrbot.core.agent.message import TextPart
    from astrbot_plugin_dynamics_learning.main import DynamicsLearning

    async def run():
        _apply_entry(store, category="style", key="tone", value="该成员常用短句交流", member="alice")
        plugin = DynamicsLearning(SimpleNamespace(), {"sessions": [SCOPE], "replace_selflearning": False})
        plugin.engine = Engine(store, plugin.config, None)
        plugin.engine.current_personas[SCOPE] = PERSONA
        await plugin.engine.refresh()
        plugin.ready = True

        scope_obj = {"platform": "qq", "umo": SCOPE, "persona_id": PERSONA, "instance_id": "inst1"}

        async def fake_char(_event):
            return scope_obj

        monkeypatch.setattr(plugin, "_character_for_event", fake_char)
        monkeypatch.setattr(
            "astrbot_plugin_dynamics_learning.main.current_persona",
            lambda _eng, _scope: PERSONA,
        )
        monkeypatch.setattr(
            "astrbot_plugin_dynamics_learning.main.replacement_allowed",
            lambda _plugin: True,
        )

        extras = {}
        event = SimpleNamespace(
            unified_msg_origin=SCOPE,
            get_sender_id=lambda: "alice",
            get_message_str=lambda: "你好",
            get_extra=lambda k, default=None: extras.get(k, default),
            set_extra=lambda k, v: extras.__setitem__(k, v),
        )
        req = SimpleNamespace(extra_user_content_parts=[])

        await plugin.inject_social_context(event, req)
        assert len(req.extra_user_content_parts) == 1
        part = req.extra_user_content_parts[0]
        assert isinstance(part, TextPart)
        assert "该成员常用短句交流" in part.text
        assert part._no_save is True
        assert part.model_dump_for_context().get("_no_save") is True
        assert extras.get("_dynamics_learning_injected_v1") is True

    asyncio.run(run())


def test_inject_social_context_handles_mock_text_part_without_mark_as_temp(store, monkeypatch):
    pytest.importorskip("astrbot.core.agent.message")
    import astrbot.core.agent.message as msg_mod
    from astrbot_plugin_dynamics_learning.main import DynamicsLearning

    class PlainPart:
        def __init__(self, text):
            self.text = text

    class NoneReturningTempPart:
        def __init__(self, text):
            self.text = text
            self.marked = False

        def mark_as_temp(self):
            self.marked = True
            return None

    async def run():
        _apply_entry(store, category="style", key="tone", value="该成员常用短句交流", member="alice")
        plugin = DynamicsLearning(SimpleNamespace(), {"sessions": [SCOPE], "replace_selflearning": False})
        plugin.engine = Engine(store, plugin.config, None)
        plugin.engine.current_personas[SCOPE] = PERSONA
        await plugin.engine.refresh()
        plugin.ready = True

        scope_obj = {"platform": "qq", "umo": SCOPE, "persona_id": PERSONA, "instance_id": "inst1"}

        async def fake_char(_event):
            return scope_obj

        monkeypatch.setattr(plugin, "_character_for_event", fake_char)
        monkeypatch.setattr("astrbot_plugin_dynamics_learning.main.current_persona", lambda _e, _s: PERSONA)
        monkeypatch.setattr("astrbot_plugin_dynamics_learning.main.replacement_allowed", lambda _p: True)

        for cls in (PlainPart, NoneReturningTempPart):
            monkeypatch.setattr(msg_mod, "TextPart", cls)
            extras = {}
            event = SimpleNamespace(
                unified_msg_origin=SCOPE,
                get_sender_id=lambda: "alice",
                get_message_str=lambda: "你好",
                get_extra=lambda k, default=None: extras.get(k, default),
                set_extra=lambda k, v: extras.__setitem__(k, v),
            )
            req = SimpleNamespace(extra_user_content_parts=[])
            await plugin.inject_social_context(event, req)
            assert len(req.extra_user_content_parts) == 1
            part = req.extra_user_content_parts[0]
            assert isinstance(part, cls)
            assert "该成员常用短句交流" in part.text
            if cls is NoneReturningTempPart:
                assert part.marked is True

    asyncio.run(run())


# ---------------------------------------------------------------------------
# DL-2: ANALYZE affected-member filtering + stable ascending order + current-first
# ---------------------------------------------------------------------------


def test_analyze_filters_current_to_affected_members_and_group_categories(store):
    async def run():
        # Member entries: alice (in selected), bob (in supplied member_sources), charlie & dave (unrelated)
        alice_p = _apply_entry(store, category="style", key="alice_style", value="爱丽丝短句", member="alice")
        bob_p = _apply_entry(store, category="style", key="bob_style", value="鲍勃幽默", member="bob")
        charlie_p = _apply_entry(store, category="style", key="charlie_style", value="查理无关", member="charlie")
        dave_p = _apply_entry(store, category="relationship", key="dave_rel", value="戴夫无关", member="dave")

        # Group-level entries: jargon, decision_guidance, participation
        jargon_p = _apply_entry(store, category="jargon", key="开黑", value="一起打游戏", member="")
        guidance_p = _apply_entry(
            store, category="decision_guidance", key="reply_length", value="闲聊时保持简短", member=""
        )
        part_events = [
            _make_event(member="alice", message_id=f"pe-{i}", turn_id=f"t-pe-{i}", text=f"msg {i}") for i in range(10)
        ]
        part_p = _apply_entry(
            store,
            category="participation",
            key="reply_probability_threshold",
            value=72,
            member="",
            evs=part_events,
        )

        # Dedicated scene_pattern entry (member="") must NOT clutter general ANALYZE payload
        s_evs = scene_evidence()
        for ev in s_evs:
            ev["scope"] = SCOPE
            ev["persona"] = PERSONA
        scene_p_raw = scene_raw_pattern(s_evs)
        sp = store.propose(candidate(scene_p_raw, s_evs, None), "llm", "seed")
        sp = store.review(sp["id"], {"verdict": "approve", "reason": "ok", "risks": []}, "llm:seed", sp["updated"])
        sp = store.apply(sp["id"], "llm:seed", sp["updated"])

        # Add a native source for bob referenced via message_ids of alice's batch event
        bob_native = {
            "scope": SCOPE,
            "member": "bob",
            "source_id": "bob-src-1",
            "text": "我也来开黑",
            "created": time.time(),
            "reply_to": "",
            "eligible": True,
        }
        store.profiles.observe_batch([bob_native])
        bob_event = _make_event(member="bob", message_id="bob-src-1", turn_id="0:bob-src-1", text="我也来开黑")
        store.profiles.bind(bob_event)

        # Ingest only alice's event (which references bob's message_id in context.message_ids)
        batch_ev = _make_event(
            member="alice",
            message_id="batch-m1",
            turn_id="0:batch-m1",
            text="准备开黑",
            context={"message_ids": ["batch-m1", "bob-src-1"]},
        )
        store.ingest(batch_ev)

        captured = []

        async def model(role, instruction, data):
            captured.append((role, instruction, data))
            return {"proposals": []}

        engine = Engine(store, {"sessions": [SCOPE], "analysis_provider": "ap"}, model)
        engine.current_personas[SCOPE] = PERSONA
        await engine.refresh()
        await engine.analyze(SCOPE, PERSONA)

        assert len(captured) == 1
        role, _, data = captured[0]
        assert role == "analysis_provider"
        current_ids = {e["id"] for e in data["current"]}
        assert alice_p["entry_id"] in current_ids
        assert bob_p["entry_id"] in current_ids
        assert jargon_p["entry_id"] in current_ids
        assert guidance_p["entry_id"] in current_ids
        assert part_p["entry_id"] in current_ids
        assert charlie_p["entry_id"] not in current_ids
        assert dave_p["entry_id"] not in current_ids
        assert sp["entry_id"] not in current_ids

    asyncio.run(run())


def test_analyze_current_stable_ascending_order_and_current_first_compact_json(store):
    async def run():
        # Seed several entries for alice & group jargon
        seeded = [
            _apply_entry(store, category="style", key=f"k{i}", value=f"观察内容 {i}", member="alice")
            for i in range(4)
        ]
        seeded.append(_apply_entry(store, category="jargon", key="摸鱼", value="划水休息", member=""))

        ev1 = _make_event(member="alice", message_id="b1", turn_id="0:b1", text="第一批消息")
        store.ingest(ev1)

        payloads = []

        async def model(role, instruction, data):
            payloads.append(data)
            return {"proposals": []}

        engine = Engine(store, {"sessions": [SCOPE], "analysis_provider": "ap"}, model)
        engine.current_personas[SCOPE] = PERSONA
        await engine.refresh()
        await engine.analyze(SCOPE, PERSONA)

        assert len(payloads) == 1
        data1 = payloads[0]
        # Dict insertion order has "current" strictly before volatile "events" and "member_sources"
        assert list(data1.keys()) == ["current", "events", "member_sources"]
        serialized1 = encode(data1)
        assert serialized1.startswith('{"current":[')
        assert serialized1.index('"current":') < serialized1.index('"events":') < serialized1.index('"member_sources":')
        # Compact separators=(",", ":") — no whitespace after structural colon/comma
        assert json.dumps(data1, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == serialized1

        # Entries in current are strictly sorted ascending by (created_at, id)
        ids1 = [e["id"] for e in data1["current"]]
        expected_order = sorted(ids1)
        assert ids1 == expected_order
        assert len(ids1) == 5

        # Update the LAST entry in ascending order; earlier entries must remain in exact same prefix order
        last_id = ids1[-1]
        last_entry = next(e for e in store.entries() if e["id"] == last_id)
        updated_entry = dict(last_entry, revision=last_entry["revision"] + 1, value="更新后的内容", state_updated_at=time.time() + 100)
        with store.transaction():
            store.db.execute("UPDATE entries SET body=? WHERE id=?", (encode(updated_entry), last_id))

        ev2 = _make_event(member="alice", message_id="b2", turn_id="0:b2", text="第二批消息")
        store.ingest(ev2)
        await engine.refresh()
        await engine.analyze(SCOPE, PERSONA)

        assert len(payloads) == 2
        data2 = payloads[1]
        ids2 = [e["id"] for e in data2["current"]]
        assert ids2 == ids1
        # The prefix covering all entries before the updated one is 100% byte-identical
        assert encode(data1["current"][:-1]) == encode(data2["current"][:-1])

    asyncio.run(run())


# ---------------------------------------------------------------------------
# DL-3: Unified static REVIEW system_prompt + batched review + fallback + locks
# ---------------------------------------------------------------------------


def test_unified_review_system_prompt_is_constant_across_all_categories(store):
    async def run():
        assert REVIEW in UNIFIED_REVIEW
        assert REVIEW_SCENES in UNIFIED_REVIEW
        assert REVIEW_STYLE in UNIFIED_REVIEW

        p_style = make(store, "tone", member="alice", approved=False, category="style")
        p_jargon_ev = _make_event(member="alice", message_id="jg1", turn_id="0:jg1", text="黑话")
        p_jargon = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "内卷",
                    "value": "过度竞争",
                    "member": "",
                    "reason": "常见群词",
                    "evidence_ids": [p_jargon_ev["id"]],
                    "days": 30,
                },
                [p_jargon_ev],
                None,
            ),
            "llm_then_human",
            "test",
        )

        s_evs = scene_evidence()
        for ev in s_evs:
            ev["scope"] = SCOPE
            ev["persona"] = PERSONA
        p_scene = store.propose(candidate(scene_raw_pattern(s_evs), s_evs, None), "llm", "test")

        e_rows = samples()
        for ev in e_rows:
            ev["scope"] = SCOPE
            ev["persona"] = PERSONA
        p_expr = store.propose(candidate(raw_style(e_rows), e_rows, None), "llm", "test")

        instructions = []

        async def model(role, instruction, data):
            instructions.append(instruction)
            if "proposals" in data:
                return {
                    "verdicts": [
                        {"proposal_id": p["id"], "verdict": "approve", "reason": "ok", "risks": []}
                        for p in data["proposals"]
                    ]
                }
            return {"verdict": "approve", "reason": "ok", "risks": []}

        engine = Engine(
            store,
            {
                "sessions": [SCOPE],
                "review_provider": "rp",
                "scene_learning_enabled": True,
                "expression_style_enabled": True,
            },
            model,
        )
        engine.current_personas[SCOPE] = PERSONA

        for p in (p_style, p_jargon, p_scene, p_expr):
            await engine.review(p["id"])

        assert len(instructions) == 4
        assert set(instructions) == {UNIFIED_REVIEW}

    asyncio.run(run())


def test_review_batch_happy_path_deduplicates_evidence_pool_and_respects_modes(store):
    async def run():
        # Shared evidence events across multiple proposals
        ev1 = _make_event(member="alice", message_id="sh1", turn_id="0:sh1", text="第一句共享证据")
        ev2 = _make_event(member="alice", message_id="sh2", turn_id="0:sh2", text="第二句共享证据")
        shared_evs = [ev1, ev2]

        p_style1 = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "concise",
                    "value": "偏好简短表达",
                    "member": "alice",
                    "reason": "重复观察",
                    "evidence_ids": [ev1["id"], ev2["id"]],
                    "days": 30,
                },
                shared_evs,
                None,
            ),
            "llm",
            "test",
        )
        p_style2 = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "casual",
                    "value": "口语化交流",
                    "member": "alice",
                    "reason": "重复观察",
                    "evidence_ids": [ev1["id"], ev2["id"]],
                    "days": 30,
                },
                shared_evs,
                None,
            ),
            "llm",
            "test",
        )
        p_rel = store.propose(
            candidate(
                {
                    "category": "relationship",
                    "key": "friendly",
                    "value": "相处融洽",
                    "member": "alice",
                    "reason": "互动友好",
                    "evidence_ids": [ev1["id"]],
                    "days": 30,
                },
                shared_evs,
                None,
            ),
            "llm_then_human",
            "test",
        )
        p_jargon = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "咕咕",
                    "value": "放鸽子",
                    "member": "",
                    "reason": "群内用词",
                    "evidence_ids": [ev2["id"]],
                    "days": 30,
                },
                shared_evs,
                None,
            ),
            "llm_then_human",
            "test",
        )

        calls = []

        async def model(role, instruction, data):
            calls.append((role, instruction, data))
            assert instruction == UNIFIED_REVIEW
            assert "evidence_pool" in data and "proposals" in data
            # Shared evidence must be deduplicated into evidence_pool (2 unique events, not 6 copies)
            assert set(data["evidence_pool"].keys()) == {ev1["id"], ev2["id"]}
            for item in data["proposals"]:
                assert "evidence" not in item
                assert item["evidence_ids"]
            verdicts = []
            for item in data["proposals"]:
                v = "reject" if item["id"] == p_jargon["id"] else "approve"
                verdicts.append({"proposal_id": item["id"], "verdict": v, "reason": "批量审查完成", "risks": []})
            return {"verdicts": verdicts}

        engine = Engine(
            store,
            {
                "sessions": [SCOPE],
                "review_provider": "rp",
                "style_review": "llm",
                "relationship_review": "llm_then_human",
                "jargon_review": "llm_then_human",
            },
            model,
        )
        engine.current_personas[SCOPE] = PERSONA

        result = await engine.review_batch([p_style1["id"], p_style2["id"], p_rel["id"], p_jargon["id"]])
        assert len(calls) == 1
        assert set(result.keys()) == {p_style1["id"], p_style2["id"], p_rel["id"], p_jargon["id"]}

        # style (mode=llm) auto-applied
        assert store.get(p_style1["id"])["state"] == "applied"
        assert store.get(p_style2["id"])["state"] == "applied"
        # relationship (mode=llm_then_human) reviewed and queued for human approval, NOT auto-applied
        assert store.get(p_rel["id"])["state"] == "reviewed"
        # jargon rejected
        assert store.get(p_jargon["id"])["state"] == "rejected"
        assert {e["key"] for e in store.entries()} == {"concise", "casual"}

    asyncio.run(run())


def test_review_batch_partial_malformed_and_omitted_verdicts_fallback_to_single_review(store):
    async def run():
        p1 = make(store, "ok", member="m1", approved=False, category="style")
        p2 = make(store, "malformed", member="m2", approved=False, category="relationship")
        p3 = make(store, "omitted", member="m3", approved=False, category="relationship")
        p4 = make(store, "duplicate", member="m4", approved=False, category="relationship")

        batch_calls = 0
        single_ids = []

        async def model(role, instruction, data):
            nonlocal batch_calls
            assert instruction == UNIFIED_REVIEW
            if "proposals" in data:
                batch_calls += 1
                return {
                    "verdicts": [
                        {"proposal_id": p1["id"], "verdict": "approve", "reason": "batch ok", "risks": []},
                        {"proposal_id": p2["id"], "verdict": "invalid_verdict", "reason": "bad", "risks": []},
                        # p3 omitted completely
                        {"proposal_id": p4["id"], "verdict": "approve", "reason": "dup1", "risks": []},
                        {"proposal_id": p4["id"], "verdict": "reject", "reason": "dup2", "risks": []},
                    ]
                }
            pid = data["proposal"]["id"]
            single_ids.append(pid)
            return {"verdict": "approve", "reason": "single fallback ok", "risks": []}

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model)
        engine.current_personas[SCOPE] = PERSONA

        await engine.review_batch([p1["id"], p2["id"], p3["id"], p4["id"]])

        assert batch_calls == 1
        # p1 committed from batch; only p2, p3, p4 fell back to single review
        assert single_ids == [p2["id"], p3["id"], p4["id"]]
        assert store.get(p1["id"])["state"] == "applied"
        assert store.get(p2["id"])["state"] == "reviewed"
        assert store.get(p3["id"])["state"] == "reviewed"
        assert store.get(p4["id"])["state"] == "reviewed"

    asyncio.run(run())


def test_review_batch_malformed_top_level_and_budget_exceeded_fallback(store):
    async def run():
        # Subcase A: completely malformed batch response falls back to single review
        a1 = make(store, "a1", member="m1", approved=False)
        a2 = make(store, "a2", member="m2", approved=False)
        modes_seen = []

        async def model_bad_batch(role, instruction, data):
            if "proposals" in data:
                modes_seen.append("batch")
                return {"unexpected": "shape"}
            modes_seen.append(("single", data["proposal"]["id"]))
            return {"verdict": "approve", "reason": "single ok", "risks": []}

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model_bad_batch)
        engine.current_personas[SCOPE] = PERSONA
        await engine.review_batch([a1["id"], a2["id"]])
        assert modes_seen == ["batch", ("single", a1["id"]), ("single", a2["id"])]
        assert store.get(a1["id"])["state"] == "reviewed"
        assert store.get(a2["id"])["state"] == "reviewed"

        # Subcase B: payload budget exceeded skips batch call and uses single review
        b1 = make(store, "b1", member="m3", approved=False)
        b2 = make(store, "b2", member="m4", approved=False)
        budget_modes = []

        async def model_budget(role, instruction, data):
            budget_modes.append("batch" if "proposals" in data else ("single", data["proposal"]["id"]))
            return {"verdict": "approve", "reason": "single ok", "risks": []}

        engine.call_model = model_budget
        await engine.review_batch([b1["id"], b2["id"]], max_payload_bytes=50)
        assert budget_modes == [("single", b1["id"]), ("single", b2["id"])]
        assert store.get(b1["id"])["state"] == "reviewed"
        assert store.get(b2["id"])["state"] == "reviewed"

    asyncio.run(run())


def test_review_batch_respects_claim_locks_and_midflight_invalidation(store):
    async def run():
        # Subcase A: pre-existing review_claim excludes claimed proposal from unowned review_batch
        claimed = make(store, "claimed", member="m1", approved=False)
        free1 = make(store, "free1", member="m2", approved=False)
        free2 = make(store, "free2", member="m3", approved=False)

        seen_batch_ids = []

        async def model_a(role, instruction, data):
            assert "proposals" in data
            seen_batch_ids.append([p["id"] for p in data["proposals"]])
            return {
                "verdicts": [
                    {"proposal_id": p["id"], "verdict": "approve", "reason": "ok", "risks": []}
                    for p in data["proposals"]
                ]
            }

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model_a)
        engine.current_personas[SCOPE] = PERSONA
        job = await engine.review_batches.prepare(SCOPE, items(claimed), "human:admin")
        assert store.review_claim(claimed["id"]) == job["id"]

        await engine.review_batch([claimed["id"], free1["id"], free2["id"]])
        assert seen_batch_ids == [[free1["id"], free2["id"]]]
        assert store.get(claimed["id"])["state"] == "pending"
        assert store.get(claimed["id"])["reviews"] == []
        assert store.get(free1["id"])["state"] == "reviewed"
        assert store.get(free2["id"])["state"] == "reviewed"

        # Release prior job so we can test mid-flight claim conflict
        await engine.review_batches.control(job["id"], "cancel", "human:admin")
        await engine.review_batches.run(job)

        # Subcase B: mid-flight claim placed on p_race during batch LLM call
        p_race = make(store, "race", member="m4", approved=False)
        p_safe = make(store, "safe", member="m5", approved=False)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def model_b(role, instruction, data):
            entered.set()
            await release.wait()
            return {
                "verdicts": [
                    {"proposal_id": p["id"], "verdict": "approve", "reason": "ok", "risks": []}
                    for p in data["proposals"]
                ]
            }

        engine.call_model = model_b
        task = asyncio.create_task(engine.review_batch([p_race["id"], p_safe["id"]]))
        await entered.wait()
        with store.transaction():
            store.db.execute("INSERT OR REPLACE INTO review_claims VALUES(?,?,?)", (p_race["id"], SCOPE, "external-job"))
        release.set()
        await task

        assert store.get(p_race["id"])["state"] == "pending"
        assert store.get(p_race["id"])["reviews"] == []
        assert store.get(p_safe["id"])["state"] == "reviewed"
        assert engine.stale_results >= 1

        # Subcase C: mid-flight provider switch invalidates entire batch result
        p_inv1 = make(store, "inv1", member="m6", approved=False)
        p_inv2 = make(store, "inv2", member="m7", approved=False)
        entered2 = asyncio.Event()
        release2 = asyncio.Event()

        async def model_c(role, instruction, data):
            entered2.set()
            await release2.wait()
            return {
                "verdicts": [
                    {"proposal_id": p["id"], "verdict": "approve", "reason": "ok", "risks": []}
                    for p in data["proposals"]
                ]
            }

        engine.call_model = model_c
        stale_before = engine.stale_results
        task2 = asyncio.create_task(engine.review_batch([p_inv1["id"], p_inv2["id"]]))
        await entered2.wait()
        engine.config["review_provider"] = "rp-switched"
        release2.set()
        await task2

        assert store.get(p_inv1["id"])["state"] == "pending"
        assert store.get(p_inv2["id"])["state"] == "pending"
        assert engine.stale_results > stale_before

    asyncio.run(run())


def test_review_batches_job_batches_llm_calls_with_durable_receipts_and_partial_fallback(store):
    async def run():
        rows = [make(store, f"k{i}", member=f"m{i}", approved=False) for i in range(4)]
        calls = []

        async def model_all_ok(role, instruction, data):
            calls.append("batch" if "proposals" in data else "single")
            assert instruction == UNIFIED_REVIEW
            assert "evidence_pool" in data
            return {
                "verdicts": [
                    {"proposal_id": p["id"], "verdict": "approve", "reason": "batch checked", "risks": []}
                    for p in data["proposals"]
                ]
            }

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model_all_ok)
        engine.current_personas[SCOPE] = PERSONA
        job = await engine.review_batches.prepare(SCOPE, items(*rows), "human:admin")
        await engine.review_batches.run(job)

        saved = engine.review_batches.get(job["id"])
        assert calls == ["batch"]
        assert saved["status"] == "completed"
        assert all(i["status"] == "reviewed" and i.get("committed") for i in saved["items"])
        assert all(store.review_claim(r["id"]) is None for r in rows)

        # Now test partial omission inside ReviewBatches.run -> 1 batch + 1 single fallback
        rows2 = [make(store, f"p{i}", member=f"u{i}", approved=False) for i in range(3)]
        calls2 = []

        async def model_partial(role, instruction, data):
            if "proposals" in data:
                calls2.append(("batch", len(data["proposals"])))
                return {
                    "verdicts": [
                        {"proposal_id": p["id"], "verdict": "approve", "reason": "batch ok", "risks": []}
                        for p in data["proposals"][:2]
                    ]
                }
            calls2.append(("single", data["proposal"]["id"]))
            return {"verdict": "approve", "reason": "fallback ok", "risks": []}

        engine.call_model = model_partial
        job2 = await engine.review_batches.prepare(SCOPE, items(*rows2), "human:admin")
        await engine.review_batches.run(job2)

        saved2 = engine.review_batches.get(job2["id"])
        assert calls2 == [("batch", 3), ("single", rows2[2]["id"])]
        assert saved2["status"] == "completed"
        assert all(i["status"] == "reviewed" and i.get("committed") for i in saved2["items"])

    asyncio.run(run())


def test_scene_pending_reviews_use_batch_and_unified_prompt(store):
    async def run():
        evs1 = scene_evidence()
        for ev in evs1:
            ev["scope"] = SCOPE
            ev["persona"] = PERSONA
        raw1 = scene_raw_pattern(evs1)
        raw2 = scene_raw_pattern(evs1)
        raw2["value"] = {**raw2["value"], "method": "顺着对方调侃补一句轻松回应"}

        p1 = store.propose(candidate(raw1, evs1, None), "llm", "llm:scene-analysis")
        p2 = store.propose(candidate(raw2, evs1, None), "llm", "llm:scene-analysis")

        calls = []

        async def model(role, instruction, data):
            calls.append((role, instruction, "proposals" in data))
            return {
                "verdicts": [
                    {"proposal_id": p["id"], "verdict": "approve", "reason": "scene verified", "risks": []}
                    for p in data["proposals"]
                ]
            }

        engine = Engine(
            store,
            {"sessions": [SCOPE], "scene_learning_enabled": True, "review_provider": "rp"},
            model,
        )
        engine.current_personas[SCOPE] = PERSONA
        await engine.scenes.analyze(SCOPE, PERSONA)

        assert calls == [("review_provider", UNIFIED_REVIEW, True)]
        assert store.get(p1["id"])["state"] == "applied"
        assert store.get(p2["id"])["state"] == "applied"

    asyncio.run(run())


def test_expression_review_pending_uses_batch_and_unified_prompt(store):
    async def run():
        rows = samples()
        for ev in rows:
            ev["scope"] = SCOPE
            ev["persona"] = PERSONA
        base_raw = raw_style(rows)
        # Create two distinct pending expression_style variants
        basic_raw = {
            **base_raw,
            "value": {
                **base_raw["value"],
                "traits": [base_raw["value"]["traits"][0]],
                "example_ids": [],
            },
        }
        p1 = store.propose(candidate(basic_raw, rows, None), "llm_then_human", "llm:expression-analysis")
        p2 = store.propose(candidate(base_raw, rows, None), "llm_then_human", "llm:expression-analysis")

        calls = []

        async def model(role, instruction, data):
            calls.append((role, instruction, "proposals" in data, len(data.get("proposals", []))))
            return {
                "verdicts": [
                    {"proposal_id": p["id"], "verdict": "approve", "reason": "style verified", "risks": []}
                    for p in data["proposals"]
                ]
            }

        engine = Engine(
            store,
            {
                "sessions": [SCOPE],
                "expression_style_enabled": True,
                "expression_style_review": "llm_then_human",
                "review_provider": "rp",
            },
            model,
        )
        engine.current_personas[SCOPE] = PERSONA
        await engine.expressions.review_pending(SCOPE, PERSONA, engine.scope_token(SCOPE))

        assert calls == [("review_provider", UNIFIED_REVIEW, True, 2)]
        assert store.get(p1["id"])["state"] == "reviewed"
        assert store.get(p2["id"])["state"] == "reviewed"

    asyncio.run(run())


def test_parse_batch_review_verdicts_edge_cases():
    valid, fallback = parse_batch_review_verdicts("not-a-dict", ["p1", "p2"])
    assert valid == {} and fallback == ["p1", "p2"]

    valid, fallback = parse_batch_review_verdicts({"verdict": "approve", "reason": "x", "risks": []}, ["p1"])
    assert valid == {} and fallback == ["p1"]

    valid, fallback = parse_batch_review_verdicts(
        {
            "verdicts": [
                {"proposal_id": "p1", "verdict": "defer", "reason": "need more evidence", "risks": ["sparse"]},
                {"proposal_id": "unknown", "verdict": "approve", "reason": "ignored", "risks": []},
                {"proposal_id": "p2", "verdict": "approve", "reason": "", "risks": []},  # empty reason invalid
            ]
        },
        ["p1", "p2", "p3"],
    )
    assert list(valid.keys()) == ["p1"]
    assert valid["p1"]["verdict"] == "defer"
    assert fallback == ["p2", "p3"]
