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
    ANALYZE,
    REVIEW,
    REVIEW_BATCH_MAX,
    REVIEW_CONCISE_RULE,
    UNIFIED_REVIEW,
    Engine,
    parse_batch_review_verdicts,
)
from astrbot_plugin_dynamics_learning.core.expressions import REVIEW_STYLE
from astrbot_plugin_dynamics_learning.core.model_projection import (
    ANALYZE_PAYLOAD_BUDGET,
    REVIEW_BATCH_PAYLOAD_BUDGET,
    format_system_prompt,
    utf8_len,
)
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

        ev1 = _make_event(member="alice", message_id="b1", turn_id="0:b1", text="第一批群聊讨论消息内容")
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

        ev2 = _make_event(member="alice", message_id="b2", turn_id="0:b2", text="第二批群聊讨论消息内容")
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


# ---------------------------------------------------------------------------
# Preserved live (+cost.1) features integrated with DL-1..DL-3
# ---------------------------------------------------------------------------


def test_preserved_source_withdrawal_archives_proposals_and_blocks_apply(store):
    ev = _make_event(member="alice", message_id="90001", turn_id="0:90001", text="待撤回消息")
    store.ingest(ev)
    p = store.propose(
        candidate(
            {
                "category": "style",
                "key": "withdrawn_case",
                "value": "基于待撤回消息的观察",
                "member": "alice",
                "reason": "test",
                "evidence_ids": [ev["id"]],
                "days": 30,
            },
            [ev],
            None,
        ),
        "llm",
        "llm:analysis",
    )
    assert store.source_withdrawals.record(SCOPE, "alice", "90001") is True
    archived = store.get(p["id"])
    assert archived["state"] == "archived"
    assert archived["archive_reason"] == "source_withdrawn"
    assert store.source_withdrawals.has_withdrawn(SCOPE, [ev]) is True

    # New proposals citing the withdrawn source are rejected
    with pytest.raises(ValueError, match="withdrawn"):
        store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "another_withdrawn",
                    "value": "不应再次建提案",
                    "member": "alice",
                    "reason": "test",
                    "evidence_ids": [ev["id"]],
                    "days": 30,
                },
                [ev],
                None,
            ),
            "llm",
            "llm:analysis",
        )


def test_preserved_defer_fingerprint_sleeps_until_evidence_refreshed(store):
    from astrbot_plugin_dynamics_learning.core.review_evidence import DEFER_FINGERPRINT

    async def run():
        ev1 = _make_event(member="alice", message_id="df1", turn_id="0:df1", text="第一条证据")
        store.ingest(ev1)
        p = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "defer_key",
                    "value": "该成员偏简短回复",
                    "member": "alice",
                    "reason": "单条证据",
                    "evidence_ids": [ev1["id"]],
                    "days": 30,
                },
                [ev1],
                None,
            ),
            "llm",
            "llm:analysis",
        )
        calls = []

        async def model(role, instruction, data):
            calls.append((role, instruction))
            return {"verdict": "defer", "reason": "证据尚不足", "risks": ["样本太少"]}

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model)
        engine.current_personas[SCOPE] = PERSONA

        first = await engine.review(p["id"], expected_updated=p["updated"], automatic=True)
        assert first["state"] == "pending"
        assert first.get(DEFER_FINGERPRINT)
        assert len(calls) == 1
        assert calls[0][1] == UNIFIED_REVIEW

        # Unchanged evidence sleeps: automatic catchup does NOT re-call the LLM
        assert engine.automatic_review_waiting(first, time.time()) is True
        second = await engine.review(p["id"], expected_updated=first["updated"], automatic=True)
        assert second["id"] == p["id"]
        assert len(calls) == 1

        # Refreshing pending evidence via store.propose clears DEFER_FINGERPRINT and wakes review
        ev2 = _make_event(member="alice", message_id="df2", turn_id="0:df2", text="第二条补充证据")
        store.ingest(ev2)
        refreshed = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "defer_key",
                    "value": "该成员偏简短回复",
                    "member": "alice",
                    "reason": "补充第二条证据",
                    "evidence_ids": [ev1["id"], ev2["id"]],
                    "days": 30,
                },
                [ev1, ev2],
                None,
            ),
            "llm",
            "llm:analysis",
        )
        assert refreshed["id"] == p["id"]
        assert DEFER_FINGERPRINT not in refreshed
        assert engine.automatic_review_waiting(refreshed, time.time()) is False

    asyncio.run(run())


def test_preserved_automatic_reviews_queue_uses_unified_prompt_and_evidence_pool(store):
    async def run():
        ev1 = _make_event(member="alice", message_id="aq1", turn_id="0:aq1", text="共享证据一")
        ev2 = _make_event(member="alice", message_id="aq2", turn_id="0:aq2", text="共享证据二")
        store.ingest(ev1)
        store.ingest(ev2)
        p1 = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "aq_k1",
                    "value": "表达习惯一",
                    "member": "alice",
                    "reason": "r1",
                    "evidence_ids": [ev1["id"], ev2["id"]],
                    "days": 30,
                },
                [ev1, ev2],
                None,
            ),
            "llm",
            "llm:analysis",
        )
        p2 = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "aq_k2",
                    "value": "表达习惯二",
                    "member": "alice",
                    "reason": "r2",
                    "evidence_ids": [ev1["id"], ev2["id"]],
                    "days": 30,
                },
                [ev1, ev2],
                None,
            ),
            "llm",
            "llm:analysis",
        )

        seen = []

        async def model(role, instruction, data):
            seen.append((role, instruction, data))
            return {
                "verdicts": [
                    {"proposal_id": item["id"], "verdict": "approve", "reason": "ok", "risks": []}
                    for item in data["proposals"]
                ]
            }

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model)
        engine.current_personas[SCOPE] = PERSONA
        engine.automatic_reviews.running = True
        engine.automatic_reviews.wait_seconds = 0

        accepted = await engine.automatic_reviews.enqueue([p1, p2])
        assert accepted == {p1["id"], p2["id"]}
        groups, _ = engine.automatic_reviews.ready(time.time(), set(), discover=False)
        assert len(groups) == 1 and len(groups[0]) == 2

        await engine.automatic_reviews.dispatch(groups[0])
        assert len(seen) == 1
        role, instruction, data = seen[0]
        assert role == "review_provider"
        assert instruction == UNIFIED_REVIEW
        assert set(data["evidence_pool"].keys()) == {ev1["id"], ev2["id"]}
        assert all("evidence" not in item and item["evidence_ids"] for item in data["proposals"])
        assert store.get(p1["id"])["state"] == "applied"
        assert store.get(p2["id"])["state"] == "applied"

        overview = await engine.overview()
        stats = overview["scheduler"]["review_dispatches"]
        assert stats["provider_invocations"] == 1
        assert stats["logical_automatic_first_reviews"] == 2

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Phase 2: Lean model-facing projections (hoisted character_context + stripped bloat)
# ---------------------------------------------------------------------------


def _live_shaped_event(idx, *, member="alice", text="我不喜欢被剧透结局", persona_summary="人设摘要" * 450):
    return event_record(
        {
            "schema_version": 1,
            "kind": "outcome",
            "scope": SCOPE,
            "persona": PERSONA,
            "member": member,
            "turn_id": f"0:live-{idx}",
            "message_id": f"live-{idx}",
            "host_version": "host-v4.29.4",
            "baseline": "b" * 64,
            "baseline_threshold": 55.0,
            "text": text,
            "mandatory": False,
            "backend": "jev",
            "timestamp": 1795100000.0 + idx,
            "policy": {
                "context": "注入的社交上下文" * 350,
                "threshold": 55.0,
                "versions": ["v1", "v2"],
            },
            "decision": {
                "action": "reply",
                "length": "short",
                "state": "engaged",
                "reason": "直接互动",
                "answers": {
                    "join": {"noul": 0.85, "type": "prob"},
                    "action": {"choice": "reply", "confidence": 0.9, "type": "cat"},
                    "completion": {"choice": "complete", "confidence": 0.95, "type": "cat"},
                    "length": {"choice": "short", "confidence": 0.8, "type": "cat"},
                },
            },
            "outcome": {
                "delivered": True,
                "final_outcome": "sent",
                "stage": "send",
                "recorded_at": 1795100001.5 + idx,
            },
            "context": {
                "persona_summary": persona_summary,
                "character_scope": {
                    "platform": "qq",
                    "umo": SCOPE,
                    "persona_id": "p1",
                    "instance_id": "inst1",
                },
                "message_ids": [f"live-{idx}"],
                "reply_to_ids": [""],
                "wake_kind": "chat",
                "interaction": {
                    "basis": "detailed internal rationale text",
                    "tentative_guidance": "long verbose tentative guidance string" * 3,
                    "communication_intent": {"choice": "sharing", "confidence": 0.9},
                    "expression_mode": {"choice": "literal", "confidence": 0.92},
                    "turn_update": {"choice": "unclear", "confidence": 0.4},
                    "reply_focus": ["剧情"],
                },
            },
        }
    )


def test_analyze_lean_projection_hoists_persona_summary_strips_policy_context_and_preserves_quote_validation(store):
    async def run():
        _apply_entry(store, category="style", key="alice_tone", value="偏简短", member="alice")
        raw_events = []
        for i in range(16):
            text = "我不喜欢被剧透结局" if i == 0 else f"日常聊天消息 {i}"
            store.profiles.observe_batch(
                [
                    {
                        "scope": SCOPE,
                        "member": "alice",
                        "source_id": f"live-{i}",
                        "text": text,
                        "created": 1795100000.0 + i,
                        "reply_to": "",
                        "eligible": True,
                        "identity_tokens": ["小爱", "alice"],
                    }
                ]
            )
            ev = _live_shaped_event(i, member="alice", text=text)
            store.ingest(ev)
            # Also bind observed profile target so member_sources has verified native rows
            obs = _make_event(member="alice", message_id=f"live-{i}", turn_id=f"0:obs-{i}", text=text, kind="observed")
            store.profiles.bind(obs)
            raw_events.append(ev)

        raw_unprojected_bytes = len(encode({"current": store.entries(), "events": raw_events, "member_sources": store.profiles.sources(SCOPE, PERSONA)}).encode("utf-8"))

        captured = []

        async def model(role, instruction, data):
            captured.append((role, instruction, data))
            if role == "analysis_provider":
                src = next(s for s in data["member_sources"] if s["text"] == "我不喜欢被剧透结局")
                return {
                    "proposals": [
                        {
                            "category": "preference",
                            "key": "spoilers",
                            "member": "alice",
                            "days": 30,
                            "reason": "成员明确自述不喜欢剧透",
                            "evidence_ids": [src["id"]],
                            "value": {
                                "topic": "剧透",
                                "stance": "dislike",
                                "condition": "",
                                "quote": "我不喜欢被剧透结局",
                                "source_id": src["id"],
                            },
                        }
                    ]
                }
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
            {"sessions": [SCOPE], "analysis_provider": "ap", "review_provider": "rp", "preference_review": "llm_then_human"},
            model,
        )
        engine.current_personas[SCOPE] = PERSONA
        await engine.refresh()
        res = await engine.analyze(SCOPE, PERSONA)

        assert res["proposals"] >= 1
        analyze_instruction, analyze_data = next((ins, d) for r, ins, d in captured if r == "analysis_provider")
        serialized = encode(analyze_data)
        projected_bytes = len(serialized.encode("utf-8"))

        # Static persona-level character_context hoisted into system_prompt (omitting scope), omitted from user JSON
        expected_char_ctx = {
            "persona_summary": "人设摘要" * 450,
            "persona": PERSONA,
            "baseline_threshold": 55.0,
        }
        assert analyze_instruction == format_system_prompt(ANALYZE, expected_char_ctx)
        assert list(analyze_data.keys()) == ["current", "events", "member_sources"]
        assert "character_context" not in analyze_data
        assert serialized.startswith('{"current":[')
        assert (
            serialized.index('"current":')
            < serialized.index('"events":')
            < serialized.index('"member_sources":')
        )

        # Bloated per-event fields stripped while signal fields remain intact
        for ev in analyze_data["events"]:
            assert "persona_summary" not in (ev.get("context") or {})
            assert "character_scope" not in (ev.get("context") or {})
            assert "context" not in (ev.get("policy") or {})
            assert ev["policy"]["versions"] == ["v1", "v2"]
            assert ev["decision"]["action"] == "reply" and ev["decision"]["join"] == 0.85
            assert "answers" not in ev["decision"]
            assert "recorded_at" not in ev["outcome"]
            assert ev["context"]["interaction"] == {
                "communication_intent": "sharing",
                "expression_mode": "literal",
                "reply_focus": ["剧情"],
            }
            assert "received_at" not in ev and "baseline" not in ev

        # member_sources preserve verbatim text & source_verified while dropping character_scope/identity_tokens
        for src in analyze_data["member_sources"]:
            assert "character_scope" not in (src.get("context") or {})
            assert "identity_tokens" not in (src.get("context") or {})
            assert src["context"]["source_verified"] is True

        # current entries omit volatile timestamps
        for entry in analyze_data["current"]:
            assert "state_updated_at" not in entry and "expires" not in entry
            assert {"id", "category", "key", "member", "value", "revision"} <= set(entry)

        # Dramatic byte reduction (>80%) on realistic 16-turn live-shaped payload
        assert projected_bytes < raw_unprojected_bytes * 0.20

        # Persisted preference proposal passed strict verbatim quote check & kept full raw evidence in SQLite
        pref = next(p for p in store.proposals() if p["category"] == "preference")
        assert pref["value"]["quote"] == "我不喜欢被剧透结局"
        assert pref["evidence"][0]["context"]["character_scope"] is not None

    asyncio.run(run())


def test_review_single_and_batch_lean_projection_reduces_payload_and_hoists_character_context(store):
    async def run():
        evs = [_live_shaped_event(i, member="alice", text=f"群内黑话例句 {i}") for i in range(4)]
        props = []
        for idx in range(4):
            p = store.propose(
                candidate(
                    {
                        "category": "jargon",
                        "key": f"梗词{idx}",
                        "value": f"群内特殊含义 {idx}",
                        "member": "",
                        "reason": "多次出现",
                        "evidence_ids": [evs[idx]["id"], evs[(idx + 1) % 4]["id"]],
                        "days": 30,
                    },
                    evs,
                    None,
                ),
                "llm_then_human",
                "llm:analysis",
            )
            props.append(p)

        raw_single_bytes = len(encode({"proposal": props[0]}).encode("utf-8"))
        raw_batch_bytes = len(encode({"proposals": props}).encode("utf-8"))

        seen = []

        async def model(role, instruction, data):
            seen.append((instruction, data))
            if "proposals" in data:
                return {
                    "verdicts": [
                        {"proposal_id": p["id"], "verdict": "approve", "reason": "verified", "risks": []}
                        for p in data["proposals"]
                    ]
                }
            return {"verdict": "approve", "reason": "verified", "risks": []}

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model)
        engine.current_personas[SCOPE] = PERSONA
        expected_review_instruction = format_system_prompt(
            UNIFIED_REVIEW,
            {
                "persona_summary": "人设摘要" * 450,
                "persona": PERSONA,
                "baseline_threshold": 55.0,
            },
        )

        # 1) Single review
        await engine.review(props[0]["id"])
        single_instruction, single_data = seen[-1]
        single_bytes = len(encode(single_data).encode("utf-8"))
        assert single_instruction == expected_review_instruction
        assert "character_context" not in single_data
        assert "evaluation" not in single_data["proposal"]
        for ev in single_data["proposal"]["evidence"]:
            assert "persona_summary" not in (ev.get("context") or {})
            assert "context" not in (ev.get("policy") or {})
            assert "character_scope" not in (ev.get("context") or {})
        assert single_bytes < raw_single_bytes * 0.45
        # Persisted SQLite evidence remains 100% untouched
        persisted = store.get(props[0]["id"])
        assert persisted["state"] == "reviewed"
        assert persisted["evidence"][0]["context"]["persona_summary"] == "人设摘要" * 450

        # 2) Batched review for remaining 3 proposals sharing overlapping evidence
        await engine.review_batch([p["id"] for p in props[1:]])
        batch_instruction, batch_data = seen[-1]
        batch_bytes = len(encode(batch_data).encode("utf-8"))
        assert batch_instruction == single_instruction == expected_review_instruction
        assert "character_context" not in batch_data
        assert len(batch_data["evidence_pool"]) == 4
        for ev in batch_data["evidence_pool"].values():
            assert "persona_summary" not in (ev.get("context") or {})
            assert "context" not in (ev.get("policy") or {})
        for item in batch_data["proposals"]:
            assert "evidence" not in item and "evaluation" not in item and "scope" not in item
            assert item["evidence_ids"]
        assert batch_bytes < raw_batch_bytes * 0.20
        assert all(store.get(p["id"])["state"] == "reviewed" for p in props[1:])

    asyncio.run(run())


def test_analyze_budget_splits_large_text_events_without_dropping_frontiers(store):
    async def run():
        # 14 events each with 3,800 chars of real text (> 53K total) must split cleanly under 36K ANALYZE_PAYLOAD_BUDGET
        for i in range(14):
            ev = _make_event(
                member="alice",
                message_id=f"big-{i}",
                turn_id=f"big-turn-{i}",
                text=f"T{i:02d}-" + ("长消息内容" * 700),
            )
            store.ingest(ev)

        batch_turn_counts = []
        seen_turns = []

        async def model(role, instruction, data):
            batch_turn_counts.append(len(data["events"]))
            seen_turns.extend(e["turn_id"] for e in data["events"])
            return {"proposals": []}

        engine = Engine(store, {"sessions": [SCOPE], "analysis_provider": "ap"}, model)
        engine.current_personas[SCOPE] = PERSONA
        while store.batches(1):
            await engine.analyze(SCOPE, PERSONA)
        assert len(batch_turn_counts) >= 2
        assert all(c < 14 for c in batch_turn_counts)
        assert seen_turns == [f"big-turn-{i}" for i in range(14)]
        assert not store.batches(1)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Phase 2 (Cache Hit >= 70%): Uniform character_context in system_prompt,
# bursty automatic-review coalescing, and configurable analysis rounds.
# ---------------------------------------------------------------------------


def test_events_and_profile_messages_reviews_share_identical_system_prompt_with_character_context(store):
    async def run():
        # Seed SQLite with an events row carrying persona_summary + baseline_threshold
        live_ev = _live_shaped_event(0, member="alice", text="群里经常用这个梗", persona_summary="稳定角色人设" * 300)
        store.ingest(live_ev)

        # 1) events-backed proposal (jargon)
        p_events = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "测试梗",
                    "value": "群内调侃用语",
                    "member": "",
                    "reason": "多次出现",
                    "evidence_ids": [live_ev["id"]],
                    "days": 30,
                },
                [live_ev],
                None,
            ),
            "llm_then_human",
            "llm:analysis",
        )

        # 2) profile_messages-backed proposals (expression_style + style with profile evidence lacking persona_summary)
        profile_rows = []
        for member in ("alice", "bob", "carol"):
            for i in range(10):
                content = ["笑死 这也太离谱了吧", "嗯", "好了 收工", "哈哈哈哈", "就这啊"][i % 5]
                profile_rows.append(
                    event_record(
                        {
                            "schema_version": 1,
                            "kind": "observed",
                            "scope": SCOPE,
                            "persona": PERSONA,
                            "member": member,
                            "message_id": f"prof-{member}-{i}",
                            "turn_id": f"0:prof-{member}-{i}",
                            "baseline": "baseline",
                            "host_version": "host",
                            "baseline_threshold": 50,
                            "mandatory": False,
                            "text": content,
                            "timestamp": time.time() - (i % 3) * 86400 - 100,
                            "context": {"source_verified": True, "reply_to": "", "reply_to_ids": []},
                        }
                    )
                )
        # Verify profile_messages evidence genuinely has NO persona_summary
        assert all("persona_summary" not in (r.get("context") or {}) for r in profile_rows)

        p_expr = store.propose(
            candidate(raw_style(profile_rows), profile_rows, None),
            "llm_then_human",
            "llm:expression",
        )
        p_member_style = store.propose(
            candidate(
                {
                    "category": "style",
                    "key": "alice_typing",
                    "value": "喜欢短句收尾",
                    "member": "alice",
                    "reason": "来自 profile_messages 样本",
                    "evidence_ids": [profile_rows[0]["id"], profile_rows[1]["id"]],
                    "days": 30,
                },
                profile_rows[:2],
                None,
            ),
            "llm_then_human",
            "llm:analysis",
        )

        seen = []

        async def model(role, instruction, data):
            seen.append((role, instruction, data))
            if role == "analysis_provider":
                return {"proposals": []}
            if "proposals" in data:
                return {
                    "verdicts": [
                        {"proposal_id": item["id"], "verdict": "approve", "reason": "ok", "risks": []}
                        for item in data["proposals"]
                    ]
                }
            return {"verdict": "approve", "reason": "ok", "risks": []}

        # Brand-new Engine instance: _character_context_cache starts empty -> proves lazy SQLite hydration
        engine = Engine(
            store,
            {"sessions": [SCOPE], "analysis_provider": "ap", "review_provider": "rp"},
            model,
        )
        engine.current_personas[SCOPE] = PERSONA
        assert engine._character_context_cache == {}

        # Review profile_messages-backed expression_style FIRST (before any events-backed call!)
        await engine.review(p_expr["id"])
        # Review events-backed jargon singly
        await engine.review(p_events["id"])
        # Batch review a fresh events-backed + profile_messages-backed pair together
        p_events_2 = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "测试梗2",
                    "value": "另一个调侃用语",
                    "member": "",
                    "reason": "再次出现",
                    "evidence_ids": [live_ev["id"]],
                    "days": 30,
                },
                [live_ev],
                None,
            ),
            "llm_then_human",
            "llm:analysis",
        )
        await engine.review_batch([p_member_style["id"], p_events_2["id"]])
        # Also trigger analyze to check ANALYZE system_prompt carries identical character_context tail
        await engine.analyze(SCOPE, PERSONA)

        expected_char_ctx = {
            "persona_summary": "稳定角色人设" * 300,
            "persona": PERSONA,
            "baseline_threshold": 55.0,
        }
        expected_review_prompt = format_system_prompt(UNIFIED_REVIEW, expected_char_ctx)
        expected_analyze_prompt = format_system_prompt(ANALYZE, expected_char_ctx)

        review_calls = [(ins, d) for role, ins, d in seen if role == "review_provider"]
        analyze_calls = [(ins, d) for role, ins, d in seen if role == "analysis_provider"]
        assert len(review_calls) == 3
        assert len(analyze_calls) == 1

        # 100% byte-identical review system_prompt across profile_messages, events, and mixed batch
        for ins, data in review_calls:
            assert ins == expected_review_prompt
            assert "character_context" not in data

        # Analyze system_prompt carries the exact same serialized character_context block
        assert analyze_calls[0][0] == expected_analyze_prompt
        assert "character_context" not in analyze_calls[0][1]

    asyncio.run(run())


def test_automatic_reviews_coalescing_window_batches_bursts_and_bypasses_on_max_or_urgent(store):
    async def run():
        ev1 = _live_shaped_event(1, member="alice", text="消息1")
        ev2 = _live_shaped_event(2, member="bob", text="消息2")
        store.ingest(ev1)
        store.ingest(ev2)

        def _make_prop(category, key, member, ev):
            return store.propose(
                candidate(
                    {
                        "category": category,
                        "key": key,
                        "value": f"值-{key}",
                        "member": member,
                        "reason": "依据明确",
                        "evidence_ids": [ev["id"]],
                        "days": 30,
                    },
                    [ev],
                    None,
                ),
                "llm",
                "llm:analysis",
            )

        p1 = _make_prop("jargon", "梗1", "", ev1)
        p2 = _make_prop("style", "alice_style", "alice", ev1)
        p3 = _make_prop("style", "bob_style", "bob", ev2)
        p4 = _make_prop("jargon", "梗2", "", ev2)

        # Urgent proposal (human source) must never wait in automatic coalescing queue
        p_urgent = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "人工纠正",
                    "value": "立即生效",
                    "member": "",
                    "reason": "管理员提交",
                    "evidence_ids": [ev1["id"]],
                    "days": 30,
                },
                [ev1],
                None,
            ),
            "llm",
            "human:admin",
        )

        seen = []

        async def model(role, instruction, data):
            seen.append((role, instruction, data))
            if "proposals" in data:
                return {
                    "verdicts": [
                        {"proposal_id": item["id"], "verdict": "approve", "reason": "ok", "risks": []}
                        for item in data["proposals"]
                    ]
                }
            return {"verdict": "approve", "reason": "ok", "risks": []}

        engine = Engine(
            store,
            {"sessions": [SCOPE], "review_provider": "rp", "jargon_review": "llm", "style_review": "llm"},
            model,
        )
        engine.current_personas[SCOPE] = PERSONA
        engine.automatic_reviews.running = True
        assert engine.automatic_reviews.coalesce_seconds() == 15.0
        assert engine.automatic_reviews.candidate(p_urgent) is False

        # Enqueue 3 bursty proposals across different members & categories at t0, t0+5, t0+10
        t0 = time.time()
        assert engine.automatic_reviews.enqueue_candidates([p1], t0) == {p1["id"]}
        assert engine.automatic_reviews.enqueue_candidates([p2], t0 + 5.0) == {p2["id"]}
        assert engine.automatic_reviews.enqueue_candidates([p3], t0 + 10.0) == {p3["id"]}

        # At t0+12 (< newest_ready + 15 = t0+25): coalescing holds all 3 together
        groups, next_at = engine.automatic_reviews.ready(t0 + 12.0, set(), discover=False)
        assert groups == []
        assert next_at == pytest.approx(t0 + 25.0)

        # At t0+25: coalescing window elapses and emits 1 combined batch of all 3
        groups, _ = engine.automatic_reviews.ready(t0 + 25.0, set(), discover=False)
        assert len(groups) == 1
        assert [item.proposal["id"] for _, item in groups[0]] == [p1["id"], p2["id"], p3["id"]]

        # When 4th item arrives at t0+12 (reaching MAX_ITEMS=4), bypass coalescing immediately
        assert engine.automatic_reviews.enqueue_candidates([p4], t0 + 12.0) == {p4["id"]}
        groups, _ = engine.automatic_reviews.ready(t0 + 12.0, set(), discover=False)
        assert len(groups) == 1
        assert len(groups[0]) == 4

        # Dispatch executes 1 batched provider call for all 4 proposals across members/categories
        await engine.automatic_reviews.dispatch(groups[0])
        assert len(seen) == 1
        assert len(seen[0][2]["proposals"]) == 4
        assert all(store.get(p["id"])["state"] == "applied" for p in (p1, p2, p3, p4))

        # When automatic_review_coalesce_seconds=0, a single item fires immediately at t0
        p5 = _make_prop("jargon", "立即触发", "", ev1)
        engine.config["automatic_review_coalesce_seconds"] = 0
        assert engine.automatic_reviews.coalesce_seconds() == 0.0
        t1 = time.time()
        assert engine.automatic_reviews.enqueue_candidates([p5], t1) == {p5["id"]}
        groups, _ = engine.automatic_reviews.ready(t1, set(), discover=False)
        assert len(groups) == 1 and len(groups[0]) == 1

    asyncio.run(run())


def test_analysis_max_rounds_per_cycle_default_two_and_configurable(store):
    schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["analysis_max_rounds_per_cycle"]["default"] == 2
    assert schema["automatic_review_coalesce_seconds"]["default"] == 15.0

    async def _run_with_config(cfg):
        engine = Engine(store, {"sessions": [SCOPE], "batch_events": 1, **cfg}, None)
        engine.current_personas[SCOPE] = PERSONA
        store.batches = lambda minimum: [{"scope": SCOPE, "persona": PERSONA, "count": 50, "oldest": 1}]
        calls = []

        async def analyze(scope, persona):
            calls.append((scope, persona))

        engine.analyze = analyze
        await engine.analysis_cycle()
        return len(calls)

    async def run():
        assert await _run_with_config({}) == 2
        assert await _run_with_config({"analysis_max_rounds_per_cycle": 1}) == 1
        assert await _run_with_config({"analysis_max_rounds_per_cycle": 3}) == 3

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Phase 3: Output-token & thinking-token reduction (concise static prompts,
# max_batch_size=4, REVIEW_BATCH_PAYLOAD_BUDGET=28_000, optional kwargs).
# ---------------------------------------------------------------------------


def test_phase3_concise_prompt_constraints_remain_static_and_preserve_prefix_cache():
    assert REVIEW_CONCISE_RULE in UNIFIED_REVIEW
    assert "35" in UNIFIED_REVIEW and "15" in UNIFIED_REVIEW and "[]" in UNIFIED_REVIEW
    assert "紧凑单行 JSON" in UNIFIED_REVIEW
    assert "40" in ANALYZE and '{"proposals":[]}' in ANALYZE and "紧凑单行 JSON" in ANALYZE

    # Strictly static prompts preserve prefix caching across calls
    char_ctx = {"persona_summary": "静态角色摘要", "scope": SCOPE, "persona": PERSONA, "baseline_threshold": 55.0}
    assert format_system_prompt(UNIFIED_REVIEW, char_ctx) == format_system_prompt(UNIFIED_REVIEW, char_ctx)
    assert format_system_prompt(ANALYZE, char_ctx) == format_system_prompt(ANALYZE, char_ctx)


def test_phase3_review_batch_size_cap_4_and_payload_budget_24k(store):
    assert REVIEW_BATCH_MAX == 4
    assert REVIEW_BATCH_PAYLOAD_BUDGET == 24_000

    async def run():
        # 1) 6 pending proposals must split into batches of [4, 2] (never > 4)
        props = [make(store, f"cap-{i}", member=f"m{i}", approved=False, category="relationship") for i in range(6)]
        chunk_sizes = []

        async def model(role, instruction, data):
            if "proposals" in data:
                chunk_sizes.append(len(data["proposals"]))
                return {
                    "verdicts": [
                        {"proposal_id": item["id"], "verdict": "approve", "reason": "依据充分", "risks": []}
                        for item in data["proposals"]
                    ]
                }
            chunk_sizes.append(1)
            return {"verdict": "approve", "reason": "依据充分", "risks": []}

        engine = Engine(store, {"sessions": [SCOPE], "review_provider": "rp"}, model)
        engine.current_personas[SCOPE] = PERSONA
        await engine.review_batch([p["id"] for p in props])
        assert chunk_sizes == [4, 2]
        assert all(store.get(p["id"])["state"] == "reviewed" for p in props)

        # 2) Large Chinese evidence proposals exceeding 24_000 UTF-8 byte budget split below 24_000 even with <= 4 proposals
        heavy_props = []
        for idx in range(3):
            # 2 events x ~1,600 Chinese chars (~4,800 UTF-8 bytes each) -> ~10KB UTF-8 per proposal; 2 fit <=24KB, 3 exceed 24KB
            evs = [
                _make_event(
                    member=f"heavy{idx}",
                    message_id=f"heavy-msg-{idx}-{j}",
                    turn_id=f"0:heavy-{idx}-{j}",
                    text=f"H{idx}-{j}-" + ("超长真实群聊内容" * 200),
                )
                for j in range(2)
            ]
            p = store.propose(
                candidate(
                    {
                        "category": "jargon",
                        "key": f"长证据梗{idx}",
                        "value": f"含义{idx}",
                        "member": "",
                        "reason": "多次出现",
                        "evidence_ids": [ev["id"] for ev in evs],
                        "days": 30,
                    },
                    evs,
                    None,
                ),
                "llm_then_human",
                "llm:analysis",
            )
            heavy_props.append(p)

        chunks = engine._pack_review_chunks(SCOPE, PERSONA, heavy_props)
        assert [len(ch) for ch in chunks] == [2, 1]
        for ch in chunks:
            assert len(ch) <= REVIEW_BATCH_MAX
            if len(ch) > 1:
                assert utf8_len(engine._build_batch_review_payload(SCOPE, PERSONA, ch)) <= REVIEW_BATCH_PAYLOAD_BUDGET

    asyncio.run(run())


def test_phase3_model_optional_max_tokens_and_reasoning_effort_passthrough():
    from astrbot_plugin_dynamics_learning.main import DynamicsLearning, MIN_SAFE_MODEL_MAX_TOKENS

    schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["model_max_tokens"]["default"] == 0
    assert schema["model_reasoning_effort"]["default"] == ""

    async def run():
        async def _call_with(provider, config):
            plugin = DynamicsLearning.__new__(DynamicsLearning)
            plugin.closed = False
            plugin.config = {"review_provider": "rp", **config}
            plugin.context = SimpleNamespace(get_provider_by_id=lambda pid: provider if pid == "rp" else None)
            plugin.engine = None
            return await plugin._model("review_provider", UNIFIED_REVIEW, {"proposal": {"id": "p1"}})

        # 1) Provider accepting **kwargs: default config passes NO extra kwargs
        kw_calls = []

        class KwargsProvider:
            def meta(self):
                return SimpleNamespace(model="deepseek-flash")

            async def text_chat(self, prompt=None, system_prompt=None, contexts=None, **kwargs):
                kw_calls.append(kwargs)
                return SimpleNamespace(completion_text='{"verdict":"approve","reason":"ok","risks":[]}')

        await _call_with(KwargsProvider(), {})
        assert kw_calls[-1] == {}

        # Configured values passed through to **kwargs provider
        await _call_with(KwargsProvider(), {"model_max_tokens": 2048, "model_reasoning_effort": "minimal"})
        assert kw_calls[-1] == {"max_tokens": 2048, "reasoning_effort": "minimal"}

        # 2) Explicit signature provider: max_tokens below 1536 clamped safely to >= 1536
        explicit_calls = []

        class ExplicitProvider:
            def meta(self):
                return SimpleNamespace(model="deepseek-flash")

            async def text_chat(
                self, prompt=None, system_prompt=None, contexts=None, max_tokens=None, reasoning_effort=None
            ):
                explicit_calls.append((max_tokens, reasoning_effort))
                return SimpleNamespace(completion_text='{"verdict":"approve","reason":"ok","risks":[]}')

        await _call_with(ExplicitProvider(), {"model_max_tokens": 512, "model_reasoning_effort": "low"})
        assert explicit_calls[-1] == (MIN_SAFE_MODEL_MAX_TOKENS, "low")

        # 3) Strict provider accepting neither kwarg: safe no-op without TypeError
        strict_calls = []

        class StrictProvider:
            def meta(self):
                return SimpleNamespace(model="strict-model")

            async def text_chat(self, prompt=None, system_prompt=None, contexts=None):
                strict_calls.append((prompt, system_prompt))
                return SimpleNamespace(completion_text='{"verdict":"approve","reason":"ok","risks":[]}')

        res = await _call_with(StrictProvider(), {"model_max_tokens": 2048, "model_reasoning_effort": "low"})
        assert res["verdict"] == "approve"
        assert len(strict_calls) == 1

        # 4) Provider accepting ONLY max_tokens: omits reasoning_effort cleanly
        partial_calls = []

        class MaxOnlyProvider:
            def meta(self):
                return SimpleNamespace(model="max-only")

            async def text_chat(self, prompt=None, system_prompt=None, contexts=None, max_tokens=None):
                partial_calls.append(max_tokens)
                return SimpleNamespace(completion_text='{"verdict":"approve","reason":"ok","risks":[]}')

        await _call_with(MaxOnlyProvider(), {"model_max_tokens": 2048, "model_reasoning_effort": "low"})
        assert partial_calls == [2048]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Phase 4: Gemini/Responses implicit-cache static 2-message contexts anchor
# & character_context wrapping on ANALYZE_SCENES / ANALYZE_STYLE.
# ---------------------------------------------------------------------------


def test_phase4_model_passes_deterministic_static_contexts_anchor():
    import hashlib

    from astrbot_plugin_dynamics_learning.main import DynamicsLearning

    async def run():
        async def _call_with(provider, instruction, data):
            plugin = DynamicsLearning.__new__(DynamicsLearning)
            plugin.closed = False
            plugin.config = {"review_provider": "rp"}
            plugin.context = SimpleNamespace(get_provider_by_id=lambda pid: provider if pid == "rp" else None)
            plugin.engine = None
            return await plugin._model("review_provider", instruction, data)

        recorded = []

        class RecordingProvider:
            def meta(self):
                return SimpleNamespace(model="gemini-3.8-flash-high")

            async def text_chat(self, prompt=None, system_prompt=None, contexts=None, **kwargs):
                recorded.append((prompt, system_prompt, contexts))
                return SimpleNamespace(completion_text='{"verdict":"approve","reason":"ok","risks":[]}')

        provider = RecordingProvider()
        instr_a = UNIFIED_REVIEW
        instr_b = ANALYZE

        # Two calls with same instruction and different payloads -> byte-identical 2-message anchor
        await _call_with(provider, instr_a, {"proposal": {"id": "p-1", "value": "alpha"}})
        await _call_with(provider, instr_a, {"proposal": {"id": "p-2", "value": "beta"}})
        expected_sha = hashlib.sha256(instr_a.encode("utf-8")).hexdigest()[:12]
        expected_contexts = [
            {"role": "user", "content": f"[cache_anchor task=dynamics_learning prefix={expected_sha}]"},
            {"role": "assistant", "content": "OK"},
        ]
        assert recorded[0][0] != recorded[1][0]
        assert recorded[0][2] == expected_contexts
        assert recorded[1][2] == expected_contexts

        # Different instruction -> different deterministic prefix hash
        await _call_with(provider, instr_b, {"events": []})
        expected_sha_b = hashlib.sha256(instr_b.encode("utf-8")).hexdigest()[:12]
        assert expected_sha_b != expected_sha
        assert recorded[2][2] == [
            {"role": "user", "content": f"[cache_anchor task=dynamics_learning prefix={expected_sha_b}]"},
            {"role": "assistant", "content": "OK"},
        ]

        # Empty instruction -> empty contexts list
        await _call_with(provider, "", {"events": []})
        assert recorded[3][2] == []

        # Strict stub provider without `contexts` parameter -> safe no-op without TypeError
        no_ctx_calls = []

        class NoContextsProvider:
            def meta(self):
                return SimpleNamespace(model="stub-no-contexts")

            async def text_chat(self, prompt=None, system_prompt=None):
                no_ctx_calls.append((prompt, system_prompt))
                return SimpleNamespace(completion_text='{"verdict":"approve","reason":"ok","risks":[]}')

        res = await _call_with(NoContextsProvider(), instr_a, {"proposal": {"id": "p-3"}})
        assert res["verdict"] == "approve"
        assert len(no_ctx_calls) == 1

    asyncio.run(run())


def test_phase4_scene_and_expression_learning_include_character_context_in_system_prompt(store):
    from astrbot_plugin_dynamics_learning.core.expressions import ANALYZE_STYLE
    from astrbot_plugin_dynamics_learning.core.scenes import ANALYZE_SCENES
    from test_scene_learning import SCOPE as SCENE_SCOPE

    async def run():
        persona = "persona"
        persona_summary = "角色核心设定与语气约束" * 300
        # Seed SQLite events row with persona_summary for lazy hydration
        ev = event_record(
            {
                "schema_version": 1,
                "kind": "outcome",
                "scope": SCENE_SCOPE,
                "persona": persona,
                "member": "alice",
                "turn_id": "0:char-seed",
                "message_id": "char-seed",
                "host_version": "host-v4.29.4",
                "baseline": "b" * 64,
                "baseline_threshold": 55.0,
                "text": "种子消息",
                "mandatory": False,
                "timestamp": time.time(),
                "context": {"persona_summary": persona_summary},
            }
        )
        store.ingest(ev)

        # Seed profile_messages for both scene learning (>=3 episodes) and expression learning (>=30 msgs, 3 members, 2 days)
        for row in [*scene_evidence(), *samples()]:
            store.db.execute(
                "INSERT OR REPLACE INTO profile_messages VALUES(?,?,?,?,?,?)",
                (SCENE_SCOPE, persona, row["message_id"], row["member"], row["timestamp"], encode(row)),
            )
        store.db.commit()

        seen = []

        async def model(role, instruction, data):
            seen.append((role, instruction, data))
            if "episodes" in data:
                return {"patterns": []}
            if "sample" in data:
                return {"style": None}
            return {"verdict": "approve", "reason": "ok", "risks": []}

        engine = Engine(
            store,
            {
                "sessions": [SCENE_SCOPE],
                "analysis_provider": "ap",
                "review_provider": "rp",
                "scene_learning_enabled": True,
                "expression_style_enabled": True,
            },
            model,
        )
        engine.current_personas[SCENE_SCOPE] = persona
        # Empty in-memory cache proves lazy SQLite hydration works for scene & expression analyzers too
        assert engine._character_context_cache == {}

        await engine.scenes.analyze(SCENE_SCOPE, persona)
        await engine.expressions.analyze(SCENE_SCOPE, persona)

        expected_char_ctx = {
            "persona_summary": persona_summary,
            "persona": persona,
            "baseline_threshold": 55.0,
        }
        scene_calls = [ins for r, ins, d in seen if r == "analysis_provider" and "episodes" in d]
        expr_calls = [ins for r, ins, d in seen if r == "analysis_provider" and "sample" in d]
        assert len(scene_calls) == 1
        assert len(expr_calls) == 1
        assert scene_calls[0] == format_system_prompt(ANALYZE_SCENES, expected_char_ctx)
        assert expr_calls[0] == format_system_prompt(ANALYZE_STYLE, expected_char_ctx)
        assert "[character_context]" in scene_calls[0] and persona_summary in scene_calls[0]
        assert "[character_context]" in expr_calls[0] and persona_summary in expr_calls[0]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Phase 5: Cross-group persona-unified system_prompt & contexts[0] anchor,
# UTF-8 byte-bounded ANALYZE (22KB) & REVIEW (24KB) budgets, and conservative
# low-signal ANALYZE pre-filter.
# ---------------------------------------------------------------------------


def test_phase5_cross_group_same_persona_produces_identical_system_prompt_and_contexts_anchor(store):
    from astrbot_plugin_dynamics_learning.core.expressions import ANALYZE_STYLE
    from astrbot_plugin_dynamics_learning.core.scenes import ANALYZE_SCENES
    from astrbot_plugin_dynamics_learning.main import _cache_anchor_contexts

    async def run():
        scope_a = "qq:GroupMessage:10001"
        scope_b = "qq:GroupMessage:20002"
        shared_persona = "wb_natsuume_anan"
        persona_summary = "共享角色夏梅安安人设摘要" * 200

        def _ev(scope, idx, text, *, with_summary=True):
            ctx = {"persona_summary": persona_summary} if with_summary else {}
            return event_record(
                {
                    "schema_version": 1,
                    "kind": "outcome",
                    "scope": scope,
                    "persona": shared_persona,
                    "member": f"user{idx}",
                    "turn_id": f"0:{scope}-{idx}",
                    "message_id": f"m-{scope}-{idx}",
                    "host_version": "host-v4.29.4",
                    "baseline": "b" * 64,
                    "baseline_threshold": 55.0,
                    "text": text,
                    "mandatory": False,
                    "timestamp": time.time() + idx,
                    "outcome": {"delivered": True, "final_outcome": "sent"},
                    "context": ctx,
                }
            )

        ev_a = _ev(scope_a, 1, "大家晚上八点一起打副本吧", with_summary=True)
        # scope_b even omits persona_summary on its event -> proves cross-scope persona hydration too!
        ev_b = _ev(scope_b, 2, "这个新版本的活动机制挺有趣的", with_summary=False)
        store.ingest(ev_a)
        store.ingest(ev_b)

        p_a = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "群A词",
                    "value": "群A含义",
                    "member": "",
                    "reason": "多次出现",
                    "evidence_ids": [ev_a["id"]],
                    "days": 30,
                },
                [ev_a],
                None,
            ),
            "llm_then_human",
            "llm:analysis",
        )
        p_b = store.propose(
            candidate(
                {
                    "category": "jargon",
                    "key": "群B词",
                    "value": "群B含义",
                    "member": "",
                    "reason": "多次出现",
                    "evidence_ids": [ev_b["id"]],
                    "days": 30,
                },
                [ev_b],
                None,
            ),
            "llm_then_human",
            "llm:analysis",
        )

        captured = {}

        async def model(role, instruction, data):
            if role == "analysis_provider":
                ev_scope = data["events"][0]["scope"]
                captured[("analyze", ev_scope)] = instruction
                return {"proposals": []}
            prop_scope = data["proposal"]["scope"]
            captured[("review", prop_scope)] = instruction
            return {"verdict": "approve", "reason": "依据明确", "risks": []}

        engine = Engine(
            store,
            {"sessions": [scope_a, scope_b], "analysis_provider": "ap", "review_provider": "rp"},
            model,
        )
        engine.current_personas[scope_a] = shared_persona
        engine.current_personas[scope_b] = shared_persona

        await engine.analyze(scope_a, shared_persona)
        await engine.analyze(scope_b, shared_persona)
        await engine.review(p_a["id"])
        await engine.review(p_b["id"])

        analyze_a = captured[("analyze", scope_a)]
        analyze_b = captured[("analyze", scope_b)]
        review_a = captured[("review", scope_a)]
        review_b = captured[("review", scope_b)]

        # 1) Cross-group 100% byte-identical system_prompt AND contexts[0] anchor
        assert analyze_a == analyze_b
        assert review_a == review_b
        assert _cache_anchor_contexts(analyze_a) == _cache_anchor_contexts(analyze_b)
        assert _cache_anchor_contexts(review_a) == _cache_anchor_contexts(review_b)
        assert scope_a not in analyze_a and scope_b not in analyze_b
        assert scope_a not in review_a and scope_b not in review_b

        # 2) Leading shared prefix ([character_context] + [personality_policy]) precedes task-specific rules
        assert analyze_a.startswith("[character_context]\n")
        assert review_a.startswith("[character_context]\n")
        shared_prefix_end = analyze_a.index("\n\n" + ANALYZE)
        assert review_a[:shared_prefix_end] == analyze_a[:shared_prefix_end]
        char_ctx = engine.character_context_for(scope_a, shared_persona)
        assert format_system_prompt(ANALYZE_SCENES, char_ctx)[:shared_prefix_end] == analyze_a[:shared_prefix_end]
        assert format_system_prompt(ANALYZE_STYLE, char_ctx)[:shared_prefix_end] == analyze_a[:shared_prefix_end]

    asyncio.run(run())


def test_phase5_utf8_byte_budget_enforced_for_analyze_22k_and_review_24k(store):
    assert ANALYZE_PAYLOAD_BUDGET == 22_000
    assert REVIEW_BATCH_PAYLOAD_BUDGET == 24_000

    async def run():
        # Seed active current entries so `current` + `events` combined must obey <= 22_000 UTF-8 bytes
        for i in range(6):
            _apply_entry(
                store,
                category="jargon",
                key=f"常用黑话{i}",
                value="群聊中经常用来调侃和表达共识的固定词组解释说明" * 12,
                member="",
            )
        # Seed 10 Chinese events (~3,600 UTF-8 bytes each, total ~36KB UTF-8 < 18,000 Python chars!)
        for i in range(10):
            ev = _make_event(
                member="alice",
                message_id=f"utf8-ev-{i}",
                turn_id=f"0:utf8-ev-{i}",
                text=f"U{i:02d}-" + ("中文聊天消息内容测试" * 115),
            )
            store.ingest(ev)

        payload_byte_sizes = []
        batch_event_counts = []
        seen_turns = []

        async def model(role, instruction, data):
            byte_size = len(encode(data).encode("utf-8"))
            payload_byte_sizes.append(byte_size)
            batch_event_counts.append(len(data["events"]))
            seen_turns.extend(e["turn_id"] for e in data["events"])
            return {"proposals": []}

        engine = Engine(store, {"sessions": [SCOPE], "analysis_provider": "ap"}, model)
        engine.current_personas[SCOPE] = PERSONA
        await engine.refresh()

        while store.batches(1):
            await engine.analyze(SCOPE, PERSONA)

        assert len(batch_event_counts) >= 2
        assert seen_turns == [f"0:utf8-ev-{i}" for i in range(10)]
        assert not store.batches(1)
        for count, size in zip(batch_event_counts, payload_byte_sizes):
            if count > 1:
                assert size <= ANALYZE_PAYLOAD_BUDGET

    asyncio.run(run())


def test_phase5_low_signal_analyze_prefilter_skips_noise_but_keeps_real_conversations(store):
    from astrbot_plugin_dynamics_learning.core.engine import has_substantive_analysis_signal

    async def run():
        # 1) Pure noise / filler / placeholder / emoji batch -> 0 LLM calls, frontiers cleanly marked processed
        noise_texts = ["[图片]", "哈哈哈哈", "嗯", "👍", "[表情]", "ok", "666", "@bot 1"]
        for i, txt in enumerate(noise_texts):
            store.ingest(
                _make_event(
                    member=f"m{i % 3}",
                    message_id=f"noise-{i}",
                    turn_id=f"0:noise-{i}",
                    text=txt,
                    kind="observed",
                )
            )

        calls = []

        async def model(role, instruction, data):
            calls.append((role, [e["turn_id"] for e in data["events"]]))
            return {"proposals": []}

        engine = Engine(store, {"sessions": [SCOPE], "analysis_provider": "ap"}, model)
        engine.current_personas[SCOPE] = PERSONA

        res_noise = await engine.analyze(SCOPE, PERSONA)
        assert calls == []
        assert res_noise["turns"] == len(noise_texts)
        assert res_noise["proposals"] == 0
        assert not store.batches(1)

        # 2) Two short non-trivial user messages (>= 4 meaningful chars each) -> MUST invoke LLM
        store.ingest(_make_event(member="alice", message_id="real-1", turn_id="0:real-1", text="晚上刷副本"))
        store.ingest(_make_event(member="bob", message_id="real-2", turn_id="0:real-2", text="八点半开组"))
        await engine.analyze(SCOPE, PERSONA)
        assert len(calls) == 1
        assert calls[-1][1] == ["0:real-1", "0:real-2"]

        # 3) Single short self-stated preference cue -> MUST invoke LLM
        store.ingest(_make_event(member="alice", message_id="pref-1", turn_id="0:pref-1", text="别剧透"))
        await engine.analyze(SCOPE, PERSONA)
        assert len(calls) == 2
        assert calls[-1][1] == ["0:pref-1"]

        # 4) Single short filler WITH delivered bot reply/outcome -> MUST invoke LLM
        store.ingest(
            event_record(
                {
                    "schema_version": 1,
                    "kind": "outcome",
                    "scope": SCOPE,
                    "persona": PERSONA,
                    "member": "alice",
                    "turn_id": "0:reply-1",
                    "message_id": "reply-1",
                    "host_version": "host1",
                    "baseline": "config1",
                    "baseline_threshold": 70,
                    "text": "嗯",
                    "mandatory": False,
                    "timestamp": time.time(),
                    "outcome": {"delivered": True, "final_outcome": "sent"},
                }
            )
        )
        await engine.analyze(SCOPE, PERSONA)
        assert len(calls) == 3
        assert calls[-1][1] == ["0:reply-1"]

        # Direct helper sanity check
        assert has_substantive_analysis_signal([{"kind": "observed", "text": "[动画表情] 哈哈哈"}]) is False
        assert has_substantive_analysis_signal([{"kind": "observed", "text": "我不喜欢被剧透"}]) is True

    asyncio.run(run())
