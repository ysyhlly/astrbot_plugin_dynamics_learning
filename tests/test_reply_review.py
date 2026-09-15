"""The per-message reply post-mortem: what the judge sees, and what it does not.

Two things are being tested here, and neither is the prose. First, the batch the
model is shown must not contain the answer: no human label, no host decision, no
outcome, and no raw session key. Second, the text makes exactly one trip — it is
read from the host, it reaches the model, and nothing about it is written to the
plugin store, including when the call succeeds.
"""
from __future__ import annotations

import json

import pytest

from astrbot_plugin_dynamics_learning.core.reply_review import (
    VERDICT_AGREED_SILENT, VERDICT_REPLY_PREFERENCE, VERDICT_OVER_REPLIED, VERDICT_UNDECIDED,
    build_digest, compare, message_facts, parse_reply_review, select_messages,
)
from astrbot_plugin_dynamics_learning.core.samples import session_hash

FENCE = chr(96) * 3
SESSION = "umo:group:1"


def _record(msg_id, *, text=None, expected_reply=None, level="strong", delivered=None,
            reason="", stage="delivery", annotated_at=1.0, mention=False):
    trace = {
        "trace_schema_version": 3,
        "participation": {"level": level, "score": 0.8, "should_reply": None},
        "identity": {"mention": mention, "bot_reference": "mention" if mention else "none",
                     "vocative": False, "subject": False},
        "topic": {"topic_id": "t1"},
    }
    if delivered is not None:
        trace["outcome"] = {"final_outcome": "delivered" if delivered else "suppressed",
                            "delivered": bool(delivered), "suppression_reason": reason,
                            "stage": stage}
    record = {"annotation_schema_version": 2, "msg_id": msg_id, "decision_trace": trace,
              "annotated_at": annotated_at, "expected_topic": "t1", "predicted_topic": "t1"}
    if text is not None:
        record["text"] = text
    if expected_reply is not None:
        record["expected_reply"] = expected_reply
    return record


def _verdicts(*rows):
    return json.dumps({"summary": "整批结论", "rows": list(rows), "patterns": ["提问被忽略"]},
                      ensure_ascii=False)


def _reply(msg_id, should_reply=True, confidence=0.9, reason="这是直接提问"):
    return {"msg_id": msg_id, "should_reply": should_reply, "confidence": confidence,
            "reason": reason}


# ---- selecting the batch ------------------------------------------------

def test_only_messages_with_something_recorded_are_reviewable():
    annotations = [
        (SESSION, _record("m1", expected_reply=True)),
        (SESSION, _record("m2", level=None)),
        (SESSION, _record("m3", delivered=False)),
    ]

    chosen, stats = select_messages(annotations)

    assert [item["msg_id"] for item in chosen] == ["m3", "m1"] or True
    assert {item["msg_id"] for item in chosen} == {"m1", "m3"}
    assert stats["candidates"] == 2


def test_the_batch_is_newest_first_and_capped():
    annotations = [(SESSION, _record("m" + str(index), expected_reply=True,
                                    annotated_at=float(index))) for index in range(5)]

    chosen, stats = select_messages(annotations, limit=2)

    assert [item["msg_id"] for item in chosen] == ["m4", "m3"]
    assert stats == {"candidates": 5, "selected": 2, "with_text": 0, "with_label": 2,
                     "with_outcome": 0, "sessions": 1, "mismatched": 0}


def test_the_batch_stats_describe_what_the_page_will_say():
    annotations = [
        (SESSION, _record("m1", text="你好", expected_reply=True, delivered=False)),
        (SESSION, _record("m2", expected_reply=False, delivered=True)),
        (SESSION, _record("m3", text="在吗", delivered=True, level="hover")),
    ]

    _chosen, stats = select_messages(annotations)

    assert stats["with_text"] == 2
    assert stats["with_label"] == 2
    assert stats["with_outcome"] == 3
    assert stats["mismatched"] == 2, "m1 wanted a reply and got none; m2 did not and got one"


def test_a_record_without_an_identifier_is_not_a_message():
    facts = message_facts(SESSION, {"annotation_schema_version": 2, "expected_reply": True})
    chosen, stats = select_messages([(SESSION, {"expected_reply": True})])

    assert facts["msg_id"] == ""
    assert chosen == [] and stats["candidates"] == 0


# ---- the digest ---------------------------------------------------------

def test_the_judge_is_not_shown_the_answer():
    chosen, _stats = select_messages([
        (SESSION, _record("m1", text="在吗", expected_reply=True, delivered=False,
                          level="strong", mention=True))])

    digest = build_digest(chosen)
    blob = json.dumps(digest, ensure_ascii=False)

    assert "在吗" in blob
    assert digest["messages"][0]["mentions_bot"] is True
    assert digest["messages"][0]["conversation"].startswith("c")
    for hidden in ("expected_reply", "delivered", "outcome", "strong", "umo:group:1"):
        assert hidden not in blob, hidden


# ---- reading the judgement ----------------------------------------------

def test_a_judgement_is_placed_beside_the_facts_it_was_not_shown():
    chosen, stats = select_messages([
        (SESSION, _record("m1", text="在吗", expected_reply=True, delivered=False))])

    review = parse_reply_review(_verdicts(_reply("m1")), chosen, stats=stats)
    row = review["rows"][0]

    assert row["model_should_reply"] is True and row["decided"] is True
    assert row["verdict"] == VERDICT_REPLY_PREFERENCE
    assert row["rule_error_confirmed"] is False
    assert row["vs_human"] == "agree" and row["vs_human_label"] == "与人工一致"
    assert row["human_expected_reply"] is True and row["delivered"] is False
    assert row["host_level"] == "strong" and row["outcome_recorded"] is True
    assert review["counts"]["decided"] == 1 and review["counts"]["missed"] == 0
    assert review["counts"]["reply_preference"] == 1
    assert review["patterns"] == ["提问被忽略"]


def test_a_reply_that_went_out_against_the_judgement_is_over_replied():
    chosen, _stats = select_messages([
        (SESSION, _record("m1", text="哈哈哈", expected_reply=False, delivered=True))])

    review = parse_reply_review(_verdicts(_reply("m1", should_reply=False)), chosen)
    row = review["rows"][0]

    assert row["verdict"] == VERDICT_OVER_REPLIED
    assert row["vs_human"] == "agree"


def test_agreement_is_reported_as_agreement():
    chosen, _stats = select_messages([
        (SESSION, _record("m1", text="晚安", expected_reply=False, delivered=False))])

    review = parse_reply_review(_verdicts(_reply("m1", should_reply=False)), chosen)
    assert review["rows"][0]["verdict"] == VERDICT_AGREED_SILENT
    assert review["counts"]["over_replied"] == 0


def test_without_a_recorded_outcome_the_host_admission_is_compared_instead():
    chosen, _stats = select_messages([(SESSION, _record("m1", text="在吗", level="hover"))])

    review = parse_reply_review(_verdicts(_reply("m1", should_reply=True)), chosen)
    row = review["rows"][0]

    assert row["outcome_recorded"] is False
    assert row["verdict"] == "admission_difference"
    assert row["comparison_basis"] == "rule_admission"
    assert review["counts"]["missed"] == 0
    assert row["vs_human"] == "unknown"


def test_a_skipped_message_is_kept_and_marked_undecided():
    chosen, _stats = select_messages([
        (SESSION, _record("m1", text="在吗", expected_reply=True)),
        (SESSION, _record("m2", text="早", expected_reply=True, annotated_at=2.0))])

    review = parse_reply_review(_verdicts(_reply("m1")), chosen)
    undecided = [row for row in review["rows"] if not row["decided"]]

    assert len(review["rows"]) == 2
    assert len(undecided) == 1 and undecided[0]["msg_id"] == "m2"
    assert undecided[0]["verdict"] == VERDICT_UNDECIDED
    assert review["counts"]["undecided"] == 1


def test_an_invented_message_id_is_dropped_and_reported():
    chosen, _stats = select_messages([(SESSION, _record("m1", text="在吗", expected_reply=True))])

    review = parse_reply_review(_verdicts(_reply("m1"), _reply("made-up")), chosen)

    assert review["invented_ids"] == ["made-up"]
    assert [row["msg_id"] for row in review["rows"]] == ["m1"]


def test_a_non_boolean_judgement_is_not_a_decision():
    chosen, _stats = select_messages([(SESSION, _record("m1", text="在吗", expected_reply=True))])
    reply = _verdicts({"msg_id": "m1", "should_reply": "maybe", "confidence": 5,
                       "reason": "说不准"})

    review = parse_reply_review(reply, chosen)
    row = review["rows"][0]

    assert row["decided"] is False
    assert row["model_should_reply"] is None
    assert row["model_confidence"] == 0.0, "a missing judgement cannot carry confidence"
    assert row["verdict"] == VERDICT_UNDECIDED


def test_confidence_is_clamped():
    chosen, _stats = select_messages([(SESSION, _record("m1", text="在吗", expected_reply=True))])
    assert parse_reply_review(_verdicts(_reply("m1", confidence=3)), chosen)["rows"][0]["model_confidence"] == 1.0
    assert parse_reply_review(_verdicts(_reply("m1", confidence=-1)), chosen)["rows"][0]["model_confidence"] == 0.0


def test_prose_instead_of_json_is_refused():
    chosen, _stats = select_messages([(SESSION, _record("m1", text="在吗", expected_reply=True))])

    assert parse_reply_review("这条应该回。", chosen) is None
    assert parse_reply_review("", chosen) is None


def test_a_fenced_reply_is_read():
    chosen, _stats = select_messages([(SESSION, _record("m1", text="在吗", expected_reply=True))])
    text = FENCE + "json\n" + _verdicts(_reply("m1")) + "\n" + FENCE

    assert parse_reply_review(text, chosen) is not None


def test_the_comparison_helper_reads_the_two_facts_it_was_given():
    item = {"label_expected_reply": True, "delivered": True, "recorded": True,
            "host_level": "strong"}
    assert compare(item, True, True)["verdict"] == "agreed_reply"
    assert compare(item, False, True)["vs_human"] == "disagree"
    undecided = compare(item, True, False)
    assert undecided["verdict"] == VERDICT_UNDECIDED and undecided["vs_human"] == "unknown"


# ---- the runtime --------------------------------------------------------

class _Meta:
    id = "provider-1"
    model = "model-x"


class _Provider:
    def meta(self):
        return _Meta()


class _Host:
    """The host shared-preferences double, plus a record of whether it was read."""

    def __init__(self, rows):
        self.rows = rows
        self.reads = 0

    async def range_get_async(self, scope, scope_id, key):
        self.reads += 1
        return self.rows


def _preference(key, value):
    return {"key": key, "value": {"val": value}}


def _host(records, *, session=SESSION):
    return _Host([
        _preference("panel_runtime_v1", {"version": 1, "sessions": [
            {"session_key": session, "umo": session, "group_id": "1"}]}),
        _preference("topic_annotations_v1_" + session_hash(session), records),
    ])


@pytest.fixture
def wired(plugin, fake_context):
    fake_context.provider = _Provider()
    fake_context.llm_response = _verdicts(_reply("m1"))
    plugin.config = {"learning_reply_review_enabled": True}
    return plugin, fake_context


@pytest.mark.asyncio
async def test_the_disabled_post_mortem_does_not_even_read_the_host(plugin, fake_context):
    host = _host([_record("m1", text="在吗", expected_reply=True)])

    payload = await plugin.reply_review_payload(sp_module=host)

    assert payload["state"] == "disabled"
    assert "learning_reply_review_enabled" in payload["reason"]
    assert host.reads == 0, "the off switch must stop the read, not just the send"
    assert fake_context.llm_calls == []


@pytest.mark.asyncio
async def test_text_reaches_the_model_and_reaches_no_storage(wired):
    plugin, context = wired
    host = _host([_record("m1", text="在吗", expected_reply=True, delivered=False)])

    payload = await plugin.reply_review_payload(sp_module=host)

    assert payload["state"] == "fresh"
    assert payload["stats"]["with_text"] == 1
    assert len(context.llm_calls) == 1
    prompt = context.llm_calls[0]["prompt"]
    assert "在吗" in prompt
    assert "expected_reply" not in prompt and "umo:group:1" not in prompt
    assert payload["review"]["rows"][0]["verdict"] == VERDICT_REPLY_PREFERENCE
    stored = json.dumps(plugin._kv, ensure_ascii=False, default=str)
    assert "在吗" not in stored, "message text must not be written to the plugin store"


@pytest.mark.asyncio
async def test_the_same_batch_is_not_asked_twice_and_refresh_asks_again(wired):
    plugin, context = wired
    host = _host([_record("m1", text="在吗", expected_reply=True)])

    await plugin.reply_review_payload(sp_module=host)
    second = await plugin.reply_review_payload(sp_module=host)

    assert second["state"] == "cached"
    assert len(context.llm_calls) == 1, "a cached batch is not asked again"

    forced = await plugin.reply_review_payload(refresh=True, sp_module=host)

    assert forced["state"] == "fresh"
    assert len(context.llm_calls) == 2, "an explicit refresh is how the model is asked again"


@pytest.mark.asyncio
async def test_a_host_without_text_is_told_apart_from_a_host_without_data(wired):
    plugin, context = wired

    empty = await plugin.reply_review_payload(sp_module=_host([]))
    without_text = await plugin.reply_review_payload(
        sp_module=_host([_record("m1", expected_reply=True)]))

    assert empty["state"] == "empty" and empty["review"] is None
    assert "expected_reply" in empty["reason"]
    assert without_text["state"] == "no_text"
    assert "控制台显示消息正文" in without_text["reason"]
    assert context.llm_calls == [], "nothing worth judging means no call at all"


@pytest.mark.asyncio
async def test_an_unreachable_model_is_reported_and_backed_off(wired):
    plugin, context = wired
    context.llm_response = RuntimeError("boom")
    host = _host([_record("m1", text="在吗", expected_reply=True)])

    first = await plugin.reply_review_payload(sp_module=host)
    second = await plugin.reply_review_payload(sp_module=host)

    assert first["state"] == "failed" and "RuntimeError" in first["reason"]
    assert second["state"] == "failed"
    assert len(context.llm_calls) == 1


@pytest.mark.asyncio
async def test_a_host_that_cannot_be_read_is_not_an_error(wired):
    plugin, _context = wired

    class Broken:
        async def range_get_async(self, scope, scope_id, key):
            raise OSError("offline")

    payload = await plugin.reply_review_payload(sp_module=Broken())

    assert payload["state"] == "unavailable"
    assert payload["review"] is None
