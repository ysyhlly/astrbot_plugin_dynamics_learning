"""The model-written contract review: what it may see, what survives.

The review is the one place in this plugin where a model writes something a
person then reads, so these tests are about the boundary rather than the prose:
the digest carries no identities, the counts come from the corpus and never from
the reply, an invented number is reported instead of printed, and every failure
lands back on the deterministic table.
"""
from __future__ import annotations

import json

import pytest

from astrbot_plugin_dynamics_learning.core.review import (
    REVIEW_PROMPT_VERSION, build_digest, build_prompt, digest_fingerprint, parse_review,
    unverified_numbers,
)

FENCE = chr(96) * 3


def _quality(**overrides):
    payload = {
        "reader_version": 1,
        "dataset": {"samples": 0, "sessions": 0, "scopes": 0, "scope_level": "session",
                    "degraded_traces": 0, "tasks": {"recipient": 0},
                    "candidate_evidence": {"full": 0}, "timestamp_semantics": "annotated_at",
                    "source_note": "样本只来自人工标注"},
        "trace": {"supported": [2, 3], "latest": 3, "observed": {"3": 4}, "unreadable": []},
        "capabilities": {
            "recipient_replay": {
                "status": "unsupported", "eligible": 0, "total": 0, "coverage": None,
                "definition": "定向阈值回放",
                "reasons": ["还没有收件人标注样本：本体记录 bot_targeted 才会产生。"],
            },
            "topic_attribution": {
                "status": "ok", "eligible": 4, "total": 5, "coverage": 0.8,
                "definition": "话题候选归因", "reasons": ["1 条没有候选集"],
            },
        },
        "blocked": [],
        "contract_findings": ["标注 schema 分布：2×3"],
        "notes": ["样本不含消息正文"],
        "dataset_gate": {"ok": False, "summary": "样本不足", "checks": [
            {"name": "samples", "status": "block", "detail": "0/40", "value": 0,
             "threshold": 40, "blocking": True}]},
    }
    payload.update(overrides)
    return payload


def _reply(**overrides):
    payload = {
        "headline": "还没有标注样本，先补收件人标注",
        "verdict": "empty",
        "rows": [
            {"capability": "recipient_replay", "status": "unsupported", "label": "缺标注",
             "explanation": "摘要里 samples 是 0", "next_action": "在场景回放标注 bot_targeted"},
        ],
        "actions": ["先标注 20 条收件人样本"],
        "caveats": ["样本只来自人工标注"],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


# ---- the digest ---------------------------------------------------------

def test_the_digest_carries_counts_and_capability_ids_but_no_identities():
    digest = build_digest(_quality())

    assert digest["prompt_version"] == REVIEW_PROMPT_VERSION
    assert [row["id"] for row in digest["capabilities"]] == ["recipient_replay",
                                                             "topic_attribution"]
    assert digest["dataset"]["samples"] == 0
    assert digest["trace"]["observed"] == {"3": 4}
    assert digest["gate"]["checks"][0]["threshold"] == 40
    # Whitelisted, not trimmed: the payload grows fields every time the plugin
    # learns something new, and none of them should start travelling to a model.
    assert "source_note" not in digest["dataset"]


def test_the_fingerprint_follows_the_data_and_not_the_clock():
    first = build_digest(_quality(generated_at=1.0))
    same = build_digest(_quality(generated_at=999.0))
    changed = build_digest(_quality(dataset={"samples": 1, "sessions": 1}))

    assert digest_fingerprint(first) == digest_fingerprint(same)
    assert digest_fingerprint(changed) != digest_fingerprint(first)


def test_the_prompt_names_the_capabilities_and_the_vocabulary():
    system, user = build_prompt(build_digest(_quality()))

    assert "recipient_replay" in user
    assert "数据摘要" in user
    assert "unsupported" in system and "insufficient" in system
    assert "只输出一个 JSON 对象" in system


# ---- the reply ----------------------------------------------------------

def test_a_reply_keeps_the_counts_from_the_digest_and_fills_what_it_skipped():
    digest = build_digest(_quality())

    parsed = parse_review(_reply(), digest)

    assert parsed is not None
    first, second = parsed["rows"]
    assert first["capability"] == "recipient_replay"
    assert first["source"] == "model"
    assert first["next_action"] == "在场景回放标注 bot_targeted"
    assert first["total"] == 0 and first["coverage"] is None
    # A row the model skipped is reported as the plugin verdict, not dropped.
    assert second["capability"] == "topic_attribution"
    assert second["source"] == "deterministic"
    assert second["status"] == "ok"
    assert second["eligible"] == 4 and second["total"] == 5
    assert parsed["model_rows"] == 1 and parsed["deterministic_rows"] == 1
    assert parsed["verdict"] == "empty" and parsed["verdict_label"] == "还没有数据"


def test_a_model_that_disagrees_is_shown_as_disagreeing():
    reply = _reply(rows=[{"capability": "topic_attribution", "status": "insufficient",
                          "label": "样本少", "explanation": "摘要里 total 是 5",
                          "next_action": "再标几条"}])

    parsed = parse_review(reply, build_digest(_quality()))
    row = next(item for item in parsed["rows"] if item["capability"] == "topic_attribution")

    assert row["status"] == "insufficient"
    assert row["deterministic_status"] == "ok"
    assert row["agrees"] is False
    assert parsed["disagreements"] == 1


def test_an_unknown_status_falls_back_to_the_plugin_verdict():
    reply = _reply(rows=[{"capability": "topic_attribution", "status": "great",
                          "label": "好", "explanation": "", "next_action": ""}])

    parsed = parse_review(reply, build_digest(_quality()))
    row = next(item for item in parsed["rows"] if item["capability"] == "topic_attribution")

    assert row["status"] == "ok"
    assert row["agrees"] is True


def test_an_invented_capability_is_reported_and_dropped():
    reply = _reply(rows=[{"capability": "made_up", "status": "ok", "label": "x",
                          "explanation": "y", "next_action": "z"}])

    parsed = parse_review(reply, build_digest(_quality()))

    assert parsed["invented_capabilities"] == ["made_up"]
    assert {row["capability"] for row in parsed["rows"]} == {"recipient_replay",
                                                             "topic_attribution"}


def test_a_number_the_digest_does_not_contain_is_reported_not_printed():
    digest = build_digest(_quality())

    parsed = parse_review(_reply(actions=["先攒到 2000 条样本再谈门槛"]), digest)

    assert parsed["unverified_numbers"] == ["2000"]
    # Numbers that are in the digest pass, including their percentage spelling.
    assert unverified_numbers("需要 20 条样本", digest) == []
    assert unverified_numbers("覆盖率 80%", digest) == []


def test_prose_instead_of_json_is_refused():
    digest = build_digest(_quality())

    assert parse_review("我觉得这批数据还行，先标注吧。", digest) is None
    assert parse_review("", digest) is None


def test_a_fenced_reply_is_read():
    digest = build_digest(_quality())
    text = f"{FENCE}json\n{_reply()}\n{FENCE}"

    parsed = parse_review(text, digest)

    assert parsed is not None and parsed["headline"]


# ---- the runtime --------------------------------------------------------

class _Meta:
    id = "provider-1"
    model = "model-x"


class _Provider:
    def meta(self):
        return _Meta()


@pytest.fixture
def wired(plugin, fake_context):
    fake_context.provider = _Provider()
    fake_context.llm_response = _reply()
    return plugin, fake_context


@pytest.mark.asyncio
async def test_the_model_is_asked_once_and_then_served_from_cache(wired):
    plugin, context = wired

    first = await plugin.contract_review_payload()
    second = await plugin.contract_review_payload()

    assert first["state"] == "fresh"
    assert first["provider_id"] == "provider-1"
    assert first["review"]["headline"]
    assert second["state"] == "cached"
    assert len(context.llm_calls) == 1
    assert context.llm_calls[0]["provider_id"] == "provider-1"
    assert "数据摘要" in context.llm_calls[0]["prompt"]


@pytest.mark.asyncio
async def test_an_explicit_refresh_asks_again(wired):
    plugin, context = wired

    await plugin.contract_review_payload()
    await plugin.contract_review_payload(refresh=True)

    assert len(context.llm_calls) == 2


@pytest.mark.asyncio
async def test_a_disabled_review_never_calls_the_model(plugin, fake_context):
    fake_context.provider = _Provider()
    fake_context.llm_response = _reply()
    plugin.config = {"learning_review_enabled": False}

    payload = await plugin.contract_review_payload()

    assert payload["state"] == "disabled"
    assert payload["review"] is None
    assert "learning_review_enabled" in payload["reason"]
    assert fake_context.llm_calls == []


@pytest.mark.asyncio
async def test_without_a_provider_the_page_is_told_so(plugin, fake_context):
    fake_context.provider = None

    payload = await plugin.contract_review_payload()

    assert payload["state"] == "unavailable"
    assert "Provider" in payload["reason"]
    assert payload["review"] is None
    assert fake_context.llm_calls == []


@pytest.mark.asyncio
async def test_an_unreadable_reply_falls_back_and_backs_off(wired):
    plugin, context = wired
    context.llm_response = "我觉得这批数据还行"

    first = await plugin.contract_review_payload()
    second = await plugin.contract_review_payload()

    assert first["state"] == "failed" and "JSON" in first["reason"]
    assert second["state"] == "failed"
    assert len(context.llm_calls) == 1, "a failed model is not asked once per refresh"

    forced = await plugin.contract_review_payload(refresh=True)

    assert forced["state"] == "failed"
    assert len(context.llm_calls) == 2, "refresh=1 is how a failed model is asked again"


@pytest.mark.asyncio
async def test_a_raising_provider_is_reported_not_swallowed(wired):
    plugin, context = wired
    context.llm_response = RuntimeError("boom")

    payload = await plugin.contract_review_payload()

    assert payload["state"] == "failed"
    assert "RuntimeError" in payload["reason"]
    assert payload["review"] is None


def test_the_review_endpoint_is_registered(plugin, fake_context):
    plugin.web.register()

    assert "review" in {route.rsplit("/", 1)[-1] for route, *_ in fake_context.routes}
