"""The annotation window: what is still labelable, and for how long.

The host prunes its per-session graph on both a node cap and a TTL, so a message
is only annotatable while it is inside that graph. These tests pin the arithmetic
that turns a snapshot into a deadline, and the two distinctions the page depends
on: "already labelled" is not "gone", and a host that does not report its limits
must not be reported as if it had.
"""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.window import (
    FALLBACK_MAX_NODES, FALLBACK_TTL_SECONDS, MAX_TEXT_IN_EXPORT, session_digest,
    window_payload,
)

NOW = 1_800_000_000.0
SESSION = "umo:group:1"


def _node(msg_id, *, offset, text="你好", user="u1"):
    """One host node, `offset` seconds before the snapshot moment."""
    return {"msg_id": msg_id, "user_id": user, "text": text, "reply_to_id": "",
            "mentioned_users": [], "timestamp": 1000.0 - offset}


def _session(key=SESSION, nodes=()):
    return {"session_key": key, "umo": key, "group_id": "1", "nodes": list(nodes)}


def _runtime(*, sessions, graph=None, saved_wall=NOW, saved_clock=1000.0):
    payload = {"version": 1, "sessions": list(sessions)}
    if saved_wall is not None:
        payload["saved_wall"] = saved_wall
    if saved_clock is not None:
        payload["saved_clock"] = saved_clock
    if graph is not None:
        payload["graph"] = graph
    return payload


GRAPH = {"max_nodes": 500, "ttl_seconds": 3600.0}


def test_the_window_reports_what_is_left_and_when_the_oldest_expires():
    runtime = _runtime(graph=GRAPH, sessions=[_session(nodes=[
        _node("m1", offset=1800), _node("m2", offset=60)])])

    payload = window_payload(runtime, {}, now_wall=NOW)
    row = payload["sessions"][0]

    assert row["messages"] == 2 and row["with_text"] == 2
    assert row["oldest_wall"] == pytest.approx(NOW - 1800)
    assert row["newest_wall"] == pytest.approx(NOW - 60)
    assert row["span_seconds"] == pytest.approx(1740)
    assert row["idle_seconds"] == pytest.approx(60)
    assert row["expires_in_seconds"] == pytest.approx(1800)
    assert row["cap_pressure"] == pytest.approx(0.004)
    assert payload["limits"] == {"max_nodes": 500, "ttl_seconds": 3600.0,
                                 "reported_by_host": True}
    assert "30 分钟" in payload["hint"]


def test_already_labelled_messages_are_counted_rather_than_dropped():
    runtime = _runtime(graph=GRAPH, sessions=[_session(nodes=[
        _node("m1", offset=10), _node("m2", offset=20)])])

    payload = window_payload(runtime, {session_digest(SESSION): {"m1"}}, now_wall=NOW)
    row = payload["sessions"][0]

    assert row["annotated"] == 1
    assert row["unlabelled"] == 1
    assert payload["totals"]["unlabelled"] == 1
    assert payload["totals"]["annotated"] == 1


def test_a_window_with_nothing_left_to_label_says_that_instead_of_nagging():
    runtime = _runtime(graph=GRAPH, sessions=[_session(nodes=[_node("m1", offset=10)])])

    payload = window_payload(runtime, {session_digest(SESSION): {"m1"}}, now_wall=NOW)

    assert payload["totals"]["unlabelled"] == 0
    assert "都已经标注过了" in payload["hint"]


def test_an_oldest_message_past_its_ttl_is_called_out_rather_than_counted_down():
    runtime = _runtime(graph=GRAPH, sessions=[_session(nodes=[_node("m1", offset=7200)])])

    payload = window_payload(runtime, {}, now_wall=NOW)

    assert payload["sessions"][0]["expires_in_seconds"] < 0
    assert "超过本体的保留期" in payload["hint"]


def test_a_host_that_does_not_report_its_limits_is_not_reported_as_if_it_had():
    runtime = _runtime(sessions=[_session(nodes=[_node("m1", offset=60)])])

    payload = window_payload(runtime, {}, now_wall=NOW)

    assert payload["limits"] == {"max_nodes": FALLBACK_MAX_NODES,
                                 "ttl_seconds": FALLBACK_TTL_SECONDS,
                                 "reported_by_host": False}


def test_a_snapshot_without_the_clock_pair_reports_no_wall_times():
    runtime = _runtime(sessions=[_session(nodes=[_node("m1", offset=60)])],
                       saved_wall=None, saved_clock=None)

    row = window_payload(runtime, {}, now_wall=NOW)["sessions"][0]

    assert row["messages"] == 1
    assert row["oldest_wall"] is None and row["newest_wall"] is None
    assert row["expires_in_seconds"] is None


def test_an_empty_snapshot_is_its_own_answer():
    payload = window_payload({}, {}, now_wall=NOW)

    assert payload["sessions"] == []
    assert payload["totals"]["messages"] == 0
    assert "还没有会话" in payload["hint"]


def test_the_export_bounds_text_and_masks_identities():
    runtime = _runtime(graph=GRAPH, sessions=[_session(nodes=[
        _node("m1", offset=10, text="x" * 1000, user="1234567890")])])

    payload = window_payload(runtime, {session_digest(SESSION): {"m1"}}, now_wall=NOW,
                             include_messages=True)
    detail = payload["sessions"][0]["messages_detail"][0]

    assert len(detail["text"]) == MAX_TEXT_IN_EXPORT
    assert detail["user"] != "1234567890" and "…" in detail["user"]
    assert detail["annotated"] is True
    assert detail["wall"] == pytest.approx(NOW - 10)
    assert detail["has_text"] is True


def test_the_export_is_only_attached_when_it_is_asked_for():
    runtime = _runtime(graph=GRAPH, sessions=[_session(nodes=[_node("m1", offset=10)])])

    assert "messages_detail" not in window_payload(runtime, {}, now_wall=NOW)["sessions"][0]


# ---- the plugin surface -------------------------------------------------

class _Sp:
    """Shared-preferences double: one snapshot plus the annotation keys."""

    def __init__(self, runtime, records, session=SESSION):
        self.runtime = runtime
        self.rows = [
            {"key": "panel_runtime_v1", "value": {"val": runtime}},
            {"key": "topic_annotations_v1_" + session_digest(session),
             "value": {"val": list(records)}},
        ]
        self.reads = []

    async def get_async(self, **kwargs):
        self.reads.append(kwargs["key"])
        return self.runtime

    async def range_get_async(self, scope, scope_id, key):
        self.reads.append("range")
        return self.rows


@pytest.mark.asyncio
async def test_the_payload_reads_the_snapshot_and_the_labels_without_writing(plugin):
    runtime = _runtime(graph=GRAPH, sessions=[_session(nodes=[
        _node("m1", offset=10), _node("m2", offset=20)])])
    sp = _Sp(runtime, [{"msg_id": "m1", "annotation_schema_version": 2}])

    payload = await plugin.annotation_window_payload(sp_module=sp)

    assert payload["source_status"] == "ok"
    assert payload["totals"]["messages"] == 2
    assert payload["sessions"][0]["annotated"] == 1
    assert sp.reads == ["panel_runtime_v1", "range"]
    assert plugin._kv == {}, "the window is a read; nothing may be written"


@pytest.mark.asyncio
async def test_a_host_that_cannot_be_read_is_reported_rather_than_raised(plugin):
    class Broken:
        async def get_async(self, **kwargs):
            raise OSError("offline")

    payload = await plugin.annotation_window_payload(sp_module=Broken())

    assert payload["source_status"] == "unavailable"
    assert payload["sessions"] == []
