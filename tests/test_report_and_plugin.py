"""The analysis snapshot and the plugin surface end to end."""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.config import LearningConfig, parse_learning_config
from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY
from astrbot_plugin_dynamics_learning.core.recommendation import KIND_CONFIG_PARAM
from astrbot_plugin_dynamics_learning.core.report import analyze, dataset_summary, error_breakdown, overview
from astrbot_plugin_dynamics_learning.core.samples import build_dataset

from .factories import ambient_record, annotated_sessions, export_payload


def _samples():
    # 16 x 18 is the smallest biased batch where the in-sample sweep and the
    # holdout iteration disagree, which is the case several tests here need.
    return build_dataset(annotated_sessions(sessions=16, per_session=18))


def test_dataset_summary_counts_tasks_and_sessions():
    samples = _samples()
    summary = dataset_summary(samples)
    assert summary["samples"] == len(samples)
    assert summary["sessions"] == len({s.session_hash for s in samples})
    assert sum(summary["tasks"].values()) == len(samples)
    assert summary["first_timestamp"] <= summary["last_timestamp"]


def test_overview_reports_undefined_accuracy_for_an_absent_task():
    record = ambient_record("m1", seed=1)
    # Drop the topic supervision so only the recipient task has samples.
    record["expected_topic"] = "UNKNOWN"
    samples = build_dataset([("s", record)])
    assert samples, "the record still carries recipient and reply supervision"
    payload = overview(samples, now=samples[0].timestamp)
    assert payload["topic"]["total"] == 0
    assert payload["topic"]["accuracy"] is None
    assert payload["recipient"]["total"] > 0


def test_error_breakdown_groups_by_task():
    samples = _samples()
    breakdown = error_breakdown(samples)
    assert set(breakdown) <= {"recipient", "topic", "reply"}
    for counts in breakdown.values():
        assert all(isinstance(value, int) for value in counts.values())


def test_analysis_runs_the_iterative_tuner_and_reports_its_verdict():
    result = analyze(_samples())
    assert [row.task for row in result.tuning] == ["recipient", "topic"]
    payload = result.as_dict()
    assert len(payload["tuning"]) == 2
    for run in payload["tuning"]:
        assert run["decision"] in {
            "strong_promote", "promote", "candidate", "reject", "rollback",
            "needs_review", "no_change", "insufficient",
        }
        assert run["rules"]["step_delta_ratio"] == 0.05
        assert run["rules"]["max_steps"] == 3
        assert "不会自动修改" in run["note"]
    assert result.promoted_runs, "the biased batch should reach a promote tier"


def test_tuning_can_be_switched_off():
    result = analyze(_samples(), with_tuning=False)
    assert result.tuning == []
    assert result.as_dict()["tuning"] == []


def test_a_proposal_that_contradicts_the_iteration_is_downgraded():
    """An in-sample sweep and a holdout iteration can point opposite ways.

    When they do, the holdout wins and the in-sample number stops being
    actionable — otherwise the console would offer two contradictory changes.
    """
    result = analyze(_samples())
    by_param = {row.param: row for row in result.recommendations if row.param}
    drift = {}
    for run in result.tuning:
        for row in (run.as_dict()["drift"] or []):
            drift[row["param"]] = row["delta"]
    conflicts = 0
    for param, recommendation in by_param.items():
        moved = drift.get(param)
        if moved is None or recommendation.before is None or recommendation.after is None:
            continue
        if (recommendation.after - recommendation.before) * moved < 0:
            conflicts += 1
            assert recommendation.actionable is False
            assert recommendation.evidence["downgrade_reason"] == \
                "样本内建议方向与留出集迭代结论相反"
    assert conflicts, "the fixture is chosen so the two directions disagree"


def test_every_parameter_proposal_carries_its_tuning_verdict():
    result = analyze(_samples())
    for row in result.recommendations:
        if row.param is None:
            continue
        assert "tuning_decision" in row.evidence
        assert row.evidence["tuning_decision"] in {
            row2.decision for row2 in result.tuning
        } or row.evidence["tuning_decision"] == "no_change"


def test_analysis_pairs_every_proposal_with_its_verdict():
    samples = _samples()
    result = analyze(samples)
    assert result.evaluation is not None
    verdict = result.evaluation.verdict
    for row in result.recommendations:
        if row.kind == KIND_CONFIG_PARAM:
            assert row.evidence.get("evaluation_verdict") == verdict
            if verdict != "accepted":
                assert row.actionable is False
    payload = result.as_dict()
    assert payload["recipient"]["task"] == "recipient"
    assert payload["topic"]["task"] == "topic"


def test_analysis_without_evaluation_still_reports_statistics():
    result = analyze(_samples(), with_evaluation=False)
    assert result.evaluation is None
    assert result.dataset["samples"] > 0


def test_config_parsing_is_total_over_host_shapes():
    assert parse_learning_config(None).source_plugin_id == "ysyhlly/astrbot_plugin_chat_dynamics"
    wrapped = parse_learning_config({"learning_max_samples": {"value": 77}})
    assert wrapped.max_samples == 77
    assert parse_learning_config({"learning_max_samples": -5}).max_samples == 50
    assert parse_learning_config({"learning_max_samples": "x"}).max_samples == 5000
    assert parse_learning_config({"learning_max_param_delta_ratio": 99}).max_param_delta_ratio == 0.20
    assert parse_learning_config({"learning_holdout_ratio": 0.99}).holdout_ratio == 0.60
    assert parse_learning_config({"source_plugin_id": "  "}).source_plugin_id == \
        "ysyhlly/astrbot_plugin_chat_dynamics"
    assert parse_learning_config({"learning_auto_analyze": "yes"}).auto_analyze is False
    assert parse_learning_config(LearningConfig(max_samples=9)).max_samples == 9
    class Dummy:
        def get(self, key):
            return {"learning_enabled": False}.get(key)
    assert parse_learning_config(Dummy()).enabled is False


# ---- plugin surface ------------------------------------------------------


@pytest.mark.asyncio
async def test_plugin_ingests_an_export_then_reports(plugin):
    rows = annotated_sessions(sessions=14, per_session=16)
    result = await plugin.ingest(source="export", payload=export_payload(rows))
    assert result["imported_samples"] > 0
    assert result["sessions"] == 14
    assert result["stored_samples"] == result["imported_samples"]
    # One annotation produces up to three samples, so the counts differ and are
    # reported separately instead of as one ambiguous number.
    assert result["annotations"] < result["imported_samples"]

    overview_payload = await plugin.overview_payload()
    assert overview_payload["dataset"]["samples"] == result["imported_samples"]
    assert overview_payload["contract"]["writes_to_host"] is False

    report = await plugin.run_analysis(with_evaluation=True)
    assert report["has_report"] is True
    assert report["limits"]["shadow_only"] is True
    assert report["report"]["evaluation"]["verdict"] in {"accepted", "rejected", "insufficient"}
    assert len(report["report"]["tuning"]) == 2

    policies = await plugin.policies_payload()
    assert policies["total"] >= 1
    # The iterative runs own the policy records; the single-shot candidate is a
    # fallback, so the same adjustment is never recorded twice.
    assert all(row["source"] == "iterative_tuning" for row in policies["rows"])
    version = policies["rows"][0]["version"]
    updated = await plugin.update_policy(version, "accept")
    assert updated["status"] == "accepted"
    assert "未发生任何变化" in updated["note"]
    with pytest.raises(ValueError):
        await plugin.update_policy("policy_v999", "accept")


@pytest.mark.asyncio
async def test_plugin_samples_page_redacts_identities(plugin):
    await plugin.ingest(source="export", payload=export_payload(
        annotated_sessions(sessions=2, per_session=4)))
    page = await plugin.samples_payload(page=1, page_size=10)
    assert page["total"] > 0
    assert len(page["rows"]) == min(10, page["total"])
    row = page["rows"][0]
    assert "umo:group" not in row["session"]
    assert "msg" not in row["msg_id"] or "…" in row["msg_id"] or "*" in row["msg_id"]
    assert "features" not in row
    filtered = await plugin.samples_payload(task="recipient")
    assert all(item["task"] == "recipient" for item in filtered["rows"])


@pytest.mark.asyncio
async def test_plugin_export_contains_no_message_text(plugin):
    await plugin.ingest(source="export", payload=export_payload(
        annotated_sessions(sessions=3, per_session=4)))
    payload = await plugin.export_payload()
    assert payload["schema"] == "dynamics_learning_export_v1"
    blob = str(payload)
    assert "text" not in blob.replace("context", "")
    assert payload["samples"]


@pytest.mark.asyncio
async def test_plugin_reset_clears_everything(plugin):
    await plugin.ingest(source="export", payload=export_payload(
        annotated_sessions(sessions=3, per_session=4)))
    await plugin.run_analysis()
    cleared = await plugin.reset_storage()
    assert cleared["removed_samples"] > 0
    assert (await plugin.overview_payload())["dataset"]["samples"] == 0
    assert (await plugin.policies_payload())["total"] == 0
    assert (await plugin.report_payload())["has_report"] is False


@pytest.mark.asyncio
async def test_plugin_reports_a_missing_host_without_raising(plugin, monkeypatch):
    """A broken host must degrade this plugin, never take the bot down with it."""
    async def unavailable(*_args, **_kwargs):
        raise RuntimeError("no host")

    monkeypatch.setattr("astrbot_plugin_dynamics_learning.main.collect_from_host", unavailable)
    result = await plugin.ingest(source="host")
    assert result["ok"] is False
    assert result["imported_samples"] == 0
    assert result["diagnostics"]["available"] is False
    assert result["diagnostics"]["error"] == "RuntimeError"
    # The plugin stays usable: an export import still works afterwards.
    recovered = await plugin.ingest(source="export", payload=export_payload(
        annotated_sessions(sessions=2, per_session=3)))
    assert recovered["imported_samples"] > 0
    assert recovered["ok"] is True


@pytest.mark.asyncio
async def test_plugin_never_writes_outside_its_own_learning_keys(plugin):
    """The read-only guarantee, asserted against every key the plugin writes."""
    writes: list[str] = []
    original = plugin.put_kv_data

    async def recording(key, value):
        writes.append(key)
        await original(key, value)

    plugin.put_kv_data = recording  # type: ignore[method-assign]
    await plugin.ingest(source="export", payload=export_payload(
        annotated_sessions(sessions=4, per_session=6)))
    await plugin.run_analysis()
    await plugin.update_policy((await plugin.policies_payload())["rows"][0]["version"], "accept")
    await plugin.reset_storage()

    assert writes, "the run must have exercised the write path"
    for key in writes:
        assert key.startswith("learning_"), key
    # None of the host's own keys may ever be targeted.
    assert not any(key.startswith("topic_annotations_") for key in writes)
    assert "panel_runtime_v1" not in writes
    assert "cooling_until" not in writes


@pytest.mark.asyncio
async def test_plugin_registers_every_endpoint_once(plugin):
    plugin.web.register()
    routes = {route for route, _handler, _methods, _desc in plugin.context.routes}
    assert routes == {
        "/astrbot_plugin_dynamics_learning/overview",
        "/astrbot_plugin_dynamics_learning/samples",
        "/astrbot_plugin_dynamics_learning/quality",
        "/astrbot_plugin_dynamics_learning/scopes",
        "/astrbot_plugin_dynamics_learning/scope",
        "/astrbot_plugin_dynamics_learning/ingest",
        "/astrbot_plugin_dynamics_learning/analyze",
        "/astrbot_plugin_dynamics_learning/report",
        "/astrbot_plugin_dynamics_learning/policies",
        "/astrbot_plugin_dynamics_learning/policy",
        "/astrbot_plugin_dynamics_learning/export",
        "/astrbot_plugin_dynamics_learning/reset",
    }
    assert plugin.web.registered is True
    plugin.web.register()
    assert len(plugin.context.routes) == 12


def test_the_page_only_calls_endpoints_the_plugin_registers(plugin):
    r"""The page is not compiled, so a mistyped endpoint fails silently in a browser.

    The map in app.js must name routes the plugin actually registers, and must not
    carry a declaration nothing calls. The reverse is deliberately not asserted:
    the plugin exposes endpoints (reset, for one) that the page has no button for.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "pages" / "learning" / "app.js") \
        .read_text(encoding="utf-8")
    block = re.search(r"const ENDPOINTS = \{(.*?)\};", source, flags=re.S)
    assert block is not None, "app.js no longer declares an ENDPOINTS map"
    declared = dict(re.findall(r"(\w+):\s*\"([^\"]+)\"", block.group(1)))
    used = set(re.findall(r"ENDPOINTS\.(\w+)", source))

    plugin.web.register()
    registered = {route.rsplit("/", 1)[-1] for route, _h, _m, _d in plugin.context.routes}

    assert used <= set(declared), "an endpoint is used but not declared"
    assert set(declared) == used, "an endpoint is declared but never used"
    assert set(declared.values()) <= registered, "the page calls a route the plugin does not serve"


def test_the_page_script_only_touches_elements_the_page_has():
    r"""A mistyped id is a null dereference in the browser and nothing at build time."""
    import re
    from pathlib import Path

    pages = Path(__file__).resolve().parents[1] / "pages" / "learning"
    markup = (pages / "index.html").read_text(encoding="utf-8")
    script = (pages / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r'id="([^"]+)"', markup))
    referenced = set(re.findall(r'\$("([^"]+)")', script))
    views = set(re.findall(r'data-view="([^"]+)"', markup))
    buttons = set(re.findall(r'data-view-btn="([^"]+)"', script))

    assert referenced <= ids, sorted(referenced - ids)
    assert buttons <= views, sorted(buttons - views)


@pytest.mark.asyncio
async def test_plugin_lifecycle_is_idempotent(plugin):
    await plugin.initialize()
    await plugin.initialize()
    assert plugin.web.registered is True
    await plugin.terminate()
    await plugin.terminate()


@pytest.mark.asyncio
async def test_auto_analysis_stays_off_unless_enabled(plugin):
    plugin.config = {"learning_auto_analyze": False}
    await plugin.initialize()
    assert plugin._analysis_task is None
    await plugin.terminate()


def test_base_policy_is_exposed_for_the_evaluator():
    assert BASE_POLICY["strong_addressivity_threshold"] == 0.70
    assert BASE_POLICY["topic_commit_threshold"] == 0.58
