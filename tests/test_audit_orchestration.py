"""Regression coverage for analysis scheduling and reset consistency."""
from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning import main
from astrbot_plugin_dynamics_learning.core.autotune import (
    DECISION_PROMOTE, TuneRun, TuneStep, _finalise, run_tuning,
)
from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.policy import BASE_POLICY
from astrbot_plugin_dynamics_learning.core.samples import build_dataset

from .factories import annotated_sessions


@pytest.mark.asyncio
async def test_analysis_keeps_event_loop_available(plugin, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()
    worker_threads = []

    def slow_analysis(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        started.set()
        assert release.wait(3), "event loop did not release the worker"
        return SimpleNamespace(as_dict=lambda: {"done": True}, tuning=[],
                               evaluation=None, dataset_gate={"ok": True})

    monkeypatch.setattr(main, "analyze", slow_analysis)
    task = asyncio.create_task(plugin.run_analysis())
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        assert not task.done()
        assert len(worker_threads) == 1
        assert worker_threads[0] != loop_thread
    finally:
        release.set()
        await task
    assert plugin._last_report == {"done": True}


@pytest.mark.asyncio
async def test_reset_clears_contract_and_review_caches(plugin):
    plugin._last_contract = {"records": 12}
    plugin._reply_review_cache = {"fingerprint": "old"}
    plugin._review_failed_at = 123
    plugin._reply_review_failed_at = 123
    await plugin.store.save_review({"fingerprint": "old"})
    await plugin.reset_storage()
    assert plugin._last_contract == {}
    assert plugin._reply_review_cache == {}
    assert plugin._review_failed_at == plugin._reply_review_failed_at == 0
    assert not await plugin.store.load_review()
    assert not (await plugin.store.load_state()).get("last_contract_stats")


def test_tuning_confidence_counts_holdout_samples_and_preserves_transition():
    samples = build_dataset(annotated_sessions(sessions=8, per_session=80))
    final = {**BASE_POLICY, "strong_addressivity_threshold": 0.665}
    run = TuneRun(task="recipient", decision=DECISION_PROMOTE, final_policy=final,
                  steps=[TuneStep(index=1, policy=final, safe=True)])
    result = _finalise(run, samples, LearningConfig(), "recipient", BASE_POLICY, (), 100)
    candidate = result.candidate
    assert candidate is not None
    assert candidate.evidence["holdout_support"] >= 40
    assert candidate.confidence in {"low", "moderate"}
    assert candidate.status_history[-1]["from"] == "proposed"
    assert candidate.status_history[-1]["to"] == "validated"


def test_unreplayable_rows_cannot_satisfy_tuning_sample_gate():
    samples = build_dataset(annotated_sessions(sessions=8, per_session=80))
    samples = [replace(sample, trace={**sample.trace, "contribution_total_recorded": False},
                       features={**sample.features, "ctx_explicit": 0.0})
               for sample in samples]
    run = run_tuning(samples, task="recipient")
    assert run.decision == "insufficient"
    assert run.candidate is None
    assert "0 条可评测样本" in run.stop_reason
