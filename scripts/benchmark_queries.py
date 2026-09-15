"""Reproducible local query benchmark; SDK/KV doubles, real plugin query methods."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Explicit test-only SDK and in-memory KV; never connects to or writes a live host.
import tests.conftest  # noqa: E402,F401
from astrbot.api.star import Context  # noqa: E402
from astrbot_plugin_dynamics_learning.main import DynamicsLearningPlugin  # noqa: E402
from astrbot_plugin_dynamics_learning.core.samples import build_dataset  # noqa: E402
from tests.factories import annotated_sessions, shadow_block  # noqa: E402


async def measured(operation):
    delays = []
    stop = asyncio.Event()
    ready = asyncio.Event()

    async def pulse():
        ready.set()
        while not stop.is_set():
            start = time.perf_counter()
            await asyncio.sleep(0.005)
            delays.append(max(0.0, time.perf_counter() - start - 0.005) * 1000)

    monitor = asyncio.create_task(pulse())
    await ready.wait()
    start = time.perf_counter()
    result = await operation()
    elapsed = (time.perf_counter() - start) * 1000
    stop.set()
    await monitor
    return result, {"elapsed_ms": round(elapsed, 3),
                    "event_loop_max_lateness_ms": round(max(delays, default=0), 3),
                    "heartbeat_observations": len(delays)}


def corpus(count):
    base = build_dataset(annotated_sessions(
        sessions=32, per_session=math.ceil(count / 96), start=1_789_480_000))[:count]
    output = []
    for index, sample in enumerate(base):
        shadow = shadow_block(baseline_reply=index % 4 == 0, shadow_reply=index % 3 == 0,
                              recorded_at=sample.timestamp)
        shadow.update(experiment_id="benchmark", candidate_hash="synthetic-candidate",
                      baseline_hash="synthetic-baseline", host_version="benchmark-host")
        output.append(replace(sample, trace={**sample.trace, "shadow": shadow}))
    assert len(output) == count
    return output


async def main():
    logging.getLogger("astrbot-test").setLevel(logging.ERROR)
    _, idle = await measured(lambda: asyncio.sleep(0.1))
    results = []
    for size in (1000, 5000, 10000):
        plugin = DynamicsLearningPlugin(Context(), {})
        plugin._samples = corpus(size)
        writes = Counter({"put": 0, "delete": 0})
        original_put = plugin.put_kv_data
        original_delete = plugin.delete_kv_data

        async def put(key, value):
            writes["put"] += 1
            await original_put(key, value)

        async def delete(key):
            writes["delete"] += 1
            await original_delete(key)

        plugin.put_kv_data = put
        plugin.delete_kv_data = delete
        item = {"samples": size, "sessions": len({s.session_hash for s in plugin._samples}),
                "tasks": dict(Counter(s.task for s in plugin._samples)),
                "configured_default_max_samples": plugin.runtime_config().max_samples,
                "queries": {}}
        for name in ("quality", "attribution", "scopes", "shadow"):
            call = getattr(plugin, name + "_payload")
            plugin._query_cache.invalidate()
            before = sum(writes.values())
            payload, cold = await measured(call)
            cold["storage_writes"] = sum(writes.values()) - before
            warm = []
            for _ in range(3):
                before = sum(writes.values())
                _, timing = await measured(call)
                timing["storage_writes"] = sum(writes.values()) - before
                warm.append(timing)
            item["queries"][name] = {
                "cold": cold, "warm_runs": warm,
                "warm_median_ms": round(statistics.median(row["elapsed_ms"] for row in warm), 3),
                "payload_json_bytes": len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))}
            if name == "shadow":
                item["queries"][name]["operational_source_status"] = payload["operational"]["source_status"]
                item["queries"][name]["labelled_shadow_rows"] = payload.get("rows")
        item["storage_writes_total"] = dict(writes)
        await plugin._query_cache.close()
        results.append(item)
        print("completed", size, flush=True)
    sources = ("main.py", "core/query_cache.py", "core/quality.py", "core/attribution.py",
               "core/scope_profile.py", "core/shadow.py")
    artifact = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "processor": platform.processor()},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "method": {"sdk": "explicit test double", "storage": "in-memory SDK KV with put/delete counters",
                   "sample_plane": "preloaded plugin._samples; excludes ingest, disk load and HTTP/JSON serialization",
                   "synthetic_shadow": "complete synthetic identity and comparisons; not real host evidence",
                   "sizes": "1000, default limit 5000, stress 10000 (preloaded beyond default ingestion cap)",
                   "cold": "invalidate QueryCache before each endpoint; one run",
                   "warm": "three immediate calls to same endpoint; defaults and 30-second cache TTL",
                   "event_loop": "5ms asyncio heartbeat lateness; includes timer/GIL/scheduler noise; short calls have only one observation",
                   "scope": "actual quality/attribution/scopes/shadow plugin methods; no old-version comparison",
                   "operational_shadow": "unavailable SDK host telemetry deliberately exercises degraded path"},
        "idle_heartbeat_baseline": idle,
        "results": results,
    }
    output = ROOT / "docs/query-benchmark.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    asyncio.run(main())
