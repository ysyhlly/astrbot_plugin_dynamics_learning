"""Offline smoke run: synthesise annotations, import them, then analyse.

This exercises the whole path a real deployment takes — ingest, sample building,
both learners, the evaluation gate and the policy store — without a running
AstrBot and without touching ChatDynamics. It is the fastest way to see what the
console will show, and to sanity-check a change end to end.

    python scripts/smoke.py
    python scripts/smoke.py --sessions 30 --per-session 24 --json out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PLUGIN_DIR.parent
for _path in (str(WORKSPACE_ROOT), str(PLUGIN_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# The unit-test double is the supported way to run this plugin outside AstrBot:
# a real Star binds its KV store to the bot's live preferences database.
from astrbot_plugin_dynamics_learning.tests.conftest import _install_astrbot_double  # noqa: E402

_install_astrbot_double()

from astrbot.api.star import Context  # noqa: E402
from astrbot_plugin_dynamics_learning.main import DynamicsLearningPlugin  # noqa: E402
from astrbot_plugin_dynamics_learning.tests.factories import (  # noqa: E402
    annotated_sessions, export_payload,
)


async def run(sessions: int, per_session: int) -> dict:
    plugin = DynamicsLearningPlugin(Context(), {})
    await plugin.initialize()
    try:
        rows = annotated_sessions(sessions=sessions, per_session=per_session, biased=True)
        ingested = await plugin.ingest(source="export", payload=export_payload(rows))
        analysis = await plugin.run_analysis(with_evaluation=True)
        policies = await plugin.policies_payload()
        quality = await plugin.quality_payload()
        scopes = await plugin.scopes_payload()
        return {"ingest": ingested, "report": analysis["report"], "policies": policies,
                "quality": quality, "scopes": scopes}
    finally:
        await plugin.terminate()


def summarise(payload: dict) -> str:
    ingest = payload["ingest"]
    report = payload["report"]
    evaluation = (report or {}).get("evaluation") or {}
    lines = [
        f"导入：{ingest['annotations']} 条标注 -> {ingest['imported_samples']} 条样本 / "
        f"{ingest['sessions']} 个会话",
        f"评测结论：{evaluation.get('verdict')}  "
        + "；".join(evaluation.get("reasons") or []),
    ]
    for run in (report or {}).get("tuning", []):
        lines.append(
            f"迭代调参 [{run['task']}] {run['decision_label']}"
            f"（采纳 {run['adopted_steps']} 步，上限 {run['rules']['max_steps']} 步）：{run['stop_reason']}")
        for row in run["drift"]:
            lines.append(f"    累计漂移 {row['param']} {row['baseline']} -> {row['value']} "
                         f"({row['delta_ratio']:+.2%})")
        for step in run["steps"]:
            moves = "、".join(f"{m['param']} {m['delta_ratio']:+.2%}" for m in step["changes"])
            lines.append(
                f"    step{step['index']} {moves} | 边际 {step['step_delta']} "
                f"累计 {step['cumulative_delta']} | 目标错误 {step['target_error']} "
                f"累计相对 {step['target_error_cumulative']} | "
                f"{'Safe' if step['safe'] else '未通过'}")
    for row in (report or {}).get("recommendations", []):
        marker = "可采纳" if row["actionable"] else "仅诊断"
        why = (row.get("evidence") or {}).get("downgrade_reason")
        suffix = f"（{why}）" if why else ""
        lines.append(f"  [{marker}] {row['title']} {row['detail']}{suffix}")
    candidates = ((report or {}).get("topic") or {}).get("candidate_metrics") or {}
    recall = candidates.get("candidate_recall") or {}
    selection = candidates.get("selection_accuracy") or {}
    if recall.get("recorded"):
        lines.append(f"话题候选：覆盖 {recall.get('coverage')} "
                     f"Recall@3 {recall.get('recall_at_3')} "
                     f"选中准确率 {selection.get('accuracy')}")
    for name, task in sorted((evaluation.get("tasks") or {}).items()):
        primary = task.get("primary_metric")
        row = (task.get("deltas") or {}).get(primary) or {}
        lines.append(
            f"  {name}: {primary} {row.get('before')} -> {row.get('after')} "
            f"({row.get('delta')})  留出 {task.get('holdout')}")
    quality = payload.get("quality") or {}
    for name, row in (quality.get("capabilities") or {}).items():
        lines.append(f"  能力 {name}: {row['status_label']} "
                     f"{row['eligible']}/{row['total']}")
    contract = quality.get("contract")
    if contract:
        lines.append(f"契约面：{contract['annotations_kept']}/{contract['annotations_seen']} 条进入样本，"
                     f"计数{'守恒' if contract['balanced'] else '不守恒'}，"
                     f"trace 缺失 {contract['decision_trace_absent']} 条")
    else:
        lines.append("契约面：这次导入没有留下原始记录计数")
    for row in (payload.get("scopes") or {}).get("rows", [])[:3]:
        lines.append(f"会话 {row['scope_label']}：{row['samples']} 条被检查样本 · "
                     f"{row['confidence_label']} · 主要问题 "
                     f"{'、'.join(row['dominant_labels']) or '—'} · {row['diagnosis_label']}")
    lines.append(f"策略记录：{payload['policies']['total']} 条")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--per-session", type=int, default=18)
    parser.add_argument("--json", type=Path, help="also write the full payload here")
    args = parser.parse_args(argv)

    payload = asyncio.run(run(args.sessions, args.per_session))
    print(summarise(payload))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"完整结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
