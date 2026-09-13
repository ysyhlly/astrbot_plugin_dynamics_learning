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
import time
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


def attach_shadow(rows, *, baseline=0.70, shadow=0.67):
    """Give every synthesised record a shadow comparison, as a real run would.

    Synthesised rather than computed, because the point of the flag is to show
    what the shadow *report* looks like — the host's own computation is what the
    cross-repo contract test exercises.
    """
    decorated = []
    for session, record in rows:
        trace = record.get("decision_trace") or {}
        participation = trace.get("participation") or {}
        score = float(participation.get("contribution_total") or 0.0)
        level = participation.get("level")
        codes = {fact.get("code") for fact in (participation.get("evidence") or [])
                 if isinstance(fact, dict)}
        structural = bool(codes & {"canonical_recipient", "bot_mention", "other_mention",
                                   "vocative", "bot_reply", "routed_bot", "routed_other",
                                   "bot_subject"})
        baseline_reply = level == "strong"
        if structural:
            shadow_reply, reason = baseline_reply, "structural"
        else:
            shadow_reply, reason = max(0.0, min(1.0, score)) >= shadow, "ambient"
        record = dict(record)
        trace = dict(trace)
        trace["shadow"] = {
            "policy_id": "policy_v1",
            "baseline_threshold": baseline,
            "shadow_threshold": shadow,
            "baseline_reply": baseline_reply,
            "shadow_reply": shadow_reply,
            "changed": baseline_reply != shadow_reply,
            "score": round(max(0.0, min(1.0, score)), 6),
            "baseline_margin": round(score - baseline, 6),
            "shadow_margin": round(score - shadow, 6),
            "reason": reason,
            "recorded_at": float(record.get("annotated_at") or 0.0),
        }
        record["decision_trace"] = trace
        decorated.append((session, record))
    return decorated


async def run(sessions: int, per_session: int, *, shadow: bool = False) -> dict:
    plugin = DynamicsLearningPlugin(Context(), {})
    await plugin.initialize()
    try:
        # Dated "now": the dataset gate blocks a corpus whose newest label is
        # older than gate_max_label_age_days, and a synthetic batch timestamped
        # in 1970 is exactly that case. The gate is part of what this script is
        # meant to exercise, so it is given a corpus that can pass it.
        rows = annotated_sessions(sessions=sessions, per_session=per_session, biased=True,
                                  start=time.time() - 7200)
        if shadow:
            rows = attach_shadow(rows)
        ingested = await plugin.ingest(source="export", payload=export_payload(rows))
        analysis = await plugin.run_analysis(with_evaluation=True)
        policies = await plugin.policies_payload()
        quality = await plugin.quality_payload()
        scopes = await plugin.scopes_payload()
        attribution = await plugin.attribution_payload()
        published = await plugin.published_payload()
        shadow_report = await plugin.shadow_payload()
        return {"ingest": ingested, "report": analysis["report"], "policies": policies,
                "quality": quality, "scopes": scopes, "attribution": attribution,
                "published": published, "shadow": shadow_report}
    finally:
        await plugin.terminate()


def summarise(payload: dict) -> str:
    ingest = payload["ingest"]
    report = payload["report"]
    evaluation = (report or {}).get("evaluation") or {}
    gate = (report or {}).get("dataset_gate") or {}
    promotion = (report or {}).get("promotion") or {}
    forward = (report or {}).get("forward") or {}
    lines = [
        f"导入：{ingest['annotations']} 条标注 -> {ingest['imported_samples']} 条样本 / "
        f"{ingest['sessions']} 个会话",
        f"数据门槛：{'通过' if gate.get('ok') else '未通过（' + '、'.join(gate.get('blocked_by') or []) + '）'}"
        f" —— {gate.get('summary', '')}",
        f"评测结论：{evaluation.get('verdict')}  "
        + "；".join(evaluation.get("reasons") or []),
        f"采纳门槛：{promotion.get('verdict')} —— "
        + "；".join(promotion.get("reasons") or []),
    ]
    if forward:
        lines.append("前向验证：" + str(forward.get("verdict")) + " "
                     + "；".join((forward.get("reasons") or [])[:1])
                     + f"（切分 {forward['split']['kind']}，"
                       f"留出 {forward['split']['holdout_samples']} 条）")
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
            f"({row.get('delta')})  留出 {task.get('holdout')}  {task.get('interval')}")
    outcome = (evaluation.get("outcome") or {})
    if outcome:
        lines.append(f"  最终发送层：{outcome.get('support')} 条记录结果"
                     f"（可回放={outcome.get('replayable')}）；"
                     f"未发送分阶段 {outcome.get('stages')}")
    attribution = payload.get("attribution") or {}
    if attribution.get("messages"):
        lines.append(f"错误归因链：{attribution['messages']} 条消息，"
                     f"模型错误 {attribution.get('model_errors')}、"
                     f"系统事件 {attribution.get('system_events')}、"
                     f"没有结果记录 {attribution.get('outcome_unavailable')}")
        for bucket, count in (attribution.get("counts") or {}).items():
            if count:
                lines.append(f"    {bucket}: {count}")
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
    shadow = payload.get("shadow") or {}
    if shadow.get("rows"):
        table = shadow.get("table") or {}
        lines.append(
            f"Shadow A/B：{shadow['rows']} 条记录（带标签 {shadow.get('labelled')}）· "
            f"分歧 {table.get('changed')} · 净收益 {table.get('net_gain')} · "
            f"总体 {table.get('overall_baseline_accuracy')} -> {table.get('overall_shadow_accuracy')}"
            f" · 子集 {table.get('subset_baseline_accuracy')} -> {table.get('subset_shadow_accuracy')}")
        lines.append(f"  配对表：都对 {table.get('both_correct')}、都错 {table.get('both_wrong')}、"
                     f"只有 baseline 对 {table.get('baseline_only')}、"
                     f"只有 shadow 对 {table.get('shadow_only')}（守恒="
                     f"{table.get('balanced')}）")
        gate = shadow.get("gate") or {}
        lines.append("  进入 active 的门槛：" + ("通过" if gate.get("ok")
                     else "未通过（" + "、".join(gate.get("blocked_by") or []) + "）"))
        for row in gate.get("checks") or []:
            lines.append(f"    [{row['status']}] {row['name']}: {row['detail']}")
    else:
        lines.append("Shadow A/B：没有记录（本体需要 learning_policy_mode=shadow；"
                     "本脚本加 --shadow 可以合成一份）")
    lines.append(f"策略记录：{payload['policies']['total']} 条 "
                 f"（状态分布 {payload['policies'].get('status_counts')}）")
    published = (payload.get("published") or {}).get("policies") or []
    lines.append(f"已发布给本体：{len(published)} 条"
                 + ("（" + "、".join(f"{row['policy_id']} shadow={row['shadow_observed']}"
                                     for row in published) + "）" if published else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--per-session", type=int, default=18)
    parser.add_argument("--shadow", action="store_true",
                        help="attach a synthesised shadow comparison to every record, "
                             "so the shadow A/B report is exercised too")
    parser.add_argument("--json", type=Path, help="also write the full payload here")
    args = parser.parse_args(argv)

    payload = asyncio.run(run(args.sessions, args.per_session, shadow=args.shadow))
    print(summarise(payload))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"完整结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
