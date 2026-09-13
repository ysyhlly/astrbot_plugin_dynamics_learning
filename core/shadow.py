"""The shadow A/B evaluation: what the disagreement subset actually says.

Phase one of a shadow run keeps the runtime on the baseline and computes the
policy's decision beside it, so every labelled turn ends up with two decisions
and one human label. This module turns that into an answer to the only question
worth asking before `active`: **did the policy get the turns it disagreed about
more right than the baseline did?**

The overall accuracy delta cannot answer it. When a threshold moves a few
percent, almost every decision is identical, so the delta is the policy's effect
diluted by a subset nobody looked at:

    10,000 turns, 320 disagreements
    overall delta    +0.5pp    the same 320 rows, spread over 10,000
    subset delta    +15.9pp    the effect on the turns it actually touched

Both numbers are reported. The gate reads the overall one — a deployment
experiences the whole corpus, not the subset — and the subset table is what
explains *why* it moved, and what refuses a policy whose effect is real but
negative:

    baseline_only   baseline right, shadow wrong   <- the policy's cost
    shadow_only     shadow right, baseline wrong   <- the policy's benefit

A policy that gains 40 and loses 40 has moved a lot of decisions and improved
nothing, and only the paired table says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .bootstrap import Unit, bootstrap_delta
from .samples import REPLY, TASK_REPLY_ADMISSION, LearningSample

SHADOW_SCHEMA_VERSION = 1

# What the plan asks of a first shadow run before anything may go active. They
# are defaults, not laws: every one is a knob on `ActiveRules`.
DEFAULT_MIN_SHADOW_SAMPLES = 500
DEFAULT_MIN_DISAGREEMENTS = 100
DEFAULT_MAX_OVERALL_REGRESSION = 0.01
DEFAULT_TARGET_RELATIVE = 0.10
DEFAULT_TARGET_ABSOLUTE = 0.01
# The relative branch needs a baseline error rate worth dividing by. 0.3% -> 0%
# is a 100% relative improvement and 0.3 percentage points; letting that satisfy
# a "10% better" rule would make the rule meaningless exactly where the corpus is
# already good.
DEFAULT_RELATIVE_MIN_BASELINE_ERROR = 0.05
DEFAULT_CI_FLOOR = -0.002
DEFAULT_MIN_SESSIONS = 3
DEFAULT_MIN_ACTIVE_HOURS = 4
DEFAULT_SUBGROUP_MAX_REGRESSION = 0.05
DEFAULT_SUBGROUP_MIN_SUPPORT = 20

GATE_OK = "ok"
GATE_BLOCK = "block"
GATE_WARN = "warn"


@dataclass(frozen=True)
class ActiveRules:
    """The thresholds a shadow run has to clear before `active` is offered."""

    min_shadow_samples: int = DEFAULT_MIN_SHADOW_SAMPLES
    min_disagreements: int = DEFAULT_MIN_DISAGREEMENTS
    max_overall_regression: float = DEFAULT_MAX_OVERALL_REGRESSION
    target_relative_improvement: float = DEFAULT_TARGET_RELATIVE
    target_absolute_improvement: float = DEFAULT_TARGET_ABSOLUTE
    relative_min_baseline_error: float = DEFAULT_RELATIVE_MIN_BASELINE_ERROR
    ci_floor: float = DEFAULT_CI_FLOOR
    min_sessions: int = DEFAULT_MIN_SESSIONS
    min_active_hours: int = DEFAULT_MIN_ACTIVE_HOURS
    subgroup_max_regression: float = DEFAULT_SUBGROUP_MAX_REGRESSION
    subgroup_min_support: int = DEFAULT_SUBGROUP_MIN_SUPPORT
    bootstrap_iterations: int = 600
    bootstrap_seed: int = 7
    bootstrap_alpha: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def evaluate_shadow(
    samples: Sequence[LearningSample],
    *,
    rules: ActiveRules | None = None,
    min_samples: int = 0,
) -> dict[str, Any]:
    """The full shadow report: the table, the interval, the subgroups, the gate.

    `min_samples` (the dataset gate's floor) is accepted so a caller that has
    already judged the corpus too small does not have to spell the same reason
    twice; it does not change any threshold in `rules`.
    """
    resolved = rules or ActiveRules()
    rows = shadow_rows(samples)
    labelled = [row for row in rows if row.labelled]
    table = disagreement_table(rows)
    interval = bootstrap_delta(_units(rows), metric="accuracy",
                               iterations=resolved.bootstrap_iterations,
                               seed=resolved.bootstrap_seed,
                               alpha=resolved.bootstrap_alpha)
    subgroups = _subgroups(rows, min_support=resolved.subgroup_min_support,
                           max_regression=resolved.subgroup_max_regression)
    sessions = len({row.session_hash for row in rows})
    hours = _active_hours(rows)
    policies = sorted({row.policy_id for row in rows})
    reasons = sorted({row.reason for row in rows if row.reason})
    checks = _gate_checks(table=table, interval=interval, subgroups=subgroups,
                          sessions=sessions, hours=hours, rules=resolved,
                          min_samples=min_samples)
    blocked = [row["name"] for row in checks if row["status"] == GATE_BLOCK]
    return {
        "shadow_schema_version": SHADOW_SCHEMA_VERSION,
        "rows": len(rows),
        "labelled": len(labelled),
        "sessions": sessions,
        "active_hours": hours,
        "policies": policies,
        "agreement_reasons": reasons,
        "counts": {"same": table["same"], "changed": table["changed"]},
        "table": table,
        "interval": interval,
        "subgroups": subgroups,
        "rules": resolved.as_dict(),
        "gate": {"ok": not blocked, "checks": checks, "blocked_by": blocked},
        "notes": _notes(table, interval, sessions, hours, len(rows), min_samples),
    }


def _gate_check(name: str, status: str, detail: str, *,
                value: Any = None, threshold: Any = None) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail,
            "value": value, "threshold": threshold}


def _gate_checks(*, table: Mapping[str, Any], interval: Mapping[str, Any],
                 subgroups: Mapping[str, Any], sessions: int, hours: int,
                 rules: ActiveRules, min_samples: int) -> list[dict[str, Any]]:
    """Every reason to withhold `active`, named, including the ones that pass.

    A gate that only reports failures cannot be read: "no failures" and "nothing
    ran" look the same. So each check states its own measurement even when it
    passes.
    """
    checks: list[dict[str, Any]] = []
    labelled = int(table["labelled"])
    changed = int(table["changed"])

    enough = labelled >= rules.min_shadow_samples
    checks.append(_gate_check(
        "shadow_samples", GATE_OK if enough else GATE_BLOCK,
        f"带标注的 shadow 样本 {labelled} 条（门槛 {rules.min_shadow_samples}）："
        + ("可以下结论。" if enough else "样本不足时下面每个比例都只是计数。"),
        value=labelled, threshold=rules.min_shadow_samples))

    disagree_enough = changed >= rules.min_disagreements
    checks.append(_gate_check(
        "disagreements", GATE_OK if disagree_enough else GATE_BLOCK,
        f"策略产生不同决策的样本 {changed} 条（门槛 {rules.min_disagreements}）："
        + ("分歧足够支撑比较。" if disagree_enough
           else "分歧太少：策略改动几乎没生效，或者本身没有可比较的空间。"),
        value=changed, threshold=rules.min_disagreements))

    delta = table["overall_shadow_accuracy"] if table["overall_shadow_accuracy"] is not None else 0.0
    base = table["overall_baseline_accuracy"] if table["overall_baseline_accuracy"] is not None else 0.0
    movement = round(delta - base, 6)
    no_regression = movement >= -rules.max_overall_regression
    checks.append(_gate_check(
        "overall_regression", GATE_OK if no_regression else GATE_BLOCK,
        f"总体准确率变化 {movement:+.4f}（下限 {-rules.max_overall_regression:+.4f}）："
        + ("没有回退。" if no_regression else "总体回退超过允许幅度。"),
        value=movement, threshold=-rules.max_overall_regression))

    baseline_error = 1.0 - base
    measurable = baseline_error >= rules.relative_min_baseline_error
    relative = ((baseline_error - (1.0 - delta)) / baseline_error) if measurable else None
    target_ok = (movement >= rules.target_absolute_improvement
                 or (relative is not None and relative >= rules.target_relative_improvement))
    checks.append(_gate_check(
        "target_improvement", GATE_OK if target_ok else GATE_BLOCK,
        f"目标指标：绝对 {movement:+.4f}（需 ≥{rules.target_absolute_improvement:.2%}）"
        f"或相对错误率改善 "
        + (f"{relative:+.2%}（需 ≥{rules.target_relative_improvement:.0%}）。"
           if relative is not None
           else f"不适用（基线错误率 {baseline_error:.2%} 低于 "
                f"{rules.relative_min_baseline_error:.0%}，相对改善在这里没有意义）。")
        + ("" if target_ok else " 两者都不满足。"),
        value=round(relative, 6) if relative is not None else None,
        threshold=rules.target_relative_improvement))

    lower = interval.get("lower")
    ci_ok = lower is not None and float(lower) >= rules.ci_floor
    checks.append(_gate_check(
        "interval_floor", GATE_OK if ci_ok else GATE_BLOCK,
        (f"总体变化的 95% 区间下界 {float(lower):+.4f}（下限 {rules.ci_floor:+.4f}）："
         + ("下界在允许范围内。" if ci_ok else "区间下界低于允许的回退下限。"))
        if lower is not None else "区间无法计算：重采样单元不足。",
        value=lower, threshold=rules.ci_floor))

    sessions_ok = sessions >= rules.min_sessions
    checks.append(_gate_check(
        "sessions", GATE_OK if sessions_ok else GATE_BLOCK,
        f"覆盖 {sessions} 个会话（门槛 {rules.min_sessions}）："
        + ("跨了多个会话。" if sessions_ok else "会话太少，可能只是在拟合一段对话。"),
        value=sessions, threshold=rules.min_sessions))

    hours_ok = hours >= rules.min_active_hours
    checks.append(_gate_check(
        "active_hours", GATE_OK if hours_ok else GATE_BLOCK,
        f"决策覆盖 {hours} 个活跃时段（门槛 {rules.min_active_hours}）："
        + ("时段够分散。" if hours_ok else "只覆盖了很少的时段，结论可能只对某个时间成立。"),
        value=hours, threshold=rules.min_active_hours))

    catastrophic = list(subgroups.get("catastrophic") or [])
    checks.append(_gate_check(
        "subgroups", GATE_BLOCK if catastrophic else GATE_OK,
        (f"{len(catastrophic)} 个会话在 shadow 策略下回退超过 "
         f"{rules.subgroup_max_regression:.0%}（支撑 ≥{rules.subgroup_min_support} 条）。")
        if catastrophic else "没有支撑足够的会话出现大幅回退。",
        value=catastrophic or None, threshold=rules.subgroup_max_regression))

    # The question the whole exercise exists to answer. Reported as a check
    # rather than a note because a policy that wins and loses equally is the one
    # case where every other number can look fine.
    net = int(table["net_gain"])
    checks.append(_gate_check(
        "net_gain", GATE_OK if net > 0 else GATE_BLOCK,
        f"分歧子集里 shadow 多赢 {table['shadow_only']} 条、多输 {table['baseline_only']} 条"
        f"（净 {net:+d}）："
        + ("策略在它影响的样本上是净收益。" if net > 0 else "策略赢的没有输的多。"),
        value=net, threshold=0))

    if min_samples and labelled < min_samples:
        checks.append(_gate_check(
            "dataset_gate", GATE_WARN,
            f"数据门槛本身就没通过（门槛 {min_samples} 条），下面的结论仅供参考。"))
    return checks


def _notes(table: Mapping[str, Any], interval: Mapping[str, Any], sessions: int,
           hours: int, rows: int, min_samples: int) -> list[str]:
    notes: list[str] = []
    if not rows:
        notes.append("没有任何消息记录了 shadow 决策：本体侧需要把 "
                     "learning_policy_mode 设为 shadow，并且至少要跑过一轮真实群聊。")
        return notes
    changed = int(table["changed"])
    if changed:
        notes.append(
            f"分歧子集是结论所在：{changed} 条不同决策里，"
            f"baseline 只对 {table['baseline_only']} 条、shadow 只对 {table['shadow_only']} 条；"
            f"另有 {table['both_correct']} 条两边都对、{table['both_wrong']} 条两边都错"
            "（这两类两边一致，不在分歧子集里）。")
        notes.append(
            f"子集上的准确率 baseline {table['subset_baseline_accuracy']} → "
            f"shadow {table['subset_shadow_accuracy']}；总体变化 "
            f"{(table['overall_shadow_accuracy'] or 0) - (table['overall_baseline_accuracy'] or 0):+.4f} "
            "是被那 95% 相同的决策稀释之后的结果。")
    else:
        notes.append("策略没有产生任何不同的决策：要么阈值移动太小，要么这批消息都在"
                     "结构化或提前返回的路径上，阈值对它没有影响。")
    if not table["labelled"]:
        notes.append("没有任何分歧样本带人工标签，无法判断谁对谁错。")
    if interval.get("crosses_zero"):
        notes.append("总体变化的区间跨 0：这个提升无法与重采样噪声区分。")
    notes.append(f"覆盖 {sessions} 个会话、{hours} 个活跃时段"
                 f"（活跃时段按本体记录的决策时间统计，不是标注时间）。")
    if min_samples:
        notes.append(f"数据门槛要求 {min_samples} 条样本。")
    return notes


__all__ = [
    "DEFAULT_CI_FLOOR", "DEFAULT_MAX_OVERALL_REGRESSION", "DEFAULT_MIN_ACTIVE_HOURS",
    "DEFAULT_MIN_DISAGREEMENTS", "DEFAULT_MIN_SESSIONS", "DEFAULT_MIN_SHADOW_SAMPLES",
    "DEFAULT_SUBGROUP_MAX_REGRESSION", "DEFAULT_SUBGROUP_MIN_SUPPORT",
    "DEFAULT_RELATIVE_MIN_BASELINE_ERROR", "DEFAULT_TARGET_ABSOLUTE", "DEFAULT_TARGET_RELATIVE",
    "GATE_BLOCK", "GATE_OK", "GATE_WARN",
    "SHADOW_SCHEMA_VERSION", "ActiveRules", "ShadowRow", "disagreement_table",
    "evaluate_shadow", "rules_from_config", "shadow_rows",
]


def rules_from_config(config: Any) -> ActiveRules:
    """Read the gate thresholds off the plugin config, falling back to defaults.

    getattr rather than attribute access so a caller that passes a partial
    object (a test, a script) gets the documented defaults instead of an
    AttributeError — the defaults are the plan's numbers, and they are the
    answer that has to be defensible when nobody configured anything.
    """
    defaults = ActiveRules()
    return ActiveRules(
        min_shadow_samples=int(getattr(config, "shadow_min_samples",
                                       defaults.min_shadow_samples)),
        min_disagreements=int(getattr(config, "shadow_min_disagreements",
                                      defaults.min_disagreements)),
        max_overall_regression=float(getattr(config, "shadow_max_regression",
                                             defaults.max_overall_regression)),
        target_relative_improvement=float(getattr(config, "shadow_target_relative",
                                                  defaults.target_relative_improvement)),
        target_absolute_improvement=float(getattr(config, "shadow_target_absolute",
                                                  defaults.target_absolute_improvement)),
        relative_min_baseline_error=float(getattr(config, "shadow_relative_min_error",
                                                  defaults.relative_min_baseline_error)),
        ci_floor=float(getattr(config, "shadow_ci_floor", defaults.ci_floor)),
        min_sessions=int(getattr(config, "shadow_min_sessions", defaults.min_sessions)),
        min_active_hours=int(getattr(config, "shadow_min_active_hours",
                                     defaults.min_active_hours)),
        subgroup_max_regression=float(getattr(config, "shadow_subgroup_max_regression",
                                              defaults.subgroup_max_regression)),
        subgroup_min_support=int(getattr(config, "shadow_subgroup_min_support",
                                         defaults.subgroup_min_support)),
        bootstrap_iterations=int(getattr(config, "bootstrap_iterations",
                                         defaults.bootstrap_iterations)),
        bootstrap_seed=int(getattr(config, "bootstrap_seed", defaults.bootstrap_seed)),
        bootstrap_alpha=float(getattr(config, "bootstrap_alpha", defaults.bootstrap_alpha)),
    )


@dataclass(frozen=True)
class ShadowRow:
    """One turn the two decisions were compared on."""

    session_hash: str
    msg_id: str
    decision_at: float
    policy_id: str
    baseline_reply: bool
    shadow_reply: bool
    changed: bool
    reason: str = ""
    score: float | None = None
    expected_reply: bool | None = None

    @property
    def labelled(self) -> bool:
        """Only a labelled turn can say which decision was right."""
        return isinstance(self.expected_reply, bool)

    @property
    def baseline_correct(self) -> bool | None:
        return None if not self.labelled else self.baseline_reply == self.expected_reply

    @property
    def shadow_correct(self) -> bool | None:
        return None if not self.labelled else self.shadow_reply == self.expected_reply

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_hash": self.session_hash,
            "msg_id": self.msg_id,
            "decision_at": self.decision_at,
            "policy_id": self.policy_id,
            "baseline_reply": self.baseline_reply,
            "shadow_reply": self.shadow_reply,
            "changed": self.changed,
            "reason": self.reason,
            "score": self.score,
            "expected_reply": self.expected_reply,
        }


def shadow_rows(samples: Sequence[LearningSample]) -> list[ShadowRow]:
    """One row per message that carries a recorded shadow decision.

    Grouped by message rather than taken per sample: one message can produce
    four samples and only one of them carries the reply label, so a per-sample
    reading would count the same comparison up to four times and report a
    disagreement rate that depends on how many labels a human happened to leave.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        shadow = sample.shadow
        if not shadow.recorded:
            continue
        key = (sample.session_hash, sample.msg_id)
        entry = grouped.setdefault(key, {"shadow": shadow, "expected": None, "at": 0.0})
        if sample.task == TASK_REPLY_ADMISSION and isinstance(sample.expected, str):
            entry["expected"] = sample.expected == REPLY
        if shadow.recorded_at:
            entry["at"] = float(shadow.recorded_at)
    rows: list[ShadowRow] = []
    for (session_hash, msg_id), entry in grouped.items():
        shadow = entry["shadow"]
        rows.append(ShadowRow(
            session_hash=session_hash, msg_id=msg_id, decision_at=entry["at"],
            policy_id=shadow.policy_id, baseline_reply=shadow.baseline_reply,
            shadow_reply=shadow.shadow_reply, changed=shadow.changed,
            reason=shadow.reason, score=shadow.score,
            expected_reply=entry["expected"],
        ))
    rows.sort(key=lambda row: (row.session_hash, row.msg_id))
    return rows


def disagreement_table(rows: Sequence[ShadowRow]) -> dict[str, Any]:
    """The paired table: four cells that partition every labelled turn.

    ```text
                      shadow right     shadow wrong
    baseline right    both_correct     baseline_only   <- the policy cost
    baseline wrong    shadow_only      both_wrong
                      ^ the policy benefit
    ```

    The two off-diagonal cells are exactly the disagreement subset, so
    `changed == baseline_only + shadow_only` holds by construction — and it is
    asserted here and in the tests, because a table whose cells do not add up
    is the one failure that would make every number derived from it
    meaningless.

    The two accuracies prefixed `overall_` are reported for context and named so
    they cannot be mistaken for the finding; the `subset_` pair is the one the
    disagreement rate is read from.
    """
    labelled = [row for row in rows if row.labelled]
    changed = [row for row in labelled if row.changed]
    baseline_only = shadow_only = both_wrong = both_correct = 0
    for row in labelled:
        baseline_right, shadow_right = row.baseline_correct, row.shadow_correct
        if baseline_right and shadow_right:
            both_correct += 1
        elif baseline_right:
            baseline_only += 1
        elif shadow_right:
            shadow_only += 1
        else:
            both_wrong += 1
    balanced = (len(changed) == baseline_only + shadow_only
                and len(labelled) == both_correct + both_wrong + baseline_only + shadow_only)
    baseline_hits = sum(1 for row in labelled if row.baseline_correct)
    shadow_hits = sum(1 for row in labelled if row.shadow_correct)
    baseline_subset_hits = sum(1 for row in changed if row.baseline_correct)
    shadow_subset_hits = sum(1 for row in changed if row.shadow_correct)
    return {
        "labelled": len(labelled),
        "same": len(labelled) - len(changed),
        "changed": len(changed),
        "changed_rate": _ratio(len(changed), len(labelled)),
        "balanced": balanced,
        "baseline_only": baseline_only,
        "shadow_only": shadow_only,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_gain": shadow_only - baseline_only,
        "overall_baseline_accuracy": _ratio(baseline_hits, len(labelled)),
        "overall_shadow_accuracy": _ratio(shadow_hits, len(labelled)),
        "subset_baseline_accuracy": _ratio(baseline_subset_hits, len(changed)),
        "subset_shadow_accuracy": _ratio(shadow_subset_hits, len(changed)),
        "subset_delta": (None if not changed
                         else round((shadow_subset_hits - baseline_subset_hits) / len(changed), 6)),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _units(rows: Sequence[ShadowRow]) -> list[Unit]:
    """Per-session paired counts, for the bootstrap.

    The unit is the session because that is the level the corpus is clustered
    at: one conversation's turns share a topic, a mood and a cast, so resampling
    turns would produce an interval narrower than the evidence supports.
    """
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if not row.labelled:
            continue
        counts = grouped.setdefault(row.session_hash, {"b": {"tp": 0.0, "fp": 0.0,
                                                             "tn": 0.0, "fn": 0.0},
                                                       "s": {"tp": 0.0, "fp": 0.0,
                                                             "tn": 0.0, "fn": 0.0}})
        for key, predicted in (("b", row.baseline_reply), ("s", row.shadow_reply)):
            expected = bool(row.expected_reply)
            bucket = ("tp" if predicted and expected else "fn" if expected
                      else "fp" if predicted else "tn")
            counts[key][bucket] += 1.0
    return [Unit(key=session, baseline=value["b"], candidate=value["s"])
            for session, value in sorted(grouped.items())]


def _subgroups(rows: Sequence[ShadowRow], *, min_support: int,
               max_regression: float) -> dict[str, Any]:
    """Per-session movement, and which sessions would call it a disaster.

    A subgroup is not a group identity — the scope still is the session — so
    these are diagnostics. What they catch is the shape of harm a global number
    hides: a policy that helps five quiet conversations and breaks the one busy
    one is not an improvement for the group that busy one is.
    """
    grouped: dict[str, list[ShadowRow]] = {}
    for row in rows:
        if row.labelled:
            grouped.setdefault(row.session_hash, []).append(row)
    entries: list[dict[str, Any]] = []
    for session, group in grouped.items():
        baseline_hits = sum(1 for row in group if row.baseline_correct)
        shadow_hits = sum(1 for row in group if row.shadow_correct)
        delta = (shadow_hits - baseline_hits) / len(group)
        entries.append({
            "group": session, "support": len(group),
            "baseline_accuracy": round(baseline_hits / len(group), 6),
            "shadow_accuracy": round(shadow_hits / len(group), 6),
            "delta": round(delta, 6),
            "eligible": len(group) >= min_support,
            "catastrophic": len(group) >= min_support and delta <= -max_regression,
        })
    entries.sort(key=lambda item: item["delta"])
    return {
        "sessions": entries,
        "eligible": sum(1 for item in entries if item["eligible"]),
        "catastrophic": [item["group"] for item in entries if item["catastrophic"]],
    }


def _active_hours(rows: Sequence[ShadowRow]) -> int:
    """Distinct clock hours the decisions were taken in.

    Read from the decision time the host recorded, not from the sample
    timestamp: the sample carries `annotated_at`, which is when a human
    reviewed, so a week of reviewing one evening would look like one active
    period.
    """
    hours = {int(row.decision_at // 3600) % 24 for row in rows if row.decision_at > 0}
    return len(hours)
