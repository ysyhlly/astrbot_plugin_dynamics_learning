"""Evaluation responsibilities derived from the replay contract, not data presence."""
from dataclasses import dataclass
from typing import Mapping

from .policy import PARAM_NAMES, normalize_policy
from .samples import TASK_RECIPIENT, TASK_REPLY_ADMISSION, TASK_TOPIC




@dataclass(frozen=True)
class EvaluationPlan:
    changed_params: tuple[str, ...]
    target_tasks: tuple[str, ...]
    affected_tasks: tuple[str, ...]
    unsupported_params: tuple[str, ...]

    def as_dict(self):
        return {name: list(getattr(self, name)) for name in
                ("changed_params", "target_tasks", "affected_tasks", "unsupported_params")}


def evaluation_plan(params: Mapping[str, float], baseline: Mapping[str, float]) -> EvaluationPlan:
    current, base = normalize_policy(params), normalize_policy(baseline)
    changed = tuple(name for name in PARAM_NAMES if abs(current[name] - base[name]) > 1e-9)
    targets, affected = [], []
    if "strong_addressivity_threshold" in changed:
        targets.append(TASK_RECIPIENT)
        affected.extend((TASK_RECIPIENT, TASK_REPLY_ADMISSION))
    if "topic_commit_threshold" in changed:
        targets.append(TASK_TOPIC)
        affected.append(TASK_TOPIC)
    # Hover only changes weak/hover, both silent and non-recipient in decide().
    # The remaining host parameters do not participate in the replay at all.
    unsupported = tuple(name for name in changed if name not in
                        ("strong_addressivity_threshold", "topic_commit_threshold"))
    return EvaluationPlan(changed, tuple(targets), tuple(affected), unsupported)
