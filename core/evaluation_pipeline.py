"""Candidate generation and final evidence have separate ownership.

One context reserves evidence before selection; every generator's output goes
through the same finalizer. This module never persists or publishes policies.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from .config import LearningConfig
from .evaluator import (
    VERDICT_ACCEPTED, EvaluationReport, Split, contract_compatibility,
    dataset_fingerprint, holdout_record, promotion_check, split_by_session,
    split_by_time, target_facts, validate_frozen,
)
from .metrics import METRIC_SCHEMA_VERSION
from .policy import PolicyCandidate, STATUS_PROPOSED, STATUS_VALIDATED
from .samples import LearningSample


@dataclass(frozen=True)
class EvaluationContext:
    """Analysis inputs and pre-reserved splits shared by all candidate paths."""

    samples: tuple[LearningSample, ...]
    temporal: Split
    reserved: Split
    config: LearningConfig
    now: float
    host_version: str | None
    baseline_source: str
    evaluation_enabled: bool
    dataset_gate_ok: bool
    dataset_fingerprint: str

    @property
    def development(self) -> tuple[LearningSample, ...]:
        return self.reserved.train

    @classmethod
    def reserve(cls, samples: Sequence[LearningSample], *, config: LearningConfig,
                now: float, host_version: str | None, baseline_source: str,
                evaluation_enabled: bool, dataset_gate_ok: bool) -> EvaluationContext:
        samples = tuple(deepcopy(samples))
        temporal = (split_by_time(samples, ratio=config.forward_holdout_ratio)
                    if config.require_forward_validation else Split(tuple(samples), ()))
        future_messages = {(row.session_hash, row.msg_id) for row in temporal.holdout}
        temporal = Split(tuple(row for row in temporal.train
                               if (row.session_hash, row.msg_id) not in future_messages),
                         tuple(row for row in samples
                               if (row.session_hash, row.msg_id) in future_messages))
        reserved = split_by_session(temporal.train, holdout_ratio=config.holdout_ratio)
        return cls(tuple(samples), temporal, reserved, config, now, host_version,
                   baseline_source, evaluation_enabled, dataset_gate_ok, dataset_fingerprint(samples))

    def finalize(self, candidate: PolicyCandidate) -> tuple[
        PolicyCandidate, EvaluationReport, EvaluationReport | None, dict[str, Any]
    ]:
        if dataset_fingerprint(self.samples) != self.dataset_fingerprint:
            raise ValueError("Evaluation snapshot changed during candidate generation")
        # Validate the exact precision that storage and the host contract preserve.
        candidate = candidate.with_fields(
            params={key: round(float(value), 4) for key, value in candidate.params.items()},
            baseline={key: round(float(value), 4) for key, value in candidate.baseline.items()})
        session = validate_frozen(candidate, self.samples, split=self.reserved, kind="final_session",
                                  config=self.config, now=self.now)
        future = (validate_frozen(candidate, self.samples,
                                  split=Split(self.development, self.temporal.holdout), kind="final_time",
                                  config=self.config, now=self.now)
                  if self.config.require_forward_validation else None)
        gate = promotion_check(session, future, config=self.config).as_dict()
        accepted = (self.evaluation_enabled and gate["verdict"] == VERDICT_ACCEPTED
                    and self.dataset_gate_ok)
        candidate = candidate.with_fields(
            training_dataset={"fingerprint": self.dataset_fingerprint,
                              "development_fingerprint": dataset_fingerprint(self.development),
                              "samples": len(self.samples)},
            holdout_result=holdout_record(session),
            forward_result=holdout_record(future) if future else {},
            compatibility=contract_compatibility(self.samples),
            target={**target_facts(self.host_version, candidate.baseline),
                    "baseline_source": self.baseline_source,
                    "baseline_verified": self.baseline_source == "host_effective"},
            evidence={**dict(candidate.evidence), "final_validation": gate,
                      "metric_schema_version": METRIC_SCHEMA_VERSION,
                      "require_forward_validation": self.config.require_forward_validation,
                      "dataset_gate_ok": bool(self.dataset_gate_ok),
                      "evaluation_enabled": self.evaluation_enabled},
        ).with_status(STATUS_VALIDATED if accepted else STATUS_PROPOSED, now=self.now,
                      reason="最终冻结候选验证通过" if accepted else "最终验证或数据门槛未通过")
        session.candidate = candidate
        if future is not None:
            future.candidate = candidate
        return candidate, session, future, gate
