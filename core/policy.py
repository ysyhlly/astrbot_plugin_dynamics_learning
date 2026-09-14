"""Policy parameters, bounded deltas, and the parameterised replay decision.

Every parameter here is a **real ChatDynamics configuration key** with the real
default taken from its `_conf_schema.json`. A recommendation that cannot be
expressed as one of these keys is reported as engineering diagnostics instead
of being dressed up as a config change.

This module deliberately contains **no write path**: nothing here can change
the host's configuration. `PolicyCandidate` objects are records, not actions.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from .trace import DecisionTrace

# 2 adds the state machine, the validation results and the compatibility block.
POLICY_SCHEMA_VERSION = 2


def _finite_or_zero(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0

# Host configuration keys this plugin is allowed to reason about, with the
# host defaults and the host's own slider ranges.
PARAM_SPECS: dict[str, dict[str, Any]] = {
    "strong_addressivity_threshold": {
        "default": 0.70, "min": 0.50, "max": 0.90, "step": 0.02,
        "label": "强指代判定阈值",
        "hint": "定向度评分达到该值判定为明确指向机器人；同时决定 legacy 准入是否回复。",
    },
    "safe_hover_threshold": {
        "default": 0.40, "min": 0.20, "max": 0.60, "step": 0.02,
        "label": "安全悬停判定阈值",
        "hint": "低于强指代且高于该值时只静默记入图谱。",
    },
    "topic_commit_threshold": {
        "default": 0.58, "min": 0.30, "max": 0.95, "step": 0.02,
        "label": "话题确定归属阈值",
        "hint": "匹配得分达到该值且满足领先间隔才确定归入已有话题。默认由 topic_join_threshold 映射而来。",
    },
    "topic_join_threshold": {
        "default": 0.48, "min": 0.30, "max": 0.85, "step": 0.02,
        "label": "话题阈值兼容设置",
        "hint": "旧兼容键：小于 0.58 时确定归属阈值为 max(0.58, 本值 + 0.10)。",
    },
    "topic_margin_threshold": {
        "default": 0.06, "min": 0.0, "max": 0.50, "step": 0.01,
        "label": "话题确定归属领先间隔",
        "hint": "最佳候选相对第二候选的最低得分差。",
    },
    "parent_accept_threshold": {
        "default": 0.72, "min": 0.50, "max": 0.95, "step": 0.02,
        "label": "推断回复边接受阈值",
        "hint": "多因子打分达到该值且领先时才建立推断回复边。",
    },
}
PARAM_NAMES = tuple(PARAM_SPECS)

BASE_POLICY: dict[str, float] = {name: float(spec["default"]) for name, spec in PARAM_SPECS.items()}

# Error kinds, named the way the review page talks about them. Kept as literals
# so this module stays free of any dependency on the sample layer.
ERROR_MISSED_BOT = "missed_bot"
ERROR_FALSE_BOT = "false_bot"
ERROR_FRAGMENTATION = "fragmentation"
ERROR_WRONG_MERGE = "wrong_merge"
ERROR_MISSED_REPLY = "missed_reply"
ERROR_PREMATURE_REPLY = "premature_reply"
# The outcome layer's two mistakes. They are *not* the admission ones: the
# admission kinds count "the router's cut was wrong", these count "nothing was
# actually sent" / "something was sent that should not have been" — and a
# 作息压制 lands here with the router having been right.
ERROR_UNDELIVERED_REPLY = "undelivered_reply"
ERROR_UNSOLICITED_REPLY = "unsolicited_reply"

# What each adjustment is *for*, keyed by parameter and the direction of the
# move. An adjustment that cannot name the error it is aimed at can only be
# judged on the global number — which is how a genuinely good change gets
# thrown away for being "only" +0.6%.
TARGET_ERROR_BY_DIRECTION: dict[tuple[str, int], str] = {
    ("strong_addressivity_threshold", -1): ERROR_MISSED_BOT,
    ("strong_addressivity_threshold", +1): ERROR_FALSE_BOT,
    ("safe_hover_threshold", -1): ERROR_MISSED_BOT,
    ("safe_hover_threshold", +1): ERROR_FALSE_BOT,
    ("topic_commit_threshold", -1): ERROR_FRAGMENTATION,
    ("topic_commit_threshold", +1): ERROR_WRONG_MERGE,
    ("topic_join_threshold", -1): ERROR_FRAGMENTATION,
    ("topic_join_threshold", +1): ERROR_WRONG_MERGE,
}

# Evidence codes whose structural outcome is "the bot is the addressee".
TARGETED_EXPLICIT_CODES = frozenset({
    "canonical_recipient", "bot_mention", "vocative", "bot_reply", "routed_bot",
})

# ---- the policy state machine ------------------------------------------
#
# proposed  -> validated -> shadow -> promoted -> superseded / rolled_back
#
# Each arrow is a different kind of evidence, and collapsing them is how a
# number becomes a configuration change nobody agreed to:
#
#   proposed   the learners produced it; nothing has been checked yet
#   validated  it beat the baseline on a holdout this plugin never fitted on
#   shadow     it has been published beside the live behaviour and observed
#   promoted   a human promoted it; ChatDynamics may read it from /published
#   superseded a newer policy replaced it
#   rolled_back it was promoted and then taken back
#
# "validated" is where every automatic path stops. The evaluator can prove an
# improvement offline; it cannot prove anything about a gate, a generator or a
# platform adapter it never observed, so it does not get to promote.
STATUS_PROPOSED = "proposed"
STATUS_VALIDATED = "validated"
STATUS_SHADOW = "shadow"
STATUS_PROMOTED = "promoted"
STATUS_SUPERSEDED = "superseded"
STATUS_ROLLED_BACK = "rolled_back"
STATUS_REJECTED = "rejected"
STATUSES = (STATUS_PROPOSED, STATUS_VALIDATED, STATUS_SHADOW, STATUS_PROMOTED,
            STATUS_SUPERSEDED, STATUS_ROLLED_BACK, STATUS_REJECTED)

STATUS_LABEL = {
    STATUS_PROPOSED: "候选（未验证）",
    STATUS_VALIDATED: "已验证（留出集通过）",
    STATUS_SHADOW: "影子观察中",
    STATUS_PROMOTED: "已采纳",
    STATUS_SUPERSEDED: "已被新版本取代",
    STATUS_ROLLED_BACK: "已回滚",
    STATUS_REJECTED: "已拒绝",
}

# The transitions this plugin will perform. Anything absent is refused rather
# than silently accepted: a state machine that accepts every arrow is a field
# with extra steps.
ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    STATUS_PROPOSED: (STATUS_VALIDATED, STATUS_REJECTED),
    # `validated -> promoted` is allowed on purpose. Requiring the shadow arrow
    # would not create shadow evidence; it would create a state people click
    # through to get where they were going. The shortcut is *recorded* instead:
    # the published record carries `shadow_observed=False`, so a host reading it
    # can see that the offline result is the whole of the evidence.
    STATUS_VALIDATED: (STATUS_SHADOW, STATUS_PROMOTED, STATUS_REJECTED, STATUS_PROPOSED),
    STATUS_SHADOW: (STATUS_PROMOTED, STATUS_REJECTED, STATUS_ROLLED_BACK, STATUS_PROPOSED),
    STATUS_PROMOTED: (STATUS_SUPERSEDED, STATUS_ROLLED_BACK),
    STATUS_REJECTED: (STATUS_PROPOSED,),
    STATUS_ROLLED_BACK: (STATUS_PROPOSED,),
    STATUS_SUPERSEDED: (),
}
# Rows written by v0.7 and earlier. `candidate` was always the un-validated
# state, and `accepted` was the evaluator's verdict — which is what `validated`
# means now, not `promoted`: those records were never published to the host.
STATUS_ALIASES = {"candidate": STATUS_PROPOSED, "accepted": STATUS_VALIDATED}

ACTION_STATUS = {
    "validate": STATUS_VALIDATED,
    "shadow": STATUS_SHADOW,
    "promote": STATUS_PROMOTED,
    "accept": STATUS_PROMOTED,
    "supersede": STATUS_SUPERSEDED,
    "rollback": STATUS_ROLLED_BACK,
    "ignore": STATUS_REJECTED,
    "reopen": STATUS_PROPOSED,
}
ACTIONS = tuple(ACTION_STATUS)


def normalize_status(value: Any) -> str:
    """Map a stored status onto the current vocabulary, or fall back to proposed."""
    if not isinstance(value, str):
        return STATUS_PROPOSED
    text = STATUS_ALIASES.get(value.strip(), value.strip())
    return text if text in STATUSES else STATUS_PROPOSED


def can_transition(current: str, target: str) -> bool:
    """Whether a move is allowed. Re-applying the current state is a no-op.

    Idempotence is deliberate: two clicks on the same button, or a retry after a
    failed request, must not turn into an error the reader has to interpret.
    Every *other* arrow is checked against the table.
    """
    source = normalize_status(current)
    destination = normalize_status(target)
    if source == destination:
        return True
    return destination in ALLOWED_TRANSITIONS.get(source, ())


def transition_error(current: str, target: str) -> str:
    allowed = ALLOWED_TRANSITIONS.get(normalize_status(current), ())
    rendered = "、".join(allowed) if allowed else "（终态，不可再变）"
    return (f"策略状态不能从 {normalize_status(current)} 变为 {normalize_status(target)}；"
            f"允许的目标：{rendered}")

SOURCE_RECIPIENT_SWEEP = "recipient_threshold_sweep"
SOURCE_TOPIC_SWEEP = "topic_threshold_sweep"
SOURCE_MANUAL = "manual"


def clamp_param(name: str, value: Any) -> float:
    spec = PARAM_SPECS.get(name)
    if spec is None:
        raise KeyError(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(spec["default"])
    number = float(value)
    if not math.isfinite(number):
        return float(spec["default"])
    return max(float(spec["min"]), min(float(spec["max"]), number))


def normalize_policy(raw: Any) -> dict[str, float]:
    """Fill missing keys with host defaults and clamp everything into range."""
    policy = dict(BASE_POLICY)
    if isinstance(raw, Mapping):
        for name in PARAM_NAMES:
            if name in raw:
                policy[name] = clamp_param(name, raw[name])
    policy["safe_hover_threshold"] = min(policy["safe_hover_threshold"],
                                         round(policy["strong_addressivity_threshold"] - 0.05, 4))
    policy["safe_hover_threshold"] = clamp_param("safe_hover_threshold", policy["safe_hover_threshold"])
    return policy


def bounded_target(name: str, base: float, target: float, max_delta_ratio: float) -> float:
    """Move `base` toward `target` by at most `max_delta_ratio` of the base value."""
    base = clamp_param(name, base)
    target = clamp_param(name, target)
    if max_delta_ratio <= 0:
        return base
    budget = abs(base) * max_delta_ratio
    if budget == 0:
        return base
    delta = max(-budget, min(budget, target - base))
    return clamp_param(name, round(base + delta, 4))


def policy_deltas(base: Mapping[str, float], candidate: Mapping[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in PARAM_NAMES:
        before = float(base.get(name, BASE_POLICY[name]))
        after = float(candidate.get(name, BASE_POLICY[name]))
        if abs(after - before) < 1e-9:
            continue
        spec = PARAM_SPECS[name]
        rows.append({
            "param": name,
            "label": spec["label"],
            "before": round(before, 4),
            "after": round(after, 4),
            "delta": round(after - before, 4),
            "delta_ratio": round((after - before) / before, 4) if before else None,
        })
    return rows


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text_tuple(value: Any, limit: int = 32) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item)[:96] for item in list(value)[:limit] if isinstance(item, str) and item)


@dataclass(frozen=True)
class PolicyCandidate:
    """A record of one proposed parameter change, and everything that judged it.

    The result fields are separate rather than one "evidence" blob because they
    answer different questions and a reader has to be able to tell which one is
    missing:

    ```text
    training_dataset  what it was fitted on        (always present)
    holdout_result    did it beat the baseline     (session holdout)
    forward_result    did it beat the *newer* data (time-ordered holdout)
    ```

    A policy with a holdout result and no forward result is not "validated"; it
    is validated on half the evidence, and the console says so.
    """

    version: str
    params: Mapping[str, float]
    baseline: Mapping[str, float] = field(default_factory=lambda: dict(BASE_POLICY))
    source: str = SOURCE_MANUAL
    rationale: str = ""
    created_at: float = 0.0
    status: str = STATUS_PROPOSED
    evidence: Mapping[str, Any] = field(default_factory=dict)
    # What the fit saw. Stored so a later reader can tell whether a result was
    # ever comparable to the corpus it is now being read against.
    training_dataset: Mapping[str, Any] = field(default_factory=dict)
    holdout_result: Mapping[str, Any] = field(default_factory=dict)
    forward_result: Mapping[str, Any] = field(default_factory=dict)
    target_error: str = ""
    collateral_regressions: tuple[str, ...] = ()
    confidence: str = ""
    compatibility: Mapping[str, Any] = field(default_factory=dict)
    # What the policy is aimed at: the host version it was replayed against, the
    # digest of the configuration it assumed, and the versions it has actually
    # been validated on. Separate from `compatibility` (which describes the data
    # it was fitted on) because "trained on what" and "valid for what" are
    # different claims, and only the second one gates adoption.
    target: Mapping[str, Any] = field(default_factory=dict)
    # When the state last changed, and why — a state machine without a history
    # cannot answer "who promoted this, and on what evidence".
    status_changed_at: float = 0.0
    status_history: tuple[Mapping[str, Any], ...] = ()

    @property
    def label(self) -> str:
        return STATUS_LABEL.get(normalize_status(self.status), self.status)

    @property
    def published(self) -> bool:
        """Only a promoted policy is offered to ChatDynamics."""
        return normalize_status(self.status) == STATUS_PROMOTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "version": self.version,
            "params": {key: round(float(value), 4) for key, value in self.params.items()},
            "baseline": {key: round(float(value), 4) for key, value in self.baseline.items()},
            "deltas": policy_deltas(self.baseline, self.params),
            "source": self.source,
            "rationale": self.rationale,
            "created_at": self.created_at,
            "status": normalize_status(self.status),
            "status_label": self.label,
            "status_changed_at": self.status_changed_at,
            "status_history": [dict(row) for row in self.status_history],
            "published": self.published,
            "evidence": dict(self.evidence),
            "training_dataset": dict(self.training_dataset),
            "holdout_result": dict(self.holdout_result),
            "forward_result": dict(self.forward_result),
            "target_error": self.target_error,
            "collateral_regressions": list(self.collateral_regressions),
            "confidence": self.confidence,
            "compatibility": dict(self.compatibility),
            "target": dict(self.target),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "PolicyCandidate | None":
        if not isinstance(raw, Mapping):
            return None
        version = raw.get("version")
        if not isinstance(version, str) or not version:
            return None
        history = raw.get("status_history")
        return cls(
            version=version[:64],
            params=normalize_policy(raw.get("params")),
            baseline=normalize_policy(raw.get("baseline")),
            source=str(raw.get("source") or SOURCE_MANUAL)[:64],
            rationale=str(raw.get("rationale") or "")[:1000],
            created_at=_finite_or_zero(raw.get("created_at")),
            status=normalize_status(raw.get("status")),
            evidence=_mapping_or_empty(raw.get("evidence")),
            training_dataset=_mapping_or_empty(raw.get("training_dataset")),
            holdout_result=_mapping_or_empty(raw.get("holdout_result")),
            forward_result=_mapping_or_empty(raw.get("forward_result")),
            target_error=str(raw.get("target_error") or "")[:64],
            collateral_regressions=_text_tuple(raw.get("collateral_regressions")),
            confidence=str(raw.get("confidence") or "")[:32],
            compatibility=_mapping_or_empty(raw.get("compatibility")),
            target=_mapping_or_empty(raw.get("target")),
            status_changed_at=_finite_or_zero(raw.get("status_changed_at")),
            status_history=tuple(
                dict(row) for row in (history if isinstance(history, list) else [])[-32:]
                if isinstance(row, Mapping)),
        )

    def with_fields(self, **changes: Any) -> "PolicyCandidate":
        return replace(self, **changes)

    def with_status(self, status: str, *, now: float | None = None,
                    reason: str = "") -> "PolicyCandidate":
        """Move to a new state, recording when and why.

        The transition is **not** validated here: this is the record's own
        setter, and the store is what refuses an illegal arrow. Validating in
        both places would make the record's copy authoritative for a question
        only the store can answer (what the state was before).
        """
        target = normalize_status(status)
        stamp = now if now is not None else time.time()
        entry = {"from": normalize_status(self.status), "to": target, "at": stamp}
        if reason:
            entry["reason"] = reason[:200]
        return replace(self, status=target, status_changed_at=stamp,
                       status_history=(*self.status_history, entry)[-32:])


def next_version(existing: Iterable[str]) -> str:
    highest = 0
    for name in existing:
        if not isinstance(name, str) or not name.startswith("policy_v"):
            continue
        suffix = name[len("policy_v"):]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"policy_v{highest + 1}"


def candidate_from(
    changes: Mapping[str, float],
    *,
    baseline: Mapping[str, float] | None = None,
    source: str = SOURCE_MANUAL,
    rationale: str = "",
    evidence: Mapping[str, Any] | None = None,
    existing_versions: Iterable[str] = (),
    now: float | None = None,
) -> PolicyCandidate:
    base = normalize_policy(baseline)
    params = dict(base)
    for name, value in changes.items():
        if name in PARAM_SPECS:
            params[name] = clamp_param(name, value)
    params = normalize_policy(params)
    return PolicyCandidate(
        version=next_version(existing_versions), params=params, baseline=base,
        source=source, rationale=rationale[:1000],
        created_at=now if now is not None else time.time(),
        evidence=dict(evidence or {}),
    )


# ---- the publish protocol (Learning -> ChatDynamics) --------------------
#
# A different protocol from the trace schema, and a different number. The trace
# schema describes what the host writes (see `core/trace.py`); this one
# describes what this plugin offers. Neither can be read off the other: a reader
# upgrade must not move the trace schema, and a host upgrade must not move this.
POLICY_CONTRACT_VERSION = 1

# The version stamped into `source.learning_version`. Kept in step with
# `register(...)` in main.py and metadata.yaml.
LEARNING_VERSION = "1.1.0"


def baseline_config_hash(policy: Mapping[str, float]) -> str:
    """A canonical hash of a resolved policy, over the whitelisted keys.

    This is the cross-repo half of "same baseline": the host computes the same
    digest over *its own* effective values for these keys and compares. The
    canonical form is therefore part of the contract, not an implementation
    detail — every parameter name, sorted, each value rounded to four decimals,
    no whitespace:

    ```text
    sha256('{"parent_accept_threshold":"0.7200","safe_hover_threshold":"0.4000",...}')
    ```

    A different rounding or key order on the host side produces a different
    digest for the same configuration, which would read as "the config moved"
    when it had not. Truncated to 16 hex characters: this detects drift, it does
    not authenticate.
    """
    resolved = normalize_policy(policy)
    canonical = json.dumps({name: f"{resolved[name]:.4f}" for name in PARAM_NAMES},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def published_policy(candidate: PolicyCandidate, *,
                     issued_at: float | None = None) -> dict[str, Any]:
    """The read-only offer ChatDynamics may consume.

    Four blocks, and the split is what keeps three different versions from being
    confused with each other:

    ```text
    policy_contract_version  这个文件本身的协议版本（Learning -> ChatDynamics）
    source.trace_schema_version  策略是在哪种 trace 上训出来的（ChatDynamics -> Learning）
    source.learning_version      哪个 Learning 版本产出的
    target.chat_dynamics_version 它是对着哪个本体版本验证的
    ```

    Every parameter is published, not only the ones that moved. A consumer that
    received a delta would have to guess the rest, and the guess would silently
    become the value it applied; publishing the resolved set means what the host
    reads is exactly what was validated here.

    What this file cannot express is deliberately absent: no KV key, no write
    target, no instruction. Whether to adopt it is ChatDynamics' decision, and
    this plugin's only power over it is to stop claiming it.
    """
    resolved = normalize_policy(candidate.params)
    base = normalize_policy(candidate.baseline)
    compatibility = dict(candidate.compatibility)
    target = dict(candidate.target)
    validated = [str(item) for item in (target.get("validated_host_versions") or []) if item]
    return {
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "policy_id": candidate.version,
        "issued_at": issued_at if issued_at is not None else time.time(),
        "state": normalize_status(candidate.status),
        "eligible_modes": (["shadow", "active"] if candidate.published else ["shadow"]),
        "source": {
            # The trace schema this policy was fitted on. A mixed corpus is not
            # hidden: the distribution travels beside it, because a policy
            # trained mostly on schema 2 rows never saw an outcome.
            "trace_schema_version": compatibility.get("trace_schema_version"),
            "trace_schema_versions": dict(compatibility.get("trace_schema_versions") or {}),
            # Provenance, not a lock: it says which corpus produced this policy.
            # Whether to *pin* it is the consumer's decision — see the note on
            # `expected_dataset_fingerprint` in docs/contract.md.
            "dataset_fingerprint": dict(candidate.training_dataset).get("fingerprint", ""),
            "learning_version": LEARNING_VERSION,
        },
        "target": {
            "chat_dynamics_version": target.get("chat_dynamics_version"),
            "baseline_config_hash": target.get("baseline_config_hash")
            or baseline_config_hash(base),
            # Compatibility is *validated*, not inferred from SemVer. Today the
            # list holds the single version this policy was replayed against (or
            # nothing, when the host did not report one); widening it later is a
            # data change, not a protocol change.
            "validated_host_versions": validated,
        },
        "params": {name: round(resolved[name], 4) for name in PARAM_NAMES},
        "baseline": {name: round(base[name], 4) for name in PARAM_NAMES},
        "changed": [name for name in PARAM_NAMES if abs(resolved[name] - base[name]) > 1e-9],
        "target_error": candidate.target_error,
        "confidence": candidate.confidence,
        # Whether anything was ever observed *beside* live behaviour. False is
        # not a failure; it is the difference between "it won offline" and "it
        # was watched", and the host gets to see which one it is being handed.
        "shadow_observed": bool(candidate.forward_result),
        "evidence": {
            "holdout": dict(candidate.holdout_result),
            "forward": dict(candidate.forward_result),
            "collateral_regressions": list(candidate.collateral_regressions),
        },
    }


def published_payload(policies: Sequence[PolicyCandidate], *,
                      issued_at: float | None = None) -> dict[str, Any]:
    """Every policy that has reached `promoted`, newest first.

    Only promoted records are offered. Earlier validated policies are exposed
    separately by candidate_payload for shadow consumers.
    """
    promoted = sorted((row for row in policies if row.published),
                      key=lambda row: row.status_changed_at or row.created_at, reverse=True)
    return {
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "generated_at": issued_at if issued_at is not None else time.time(),
        "policies": [published_policy(row, issued_at=issued_at) for row in promoted],
        "note": "本插件只发布策略，不写入 ChatDynamics 任何配置；是否采用由本体决定。",
    }


def candidate_payload(policies: Sequence[PolicyCandidate], *,
                      issued_at: float | None = None) -> dict[str, Any]:
    """Validated offers for observation, without requiring early promotion."""
    eligible = sorted(
        (row for row in policies if normalize_status(row.status) in
         {STATUS_VALIDATED, STATUS_SHADOW, STATUS_PROMOTED}),
        key=lambda row: row.status_changed_at or row.created_at, reverse=True)
    return {
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "generated_at": issued_at if issued_at is not None else time.time(),
        "eligible_modes": ["shadow"],
        "policies": [published_policy(row, issued_at=issued_at) for row in eligible],
        "note": "Shadow observation only; promotion is required for active use.",
    }


@dataclass(frozen=True)
class ReplayDecision:
    """What the parameterised decision function produces for one trace."""

    targeted: bool
    level: str
    score: float
    topic_committed: bool

    @property
    def recipient_label(self) -> str:
        return "bot" if self.targeted else "other"

    @property
    def reply_label(self) -> str:
        return "reply" if self.level == "strong" else "silent"


def ambient_score(trace: DecisionTrace) -> float:
    """Reproduce the host's ambient additive score from the recorded ledger.

    `contribution_total` is the sum of the recorded evidence contributions, which
    is exactly the host's pre-clamp score. Re-clamping here reproduces
    `ParticipationPolicy.evaluate` without re-deriving any weight.
    """
    return max(0.0, min(1.0, float(trace.contribution_total)))


def decide(
    trace: DecisionTrace,
    policy: Mapping[str, float],
    *,
    model_score: float | None = None,
) -> ReplayDecision:
    """Replay one trace under a policy.

    `model_score`, when given, replaces the additive score for ambient turns
    with a fitted probability in [0, 1]. Structural (explicit) turns and the
    no-prior-bot early return are threshold-independent, exactly as in the host.
    """
    policy = normalize_policy(policy)
    strong = float(policy["strong_addressivity_threshold"])
    hover = float(policy["safe_hover_threshold"])
    commit = float(policy["topic_commit_threshold"])

    topic_committed = bool(
        trace.topic_id and not trace.topic_ambiguous and trace.topic_confidence >= commit
    )

    if trace.is_explicit:
        targeted = bool(trace.bot_targeted)
        return ReplayDecision(targeted, "strong" if targeted else "weak",
                              float(trace.participation_score or 0.0), topic_committed)

    if trace.prior_bot_proxy == 0:
        # Host early return: no prior bot message, level stays weak.
        score = min(1.0, ambient_score(trace))
        return ReplayDecision(False, "weak", score, topic_committed)

    score = ambient_score(trace) if model_score is None else max(0.0, min(1.0, float(model_score)))
    level = "strong" if score >= strong else "hover" if score >= hover else "weak"
    return ReplayDecision(level == "strong", level, score, topic_committed)


def clamp_cumulative(original: Mapping[str, float], proposed: Mapping[str, float],
                     max_ratio: float) -> tuple[dict[str, float], list[str]]:
    """Bound the total drift from the original baseline.

    One step is capped at ±5%, but three of them compound to -14.3%. The
    cumulative cap is what actually stops an iteration from walking a parameter
    somewhere nobody agreed to, so it is enforced separately from the per-step
    budget and every clamped parameter is named.
    """
    base = normalize_policy(original)
    result = normalize_policy(proposed)
    clamped: list[str] = []
    if max_ratio <= 0:
        return base, list(PARAM_NAMES)
    for name in PARAM_NAMES:
        budget = abs(base[name]) * max_ratio
        low, high = base[name] - budget, base[name] + budget
        if result[name] < low - 1e-9:
            result[name] = clamp_param(name, low)
            clamped.append(name)
        elif result[name] > high + 1e-9:
            result[name] = clamp_param(name, high)
            clamped.append(name)
    return result, clamped


def drift_from(original: Mapping[str, float],
               policy: Mapping[str, float]) -> list[dict[str, Any]]:
    """Per-parameter cumulative drift, for reporting and for the ±15% stop."""
    base, current = normalize_policy(original), normalize_policy(policy)
    rows: list[dict[str, Any]] = []
    for name in PARAM_NAMES:
        before, after = base[name], current[name]
        if abs(after - before) < 1e-9:
            continue
        rows.append({"param": name, "label": PARAM_SPECS[name]["label"],
                     "baseline": round(before, 4), "value": round(after, 4),
                     "delta": round(after - before, 4),
                     "delta_ratio": round((after - before) / before, 4) if before else None})
    return rows


def target_error_for(changes: Mapping[str, float],
                     baseline: Mapping[str, float]) -> str | None:
    """Name the error kind the dominant change in `changes` is aimed at."""
    base = normalize_policy(baseline)
    dominant: tuple[float, str, int] | None = None
    for name, value in changes.items():
        if name not in PARAM_SPECS:
            continue
        delta = clamp_param(name, value) - base[name]
        if abs(delta) < 1e-9:
            continue
        magnitude = abs(delta) / abs(base[name]) if base[name] else abs(delta)
        direction = 1 if delta > 0 else -1
        if dominant is None or magnitude > dominant[0]:
            dominant = (magnitude, name, direction)
    if dominant is None:
        return None
    return TARGET_ERROR_BY_DIRECTION.get((dominant[1], dominant[2]))


def policy_summary(policy: Mapping[str, float]) -> list[dict[str, Any]]:
    normalized = normalize_policy(policy)
    return [{"param": name, "label": PARAM_SPECS[name]["label"], "value": normalized[name],
             "default": BASE_POLICY[name], "min": PARAM_SPECS[name]["min"],
             "max": PARAM_SPECS[name]["max"]}
            for name in PARAM_NAMES]


def sweep_values(name: str, *, steps: int = 21) -> list[float]:
    spec = PARAM_SPECS[name]
    low, high = float(spec["min"]), float(spec["max"])
    if steps <= 1:
        return [low]
    return [round(low + (high - low) * index / (steps - 1), 4) for index in range(steps)]


def sequences_equal(left: Sequence[float], right: Sequence[float], tol: float = 1e-6) -> bool:
    return len(left) == len(right) and all(abs(a - b) <= tol for a, b in zip(left, right))


__all__ = [
    "ACTION_STATUS", "ACTIONS", "ALLOWED_TRANSITIONS", "BASE_POLICY", "ERROR_FALSE_BOT",
    "ERROR_FRAGMENTATION", "ERROR_MISSED_BOT", "ERROR_MISSED_REPLY", "ERROR_PREMATURE_REPLY",
    "ERROR_UNDELIVERED_REPLY", "ERROR_UNSOLICITED_REPLY", "ERROR_WRONG_MERGE", "PARAM_NAMES",
    "LEARNING_VERSION", "PARAM_SPECS", "POLICY_CONTRACT_VERSION", "POLICY_SCHEMA_VERSION",
    "PolicyCandidate",
    "ReplayDecision", "SOURCE_MANUAL", "SOURCE_RECIPIENT_SWEEP", "SOURCE_TOPIC_SWEEP",
    "STATUS_ALIASES", "STATUS_LABEL", "STATUS_PROPOSED", "STATUS_PROMOTED", "STATUS_REJECTED",
    "STATUS_ROLLED_BACK", "STATUS_SHADOW", "STATUS_SUPERSEDED", "STATUS_VALIDATED", "STATUSES",
    "TARGETED_EXPLICIT_CODES", "TARGET_ERROR_BY_DIRECTION", "ambient_score", "baseline_config_hash",
    "bounded_target",
    "can_transition", "candidate_from", "candidate_payload", "clamp_cumulative", "clamp_param", "decide", "drift_from",
    "next_version", "normalize_policy", "normalize_status", "policy_deltas", "policy_summary",
    "published_payload", "published_policy", "sequences_equal", "sweep_values",
    "target_error_for", "transition_error",
]
