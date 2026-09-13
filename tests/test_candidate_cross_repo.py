"""Real Learning candidate offers cross the host shadow/active boundary."""
import pytest

from astrbot_plugin_dynamics_learning.core.policy import (
    BASE_POLICY, candidate_from, candidate_payload,
)

consumer = pytest.importorskip("astrbot_plugin_chat_dynamics.core.learning_policy")


@pytest.mark.parametrize("state", ["validated", "shadow"])
def test_unpromoted_candidate_can_be_observed_but_never_applied(state):
    version = "v1.8.0"
    candidate = candidate_from({"strong_addressivity_threshold": 0.67}).with_fields(
        training_dataset={"fingerprint": "candidate-corpus"},
        compatibility={"trace_schema_version": 3},
        target={"chat_dynamics_version": version,
                "baseline_config_hash": consumer.baseline_config_hash(BASE_POLICY),
                "validated_host_versions": [version]},
    ).with_status(state)
    payload = candidate_payload([candidate])
    shadow = consumer.resolve(payload, mode="shadow", host_version=version,
                              effective_config=BASE_POLICY)
    assert shadow.policy_id == candidate.version
    assert shadow.overrides["strong_addressivity_threshold"] == pytest.approx(0.67)
    assert shadow.applied is False
    active = consumer.resolve(payload, mode="active", host_version=version,
                              effective_config=BASE_POLICY)
    assert active.applied is False
    assert active.status == "incompatible"
    assert active.overrides == {}
