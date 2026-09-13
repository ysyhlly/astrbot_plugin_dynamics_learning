"""Operational telemetry retains its own denominator and no human labels."""

from astrbot_plugin_dynamics_learning.core.shadow_coverage import evaluate_shadow_coverage


def observation(identity="a", **overrides):
    return (
        dict(
            observation_id=identity,
            recorded_at=1000,
            policy_id="p",
            host_version="1.8",
            session_hash="opaque",
            reason="ambient",
            baseline_reply=False,
            shadow_reply=True,
            changed=True,
            score=0.6,
            baseline_threshold=0.7,
            shadow_threshold=0.5,
        )
        | overrides
    )


def snapshot(rows, **overrides):
    return dict(schema_version=1, retention_seconds=100, max_records=20, updated_at=1000, observations=rows) | overrides


def test_coverage_deduplicates_and_preserves_policy_version_buckets():
    a = observation()
    report = evaluate_shadow_coverage(
        snapshot([a, a, observation("b", host_version="1.9", shadow_reply=False, changed=False)]), now=1000
    )
    assert report["comparisons"] == 2
    assert report["disagreements"] == 1
    assert report["disagreement_rate"] == 0.5
    assert report["duplicates"] == 1
    assert len(report["buckets"]) == 2
    assert sum(row["comparisons"] for row in report["by_hour"]) == 2
    assert "accuracy" not in report


def test_invalid_conflicting_and_expired_rows_do_not_inflate_denominator():
    rows = [
        observation(),
        observation(score=0.3),
        observation("old", recorded_at=800),
        observation("future", recorded_at=1001),
        observation("nan", score=float("nan")),
        observation("bad", changed=False),
        observation("truthy", baseline_reply=0),
        observation("valid"),
    ]
    report = evaluate_shadow_coverage(snapshot(rows), now=1000)
    assert report["comparisons"] == 1
    assert report["invalid"] == 5
    assert report["expired"] == 1


def test_retention_count_limit_and_empty_payload():
    report = evaluate_shadow_coverage(
        snapshot([observation("a", recorded_at=999), observation("b")], max_records=1), now=1000
    )
    assert report["comparisons"] == 1
    assert report["window"]["from"] == 1000
    assert report["window"]["denominator"] == "retained_unique_valid_observations"
    assert evaluate_shadow_coverage(None)["available"] is False
    assert evaluate_shadow_coverage(snapshot([], schema_version=True))["available"] is False
    empty = evaluate_shadow_coverage(snapshot([]), now=1000)
    assert empty["available"] is True
    assert empty["disagreement_rate"] is None
