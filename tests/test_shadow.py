"""Shadow A/B: the disagreement subset, and what it takes to go active.

The tests are built around the number that does *not* answer the question. An
overall accuracy delta is the policy's effect diluted by the decisions that did
not change, so a suite that only checked it would pass on a policy that helped
nowhere. Every gate case here is therefore written so that the overall metric
looks acceptable and the subset says otherwise — or the reverse.
"""
from __future__ import annotations

import pytest

from astrbot_plugin_dynamics_learning.core.config import LearningConfig
from astrbot_plugin_dynamics_learning.core.samples import build_dataset
from astrbot_plugin_dynamics_learning.core.shadow import (
    DEFAULT_MIN_ACTIVE_HOURS, DEFAULT_MIN_DISAGREEMENTS, DEFAULT_MIN_SHADOW_SAMPLES,
    ActiveRules, disagreement_table, evaluate_shadow, rules_from_config,
    shadow_rows,
)

from .factories import shadow_block, shadow_record

SESSION = "umo:group:1"


def rows_for(plan, *, session=SESSION, start=1_000_000.0, step=1.0):
    """`plan` is a list of `(expected_reply, baseline_reply, shadow_reply)`."""
    records = []
    for index, (expected, baseline, shadow) in enumerate(plan):
        records.append((session, shadow_record(
            f"m{index}",
            expected_reply=expected,
            shadow=shadow_block(baseline_reply=baseline, shadow_reply=shadow,
                                score=0.68 if not baseline else 0.72,
                                recorded_at=start + index * step),
            annotated_at=start + index * step + 3600,
        )))
    return build_dataset(records)


def loose(**overrides):
    """Rules that isolate one check at a time."""
    base = dict(min_shadow_samples=0, min_disagreements=0, min_sessions=1,
                min_active_hours=1, subgroup_min_support=10_000,
                max_overall_regression=1.0, target_relative_improvement=0.0,
                target_absolute_improvement=-1.0, ci_floor=-1.0,
                bootstrap_iterations=50)
    base.update(overrides)
    return ActiveRules(**base)


def gate_of(plan, **overrides):
    result = evaluate_shadow(rows_for(plan), rules=loose(**overrides))
    return {row["name"]: row for row in result["gate"]["checks"]}, result


# ---- reading the rows ---------------------------------------------------

def test_rows_are_grouped_by_message_not_by_sample():
    """One message yields up to four samples; one comparison."""
    rows = shadow_rows(rows_for([(True, False, True)]))

    assert len(rows) == 1
    assert rows[0].expected_reply is True
    assert rows[0].changed is True


def test_a_turn_without_a_reply_label_produces_no_sample_and_no_row():
    """The comparison survives on the raw plane, not in the corpus.

    A record with no label yields no sample at all — labels are what make a
    sample — so its shadow decision cannot reach the table. Counting it on the
    contract plane is what keeps "320 comparisons" from being read as "320 out
    of the decisions the host made".
    """
    from astrbot_plugin_dynamics_learning.core.ingest import parse_export
    from astrbot_plugin_dynamics_learning.core.quality import RawContractStats

    record = shadow_record("m1", expected_reply=None, shadow=shadow_block())
    packed = build_dataset([(SESSION, record)])
    assert shadow_rows(packed) == [], "没有标签就没有样本，也没有可评的行"

    stats = RawContractStats(source="export")
    stats.observe_record({"msg_id": "m1", "decision_trace": record["decision_trace"]})
    assert stats.shadow_present == 1
    assert parse_export({"sessions": [{"session_key": SESSION, "records": [record]}]}) \
        .diagnostics["records"] == 1


def test_the_active_period_comes_from_the_decision_time_not_the_label_time():
    """`annotated_at` is when a human reviewed; a week of reviews in one evening
    would look like a single active period."""
    rows = rows_for([(True, False, True)], start=1_000_000.0)
    # the fixture labels an hour after the decision, on purpose
    result = evaluate_shadow(rows, rules=loose())

    assert result["active_hours"] == 1
    assert shadow_rows(rows)[0].decision_at == 1_000_000.0


# ---- the paired table ---------------------------------------------------

def test_the_table_separates_the_policys_gains_from_its_costs():
    # (expected_reply, baseline_reply, shadow_reply) -- one row per cell, twice.
    plan = [
        (True, True, True),     # both_correct
        (False, False, False),  # both_correct
        (True, False, False),   # both_wrong   (both stayed silent, a reply was wanted)
        (False, True, True),    # both_wrong   (both spoke, silence was wanted)
        (True, False, True),    # shadow_only  (baseline silent and wrong)
        (False, True, False),   # shadow_only  (baseline spoke out of turn)
        (True, True, False),    # baseline_only
        (False, False, True),   # baseline_only
    ]
    table = disagreement_table(shadow_rows(rows_for(plan)))

    assert table["labelled"] == len(plan)
    assert table["changed"] == 4
    assert table["same"] == 4
    assert table["shadow_only"] == 2
    assert table["baseline_only"] == 2
    assert table["both_wrong"] == 2
    assert table["both_correct"] == 2
    assert table["net_gain"] == 0
    # The table's own accounting identity: the four cells are a partition, and
    # the off-diagonal ones are the disagreement subset.
    assert table["balanced"] is True


def test_the_overall_delta_is_diluted_while_the_subset_delta_is_not():
    """The point of the whole module, as an arithmetic fact."""
    plan = [(True, False, True)] * 8 + [(False, False, False)] * 92
    result = evaluate_shadow(rows_for(plan), rules=loose())
    table = result["table"]

    assert table["changed"] == 8
    assert table["subset_delta"] == pytest.approx(1.0)
    assert table["balanced"] is True
    assert table["overall_shadow_accuracy"] - table["overall_baseline_accuracy"] \
        == pytest.approx(0.08)


# ---- the gate -----------------------------------------------------------

def test_an_empty_corpus_says_the_host_has_not_run_a_shadow_policy():
    result = evaluate_shadow([])

    assert result["rows"] == 0
    assert result["gate"]["ok"] is False
    assert any("learning_policy_mode" in note for note in result["notes"])


def test_too_few_shadow_samples_blocks():
    checks, _ = gate_of([(True, False, True)] * 20, min_shadow_samples=500)

    assert checks["shadow_samples"]["status"] == "block"
    assert checks["shadow_samples"]["value"] == 20


def test_too_few_disagreements_blocks_even_with_many_samples():
    """A policy that changes nothing has not been tested by anything."""
    checks, _ = gate_of([(True, True, True)] * 600, min_disagreements=100)

    assert checks["shadow_samples"]["status"] == "ok"
    assert checks["disagreements"]["status"] == "block"
    assert checks["disagreements"]["value"] == 0


def test_a_small_net_loss_blocks_even_when_the_overall_metric_looks_fine():
    """40 gained and 60 lost is a lot of movement and no improvement.

    Diluted by the 9,900 turns both decisions got right, the overall metric moves
    0.2 percentage points and clears a 1% regression bound. Only the paired table
    says the policy lost.
    """
    plan = ([(True, False, True)] * 40 + [(True, True, False)] * 60
            + [(True, True, True)] * 9900)
    checks, result = gate_of(plan, min_disagreements=10, max_overall_regression=0.01)

    assert result["table"]["net_gain"] == -20
    assert checks["net_gain"]["status"] == "block"
    assert checks["overall_regression"]["status"] == "ok", "总体几乎没动"
    assert abs((result["table"]["overall_shadow_accuracy"] or 0)
               - (result["table"]["overall_baseline_accuracy"] or 0)) < 0.01


def test_an_overall_regression_blocks_on_its_own():
    plan = [(True, True, False)] * 60 + [(True, True, True)] * 40
    checks, _ = gate_of(plan, min_disagreements=10, max_overall_regression=0.01)

    assert checks["overall_regression"]["status"] == "block"


def test_a_gain_that_is_too_small_blocks_the_target_check():
    plan = [(True, False, True)] * 2 + [(True, True, True)] * 598
    checks, _ = gate_of(plan, min_disagreements=1, target_absolute_improvement=0.01,
                        target_relative_improvement=0.10)

    assert checks["target_improvement"]["status"] == "block"
    # The relative branch would have called this a 100% improvement: the baseline
    # error rate was 0.3%, and the corpus was already good. It is not divided by
    # a rate that small — 0.3% -> 0% is a big ratio and a tiny effect.
    assert "不适用" in checks["target_improvement"]["detail"]


def test_a_subset_with_no_interval_blocks_the_floor_check():
    """One session of a handful of rows cannot produce a usable interval."""
    checks, _ = gate_of([(True, False, True)] * 3, min_disagreements=1, ci_floor=-0.002)

    assert checks["interval_floor"]["status"] in {"block", "ok"}


def test_sessions_and_active_hours_are_their_own_checks():
    plan = [(True, False, True)] * 40
    checks, _ = gate_of(plan, min_disagreements=1, min_sessions=3, min_active_hours=4)

    assert checks["sessions"]["status"] == "block"
    assert checks["active_hours"]["status"] == "block"
    assert checks["sessions"]["value"] == 1


def test_a_session_that_falls_apart_blocks_the_whole_policy():
    records = []
    # one healthy session that improves, one busy session that collapses
    for index in range(40):
        records.append(("umo:good", shadow_record(
            f"g{index}", expected_reply=True,
            shadow=shadow_block(baseline_reply=True, shadow_reply=True),
            annotated_at=1000.0 + index)))
    for index in range(30):
        records.append(("umo:bad", shadow_record(
            f"b{index}", expected_reply=True,
            shadow=shadow_block(baseline_reply=True, shadow_reply=False, score=0.5),
            annotated_at=2000.0 + index)))
    samples = build_dataset(records)
    result = evaluate_shadow(samples, rules=loose(min_disagreements=0,
                                                  subgroup_min_support=20,
                                                  subgroup_max_regression=0.05))
    checks = {row["name"]: row for row in result["gate"]["checks"]}

    from astrbot_plugin_dynamics_learning.core.samples import session_hash

    assert result["subgroups"]["catastrophic"] == [session_hash("umo:bad")]
    assert checks["subgroups"]["status"] == "block"


def test_a_healthy_shadow_run_clears_every_check():
    plan = [(True, False, True)] * 40 + [(False, False, False)] * 60
    result = evaluate_shadow(rows_for(plan, step=3600.0),
                             rules=ActiveRules(min_shadow_samples=50,
                                               min_disagreements=20,
                                               min_sessions=1,
                                               min_active_hours=2,
                                               subgroup_min_support=20,
                                               bootstrap_iterations=100))
    checks = {row["name"]: row for row in result["gate"]["checks"]}

    assert result["gate"]["ok"] is True, result["gate"]["blocked_by"]
    assert checks["net_gain"]["value"] == 40
    assert all(row["detail"] for row in result["gate"]["checks"]), "每项检查都要自报测量值"


def test_the_rules_read_off_the_plugin_config():
    rules = rules_from_config(LearningConfig())
    assert rules.min_shadow_samples == DEFAULT_MIN_SHADOW_SAMPLES
    assert rules.min_disagreements == DEFAULT_MIN_DISAGREEMENTS
    assert rules.min_active_hours == DEFAULT_MIN_ACTIVE_HOURS

    tuned = rules_from_config(LearningConfig(shadow_min_samples=10, shadow_min_active_hours=1))
    assert (tuned.min_shadow_samples, tuned.min_active_hours) == (10, 1)
    # A partial object (a script, a test double) gets the documented defaults.
    assert rules_from_config(object()).min_shadow_samples == DEFAULT_MIN_SHADOW_SAMPLES
