"""Metric agreement and observational reporting regressions."""
import pytest

from astrbot_plugin_dynamics_learning.core.bootstrap import Unit, bootstrap_delta
from astrbot_plugin_dynamics_learning.core.metrics import binary_report, topic_pair_metrics
from astrbot_plugin_dynamics_learning.core.samples import LearningSample, TASK_TOPIC
from astrbot_plugin_dynamics_learning.core.topic_learner import learn


def test_perfect_partition_beats_all_merged_and_is_label_invariant():
    perfect = topic_pair_metrics([[('X', 'A'), ('X', 'A'), ('Y', 'B'), ('Y', 'B')]])
    merged = topic_pair_metrics([[('X', 'A'), ('X', 'A'), ('X', 'B'), ('X', 'B')]])
    assert perfect['pair_accuracy'] == 1.0
    assert merged['pair_accuracy'] == pytest.approx(1 / 3, abs=0.0001)
    assert topic_pair_metrics([[('Z', 'A'), ('Z', 'A'), ('W', 'B'), ('W', 'B')]]) == perfect
    keys = ('pairs', 'true_positive', 'wrong_merge', 'fragmentation')
    interval = bootstrap_delta([Unit('s', {k: merged[k] for k in keys},
                                    {k: perfect[k] for k in keys})],
                               metric='pair_accuracy', iterations=10)
    assert interval['delta'] == pytest.approx(2 / 3, abs=0.000001)
    assert interval['lower'] == interval['upper'] == interval['delta']


def test_zero_f1_remains_defined_in_report_and_bootstrap():
    before = {'tp': 2, 'fp': 0, 'tn': 0, 'fn': 0}
    after = {'tp': 0, 'fp': 3, 'tn': 0, 'fn': 2}
    assert binary_report(after)['f1'] == 0.0
    assert binary_report({'tn': 3})['f1'] is None
    assert binary_report({'fn': 2})['f1'] == 0.0
    interval = bootstrap_delta([Unit('s', before, after)], metric='f1', iterations=10)
    assert interval['delta'] == -1.0
    assert interval['dropped'] == 0
    assert topic_pair_metrics([[('X', 'A'), ('Y', 'A')]])['f1'] == 0.0


def test_topic_observations_do_not_relax_unassigned_messages():
    samples = [LearningSample(str(i), 'session', 'hash', str(i), float(i), TASK_TOPIC,
                              predicted, 'A', 0.9, 'human',
                              trace={'topic_candidates': [{'topic_id': 'X', 'final_score': 0.9}]})
               for i, predicted in enumerate(('X', ''))]
    report = learn(samples)
    assert report.pair_metrics['fragmentation'] == 1
    assert report.pair_metrics['pair_accuracy'] == 0.0
    assert report.threshold_sweep['baseline']['source'] == 'recorded'
    assert report.threshold_sweep['baseline']['fragmentation'] == 1
