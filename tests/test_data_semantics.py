from dataclasses import replace

from astrbot_plugin_dynamics_learning.core.samples import LearningSample, samples_from_annotation
from astrbot_plugin_dynamics_learning.core.quality import dataset_gate, label_freshness
from astrbot_plugin_dynamics_learning.core.config import LearningConfig


def sample(**changes):
    row = LearningSample('id', 'session', 'hash', 'msg', 0, 'recipient', 'bot', 'bot', 1, 'manual_replay')
    return replace(row, **changes)


def test_import_does_not_invent_label_time_or_origin():
    row = samples_from_annotation({'msg_id': 'm', 'bot_targeted': True}, 'session', now=900)[0]
    assert row.annotated_at == row.timestamp == row.event_at == 0
    assert row.ingested_at == 900
    assert row.label_source == 'unknown'


def test_independent_clocks_and_provenance_round_trip():
    row = samples_from_annotation({'msg_id': 'm', 'bot_targeted': True, 'annotated_at': 400,
        'event_at': 100, 'accepted_from': 'ai', 'label_revision': 'rev-2'}, 'session', now=900)[0]
    restored = LearningSample.from_dict(row.as_dict(include_trace=True))
    assert restored == row
    assert (restored.event_at, restored.annotated_at, restored.ingested_at) == (100, 400, 900)
    assert restored.label_source == 'ai'
    assert restored.label_revision == 'rev-2'


def test_legacy_rows_keep_unknown_event_and_label_origin():
    payload = sample(timestamp=123).as_dict()
    for name in ('event_at', 'ingested_at', 'label_source', 'label_revision'):
        payload.pop(name)
    row = LearningSample.from_dict(payload)
    assert row.event_at == row.ingested_at == 0
    assert row.label_source == 'unknown'
    assert label_freshness([row], now=130, max_age_days=1)['recent_count'] == 1


def test_fresh_import_and_one_fresh_label_do_not_refresh_old_corpus():
    rows = [sample(timestamp=1, ingested_at=1000000) for _ in range(40)]
    rows += [sample(annotated_at=1000000)]
    config = LearningConfig(gate_min_samples=20, gate_max_label_age_days=1)
    report = dataset_gate(rows, config=config, now=1000000)
    assert 'label_age' in report['blocked_by']
    freshness = next(row for row in report['checks'] if row['name'] == 'label_age')['freshness']
    assert freshness['recent_count'] == 1
    assert freshness['recent_ratio'] == 1 / 41


def test_invalid_and_unknown_times_do_not_pass_freshness():
    rows = [sample(timestamp=value) for value in (0, -1, float('nan'), float('inf'), 110)]
    report = label_freshness(rows, now=100, max_age_days=1)
    assert report['recent_count'] == 0
    assert report['missing_ratio'] == 1
