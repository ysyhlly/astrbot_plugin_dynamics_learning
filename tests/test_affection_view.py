import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning.core.contract import encode, entry_key
from astrbot_plugin_dynamics_learning.core.social import character_key
from astrbot_plugin_dynamics_learning.core.store import Store

SCOPE = 'qq:GroupMessage:group'


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / 'affection.sqlite3')
    yield result
    result.close()


def identity(scope=SCOPE, role='bot-a'):
    return {'platform': scope.split(':')[0], 'umo': scope, 'persona_id': role, 'instance_id': role + '-instance'}


def put(store, persona, member='member', score=118.33643134839393, *, scope=SCOPE, role='bot-a', active=True, stamp=0):
    character = identity(scope, role)
    key = character_key(character)
    row = {'id': entry_key(scope, persona, member, 'affection', 'bot_affection'), 'scope': scope, 'persona': persona,
           'member': member, 'category': 'affection', 'key': 'bot_affection', 'value': {'character_key': key, 'affection_level': score},
           'active': active, 'expires': 1, 'revision': 1, 'proposal_id': 'legacy:' + persona + member,
           'imported': True, 'state_updated_at': stamp}
    with store.transaction():
        store.db.execute('INSERT OR REPLACE INTO entries VALUES(?,?)', (row['id'], encode(row)))
        store.db.execute('INSERT OR REPLACE INTO social_bindings VALUES(?,?,?)', (scope, persona, key))
        store.db.execute('INSERT OR REPLACE INTO native_scope_registry VALUES(?,?,?)', (scope, persona, encode(character)))
    return row


def read(store, **kwargs):
    return store.console.entries(SCOPE, 'affection', current_personas={SCOPE: 'current'}, **kwargs)


def test_four_text_fingerprints_display_one_exact_current_value_without_writes(store):
    for version in ('old1', 'old2', 'old3', 'current'):
        put(store, version)
    before = store.db.total_changes
    bodies = store.db.execute('SELECT id,body FROM entries ORDER BY id').fetchall()
    current = read(store)
    assert current['total'] == 1
    assert current['entries'][0]['persona'] == 'current'
    assert current['entries'][0]['value']['affection_level'] == 118.33643134839393
    assert current['entries'][0]['affection_versions'] == 4
    history = read(store, state='all', affection_view='history')
    assert history['total'] == 3 and all(not r['affection_current'] for r in history['entries'])
    assert read(store, state='all', affection_view='all')['total'] == 4
    assert store.db.total_changes == before
    assert store.db.execute('SELECT id,body FROM entries ORDER BY id').fetchall() == bodies


def test_current_lower_score_wins_even_when_old_version_is_later_updated(store):
    put(store, 'old', score=999.1234, stamp=time.time() + 10000)
    put(store, 'current', score=7.25, stamp=1)
    assert read(store)['entries'][0]['value']['affection_level'] == 7.25
    assert read(store, query='999.1234')['total'] == 0
    assert read(store, query='999.1234', state='all', affection_view='history')['total'] == 1


def test_dedup_before_pagination_and_generic_active_view(store):
    for i in range(31):
        for version in ('old1', 'old2', 'old3', 'current'):
            put(store, version, member=str(i), score=float(i))
    first = read(store)
    second = read(store, page=2)
    assert first['total'] == second['total'] == 31
    assert len(first['entries']) == 25 and len(second['entries']) == 6
    assert len({r['member'] for r in first['entries'] + second['entries']}) == 31
    assert store.console.entries(SCOPE, current_personas={SCOPE: 'current'})['total'] == 31


def test_distinct_stable_roles_groups_and_members_are_not_combined(store):
    put(store, 'old')
    put(store, 'current')
    put(store, 'role-b-old', role='bot-b', score=20)
    put(store, 'role-b-new', role='bot-b', score=30)
    put(store, 'current', member='another', score=40)
    put(store, 'other', scope='qq:GroupMessage:other', score=50)
    rows = read(store)['entries']
    assert len(rows) == 3
    assert sorted(r['value']['affection_level'] for r in rows) == [30, 40, 118.33643134839393]
    all_groups = store.console.entries(category='affection', current_personas={SCOPE: 'current', 'qq:GroupMessage:other': 'other'})
    assert all_groups['total'] == 4


@pytest.mark.parametrize('missing', [False, True])
def test_inactive_or_missing_current_record_does_not_revive_old_active_copy(store, missing):
    put(store, 'old')
    row = put(store, 'current', active=False)
    if missing:
        with store.transaction():
            store.db.execute('DELETE FROM entries WHERE id=?', (row['id'],))
    assert read(store)['total'] == 0
    assert read(store, state='all', affection_view='history')['total'] == (1 if missing else 2)


def test_unknown_scope_uses_observed_version_not_largest_score(store):
    put(store, 'first', score=300)
    put(store, 'second', score=20)
    with store.transaction():
        for i, persona in enumerate(('first', 'second')):
            body = {'kind': 'observed', 'scope': SCOPE, 'persona': persona}
            store.db.execute('INSERT INTO events VALUES(?,?,?,?,?)', (persona, SCOPE, persona, 100 + i, encode(body)))
    result = store.console.entries(SCOPE, 'affection')
    assert result['entries'][0]['persona'] == 'second'
    assert result['entries'][0]['value']['affection_level'] == 20


def test_identity_conflict_is_visible_in_history_not_current(store):
    row = put(store, 'current')
    row['value']['character_key'] = character_key(identity(role='wrong-role'))
    with store.transaction():
        store.db.execute('UPDATE entries SET body=? WHERE id=?', (encode(row), row['id']))
    assert read(store)['total'] == 0
    hist = read(store, state='all', affection_view='history')
    assert hist['entries'][0]['affection_selection'] == 'identity_conflict'


def test_non_affection_listing_is_unchanged(store):
    row = put(store, 'current')
    row.update(id='a-style', category='style', value='该成员喜欢短句', expires=time.time() + 100)
    with store.transaction():
        store.db.execute('INSERT INTO entries VALUES(?,?)', (row['id'], encode(row)))
    result = store.console.entries(SCOPE, 'style')
    assert result['total'] == 1 and 'affection_current' not in result['entries'][0]
    with pytest.raises(ValueError):
        store.console.entries(SCOPE, 'style', affection_view='history')


def test_entries_route_passes_view_and_current_persona(store, monkeypatch):
    pytest.importorskip('astrbot.api.star')
    from astrbot_plugin_dynamics_learning import main

    put(store, 'older', score=200, stamp=2)
    put(store, 'current', score=98, stamp=1)
    plugin = main.DynamicsLearning(SimpleNamespace(), {'sessions': [SCOPE]})
    plugin.engine = SimpleNamespace(store=store, current_personas={SCOPE: 'current'})
    body = {'scope': SCOPE, 'category': 'affection', 'state': 'active', 'affection_view': 'current'}

    async def authorize():
        return 'human:audit'

    async def request_body():
        return json.dumps(body).encode()

    monkeypatch.setattr(plugin, '_authorized', authorize)
    monkeypatch.setattr(main, 'request', SimpleNamespace(_request=SimpleNamespace(body=request_body)))
    response = asyncio.run(plugin.entries())
    assert response.status_code == 200
    data = json.loads(response.body)['data']
    assert data['affection_view'] == 'current'
    assert [e['value']['affection_level'] for e in data['entries']] == [98]

    body.update(state='all', affection_view='history')
    response = asyncio.run(plugin.entries())
    assert response.status_code == 200
    data = json.loads(response.body)['data']
    assert data['affection_view'] == 'history'
    assert [e['value']['affection_level'] for e in data['entries']] == [200]

    body.update(scope='qq:GroupMessage:disabled')
    assert asyncio.run(plugin.entries()).status_code == 400
