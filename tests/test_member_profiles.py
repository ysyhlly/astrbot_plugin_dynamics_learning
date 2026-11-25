import asyncio
import json
import time

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, encode, PERSONALITY_POLICY
from astrbot_plugin_dynamics_learning.core.engine import Engine, ANALYZE, REVIEW
from astrbot_plugin_dynamics_learning.core.store import Store
from test_learning import event, raw


@pytest.mark.parametrize('category', ['style', 'typing_habit', 'preference'])
def test_profile_member_evidence_and_scoped_context(tmp_path, category):
    async def run():
        store = Store(tmp_path / 'learning.sqlite3')
        e = event()
        evidence_list = [e]
        r = {**raw(), 'category': category, 'value': '该成员明确说不喜欢剧透。'}
        if category == 'preference':
            e.update(text='我不喜欢剧透', context={'source_verified':True})
            r['value'] = dict(topic='剧透', stance='dislike', condition='', quote=e['text'], source_id=e['id'])
        elif category == 'typing_habit':
            from astrbot_plugin_dynamics_learning.core.contract import event_record
            evidence_list = []
            for idx in range(10):
                base = event()
                base.pop('id', None)
                base.pop('received_at', None)
                base.update(
                    turn_id=f'0:th{idx}',
                    message_id=f'th{idx}',
                    text=f'习惯消息{idx}',
                    timestamp=1000.0 + idx,
                    context={'source_verified': True},
                )
                evidence_list.append(event_record(base))
            e = evidence_list[0]
            r['evidence_ids'] = [ev['id'] for ev in evidence_list]
        for member in ['', 'bob']:
            with pytest.raises(ValueError):
                candidate({**r, 'member': member}, evidence_list, None)
        p = store.propose(candidate(r, evidence_list, None), 'llm_then_human', 'test')
        with pytest.raises(ValueError, match='LLM approval'):
            store.apply(p['id'], 'human:admin', p['updated'])
        p = store.review(p['id'], dict(verdict='approve', reason='明确自述', risks=[]), 'llm:test', p['updated'])
        with pytest.raises(ValueError, match='human approval'):
            store.apply(p['id'], 'llm:test', p['updated'])
        store.apply(p['id'], 'human:admin', p['updated'])
        engine = Engine(store, {'sessions': [e['scope'], 'other:GroupMessage:2']}, None)
        await engine.refresh()
        engine.current_personas[e['scope']] = e['persona']
        args = {k: e[k] for k in ['scope', 'persona', 'member', 'baseline', 'host_version']}
        result = engine.context(**args)
        assert PERSONALITY_POLICY in result['context']
        assert ('我不喜欢剧透' if category == 'preference' else '该成员明确说不喜欢剧透') in result['context']
        assert '"member":"alice"' in result['context'] and '群友' in result['context']
        assert p['id'] in result['versions']
        assert engine.member_profile(e['scope'], 'alice')['observations'][0]['category'] == category
        assert not engine.member_profile(e['scope'], 'bob')['available']
        assert not engine.member_profile('other:GroupMessage:2', 'alice')['available']
        for key, value in [('member', 'bob'), ('scope', 'other:GroupMessage:2'), ('persona', 'other')]:
            assert not engine.context(**{**args, key: value})['context']
        store.rollback(p['id'], 'human:admin')
        await engine.refresh()
        assert not engine.context(**args)['context']
        assert not engine.member_profile(e['scope'], 'alice')['available']
        store.close()
    asyncio.run(run())


def test_old_persona_and_style_are_revoked_and_cannot_be_applied(tmp_path):
    store = Store(tmp_path / 'old.sqlite3')
    e = event()
    with pytest.raises(ValueError, match='category'):
        candidate({**raw(), 'category': 'persona'}, [e], None)
    for category in ['persona', 'style']:
        data = candidate(raw(), [e], None)
        data.update(category=category)
        data.pop('profile_policy')
        with pytest.raises(ValueError, match='fixed personality'):
            store.propose(data, 'human', 'test')
        # Seed an actual pre-upgrade proposal/entry without the new contract.
        p = dict(data, id=category, entry_id=category, state='pending', updated=1, created=1, reviews=[],
                 mode='human', base_revision=0, before=None)
        with store.db:
            store.db.execute('INSERT INTO proposals VALUES(?,?,?,?,?,?,?)',
                (p['id'], p['scope'], p['persona'], 'pending', 1, 1, encode(p)))
            entry = dict(data, id=category, revision=1, active=True, expires=time.time()+100,
                         proposal_id=category)
            store.db.execute('INSERT INTO entries VALUES(?,?)', (category, encode(entry)))
        with pytest.raises(ValueError, match='fixed personality'):
            store.apply(category, 'human:admin', 1)
    engine = Engine(store, {'sessions': [e['scope']]}, None)
    engine.cache = tuple(store.entries())
    args = {k: e[k] for k in ['scope', 'persona', 'member', 'baseline', 'host_version']}
    assert not engine.context(**args)['context']
    assert store.enforce_personality_policy() == {'entries': 2, 'proposals': 2}
    assert store.enforce_personality_policy() == {'entries': 0, 'proposals': 0}
    assert all(not row['active'] for row in store.entries())
    assert all(row['state'] == 'rejected' for row in store.proposals())
    assert store.db.execute("SELECT count(*) FROM audit WHERE action='policy_disabled'").fetchone()[0] == 4
    store.close()


def test_rollback_cannot_restore_old_personality_material(tmp_path):
    store = Store(tmp_path / 'rollback.sqlite3')
    p = store.propose(candidate(raw(), [event()], None), 'human', 'test')
    # A prior version came from the old untyped style channel.
    p['before'] = dict(p, id=p['entry_id'], active=True, revision=0, proposal_id='legacy:old')
    p['before'].pop('profile_policy')
    with store.db:
        store._save(p)
    p = store.apply(p['id'], 'human:admin', p['updated'])
    store.rollback(p['id'], 'human:admin')
    assert not store.entries()[0]['active']
    store.close()


def test_profile_policy_is_shared_by_analysis_review_and_ui():
    from pathlib import Path
    assert PERSONALITY_POLICY in ANALYZE and PERSONALITY_POLICY in REVIEW
    assert '多个独立消息' in ANALYZE and '多个独立消息' in REVIEW
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / '_conf_schema.json').read_text(encoding='utf-8'))
    assert 'persona_review' not in schema
    assert all(schema[c+'_review']['default'] == 'llm' for c in ['style', 'typing_habit', 'preference','expression_example'])
