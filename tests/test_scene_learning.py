import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_dynamics_learning.core.contract import candidate, encode, event_record, policy_allowed
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.scenes import episodes, scene_labels, select_patterns, sanitize
from astrbot_plugin_dynamics_learning.core.store import Store

SCOPE = 'qq:GroupMessage:scene-test'
SCENE = {'communication_intent': 'banter', 'expression_mode': 'joke', 'turn_update': 'continues_joke'}


def source(i, member='alice', reply='', stamp=None, content='又把大招交给空气了'):
    row = event_record({'schema_version': 1, 'kind': 'observed', 'scope': SCOPE, 'persona': 'persona',
                        'member': member, 'message_id': str(i), 'turn_id': '0:' + str(i), 'baseline': 'baseline',
                        'host_version': 'host', 'baseline_threshold': 50, 'mandatory': False, 'text': content,
                        'timestamp': stamp or time.time(), 'context': {'source_verified': True, 'reply_to': reply,
                        'reply_to_ids': [reply] if reply else []}})
    return row


def evidence():
    result = []
    for i in range(3):
        stamp = time.time() - (3 - i) * 86400
        result.extend([source(f'q{i}', stamp=stamp), source(f'a{i}', 'bob' if i % 2 else 'carol', f'q{i}', stamp + 1,
                       '@群友(123456789) 空气这把吃得真好')])
    return result


def raw_pattern(events=None):
    events = events or evidence()
    return {'category': 'scene_pattern', 'key': 'ignored', 'member': '', 'days': 30,
            'value': {'scene': dict(SCENE), 'method': '沿着具体失误轻轻接一句', 'conditions': '对方明确在自嘲',
                      'avoid': '真实求助或认真难过时不用', 'episode_ids': [e['id'] for e in episodes(events)]},
            'reason': '三段独立真人对话', 'evidence_ids': [e['id'] for e in events]}


def approved(store, events=None):
    events = events or evidence()
    p = store.propose(candidate(raw_pattern(events), events, None), 'llm', 'test')
    p = store.review(p['id'], {'verdict': 'approve', 'reason': 'independent review', 'risks': []}, 'llm:test', p['updated'])
    return store.apply(p['id'], 'llm:test', p['updated'])


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / 'test.sqlite3')
    yield s
    s.close()


def test_native_episodes_and_derived_examples_keep_original_sources():
    rows = evidence()
    data = candidate(raw_pattern(rows), rows, None)
    assert data['scene_policy'] == 1 and policy_allowed(data)
    assert data['value']['support'] == {'episodes': 3, 'authors': 2, 'days': 3}
    assert len(data['value']['examples']) == 2
    assert all(r['derived'] for r in data['value']['examples'])
    assert '123456789' not in encode(data['value'])
    assert '123456789' in encode(data['evidence'])


@pytest.mark.parametrize('mutation', ['one_author', 'one_day', 'two_episodes', 'unverified', 'missing_parent', 'cross_scope', 'member', 'invented_episode'])
def test_invalid_group_generalizations_are_rejected(mutation):
    rows = evidence()
    if mutation == 'one_author':
        for r in rows[1::2]:
            r['member'] = 'bob'
    elif mutation == 'one_day':
        for i, r in enumerate(rows):
            r['timestamp'] = 1700000000 + i
    elif mutation == 'two_episodes':
        rows = rows[:4]
    elif mutation == 'unverified':
        rows[1]['context']['source_verified'] = False
    elif mutation == 'missing_parent':
        rows[0]['context']['reply_to'] = 'missing'
    elif mutation == 'cross_scope':
        rows[0]['scope'] = 'qq:GroupMessage:elsewhere'
    raw = raw_pattern(rows)
    if mutation == 'member':
        raw['member'] = 'alice'
    elif mutation == 'invented_episode':
        raw['value']['episode_ids'][0] = 'invented'
    with pytest.raises(ValueError):
        candidate(raw, rows, None)


def test_overlapping_reply_tree_is_not_independent_evidence():
    rows = evidence()
    rows.append(source('branch', 'bob', 'a0', rows[1]['timestamp'] + 1, '再接一句'))
    raw = raw_pattern(rows)
    with pytest.raises(ValueError, match='overlap'):
        candidate(raw, rows, None)


def test_scene_matching_does_not_depend_on_identical_words(store):
    approved(store)
    entries = store.entries()
    assert select_patterns(entries, SCOPE, 'persona', SCENE, time.time())
    for key, value in [('expression_mode', 'sarcasm'), ('expression_mode', 'literal'), ('expression_mode', 'mixed'), ('communication_intent', 'task'), ('turn_update', 'corrects_previous'), ('turn_update', 'acknowledges')]:
        assert not select_patterns(entries, SCOPE, 'persona', {**SCENE, key: value}, time.time())
    assert not select_patterns(entries, 'qq:GroupMessage:another', 'persona', SCENE, time.time())
    assert not select_patterns(entries, SCOPE, 'other-persona', SCENE, time.time())


def test_scene_confidence_thresholds():
    assert scene_labels({'expression_mode': {'choice': 'sarcasm', 'confidence': .79}})['expression_mode'] == 'unclear'
    assert scene_labels({'communication_intent': {'choice': 'banter', 'confidence': True}})['communication_intent'] == 'unclear'
    assert sanitize('跟小甲说 @小乙(123456789) https://example.com', ['小甲']) == '跟[成员]说 [成员] [链接]'


def test_feature_off_v1_private_isolation_and_revocation(store):
    async def run():
        p = approved(store)
        engine = Engine(store, {'sessions': [SCOPE]}, None)
        engine.current_personas[SCOPE] = 'persona'
        await engine.refresh()
        args = dict(scope=SCOPE, persona='persona', member='alice', scene=SCENE)
        assert engine.scenes.context(**args)['items'] == []
        engine.config['scene_learning_enabled'] = True
        packet = engine.scenes.context(**args)
        assert packet['items'] and packet['items'][0]['member'] == ''
        assert not engine.context(scope=SCOPE, persona='persona', member='alice', baseline='baseline', host_version='host')['reply_projection']['items']
        assert not engine.scenes.context(**{**args, 'scope': 'qq:FriendMessage:alice'})['items']
        frozen = [{'source_id': p['id'], 'revision': 1}]
        assert engine.scenes.context(**args, frozen=frozen)['items'] == packet['items']
        await engine.action(p['id'], 'rollback', 'human:test', p['updated'])
        assert not engine.scenes.context(**args, frozen=frozen)['items']
    asyncio.run(run())


def receipt(store, p, turn='turn1', stage='provider'):
    store.scenes.receipt({'scope': SCOPE, 'persona': 'persona', 'member': 'alice', 'turn_id': turn,
                         'stage': stage, 'refs': [{'source_id': p['id'], 'revision': 1}]})


def delivery(store, turn='turn1', message='bot1'):
    row = source('human-' + turn)
    row.update(kind='outcome', turn_id=turn, reply={'delivered': True, 'message_id': message, 'text': 'some sent reply'})
    store.ingest(row)
    store.scenes.observe(row)


def feedback(store, content='不是这个意思', reply='bot1', identity='feedback1', member='alice'):
    row = source(identity, member, reply, content=content)
    store.ingest(row)
    store.scenes.observe(row)
    return row


def test_delivery_and_silence_never_imply_positive_feedback(store):
    p = approved(store)
    receipt(store, p)
    delivery(store)
    assert store.scenes.summary()['delivered'] == 1
    assert store.scenes.summary()['feedback'] == 0
    feedback(store, '哈哈')
    assert store.scenes.summary()['feedback'] == 0
    assert not store.scenes.adjustment_evidence(SCOPE, 'persona')


@pytest.mark.parametrize('content,member,reply', [('不是这个意思', 'bob', 'bot1'), ('不是这个意思', 'alice', ''), ('“不是这个意思”', 'alice', 'bot1')])
def test_unattributable_or_quoted_feedback_does_not_penalize(store, content, member, reply):
    p = approved(store)
    receipt(store, p)
    delivery(store)
    feedback(store, content, reply, member=member)
    assert not store.scenes.adjustment_evidence(SCOPE, 'persona')


def test_late_callbacks_provider_proof_and_reviewed_penalty(store):
    async def run():
        p = approved(store)
        receipt(store, p, stage='selected')
        feedback(store)  # Feedback arrives before successful delivery callback.
        delivery(store)
        assert not store.scenes.adjustment_evidence(SCOPE, 'persona')
        receipt(store, p)
        receipt(store, p, turn='turn2')
        delivery(store, 'turn2', 'bot2')
        feedback(store, reply='bot2', identity='feedback2')
        facts = store.scenes.adjustment_evidence(SCOPE, 'persona')
        assert len(facts) == 2
        engine = Engine(store, {'sessions': [SCOPE], 'scene_learning_enabled': True}, None)
        engine.current_personas[SCOPE] = 'persona'
        await engine.refresh()
        await engine.scenes.propose_adjustments(SCOPE, 'persona', engine.scope_token(SCOPE))
        proposal = next(p for p in store.proposals() if p['category'] == 'scene_adjustment')
        assert proposal['state'] == 'pending'
        assert not any(e['category'] == 'scene_adjustment' for e in store.entries())
        reviewed = store.review(proposal['id'], {'verdict': 'approve', 'reason': 'two independent corrections', 'risks': []}, 'llm:test', proposal['updated'])
        store.apply(reviewed['id'], 'llm:test', reviewed['updated'])
        assert any(e['category'] == 'scene_adjustment' for e in store.entries())
        store.purge(SCOPE)
        assert store.scenes.summary()['delivered'] == 0
        assert store.scenes.recent() == {}
    asyncio.run(run())


def test_background_proposal_review_and_no_repeat_model_call(store):
    async def run():
        rows = evidence()
        for e in rows:
            store.db.execute('INSERT INTO profile_messages VALUES(?,?,?,?,?,?)', (SCOPE, 'persona', e['message_id'], e['member'], e['timestamp'], encode(e)))
        store.db.commit()
        calls = []
        async def model(role, instruction, data):
            calls.append(role)
            return {'patterns': [raw_pattern(rows)['value']]} if role == 'analysis_provider' else {'verdict': 'approve', 'reason': 'supported', 'risks': []}
        engine = Engine(store, {'sessions': [SCOPE], 'scene_learning_enabled': True}, model)
        engine.current_personas[SCOPE] = 'persona'
        await engine.scenes.analyze(SCOPE, 'persona')
        assert engine.scenes.context(scope=SCOPE, persona='persona', member='alice', scene=SCENE)['items']
        await engine.scenes.analyze(SCOPE, 'persona')
        assert calls == ['analysis_provider', 'review_provider']
    asyncio.run(run())


def test_defaults_disabled():
    schema = json.loads((Path(__file__).parents[1] / '_conf_schema.json').read_text(encoding='utf-8'))
    assert schema['scene_learning_enabled']['default'] is False
    assert schema['scene_disambiguation_enabled']['default'] is False


def test_natural_correction_clause_and_laughter_are_not_confused(store):
    p = approved(store)
    receipt(store, p)
    delivery(store)
    feedback(store, '我是在开玩笑，不是真的要辞职')
    assert store.scenes.adjustment_evidence(SCOPE, 'persona')[0]['context']['scene_feedback']['kind'] == 'joke_correction'
    laugh = source('laugh', reply='bot1', content='哈哈哈哈哈哈哈！')
    laugh['context']['interaction'] = SCENE
    store.ingest(laugh)
    store.scenes.observe(laugh)
    assert store.scenes.summary()['feedback'] == 1


def test_scene_review_cannot_apply_after_switch_off(store):
    async def run():
        events = evidence()
        p = store.propose(candidate(raw_pattern(events), events, None), 'llm', 'test')
        async def review(role, instruction, data):
            engine.config['scene_learning_enabled'] = False
            return {'verdict': 'approve', 'reason': 'passed before switch changed', 'risks': []}
        engine = Engine(store, {'sessions': [SCOPE], 'scene_learning_enabled': True}, review)
        engine.current_personas[SCOPE] = 'persona'
        with pytest.raises(ValueError, match='disabled'):
            await engine.review(p['id'])
        assert not store.entries()
        assert store.get(p['id'])['state'] == 'reviewed'
    asyncio.run(run())
