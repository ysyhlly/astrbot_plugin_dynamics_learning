import asyncio
import time

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, encode, policy_allowed
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.expressions import balanced_sample, group_statistics, expression_feedback
from test_scene_learning import SCOPE, source, store as store


def samples():
    rows = []
    for member in ('alice', 'bob', 'carol'):
        for i in range(10):
            content = ['笑死 这也太离谱了吧', '嗯', '好了 收工', '哈哈哈哈', '就这啊'][i % 5]
            rows.append(source(f'{member}-{i}', member, stamp=time.time() - (i % 3) * 86400 - 100, content=content))
    return rows


def raw_style(rows):
    refs = [e['id'] for e in rows if '吧' in e['text']]
    return {'category': 'expression_style', 'key': 'ignored', 'member': '', 'days': 30, 'reason': '均衡样本',
            'evidence_ids': [e['id'] for e in rows], 'value': {'sample_ids': [e['id'] for e in rows],
            'traits': [{'kind': 'length', 'observation': '经常用短句回应', 'guidance': '简单闲聊可短句收住，不补无必要总结', 'source_ids': [e['id'] for e in rows[:3] + rows[10:13]], 'marker': ''},
                       {'kind': 'particle', 'observation': '多位成员会用吧收尾', 'guidance': '合适时可用吧软化语气，不要每句都用', 'source_ids': refs, 'marker': '吧'}],
            'example_ids': [rows[0]['id'], rows[2]['id'], rows[1]['id']]}}


def apply_style(store):
    rows = samples()
    p = store.propose(candidate(raw_style(rows), rows, None), 'llm', 'test')
    p = store.review(p['id'], {'verdict': 'approve', 'reason': 'verified', 'risks': []}, 'llm:test', p['updated'])
    return store.apply(p['id'], 'llm:test', p['updated'])


def test_group_balance_and_short_reactions_are_not_discarded():
    rows = samples()
    rows.extend(source('flood-' + str(i), 'alice', content='很长的正式说明' * 20) for i in range(100))
    selected = balanced_sample(rows)
    stats = group_statistics(selected)
    assert stats['per_member'] == {'alice': 10, 'bob': 10, 'carol': 10}
    assert stats['balanced']['short_rate'] > .5
    assert any(e['text'] == '嗯' for e in selected)


@pytest.mark.parametrize('change', ['few_messages', 'few_members', 'one_day', 'one_off_marker', 'unsupported_humor', 'foreign_group'])
def test_group_style_requires_real_repeated_evidence(change):
    rows = samples()
    raw = raw_style(rows)
    if change == 'few_messages':
        raw['value']['sample_ids'] = raw['value']['sample_ids'][:29]
    elif change == 'few_members':
        for e in rows:
            if e['member'] == 'carol':
                e['member'] = 'alice'
    elif change == 'one_day':
        for e in rows:
            e['timestamp'] = time.time()
    elif change == 'one_off_marker':
        raw['value']['traits'][1]['marker'] = '离谱捏'
    elif change == 'unsupported_humor':
        raw['value']['traits'][0]['kind'] = 'humor'
    else:
        rows[0]['scope'] = 'qq:GroupMessage:other'
    with pytest.raises(ValueError):
        candidate(raw, rows, None)


def test_style_observations_are_separate_from_bot_guidance(store):
    p = apply_style(store)
    assert policy_allowed(p) and p['member'] == '' and p['expression_policy'] == 1
    assert p['value']['traits'][0]['observation'] != p['value']['traits'][0]['guidance']
    assert all(e['derived'] for e in p['value']['examples'])
    assert not any(e['category'] == 'style' for e in store.entries())


def test_daily_schedule_requires_twenty_new_messages_and_survives_reopen(store):
    for e in samples():
        store.db.execute('INSERT INTO profile_messages VALUES(?,?,?,?,?,?)', (SCOPE, 'persona', e['message_id'], e['member'], e['timestamp'], encode(e)))
    store.db.commit()
    _, _, frontier = store.expressions.sample(SCOPE, 'persona')
    assert store.expressions.reserve(SCOPE, 'persona', frontier, 100000)
    assert not store.expressions.reserve(SCOPE, 'persona', frontier, 200000)
    for i in range(20):
        e = source('new' + str(i))
        store.db.execute('INSERT INTO profile_messages VALUES(?,?,?,?,?,?)', (SCOPE, 'persona', e['message_id'], e['member'], e['timestamp'], encode(e)))
    store.db.commit()
    _, _, newest = store.expressions.sample(SCOPE, 'persona')
    assert not store.expressions.reserve(SCOPE, 'persona', newest, 100100)
    assert store.expressions.reserve(SCOPE, 'persona', newest, 200000)


def test_reply_modes_isolation_repetition_freeze_and_revoke(store):
    async def run():
        p = apply_style(store)
        engine = Engine(store, {'sessions': [SCOPE]}, None)
        engine.current_personas[SCOPE] = 'persona'
        await engine.refresh()
        args = {'scope': SCOPE, 'persona': 'persona', 'member': 'alice', 'scene': {'communication_intent': 'sharing', 'expression_mode': 'literal', 'turn_update': 'adds_detail'}, 'action': 'reply'}
        assert not engine.expressions.context(**args)['items']
        engine.config['expression_style_enabled'] = True
        packet = engine.expressions.context(**args)
        row = packet['items'][0]
        assert row['value']['mode'] == 'casual' and row['value']['examples']
        assert not engine.context(scope=SCOPE, persona='persona', member='alice', baseline='b', host_version='h')['reply_projection']['items']
        for scene in [{'communication_intent': 'task'}, {'communication_intent': 'support'}, {'turn_update': 'corrects_previous'}]:
            assert not engine.expressions.context(**{**args, 'scene': scene})['items']
        for scope, persona in [('qq:FriendMessage:alice', 'persona'), (SCOPE, 'other')]:
            assert not engine.expressions.context(**{**args, 'scope': scope, 'persona': persona})['items']
        assert not engine.expressions.context(**{**args, 'scene': {}})['items']
        engine.expressions.recent[(SCOPE, 'persona')] = {'markers': {'吧'}, 'examples': {v['id'] for v in row['value']['examples']}, 'expires': time.time() + 100}
        fresh = engine.expressions.context(**args)['items'][0]
        assert not any(t['marker'] == '吧' for t in fresh['value']['traits'])
        assert set(v['id'] for v in fresh['value']['examples']).isdisjoint(v['id'] for v in row['value']['examples'])
        frozen = {k: row[k] for k in ('source_id', 'revision', 'selection')}
        assert engine.expressions.context(**args, frozen=frozen)['items'] == packet['items']
        await engine.action(p['id'], 'rollback', 'human:test', p['updated'])
        assert not engine.expressions.context(**args, frozen=frozen)['items']
    asyncio.run(run())


@pytest.mark.parametrize('phrase,kind', [('太正式了', 'too_formal'), ('别解释，这个我知道', 'overexplaining'), ('别每次都问', 'repeated_question'), ('不要老用这个词', 'repeated_phrase'), ('他说“太正式了”', None), ('太正式了？', None), ('哈哈哈', None)])
def test_feedback_is_explicit_not_silence_or_third_party(phrase, kind):
    assert expression_feedback(phrase) == kind


def test_expression_feedback_proposes_only_reviewed_downgrade(store):
    from test_scene_learning import receipt, delivery, feedback
    async def run():
        p = apply_style(store)
        for i in range(2):
            receipt(store, p, turn='t' + str(i))
            delivery(store, 't' + str(i), 'b' + str(i))
            feedback(store, '太正式了', 'b' + str(i), 'f' + str(i))
        engine = Engine(store, {'sessions': [SCOPE], 'expression_style_enabled': True}, None)
        engine.current_personas[SCOPE] = 'persona'
        await engine.refresh()
        await engine.scenes.propose_adjustments(SCOPE, 'persona', engine.scope_token(SCOPE), expression=True)
        assert any(p['category'] == 'expression_adjustment' and p['state'] == 'pending' for p in store.proposals())
        assert not any(p['category'] == 'expression_adjustment' for p in store.entries())
        assert not any(p['category'] == 'scene_adjustment' for p in store.proposals())
    asyncio.run(run())


def test_background_style_analysis_does_not_call_model_each_cycle(store):
    async def run():
        rows = samples()
        for e in rows:
            store.db.execute('INSERT INTO profile_messages VALUES(?,?,?,?,?,?)', (SCOPE, 'persona', e['message_id'], e['member'], e['timestamp'], encode(e)))
        store.db.commit()
        calls = []
        async def model(role, instruction, data):
            calls.append(role)
            return {'style': raw_style(rows)['value']} if role == 'analysis_provider' else {'verdict': 'approve', 'reason': 'supported', 'risks': []}
        engine = Engine(store, {'sessions': [SCOPE], 'expression_style_enabled': True}, model)
        engine.current_personas[SCOPE] = 'persona'
        await engine.expressions.analyze(SCOPE, 'persona')
        await engine.expressions.analyze(SCOPE, 'persona')
        assert calls == ['analysis_provider', 'review_provider', 'review_provider']
        assert len(engine.cache) == 1 and engine.cache[0]['category'] == 'expression_style'
        assert store.db.execute("SELECT count(*) FROM audit WHERE action='applied'").fetchone()[0] == 1
        store.purge(SCOPE)
        assert store.db.execute('SELECT count(*) FROM expression_analysis').fetchone()[0] == 0
    asyncio.run(run())


def test_busy_current_day_does_not_erase_cross_day_evidence():
    rows = samples()
    for member in ('alice', 'bob', 'carol'):
        rows.extend(source(member + '-busy-' + str(i), member, content='今天好多消息') for i in range(100))
    chosen = balanced_sample(rows)
    assert group_statistics(chosen)['days'] >= 2
    assert set(group_statistics(chosen)['per_member'].values()) == {10}


def test_rejected_enrichment_preserves_independently_approved_basic_style(store):
    async def run():
        rows = samples()
        for e in rows:
            store.db.execute('INSERT INTO profile_messages VALUES(?,?,?,?,?,?)', (SCOPE, 'persona', e['message_id'], e['member'], e['timestamp'], encode(e)))
        store.db.commit()
        async def model(role, instruction, data):
            if role == 'analysis_provider':
                return {'style': raw_style(rows)['value']}
            enriched = len(data['proposal']['value']['traits']) > 1
            return {'verdict': 'reject' if enriched else 'approve', 'reason': 'voice not supported' if enriched else 'format supported', 'risks': []}
        engine = Engine(store, {'sessions': [SCOPE], 'expression_style_enabled': True}, model)
        engine.current_personas[SCOPE] = 'persona'
        await engine.expressions.analyze(SCOPE, 'persona')
        assert len(engine.cache[0]['value']['traits']) == 1
        assert store.db.execute("SELECT count(*) FROM audit WHERE action='applied'").fetchone()[0] == 1
        assert {p['state'] for p in store.proposals()} == {'applied', 'rejected'}
    asyncio.run(run())


def test_only_actual_sent_markers_count_as_output_repetition(store):
    p = apply_style(store)
    store.scenes.provider_receipt({'stage': 'provider_kwargs_after_boundaries', 'scope': SCOPE, 'member': 'alice',
        'scene_trace': {'scope': SCOPE, 'persona': 'persona', 'member': 'alice', 'turn_id': 't'},
        'learning': [{'source_id': p['id'], 'revision': 1, 'category': 'expression_style', 'example_ids': ['example1']}]})
    recent = store.expressions.recent()[(SCOPE, 'persona')]
    assert recent['examples'] == {'example1'} and not recent['markers']
    e = source('reply-source')
    e['reply'] = {'delivered': False, 'message_id': 'sent', 'text': '那就这样吧'}
    store.expressions.delivered(e, store.entries())
    assert not store.expressions.recent()[(SCOPE, 'persona')]['markers']
    e['reply']['delivered'] = True
    store.expressions.delivered(e, store.entries())
    assert store.expressions.recent()[(SCOPE, 'persona')]['markers'] == {'吧'}
    assert store.expressions.recent()[(SCOPE, 'persona')]['expires'] > time.time() + 3500
