import asyncio
import time
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.profiles import native_message, statistics, valid_pair
from astrbot_plugin_dynamics_learning.core.store import Store

SCOPE = 'account:GroupMessage:test'
PERSONA = 'card-fingerprint'


@pytest.fixture
def store(tmp_path):
    store = Store(tmp_path / 'layers.sqlite3')
    yield store
    store.close()


def target(ids, member='alice', persona=PERSONA, scope=SCOPE):
    return event_record(dict(schema_version=1, kind='observed', scope=scope, persona=persona, member=member,
        turn_id='turn:'+ids[-1], message_id=ids[-1], host_version='host', baseline='base', baseline_threshold=70,
        mandatory=False, text='合并回合，不可当单条原话', context={'message_ids':ids}))


def raw(source, text, member='alice', offset=0, eligible=True, reply='', scope=SCOPE):
    return dict(scope=scope, member=member, source_id=source, text=text, eligible=eligible,
                reply_to=reply, created=time.time()-100+offset)


def add(store, source, text, member='alice', offset=0, **kwargs):
    store.profiles.observe(raw(source, text, member, offset, **kwargs))
    store.ingest(target([source], member=member, scope=kwargs.get('scope', SCOPE)))


def test_original_messages_survive_both_callback_orders_and_merged_turns(store):
    # One callback can contain multiple messages, but each original counts once.
    store.ingest(target(['a', 'b']))
    store.profiles.observe(raw('a', '第一条'))
    store.profiles.observe(raw('b', '第二条', offset=2))
    store.profiles.observe(raw('a', '第一条'))
    store.ingest(target(['a', 'b']))
    store.profiles.observe(raw('c', '第三条', offset=3))
    store.ingest(target(['c']))
    # The callback's claimed speaker cannot take another member's source.
    store.profiles.observe(raw('d', 'Bob的原话', 'bob', 4))
    store.ingest(target(['d'], member='alice'))
    sources = store.profiles.sources(SCOPE, PERSONA)
    assert [e['text'] for e in sources] == ['第一条', '第二条', '第三条']
    assert statistics(sources + sources)['sample_count'] == 3
    assert not store.profiles.sources('other:GroupMessage:test', PERSONA)
    assert not store.profiles.sources(SCOPE, 'other-card')


def test_exact_pair_sources_and_no_bridging_excluded_messages(store):
    add(store, 'a', '游戏又失败了', 'bob')
    add(store, 'cmd', '/status', 'bob', 1, eligible=False)
    add(store, 'b', '绷不住了', offset=2)
    assert not store.profiles.bundle(SCOPE, PERSONA)['examples']
    add(store, 'c', '怎么又是你', offset=3, reply='a')
    pairs = store.profiles.bundle(SCOPE, PERSONA)['examples']
    assert len(pairs) == 1 and pairs[0]['value']['link'] == 'reply'
    p = pairs[0]
    value = p['value']
    r = dict(category='expression_example', key='sample', value=value, member='alice',
             evidence_ids=[e['id'] for e in p['evidence']], reason='原话', days=30)
    normalized = candidate(r, p['evidence'], None)
    assert normalized['value']['expression'] == '怎么又是你'
    with pytest.raises(ValueError, match='edited'):
        candidate({**r, 'value':{**value, 'expression':'模型编造'}}, p['evidence'], None)
    with pytest.raises(ValueError, match='conversation pair'):
        candidate({**r, 'member':'bob'}, p['evidence'], None)
    context, expression = p['evidence']
    assert not valid_pair(context, {**expression, 'persona':'other'})
    assert not valid_pair(context, {**expression, 'timestamp':context['timestamp']+301})


def test_statistics_are_reproducible_not_model_supplied(store):
    for i in range(10):
        add(store, str(i), '短句！' if i < 5 else '另一行\n表情😀', offset=i)
    sources = store.profiles.sources(SCOPE, PERSONA)
    stats = statistics(sources)
    assert stats['sample_count'] == 10
    assert stats['short_rate'] == 1 and stats['multiline_rate'] == .5
    assert stats['punctuation_rate'] == .5 and stats['emoji_rate'] == .5
    assert stats['consecutive_rate'] == .9
    r = dict(category='typing_habit', key='stats', value={'kind':'observed_typing'}, member='alice',
             evidence_ids=[e['id'] for e in sources], reason='统计', days=30)
    assert candidate(r, sources, None)['value'] == stats
    with pytest.raises(ValueError, match='statistics must match'):
        candidate({**r, 'value':{'sample_count':999}}, sources, None)
    with pytest.raises(ValueError, match='ten distinct'):
        candidate({**r, 'evidence_ids':[sources[0]['id']]*10}, sources, None)


def test_preference_quote_validation_and_reversible_conflict(store):
    add(store, 'a', '我不喜欢剧透', offset=1)
    source = store.profiles.sources(SCOPE, PERSONA)[0]
    value = dict(topic='剧透', stance='dislike', condition='看完之前', quote='我不喜欢剧透', source_id=source['id'])
    r = dict(category='preference', key='wrong-model-key', value=value, member='alice',
             evidence_ids=[source['id']], reason='明确自述', days=30)
    data = candidate(r, [source], None)
    p = store.propose(data, 'human', 'test')
    store.apply(p['id'], 'human:admin', p['updated'])
    for update in [dict(quote='我喜欢剧透'), dict(source_id='invented')]:
        with pytest.raises(ValueError, match='quote must match'):
            candidate({**r, 'value':{**value, **update}}, [source], None)
    add(store, 'b', '我现在喜欢先看剧透', offset=2)
    new = store.profiles.sources(SCOPE, PERSONA)[-1]
    replacement = {**value, 'stance':'like', 'quote':new['text'], 'source_id':new['id']}
    data2 = candidate({**r, 'value':replacement, 'evidence_ids':[new['id']]}, [new], None)
    assert data2['key'] == data['key']
    p2 = store.propose(data2, 'human', 'test')
    p2 = store.apply(p2['id'], 'human:admin', p2['updated'])
    assert store.entries()[0]['value']['stance'] == 'like'
    store.rollback(p2['id'], 'human:admin')
    assert store.entries()[0]['value']['stance'] == 'dislike'


def test_three_layers_require_review_and_only_relevant_approved_samples_reach_reply(store):
    async def run():
        add(store, 'context', '游戏又失败了', 'bob')
        for i in range(10):
            add(store, str(i), '绷不住了，游戏又失败' if i == 0 else '我不喜欢剧透', offset=i+1,reply='context' if i==0 else '')
        async def model(role, prompt, data):
            if role == 'review_provider':
                return dict(verdict='approve', reason='测试审查', risks=[])
            e = next(e for e in data['member_sources'] if e['text']=='我不喜欢剧透')
            return {'proposals':[dict(category='preference', key='spoilers', member='alice', days=30,
                reason='明确自述', evidence_ids=[e['id']], value=dict(topic='剧透', stance='dislike', condition='', quote=e['text'], source_id=e['id']))]}
        engine = Engine(store, {'sessions':[SCOPE], 'review_provider':'test',
            **{c+'_review':'llm_then_human' for c in ['style','typing_habit','preference','expression_example']}}, model)
        await engine.refresh()
        await engine.analyze(SCOPE, PERSONA)
        assert {p['category'] for p in store.proposals()} == {'expression_example', 'typing_habit', 'preference'}
        assert all(p['state']=='reviewed' for p in store.proposals())
        assert not engine.member_profile(SCOPE,'alice')['available']
        args = dict(scope=SCOPE, persona=PERSONA, member='alice', baseline='base', host_version='host')
        assert engine.context(**args,query='游戏失败')['context']==''
        for p in store.proposals():
            await engine.action(p['id'], 'approve', 'human:admin', p['updated'])
        profile = engine.member_profile(SCOPE, 'alice')
        assert all(profile['layers'][layer] for layer in ('examples', 'habits', 'preferences'))
        assert '绷不住' in engine.context(**args, query='游戏失败')['context']
        assert '绷不住' not in engine.context(**args, query='明天天气')['context']
        assert '我不喜欢剧透' in engine.context(**args, query='明天天气')['context']
        assert not engine.context(**{**args,'member':'bob'}, query='游戏失败')['context']
        sample = next(p for p in store.proposals() if p['category']=='expression_example')
        await engine.action(sample['id'], 'rollback', 'human:admin', sample['updated'])
        assert '绷不住' not in engine.context(**args, query='游戏失败')['context']
        assert len(engine.context(**args)['context']) <= 5000
        await engine.purge(SCOPE)
        assert not store.profiles.sources(SCOPE, PERSONA)
        assert store.db.execute('SELECT count(*) FROM profile_raw').fetchone()[0] == 0
        assert not engine.member_profile(SCOPE,'alice')['available']
    asyncio.run(run())


def test_native_observer_excludes_commands_and_bot_but_keeps_boundary():
    class Reply:
        id = 'original'
    event = SimpleNamespace(unified_msg_origin=SCOPE, get_sender_id=lambda:'alice', get_message_str=lambda:'你好！',
        get_messages=lambda:[Reply()], message_obj=SimpleNamespace(message_id='1', self_id='bot'))
    assert native_message(event)['eligible'] and native_message(event)['reply_to']=='original'
    event.get_message_str=lambda:'/status'
    assert not native_message(event)['eligible']
    event.get_sender_id=lambda:'bot'
    event.get_message_str=lambda:'Bot的话'
    assert not native_message(event)['eligible']
    event.unified_msg_origin='account:FriendMessage:test'
    assert native_message(event) is None


def test_profile_raw_retention_and_rejected_sample_not_resubmitted(store):
    add(store, 'a', '游戏失败了', 'bob')
    add(store, 'b', '绷不住了', offset=1,reply='a')
    r, evidence = store.profiles.proposals(SCOPE,PERSONA)[0]
    p = store.propose(candidate(r,evidence,None), 'human', 'test')
    store.review(p['id'], dict(verdict='reject', reason='错配语境', risks=[]), 'human:admin', p['updated'])
    assert not store.profiles.proposals(SCOPE,PERSONA)
    with store.db:
        for table in ['profile_raw','profile_targets','profile_messages']:
            store.db.execute('UPDATE '+table+' SET created=1')
    store.checkpoint(SCOPE,PERSONA,time.time())
    store.prune(30)
    assert not store.profiles.sources(SCOPE,PERSONA)
    assert store.get(p['id'])['evidence']  # Audit evidence is not silently erased by raw retention.
