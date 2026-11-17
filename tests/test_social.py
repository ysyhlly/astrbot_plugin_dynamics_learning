import asyncio
from pathlib import Path
import sqlite3
import sys
import time

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from astrbot_plugin_dynamics_learning.core.affection import state, adjustment
from astrbot_plugin_dynamics_learning.core.contract import candidate, encode, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.social import Social, character_key
from astrbot_plugin_dynamics_learning.core.store import Store, Conflict
from astrbot_plugin_dynamics_learning.migration_social import inspect_sources, migrate

SCOPE='account:GroupMessage:123'
CHARACTER={'platform':'account','umo':SCOPE,'persona_id':'persona','instance_id':'instance'}


@pytest.fixture
def store(tmp_path):
    value=Store(tmp_path/'learning.sqlite3')
    yield value
    value.close()


def legacy(store, *, category='affection', member='alice', value=None, key='bot_affection', source_key=None):
    value = state({'affection_level':125}) if value is None else value
    with store.lock,store.db:
        store.db.execute('INSERT INTO legacy_social VALUES(?,?,?,?,?,?)',
            (SCOPE,source_key or character_key(CHARACTER),category,member,key,encode({'value':value,'provenance':{'row_id':1}})))


def evidence(mid='m1',member='alice'):
    return {'id':'event-'+mid,'scope':SCOPE,'persona':'fingerprint','member':member,'message_id':mid,
            'turn_id':mid,'baseline':'config','host_version':'host','baseline_threshold':70,
            'kind':'observed','text':'谢谢你认真陪我讨论','context':{'character_scope':CHARACTER}}


def affection_candidate(store, mid='m1',delta=2):
    entries=store.entries()
    old=next((e for e in entries if e['category']=='affection'),None)
    raw={'category':'affection','key':'bot_affection','member':'alice','value':{'experience_delta':delta,'tone':.1},
         'reason':'有根据的互动','evidence_ids':['event-'+mid],'days':30}
    return store.propose(candidate(raw,[evidence(mid)],old),'human','test')


def test_uncapped_growth_preserves_legacy_score_and_experience():
    before=state({'affection_level':125})
    assert before['affection_level'] == pytest.approx(125)
    assert before['max_affection'] is None
    after=state(before,delta=20)
    assert 125<after['affection_level']<126
    assert state(after,delta=-20)['affection_level'] == pytest.approx(125)
    assert state({'affection_level':0},delta=-2)['affection_level']==0


@pytest.mark.parametrize('bad',[True,float('nan'),float('inf'),'not-a-number'])
def test_affection_rejects_bad_numbers(bad):
    with pytest.raises(ValueError):
        adjustment({'experience_delta':bad})


def test_affection_requires_member_and_same_source_author():
    raw={'category':'affection','key':'bot_affection','value':{'experience_delta':1},'member':'bob',
         'reason':'test','evidence_ids':['event-m1'],'days':30}
    with pytest.raises(ValueError,match='member evidence'):
        candidate(raw,[evidence()],None)


def test_native_legacy_scope_wins_and_changed_persona_cannot_read_it(store):
    legacy(store,source_key='123',value=state({'affection_level':2}))
    legacy(store,value=state({'affection_level':125}))
    store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
    assert store.entries()[0]['value']['affection_level']==pytest.approx(125)
    async def run():
        engine=Engine(store,{'sessions':[SCOPE]},None)
        await engine.refresh()
        social=Social(engine)
        engine.social=social
        await social.initialize()
        context,_=social.context(SCOPE,'fingerprint','alice')
        assert '125' in context
        assert 'bot_to_current_member' not in social.context(SCOPE,'other-persona','alice')[0]
        assert not social.context('account:GroupMessage:other','fingerprint','alice')[0]
    asyncio.run(run())


def test_affection_apply_is_evidence_idempotent_and_rollback_restores_import(store):
    legacy(store)
    store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
    before=store.entries()[0]['value']['affection_exp']
    p=affection_candidate(store)
    store.apply(p['id'],'human:admin',p['updated'])
    with pytest.raises(Conflict,match='already used'):
        affection_candidate(store)
    store.rollback(p['id'],'human:admin')
    assert store.entries()[0]['value']['affection_exp']==before
    assert store.entries()[0]['active']


def test_persona_text_revision_keeps_same_character_relationship_without_reusing_sources(store):
    legacy(store)
    store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
    p=affection_candidate(store)
    store.apply(p['id'],'human:admin',p['updated'])
    first=store.entries()[0]['value']['affection_exp']
    store.bind_legacy_social(SCOPE,'new-text-fingerprint',character_key(CHARACTER))
    inherited=next(e for e in store.entries() if e['persona']=='new-text-fingerprint')
    assert inherited['value']['affection_exp']==first
    assert inherited['value']['character_key']==character_key(CHARACTER)


def test_new_member_relationship_survives_persona_text_revision(store):
    store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
    p=affection_candidate(store)
    store.apply(p['id'],'human:admin',p['updated'])
    before=store.entries()[0]['value']['affection_exp']
    store.bind_legacy_social(SCOPE,'revised',character_key(CHARACTER))
    assert next(e for e in store.entries() if e['persona']=='revised')['value']['affection_exp']==before


def test_old_persona_proposal_cannot_overwrite_current_relationship(store):
    legacy(store)
    store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
    p=affection_candidate(store)
    async def run():
        engine=Engine(store,{'sessions':[SCOPE],'affection_review':'human'},None)
        await engine.refresh()
        engine.current_personas[SCOPE]='new-persona-text'
        with pytest.raises(ValueError,match='角色资料已更新'):
            await engine.action(p['id'],'approve','human:admin',p['updated'])
    asyncio.run(run())


def test_different_character_cannot_reuse_legacy_bare_group_affection(store):
    legacy(store,source_key='123')
    store.bind_legacy_social(SCOPE,'first',character_key(CHARACTER))
    other={**CHARACTER,'persona_id':'different-character'}
    store.bind_legacy_social(SCOPE,'different',character_key(other))
    assert not [e for e in store.entries() if e['persona']=='different' and e['category']=='affection']


def test_slang_only_in_matching_group_query_and_has_provenance(store):
    legacy(store,category='jargon',member='',key='老六',value='本群对出其不意操作的戏称')
    store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
    async def run():
        engine=Engine(store,{'sessions':[SCOPE]},None)
        await engine.refresh()
        args=dict(scope=SCOPE,persona='fingerprint',member='alice',baseline='config',host_version='host')
        assert '本群对' in engine.context(**args,query='你这个老六')['context']
        assert not engine.context(**args,query='今天天气怎么样')['context']
        assert store.entries()[0]['provenance']['row_id']==1
    asyncio.run(run())


def test_operator_slang_correction_creates_reviewable_proposal(store):
    legacy(store,category='jargon',member='',key='老六',value='原释义')
    store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
    async def run():
        engine=Engine(store,{'sessions':[SCOPE],'jargon_review':'human'},None)
        await engine.refresh()
        entry=engine.cache[0]
        p=await engine.correct_jargon(entry['id'],entry['revision'],'纠正后的本群含义','管理员核对来源','human:admin')
        assert p['state']=='pending' and engine.cache[0]['value']=='原释义'
        await engine.action(p['id'],'approve','human:admin',p['updated'])
        assert engine.cache[0]['value']=='纠正后的本群含义'
        await engine.action(p['id'],'rollback','human:admin',store.get(p['id'])['updated'])
        assert engine.cache[0]['value']=='原释义'
    asyncio.run(run())


def test_embedded_graph_counts_once_and_preserves_directed_evidence(store):
    async def run():
        engine=Engine(store,{'sessions':[SCOPE]},None)
        social=Social(engine)
        kwargs=dict(window_seconds=300,max_recent=60,pair_weight=.25,mention_weight=5,same_topic_weight=.8,
                    directed=[('bob','reply',8.0)],source_id='m1')
        await social.graph.record_message(SCOPE,'alice','Alice','谢谢你的帮助',**kwargs)
        await social.graph.record_message(SCOPE,'alice','Alice','谢谢你的帮助',**kwargs)
        full=await social.graph.overview(SCOPE,top_limit=10000,daily_decay=.985)
        assert full['users']['alice']['lifetime_count']==1
        assert full['all_relations'][0]['reply_hits']==1
        with pytest.raises(ValueError,match='collision'):
            await social.graph.record_message(SCOPE,'alice','Alice','different text',**kwargs)
        assert not (await social.graph.overview('other:GroupMessage:123',top_limit=10,daily_decay=.985))['users']
    asyncio.run(run())


def test_social_purge_invalidates_old_queue_and_keeps_other_scopes(store):
    async def run():
        engine=Engine(store,{'sessions':[SCOPE]},None)
        social=Social(engine)
        engine.social=social
        social.observe(dict(scope=SCOPE,member='alice',source_id='m1',name='A',text='hi',directed=[],bot_id='bot'))
        await engine.purge(SCOPE)
        consumer=asyncio.create_task(social.consume())
        await social.queue.join()
        consumer.cancel()
        await asyncio.gather(consumer,return_exceptions=True)
        assert not (await social.graph.overview(SCOPE,top_limit=10,daily_decay=.985))['users']
    asyncio.run(run())


def test_cancelled_social_worker_finishes_owned_write_before_shutdown(store):
    async def run():
        social=Social(Engine(store,{'sessions':[SCOPE]},None))
        started,release=asyncio.Event(),asyncio.Event()
        completed=[]
        async def write():
            started.set()
            await release.wait()
            completed.append(True)
        worker=asyncio.create_task(social.write(write()))
        await started.wait()
        worker.cancel()
        await asyncio.sleep(0)
        assert not worker.done()
        release.set()
        await asyncio.gather(worker,return_exceptions=True)
        await social.drain()
        assert completed==[True] and not social.owned_tasks
    asyncio.run(run())


def test_analysis_budget_keeps_distinct_turns_instead_of_repeated_phases(store):
    async def run():
        for index in range(15):
            for kind in ('observed','decision','outcome'):
                row={'schema_version':1,'kind':kind,'scope':SCOPE,'persona':'fingerprint','member':'alice',
                     'turn_id':str(index),'message_id':str(index),'host_version':'host','baseline':'config',
                     'baseline_threshold':70,'mandatory':False,'text':'真实问题',
                     'context':{'persona_summary':'A'*2000},'policy':{'context':'B'*2500}}
                store.ingest(event_record(row))
        async def model(role,prompt,data):
            assert len(data['events'])==15
            assert len({e['turn_id'] for e in data['events']})==15
            assert all(e['kind']=='outcome' for e in data['events'])
            return {'proposals':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        await engine.analyze(SCOPE,'fingerprint')
    asyncio.run(run())


def test_migration_is_previewed_idempotent_and_retains_source(tmp_path):
    sl=tmp_path/'legacy.db'
    with sqlite3.connect(sl) as db:
        db.executescript('''CREATE TABLE user_affections(id INTEGER,group_id TEXT,user_id TEXT,affection_level REAL,
            affection_exp TEXT,affection_growth_interval REAL,updated_at REAL);
            CREATE TABLE jargon(id INTEGER,chat_id TEXT,content TEXT,meaning TEXT,is_jargon INT,is_complete INT,is_global INT,forget_stage INT,count INT,updated_at REAL);
            CREATE TABLE narrative_observations(scope_key TEXT,envelope TEXT,timestamp REAL);''')
        db.execute('INSERT INTO user_affections VALUES(?,?,?,?,?,?,?)',(1,'123','alice',125,None,20,time.time()))
        db.execute('INSERT INTO jargon VALUES(?,?,?,?,?,?,?,?,?,?)',(1,'123','老六','出其不意',1,1,0,0,4,time.time()))
        db.execute('INSERT INTO jargon VALUES(?,?,?,?,?,?,?,?,?,?)',(2,'123','普通词','普通含义',0,1,0,0,4,time.time()))
    graph=tmp_path/'legacy-relations.db'
    target=tmp_path/'new.db'
    legacy_store=Store(graph)
    relation=Social(Engine(legacy_store,{'sessions':[SCOPE]},None)).graph
    asyncio.run(relation.record_message('123','alice','Alice','hi',window_seconds=300,max_recent=20,pair_weight=.25,mention_weight=5,same_topic_weight=.8))
    before=sl.read_bytes()
    preview=inspect_sources(sl,graph,[SCOPE])
    assert len(preview['records'])==2
    result=migrate(sl,graph,target,[SCOPE],preview['fingerprint'])
    assert result['affection_records']==1 and result['confirmed_jargon']==1
    assert migrate(sl,graph,target,[SCOPE],preview['fingerprint'])['duplicate']
    assert sl.read_bytes()==before
    with pytest.raises(ValueError,match='ambiguous'):
        inspect_sources(sl,graph,[SCOPE,'other:GroupMessage:123'])
    legacy_store.close()
