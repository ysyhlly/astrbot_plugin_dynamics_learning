import asyncio
import sqlite3
import time

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, event_record
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from test_learning import event, raw
from test_three_layers import add, SCOPE, PERSONA


@pytest.fixture
def store(tmp_path):
    store=Store(tmp_path/'optimization.sqlite3')
    yield store
    store.close()


def callback(index,kind='observed',scope='qq:GroupMessage:1',size=0):
    original={k:v for k,v in event().items() if k not in {'id','received_at'}}
    original.update(scope=scope,turn_id=str(index),message_id=str(index),kind=kind,context={'padding':'x'*size})
    row=event_record(original)
    row['received_at']=1000.0  # Equal timestamps must not make a frontier ambiguous.
    return row


def test_incremental_batches_merge_phases_before_limit_and_never_skip_old_turns(store):
    async def run():
        for index in range(120):
            for kind in ['observed','decision','outcome']:
                store.ingest(callback(index,kind))
        seen=[]
        async def model(role,prompt,data):
            seen.extend(e['turn_id'] for e in data['events'])
            assert all(e['kind']=='outcome' for e in data['events'])
            return {'proposals':[]}
        engine=Engine(store,{'sessions':['qq:GroupMessage:1']},model)
        first=await engine.analyze('qq:GroupMessage:1','persona-a')
        assert first['turns']==80 and first['callbacks']==240
        assert store.batches(1)[0]['count']==40
        second=await engine.analyze('qq:GroupMessage:1','persona-a')
        assert second['turns']==40 and second['callbacks']==120
        assert seen==list(map(str,range(120)))
        await engine.analyze('qq:GroupMessage:1','persona-a')
        assert len(seen)==120 and not store.batches(1)
    asyncio.run(run())


def test_model_budget_preserves_contiguous_pending_frontiers(store):
    async def run():
        for index in range(12):
            store.ingest(callback(index,size=22000))
        seen=[]
        async def model(role,prompt,data):
            seen.extend(e['turn_id'] for e in data['events'])
            return {'proposals':[]}
        engine=Engine(store,{'sessions':['qq:GroupMessage:1']},model)
        for _ in range(5):
            await engine.analyze('qq:GroupMessage:1','persona-a')
        assert seen==list(map(str,range(12)))
        assert not store.batches(1)
    asyncio.run(run())


def test_model_failure_and_transaction_failure_do_not_advance_progress(store,monkeypatch):
    async def run():
        e=event()
        store.ingest(e)
        async def fail(*args):
            raise TimeoutError()
        engine=Engine(store,{'sessions':[e['scope']]},fail)
        with pytest.raises(TimeoutError):
            await engine.analyze(e['scope'],e['persona'])
        assert store.batches(1)[0]['count']==1
        async def success(*args):
            return {'proposals':[raw()]}
        engine.call_model=success
        def audit_failure(*args):
            raise RuntimeError('storage failure')
        monkeypatch.setattr(store,'_audit',audit_failure)
        with pytest.raises(RuntimeError):
            await engine.analyze(e['scope'],e['persona'])
        assert not store.proposals() and not store.scheduling.recent()
        assert store.batches(1)[0]['count']==1
    asyncio.run(run())


def test_ingress_and_other_groups_continue_while_model_waits(store):
    async def run():
        scopes=['qq:GroupMessage:1','qq:GroupMessage:2']
        for scope in scopes:
            store.ingest(callback(0,scope=scope))
        entered=asyncio.Event()
        both=asyncio.Event()
        release=asyncio.Event()
        calls=[]
        async def model(role,prompt,data):
            calls.append(data['events'][0]['scope'])
            entered.set()
            if len(calls)==2:
                both.set()
            await release.wait()
            return {'proposals':[]}
        engine=Engine(store,{'sessions':scopes},model)
        consumer=asyncio.create_task(engine.consume())
        first=asyncio.create_task(engine.analyze(scopes[0],'persona-a'))
        await asyncio.wait_for(entered.wait(),1)
        incoming={k:v for k,v in callback(1).items() if k not in {'id','received_at'}}
        assert engine.ingest(incoming)
        await asyncio.wait_for(engine.queue.join(),1)
        assert len(store.events(scopes[0],'persona-a'))==2
        second=asyncio.create_task(engine.analyze(scopes[1],'persona-a'))
        await asyncio.wait_for(both.wait(),1)
        release.set()
        # Ingress changed a previously unknown current persona: discard that
        # snapshot once, preserving its evidence for a fresh run.
        results=await asyncio.gather(first,second,return_exceptions=True)
        assert any(isinstance(r,ValueError) for r in results)
        await engine.analyze(scopes[0],'persona-a')
        assert not store.batches(1)
        consumer.cancel()
        await asyncio.gather(consumer,return_exceptions=True)
    asyncio.run(run())


def test_feedback_arriving_during_model_call_remains_pending(store):
    async def run():
        first=callback(0)
        store.ingest(first)
        entered=asyncio.Event()
        release=asyncio.Event()
        async def model(*args):
            entered.set()
            await release.wait()
            return {'proposals':[]}
        engine=Engine(store,{'sessions':[first['scope']]},model)
        task=asyncio.create_task(engine.analyze(first['scope'],first['persona']))
        await entered.wait()
        store.ingest(callback(0,'outcome'))
        store.ingest(callback(1))
        release.set()
        result=await task
        assert result['callbacks']==1
        assert store.batches(1)[0]['callbacks']==2
        await engine.analyze(first['scope'],first['persona'])
        assert not store.batches(1)
    asyncio.run(run())


def test_reject_and_purge_during_model_wait_cannot_be_undone(store):
    async def run():
        e=event()
        p=store.propose(candidate(raw(),[e],None),'llm_then_human','test')
        entered=asyncio.Event()
        release=asyncio.Event()
        async def model(*args):
            entered.set()
            await release.wait()
            return {'verdict':'approve','reason':'late','risks':[]}
        engine=Engine(store,{'sessions':[e['scope']],'style_review':'llm_then_human'},model)
        task=asyncio.create_task(engine.review(p['id']))
        await entered.wait()
        await asyncio.wait_for(engine.action(p['id'],'reject','human:admin',p['updated']),1)
        release.set()
        with pytest.raises(ValueError):
            await task
        assert store.get(p['id'])['state']=='rejected' and not store.entries()
        assert len(store.get(p['id'])['reviews'])==1
        store.ingest(e)
        entered.clear()
        release.clear()
        async def analysis(*args):
            entered.set()
            await release.wait()
            return {'proposals':[raw()]}
        engine.call_model=analysis
        task=asyncio.create_task(engine.analyze(e['scope'],e['persona']))
        await entered.wait()
        await asyncio.wait_for(engine.purge(e['scope']),1)
        release.set()
        with pytest.raises(ValueError):
            await task
        assert not store.entries() and not store.proposals() and not store.batches(1)
    asyncio.run(run())


def approved(store,key,member='alice'):
    e=event()
    e['member']=member
    data={**raw(),'key':key,'member':member}
    p=store.propose(candidate(data,[e],None),'llm_then_human','test')
    return store.review(p['id'],{'verdict':'approve','reason':'ok','risks':[]},'llm:test',p['updated'])


def test_bulk_apply_is_atomic_and_member_scoped(store):
    async def run():
        first=approved(store,'one')
        second=approved(store,'two')
        engine=Engine(store,{'sessions':[first['scope']]},None)
        items=[{'id':p['id'],'updated':p['updated']} for p in [first,second]]
        with pytest.raises(ValueError):
            await engine.batch_action(first['scope'],first['persona'],'bob',items,'human:admin')
        assert not store.entries()
        with pytest.raises(ValueError):
            store.apply_batch([(first['id'],first['updated'],False),(second['id'],0,False)],'human:admin')
        assert not store.entries() and store.get(first['id'])['state']=='reviewed'
        result=await engine.batch_action(first['scope'],first['persona'],'alice',items,'human:admin')
        assert result['count']==2 and len(store.entries())==2
        p=store.get(first['id'])
        await engine.action(p['id'],'rollback','human:admin',p['updated'])
        assert len([e for e in store.entries() if e['active']])==1
    asyncio.run(run())


def test_unapproved_item_blocks_entire_bulk_apply(store):
    async def run():
        first=approved(store,'one')
        e=event()
        second=store.propose(candidate({**raw(),'key':'two'},[e],None),'llm_then_human','test')
        engine=Engine(store,{'sessions':[first['scope']]},None)
        with pytest.raises(ValueError,match='LLM approval'):
            await engine.batch_action(first['scope'],first['persona'],'alice',
                [{'id':p['id'],'updated':p['updated']} for p in [first,second]],'human:admin')
        assert not store.entries()
    asyncio.run(run())


def test_quality_gate_prefers_direct_reply_and_requires_repeated_topic_for_adjacency(store):
    add(store,'a','游戏失败了','bob')
    add(store,'b','游戏又失败',offset=1)
    assert not store.profiles.bundle(SCOPE,PERSONA)['examples']
    add(store,'c','游戏失败了','bob',65)
    add(store,'d','游戏又失败',offset=66)
    assert len(store.profiles.bundle(SCOPE,PERSONA)['examples'])==1
    add(store,'e','绷不住了',offset=67,reply='c')
    add(store,'f','好的',offset=68,reply='c')
    add(store,'g','看看 https://example.com',offset=69,reply='c')
    examples=store.profiles.bundle(SCOPE,PERSONA)['examples']
    assert {p['value']['expression'] for p in examples}=={'游戏又失败','绷不住了'}


def test_pending_evidence_survives_retention_until_committed(store):
    e=callback(0)
    store.ingest(e)
    store.prune(1)
    assert store.events(e['scope'],e['persona'])
    batch=store.scheduling.batch(e['scope'],e['persona'])
    store.scheduling.commit(batch,[],0)
    store.prune(1)
    assert not store.events(e['scope'],e['persona'])


def test_fair_cycle_processes_small_aged_groups_and_limits_models(store):
    async def run():
        scopes=['qq:GroupMessage:'+str(n) for n in range(3)]
        for scope in scopes:
            store.ingest(callback(0,scope=scope))
        active=0
        maximum=0
        calls=[]
        overlapping=asyncio.Event()
        async def model(role,prompt,data):
            nonlocal active,maximum
            active+=1
            maximum=max(maximum,active)
            calls.append(data['events'][0]['scope'])
            if len(calls)>=2:
                overlapping.set()
            await asyncio.wait_for(overlapping.wait(),2)
            active-=1
            return {'proposals':[]}
        engine=Engine(store,{'sessions':scopes,'batch_events':20,'analysis_interval':600},model)
        await engine.analysis_cycle()
        assert set(calls)==set(scopes) and maximum==2
        assert not store.batches(1)
        assert engine.model_queued==engine.model_active==0
    asyncio.run(run())


def test_legacy_checkpoint_seed_and_restart_preserve_pending_frontier(tmp_path):
    path=tmp_path/'legacy.sqlite3'
    with sqlite3.connect(path) as db:
        db.executescript('CREATE TABLE events(id TEXT PRIMARY KEY,scope TEXT,persona TEXT,created REAL,body TEXT); CREATE TABLE checkpoints(scope TEXT,persona TEXT,cursor REAL,PRIMARY KEY(scope,persona));')
        from astrbot_plugin_dynamics_learning.core.contract import encode
        for i in range(3):
            row=callback(i)
            row['received_at']=i+1
            db.execute('INSERT INTO events VALUES(?,?,?,?,?)',(row['id'],row['scope'],row['persona'],i+1,encode(row)))
        db.execute("INSERT INTO checkpoints VALUES('qq:GroupMessage:1','persona-a',1)")
    store=Store(path)
    batch=store.scheduling.batch('qq:GroupMessage:1','persona-a')
    assert [e['turn_id'] for e in batch['events']]==['1','2']
    batch['events']=batch['events'][:1]
    batch['frontiers']=batch['frontiers'][:1]
    store.scheduling.commit(batch,[],0)
    store.close()
    store=Store(path)
    assert [e['turn_id'] for e in store.scheduling.batch('qq:GroupMessage:1','persona-a')['events']]==['2']
    store.close()


def test_cancelled_model_waiter_does_not_leak_queue_count(store):
    async def run():
        release=asyncio.Event()
        async def model(*args):
            await release.wait()
            return {}
        engine=Engine(store,{},model)
        running=[asyncio.create_task(engine.invoke_model('x','',{})) for _ in range(3)]
        for _ in range(100):
            if engine.model_active==2 and engine.model_queued==1:
                break
            await asyncio.sleep(.001)
        assert engine.model_active==2 and engine.model_queued==1
        running[-1].cancel()
        await asyncio.gather(running[-1],return_exceptions=True)
        assert engine.model_queued==0
        release.set()
        await asyncio.gather(*running[:2])
        assert engine.model_active==0 and not engine.operation_locks
    asyncio.run(run())


def test_pending_native_sources_are_not_pruned_before_analysis(store):
    add(store,'old','我不喜欢剧透')
    with store.db:
        for table in ('events','analysis_inbox','profile_raw','profile_targets','profile_messages'):
            store.db.execute('UPDATE '+table+' SET created=1')
    store.prune(1)
    assert store.db.execute('SELECT count(*) FROM profile_messages').fetchone()[0]==1
    store.checkpoint(SCOPE,PERSONA,time.time())
    store.prune(1)
    assert store.db.execute('SELECT count(*) FROM profile_messages').fetchone()[0]==0


def test_consume_batch_drain_and_per_scope_context_revision(store):
    from types import SimpleNamespace
    from astrbot_plugin_dynamics_learning import main

    async def run():
        s1='qq:GroupMessage:1'
        s2='qq:GroupMessage:2'
        engine=Engine(store,{'sessions':[s1,s2],'batch_events':20},None)
        plugin=main.DynamicsLearning(SimpleNamespace(),{'sessions':[s1,s2]})
        plugin.engine=engine
        plugin.ready=True
        for i in range(6):
            engine.ingest({k:v for k,v in callback(i,scope=s1 if i<4 else s2).items() if k not in {'id','received_at'}})
        consumer=asyncio.create_task(engine.consume())
        try:
            await asyncio.wait_for(engine.queue.join(),3)
        finally:
            engine.closed=True
            consumer.cancel()
            await asyncio.gather(consumer,return_exceptions=True)
        engine.closed=False
        b1={r['scope']:r for r in store.batches(1)}
        assert b1[s1]['count']==4 and b1[s2]['count']==2
        fetched=store.scheduling.batch(s1,'persona-a')
        assert len(fetched['events'])==4
        store.scheduling.commit(fetched,[],0)
        b0={r['scope']:r for r in store.batches(0)}
        assert b0[s1]['count']==0 and b0[s1]['cursor']>0 and b0[s2]['count']==2
        gen_s1_before=plugin.peek_context_revision({'umo':s1})['generation']
        gen_s2_before=plugin.peek_context_revision({'umo':s2})['generation']
        p=store.propose(candidate({**raw(),'category':'jargon','member':'','key':'黑话甲','evidence_ids':[fetched['events'][0]['id']]},[fetched['events'][0]],None),'human','t')
        store.apply(p['id'],'human:admin',p['updated'])
        await engine.refresh()
        assert plugin.peek_context_revision({'umo':s1})['generation']>gen_s1_before
        assert plugin.peek_context_revision({'umo':s2})['generation']==gen_s2_before
    asyncio.run(run())


def test_scene_disambiguation_fast_lane_when_background_slots_full(store):
    async def run():
        gate=asyncio.Event()
        async def model(role,instruction,data):
            if data.get('text'):
                return {'communication_intent':'banter','expression_mode':'joke','turn_update':'continues_joke'}
            await gate.wait()
            return {'proposals':[]}
        engine=Engine(store,{'sessions':[SCOPE],'scene_learning_enabled':True,'scene_disambiguation_enabled':True,'analysis_provider':'m'},model)
        engine.current_personas[SCOPE]=PERSONA
        bg=[asyncio.create_task(engine.invoke_model('analysis_provider','x',{'scope':SCOPE})) for _ in range(2)]
        for _ in range(50):
            if engine.model_active==2:
                break
            await asyncio.sleep(.002)
        assert engine.model_active==2
        res=await asyncio.wait_for(engine.scenes.disambiguate(scope=SCOPE,persona=PERSONA,member='alice',turn_id='t1',text='哈哈哈哈',context=[]),1.0)
        assert res['expression_mode']=='joke'
        gate.set()
        await asyncio.gather(*bg)
    asyncio.run(run())


def test_jargon_propose_overflow_and_catchup_review(store,monkeypatch):
    import json
    from types import SimpleNamespace
    from astrbot_plugin_dynamics_learning import main

    async def run():
        e=event()
        store.ingest(e)
        p=store.propose(candidate({**raw(),'category':'jargon','member':'','key':'老梗'},[e],None),'human','fixture')
        p=store.apply(p['id'],'human:admin',p['updated'])
        entry=store.entries()[0]
        calls=[]
        async def model(role,instruction,data):
            calls.append((role,data['proposal']['id']))
            return {'verdict':'approve','reason':'依据充分','risks':[]}
        plugin=main.DynamicsLearning(SimpleNamespace(),{'sessions':[e['scope']],'jargon_review':'llm','review_provider':'rm'})
        plugin.engine=Engine(store,plugin.config,model)
        plugin.engine.current_personas[e['scope']]=e['persona']
        plugin.ready=True
        await plugin.engine.refresh()
        plugin.jobs={'busy-1','busy-2'}
        monkeypatch.setattr(plugin,'_authorized',lambda: asyncio.sleep(0,result='human:admin'))
        body=json.dumps({'id':entry['id'],'revision':entry['revision'],'value':'更新含义','reason':'管理员纠正'}).encode()
        monkeypatch.setattr(main,'request',SimpleNamespace(_request=SimpleNamespace(body=lambda: asyncio.sleep(0,result=body))))
        resp=await plugin.jargon_propose()
        payload=json.loads(resp.body)
        assert resp.status_code==200 and payload['data']['state']=='pending' and payload['data']['queued'] is False
        pid=payload['data']['id']
        await plugin.engine.review_pending_proposals()
        assert calls==[('review_provider',pid)]
        assert store.get(pid)['state']=='applied'
        assert any(row['value']=='更新含义' and row['active'] for row in store.entries())
    asyncio.run(run())


def test_migrations_skip_private_umos_without_raising(tmp_path):
    from astrbot_plugin_dynamics_learning.core.social import Social
    from astrbot_plugin_dynamics_learning.migration_replacement import plan
    from astrbot_plugin_dynamics_learning.migration_social import inspect_sources

    src=tmp_path/'self.sqlite3'
    with sqlite3.connect(src) as db:
        db.executescript('CREATE TABLE user_affections(id INTEGER PRIMARY KEY,group_id TEXT,user_id TEXT,affection_level REAL,affection_exp TEXT,affection_growth_interval REAL,updated_at REAL); CREATE TABLE jargon(id INTEGER PRIMARY KEY,chat_id TEXT,content TEXT,meaning TEXT,is_jargon INT,is_complete INT,is_global INT,forget_stage INT,count INT,updated_at REAL); CREATE TABLE narrative_observations(scope_key TEXT,envelope TEXT,timestamp REAL);')
        db.execute('INSERT INTO user_affections VALUES(1,"123","alice",80,NULL,20,0)')
    rel=tmp_path/'rel.sqlite3'
    rel_store=Store(rel)
    Social(Engine(rel_store,{'sessions':['qq:GroupMessage:123']},None)).graph
    sessions=['qq:GroupMessage:123','qq:FriendMessage:999']
    p_res=plan(src,sessions)
    assert p_res['skipped_private']==['qq:FriendMessage:999'] and len(p_res['records'])==1
    s_res=inspect_sources(src,rel,sessions)
    assert s_res['skipped_private']==['qq:FriendMessage:999'] and s_res['scopes']==['qq:GroupMessage:123']
    with pytest.raises(ValueError,match='ambiguous'):
        inspect_sources(src,rel,['qq:GroupMessage:123','other:GroupMessage:123','qq:FriendMessage:999'])
    rel_store.close()


def test_scene_summary_cache_invalidates_on_write(store):
    first=store.scenes.summary()
    second=store.scenes.summary()
    assert first==second and first['attempts']==0
    store.scenes.receipt({'scope':SCOPE,'persona':PERSONA,'turn_id':'t1','member':'alice','stage':'disambiguation','status':'succeeded','elapsed_ms':120,'refs':[]})
    updated=store.scenes.summary()
    assert updated['attempts']==1 and updated['extra_wait_ms']==120


def test_same_session_rotates_personas_even_if_one_fails(store):
    async def run():
        engine = Engine(store, {"sessions": [SCOPE], "batch_events": 1, "analysis_max_rounds_per_cycle": 4}, None)
        engine.current_personas[SCOPE] = "current"
        batches = [{"scope": SCOPE, "persona": name, "count": 50, "oldest": i}
                   for i, name in enumerate(("old", "older", "current"))]
        store.batches = lambda minimum: batches
        calls = []

        async def analyze(scope, persona):
            calls.append(persona)
            if persona == "old":
                raise ValueError("fixture failure")

        engine.analyze = analyze
        await engine.analysis_cycle()
        assert calls[0] == "current"
        assert set(calls) == {"old", "older", "current"}
        assert calls.count("old") == 1
        assert len(calls) == 4

    asyncio.run(run())
