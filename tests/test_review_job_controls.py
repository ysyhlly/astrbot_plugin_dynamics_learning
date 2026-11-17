import asyncio
import copy

import pytest

from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from test_group_review import SCOPE, make, items


@pytest.fixture
def store(tmp_path):
    result=Store(tmp_path/'controls.sqlite3')
    yield result
    result.close()


def test_pause_finishes_current_then_resume_only_unfinished_across_restart(store):
    async def run():
        rows=[make(store,str(i),member=str(i),approved=False) for i in range(3)]
        entered=asyncio.Event()
        release=asyncio.Event()
        calls=[]
        async def model(role,instruction,data):
            key=data['proposal']['key']
            calls.append(key)
            if key=='1':
                entered.set()
                await release.wait()
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(*rows),'human:a')
        task=asyncio.create_task(manager.run(job))
        await entered.wait()
        response=await manager.control(job['id'],'pause','human:a')
        assert response['status']=='pausing'
        release.set()
        await task
        saved=manager.get(job['id'])
        assert saved['status']=='paused' and calls==['0','1']
        assert [i['status'] for i in saved['items']]==['reviewed','reviewed','queued']
        completed=copy.deepcopy(saved['items'][:2])
        with store.transaction():
            store.db.execute('UPDATE review_batches SET created=1')
        store.console.prune(1)
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        assert manager.get(job['id'])['status']=='paused'
        resumed=await manager.control(job['id'],'resume','human:a')
        await manager.run(resumed)
        assert calls==['0','1','2'] and resumed['status']=='completed'
        assert resumed['items'][:2]==completed
    asyncio.run(run())


def test_cancel_discards_inflight_result_and_keeps_completed_results(store):
    async def run():
        rows=[make(store,str(i),member=str(i),approved=False) for i in range(3)]
        entered=asyncio.Event()
        release=asyncio.Event()
        calls=[]
        async def model(role,instruction,data):
            key=data['proposal']['key']
            calls.append(key)
            if key=='1':
                entered.set()
                await release.wait()
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        job=await engine.review_batches.prepare(SCOPE,items(*rows),'human:a')
        task=asyncio.create_task(engine.review_batches.run(job))
        await entered.wait()
        original=copy.deepcopy(engine.review_batches.get(job['id'])['items'][0])
        assert (await engine.review_batches.control(job['id'],'cancel','human:a'))['status']=='cancelling'
        release.set()
        await task
        saved=engine.review_batches.get(job['id'])
        assert saved['status']=='cancelled' and calls==['0','1']
        assert saved['items'][0]==original
        assert [i['status'] for i in saved['items']]==['reviewed','cancelled','cancelled']
        assert store.get(rows[1]['id'])['reviews']==[]
        assert store.get(rows[2]['id'])['state']=='pending'
        with pytest.raises(ValueError):
            await engine.review_batches.control(job['id'],'resume','human:a')
    asyncio.run(run())


def test_cancel_before_model_slot_prevents_provider_call(store):
    async def run():
        p=make(store,'one',approved=False)
        calls=[]
        async def model(*args):
            calls.append(1)
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        await engine.model_slots.acquire()
        await engine.model_slots.acquire()
        job=await engine.review_batches.prepare(SCOPE,items(p),'human:a')
        task=asyncio.create_task(engine.review_batches.run(job))
        for _ in range(100):
            if engine.model_queued:
                break
            await asyncio.sleep(.01)
        assert engine.model_queued==1
        await engine.review_batches.control(job['id'],'cancel','human:a')
        engine.model_slots.release()
        engine.model_slots.release()
        await task
        assert calls==[] and job['status']=='cancelled'
    asyncio.run(run())


def test_retry_failed_only_preserves_parent_and_profile_adoption_stamp(store):
    async def run():
        first=make(store,'profile',approved=False,category='style')
        second=make(store,'success','bob',approved=False)
        calls=[]
        async def model(role,instruction,data):
            key=data['proposal']['key']
            calls.append(key)
            if key=='profile' and calls.count(key)==1:
                raise TimeoutError()
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(first,second),'human:a')
        await manager.run(job)
        parent=copy.deepcopy(manager.get(job['id']))
        assert parent['items'][0]['updated']==store.get(first['id'])['updated']
        child=await manager.control(job['id'],'retry','human:a')
        assert child['parent_id']==job['id'] and [i['id'] for i in child['items']]==[first['id']]
        with pytest.raises(ValueError):
            await manager.control(job['id'],'retry','human:a')
        await manager.run(child)
        assert calls==['profile','success','profile']
        assert child['items'][0]['status']=='applied'
        assert manager.get(job['id'])['items']==parent['items']
        with pytest.raises(ValueError):
            await manager.control(job['id'],'retry','human:a')
    asyncio.run(run())


def test_retry_skips_modified_failed_records(store):
    async def run():
        rows=[make(store,str(i),member=str(i),approved=False) for i in range(2)]
        async def model(*args):
            raise TimeoutError()
        engine=Engine(store,{'sessions':[SCOPE]},model)
        job=await engine.review_batches.prepare(SCOPE,items(*rows),'human:a')
        await engine.review_batches.run(job)
        await engine.group_batch_reject(SCOPE,items(rows[0]),'human:a')
        child=await engine.review_batches.control(job['id'],'retry','human:a')
        assert [i['id'] for i in child['items']]==[rows[1]['id']]
        assert child['retry_skipped']==[rows[0]['id']]
        assert all(i['status']=='failed' for i in engine.review_batches.get(job['id'])['items'])
    asyncio.run(run())


def test_queued_pause_cancel_and_disabled_scope_are_safe(store):
    async def run():
        p=make(store,'one',approved=False)
        engine=Engine(store,{'sessions':[SCOPE]},None)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(p),'human:a')
        with pytest.raises(ValueError):
            await manager.control(job['id'],'cancel','llm:test')
        await manager.control(job['id'],'pause','human:a')
        await manager.run(job)
        assert manager.get(job['id'])['status']=='paused'
        with pytest.raises(ValueError):
            await manager.prepare(SCOPE,items(p),'human:a')
        engine.config['sessions']=[]
        with pytest.raises(ValueError):
            await manager.control(job['id'],'resume','human:a')
        engine.config['sessions']=[SCOPE]
        await manager.control(job['id'],'cancel','human:a')
        await manager.run(job)
        assert job['status']=='cancelled' and store.get(p['id'])['state']=='pending'
    asyncio.run(run())
