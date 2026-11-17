import asyncio

import pytest

from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from test_group_review import SCOPE, OTHER, make, items


@pytest.fixture
def store(tmp_path):
    value=Store(tmp_path/'batch-actions.sqlite3')
    yield value
    value.close()


def test_reject_batch_atomic_and_records_reason(store):
    async def run():
        first=make(store,'one',approved=False)
        second=make(store,'two','bob')
        engine=Engine(store,{'sessions':[SCOPE]},None)
        invalid=items(first,second)
        invalid[1]['updated']=0
        with pytest.raises(ValueError):
            await engine.group_batch_reject(SCOPE,invalid,'human:admin','证据不足')
        assert store.get(first['id'])['state']=='pending'
        result=await engine.group_batch_reject(SCOPE,items(first,second),'human:admin','证据不足')
        assert result['count']==2 and not store.entries()
        for p in (first,second):
            saved=store.get(p['id'])
            assert saved['state']=='rejected' and saved['reviews'][-1]['reason']=='证据不足'
    asyncio.run(run())


def test_reject_transaction_rolls_back_partial_write(store,monkeypatch):
    async def run():
        rows=[make(store,'one'),make(store,'two','bob')]
        engine=Engine(store,{'sessions':[SCOPE]},None)
        review=store.review
        def fail_second(pid,*args,**kwargs):
            if pid==rows[1]['id']:
                raise RuntimeError('disk failure')
            return review(pid,*args,**kwargs)
        monkeypatch.setattr(store,'review',fail_second)
        with pytest.raises(RuntimeError):
            await engine.group_batch_reject(SCOPE,items(*rows),'human:a')
        assert all(store.get(p['id'])['state']=='reviewed' for p in rows)
    asyncio.run(run())


def test_llm_batch_continues_after_provider_failure_and_reports_each_result(store):
    async def run():
        rows=[make(store,key,member=str(i),approved=False) for i,key in enumerate(['ok','fail','reject','defer'])]
        calls=[]
        async def model(role,instruction,data):
            key=data['proposal']['key']
            calls.append(key)
            if key=='fail':
                raise TimeoutError('secret provider error must not be stored')
            return {'verdict':{'ok':'approve','reject':'reject','defer':'defer'}[key],'reason':'checked','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE],'review_provider':'test'},model)
        job=await engine.review_batches.prepare(SCOPE,items(*rows),'human:a')
        with pytest.raises(ValueError,match='已有'):
            await engine.review_batches.prepare(SCOPE,items(*rows),'human:a')
        await engine.review_batches.run(job)
        saved=engine.review_batches.recent(SCOPE)[0]
        assert calls==['ok','fail','reject','defer']
        assert [r['status'] for r in saved['items']]==['reviewed','failed','rejected','deferred']
        assert saved['items'][1]['error']=='TimeoutError'
        assert saved['status']=='completed_with_issues' and not store.entries()
    asyncio.run(run())


def test_llm_batch_member_profiles_keep_auto_apply_rule(store):
    async def run():
        p=make(store,'表达习惯',approved=False,category='style')
        async def model(*args):
            return {'verdict':'approve','reason':'已核对','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE],'review_provider':'test'},model)
        job=await engine.review_batches.prepare(SCOPE,items(p),'human:a')
        await engine.review_batches.run(job)
        assert job['items'][0]['status']=='applied' and store.get(p['id'])['state']=='applied'
    asyncio.run(run())


def test_changes_during_llm_call_and_before_start_cannot_be_overwritten(store):
    async def run():
        first=make(store,'first',approved=False)
        second=make(store,'second','bob',approved=False)
        entered=asyncio.Event()
        release=asyncio.Event()
        calls=[]
        async def model(role,instruction,data):
            calls.append(data['proposal']['id'])
            entered.set()
            await release.wait()
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE],'review_provider':'test'},model)
        job=await engine.review_batches.prepare(SCOPE,items(first,second),'human:a')
        task=asyncio.create_task(engine.review_batches.run(job))
        await entered.wait()
        await asyncio.wait_for(engine.group_batch_reject(SCOPE,items(first,second),'human:a'),1)
        release.set()
        await task
        assert calls==[first['id']]
        assert all(i['status']=='skipped' for i in job['items'])
        assert all(store.get(p['id'])['state']=='rejected' for p in (first,second))
    asyncio.run(run())


def test_cancellation_persists_interrupted_and_restart_does_not_retry(store):
    async def run():
        p=make(store,'one',approved=False)
        entered=asyncio.Event()
        async def model(*args):
            entered.set()
            await asyncio.Event().wait()
        engine=Engine(store,{'sessions':[SCOPE],'review_provider':'test'},model)
        job=await engine.review_batches.prepare(SCOPE,items(p),'human:a')
        task=asyncio.create_task(engine.review_batches.run(job))
        await entered.wait()
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        assert job['status']=='interrupted'
        reloaded=Engine(store,{'sessions':[SCOPE]},None)
        assert reloaded.review_batches.recent()[0]['items'][0]['status']=='interrupted'
        assert store.get(p['id'])['state']=='pending'
    asyncio.run(run())


def test_purge_during_job_does_not_restore_job_or_proposals(store):
    async def run():
        p=make(store,'one',approved=False)
        entered=asyncio.Event()
        release=asyncio.Event()
        async def model(*args):
            entered.set()
            await release.wait()
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE],'review_provider':'test'},model)
        job=await engine.review_batches.prepare(SCOPE,items(p),'human:a')
        task=asyncio.create_task(engine.review_batches.run(job))
        await entered.wait()
        await engine.purge(SCOPE)
        release.set()
        await task
        assert engine.review_batches.recent()==[] and not store.proposals()
    asyncio.run(run())


@pytest.mark.parametrize('action',['reject','review'])
def test_new_batch_actions_reject_foreign_and_duplicate_records(store,action):
    async def run():
        p=make(store,'one',approved=False)
        foreign=make(store,'two',scope=OTHER,approved=False)
        engine=Engine(store,{'sessions':[SCOPE,OTHER]},None)
        fn=engine.group_batch_reject if action=='reject' else engine.review_batches.prepare
        for batch,actor in [(items(p,foreign),'human:a'),(items(p,p),'human:a'),(items(p),'llm:test')]:
            with pytest.raises(ValueError):
                await fn(SCOPE,batch,actor)
        assert not engine.review_batches.recent() and store.get(p['id'])['state']=='pending'
    asyncio.run(run())


def test_bad_model_output_is_a_failure_not_a_stale_record(store):
    async def run():
        p=make(store,'one',approved=False)
        async def model(*args):
            return {'unexpected':'output'}
        engine=Engine(store,{'sessions':[SCOPE],'review_provider':'test'},model)
        job=await engine.review_batches.prepare(SCOPE,items(p),'human:a')
        await engine.review_batches.run(job)
        assert job['items'][0]['status']=='failed'
        assert store.get(p['id'])['state']=='pending'
    asyncio.run(run())
