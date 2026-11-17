import asyncio
import copy
import threading
import time

import pytest

from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from test_group_review import SCOPE, make, items


@pytest.fixture
def store(tmp_path):
    s=Store(tmp_path/'audit.sqlite3')
    yield s
    s.close()


def test_committed_result_survives_cancel_before_runner_receipt(store,monkeypatch):
    async def run():
        p=make(store,'one',approved=False,category='style')
        calls=[]
        async def model(*args):
            calls.append(1)
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(p),'human:a')
        review=engine.review
        async def crash_after_commit(*args,**kwargs):
            await review(*args,**kwargs)
            raise asyncio.CancelledError()
        monkeypatch.setattr(engine,'review',crash_after_commit)
        await asyncio.gather(asyncio.create_task(manager.run(job)),return_exceptions=True)
        assert store.get(p['id'])['state']=='applied'
        assert manager.get(job['id'])['items'][0]['status']=='applied'
        assert calls==[1]
        assert manager.get(job['id'])['status']=='completed'
    asyncio.run(run())


def test_interrupted_mode_adoption_does_not_lose_unstarted_item(store,monkeypatch):
    async def run():
        p=make(store,'profile',approved=False,category='style')
        calls=[]
        async def model(*args):
            calls.append(1)
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(p),'human:a')
        started=manager.started
        def crash(*args):
            raise asyncio.CancelledError()
        monkeypatch.setattr(manager,'started',crash)
        await asyncio.gather(asyncio.create_task(manager.run(job)),return_exceptions=True)
        assert calls==[]
        monkeypatch.setattr(manager,'started',started)
        resumed=await manager.control(job['id'],'resume','human:a')
        await manager.run(resumed)
        assert calls==[1] and store.get(p['id'])['state']=='applied'
    asyncio.run(run())


def test_refresh_failure_cannot_relabel_applied_result_as_failed(store,monkeypatch):
    async def run():
        p=make(store,'profile',approved=False,category='style')
        async def model(*args):
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        async def broken_refresh():
            raise RuntimeError('refresh failed')
        monkeypatch.setattr(engine,'refresh',broken_refresh)
        job=await engine.review_batches.prepare(SCOPE,items(p),'human:a')
        await engine.review_batches.run(job)
        assert store.get(p['id'])['state']=='applied'
        saved=engine.review_batches.get(job['id'])
        assert saved['items'][0]['status']=='applied'
        assert saved['items'][0]['error']
    asyncio.run(run())


def test_retry_ancestor_cannot_fork_duplicate_failed_attempts(store):
    async def run():
        p=make(store,'one',approved=False)
        async def model(*args):
            raise TimeoutError()
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        parent=await manager.prepare(SCOPE,items(p),'human:a')
        await manager.run(parent)
        child=await manager.control(parent['id'],'retry','human:a')
        await manager.run(child)
        before=copy.deepcopy(manager.get(parent['id'])['items'])
        with pytest.raises(ValueError,match='最新'):
            await manager.control(parent['id'],'retry','human:a')
        assert manager.get(parent['id'])['items']==before
    asyncio.run(run())


def test_cancel_waits_for_database_worker_and_durable_receipt(store,monkeypatch):
    async def run():
        p=make(store,'one',approved=False)
        entered=threading.Event()
        release=threading.Event()
        async def model(*args):
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(p),'human:a')
        review=store.review
        def slow_commit(*args,**kwargs):
            result=review(*args,**kwargs)
            entered.set()
            assert release.wait(5)
            return result
        monkeypatch.setattr(store,'review',slow_commit)
        task=asyncio.create_task(manager.run(job))
        try:
            assert await asyncio.to_thread(entered.wait,3)
            task.cancel()
            await asyncio.sleep(.01)
            task.cancel()  # Repeated cancellation must still drain the writer.
            await asyncio.sleep(.01)
            assert not task.done() and engine.lock.locked()
        finally:
            release.set()
        await asyncio.gather(task,return_exceptions=True)
        assert store.get(p['id'])['state']=='reviewed'
        assert manager.get(job['id'])['items'][0]['status']=='reviewed'
        assert manager.get(job['id'])['status']=='completed'
    asyncio.run(run())


def test_result_and_receipt_rollback_together_on_storage_failure(store,monkeypatch):
    async def run():
        p=make(store,'one',approved=False)
        async def model(*args):
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(p),'human:a')
        def fail_receipt(*args):
            raise OSError('simulated write failure')
        monkeypatch.setattr(manager,'committed',fail_receipt)
        await manager.run(job)
        assert store.get(p['id'])['reviews']==[]
        assert store.get(p['id'])['state']=='pending'
        assert manager.get(job['id'])['items'][0]['status']=='failed'
    asyncio.run(run())


def test_apply_failure_keeps_review_and_rolls_back_partial_entry(store,monkeypatch):
    async def run():
        p=make(store,'profile',approved=False,category='style')
        async def model(*args):
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        original=store.apply
        def fail_after_write(*args,**kwargs):
            original(*args,**kwargs)
            raise OSError('simulated apply failure')
        monkeypatch.setattr(store,'apply',fail_after_write)
        job=await engine.review_batches.prepare(SCOPE,items(p),'human:a')
        await engine.review_batches.run(job)
        row=engine.review_batches.get(job['id'])['items'][0]
        assert store.get(p['id'])['state']=='reviewed' and not store.entries()
        assert row['status']=='reviewed' and row['committed'] and row['auto_apply_pending'] and row['error']
    asyncio.run(run())


def test_completion_save_failure_does_not_strand_scope(store,monkeypatch):
    async def run():
        p=make(store,'one',approved=False)
        async def model(*args):
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(p),'human:a')
        original=manager.save
        failed=False
        def fail_once(row):
            nonlocal failed
            if row['status']=='completed' and not failed:
                failed=True
                raise OSError('one-shot failure')
            return original(row)
        monkeypatch.setattr(manager,'save',fail_once)
        with pytest.raises(OSError):
            await manager.run(job)
        assert manager.get(job['id'])['status']=='completed'
        manager.ensure_available(SCOPE)
    asyncio.run(run())


def test_retention_preserves_recoverable_and_recently_completed_old_jobs(store):
    async def run():
        p=make(store,'one',approved=False)
        async def model(*args):
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        manager=engine.review_batches
        job=await manager.prepare(SCOPE,items(p),'human:a')
        job['created']=1
        manager.save(job)
        with store.transaction():
            store.db.execute('UPDATE review_batches SET created=1')
        manager.interrupt(job)
        store.console.prune(1)
        assert manager.get(job['id'])['status']=='interrupted'
        resumed=await manager.control(job['id'],'resume','human:a')
        await manager.run(resumed)
        store.console.prune(1)
        assert manager.get(job['id'])['status']=='completed'
        assert manager.get(job['id'])['updated']>time.time()-60
    asyncio.run(run())
