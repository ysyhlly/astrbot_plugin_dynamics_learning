import asyncio
import copy
import threading

import pytest

from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.workbench import govern,detail
from test_group_review import make,SCOPE


@pytest.fixture
def store(tmp_path):
    s=Store(tmp_path/'integrity.sqlite3')
    yield s
    s.close()


def test_cancelled_rollback_drains_commit_and_removes_revoked_cache(store,monkeypatch):
    async def run():
        p=make(store,'one')
        p=store.apply(p['id'],'human:a',p['updated'])
        engine=Engine(store,{'sessions':[SCOPE]},None)
        await engine.refresh()
        entered=threading.Event()
        release=threading.Event()
        finished=threading.Event()
        rollback=store.rollback
        def delayed(*args,**kwargs):
            result=rollback(*args,**kwargs)
            entered.set()
            assert release.wait(5)
            finished.set()
            return result
        monkeypatch.setattr(store,'rollback',delayed)
        task=asyncio.create_task(engine.action(p['id'],'rollback','human:a',p['updated']))
        try:
            assert await asyncio.to_thread(entered.wait,3)
            task.cancel()
            await asyncio.sleep(.02)
            drained_before_return=not task.done()
        finally:
            release.set()
        await asyncio.gather(task,return_exceptions=True)
        assert await asyncio.to_thread(finished.wait,3)
        assert store.get(p['id'])['state']=='rolled_back'
        assert not any(e['active'] and e['proposal_id']==p['id'] for e in engine.cache)
        assert drained_before_return
    asyncio.run(run())


def test_archive_failure_rolls_back_proofs_entries_claims_and_history(store,monkeypatch):
    async def run():
        rows=[make(store,str(i),approved=False) for i in range(2)]
        engine=Engine(store,{'sessions':[SCOPE]},None)
        job=await engine.review_batches.prepare(SCOPE,[{'id':p['id'],'updated':p['updated']} for p in rows],'human:a')
        await engine.review_batches.control(job['id'],'cancel','human:a')
        engine.current_personas[SCOPE]='new-persona'
        original=copy.deepcopy(rows)
        audit=store._audit
        def fail(p,action,*args,**kwargs):
            if p['id']==rows[1]['id'] and action=='archive':
                raise OSError('simulated disk error')
            return audit(p,action,*args,**kwargs)
        monkeypatch.setattr(store,'_audit',fail)
        with pytest.raises(OSError):
            govern(engine,SCOPE,rows,'archive','human:a')
        assert [store.get(p['id']) for p in rows]==original
        assert all(store.review_claim(p['id'])==job['id'] for p in rows)
        monkeypatch.setattr(store,'_audit',audit)
        govern(engine,SCOPE,rows,'archive','human:a')
        for p in rows:
            store.ingest(p['evidence'][0])
        with store.transaction():
            store.db.execute('UPDATE events SET created=1')
            store.db.execute('UPDATE analysis_inbox SET processed=1')
        store.prune(1)
        assert store.db.execute('SELECT count(*) FROM events').fetchone()[0]==0
        for p in rows:
            actual=detail(store,[SCOPE],proposal_id=p['id'])
            assert actual['proposal']['state']=='archived'
            assert actual['proposal']['evidence']==p['evidence'] and actual['proposal']['reviews']==p['reviews']
            assert any(a['action']=='archive' for a in actual['timeline'])
        assert not store.entries()
    asyncio.run(run())


def test_rollback_failure_cannot_erase_current_or_prior_versions(store,monkeypatch):
    first=make(store,'item',value='first')
    store.apply(first['id'],'human:a',first['updated'])
    second=make(store,'item',value='second')
    second=store.apply(second['id'],'human:a',second['updated'])
    before=copy.deepcopy(store.entries())
    original=store._audit
    def fail(p,action,*args,**kwargs):
        if action=='rolled_back':
            raise OSError('audit failure')
        return original(p,action,*args,**kwargs)
    monkeypatch.setattr(store,'_audit',fail)
    with pytest.raises(OSError):
        store.rollback(second['id'],'human:a')
    assert store.entries()==before and store.get(second['id'])['state']=='applied'
    monkeypatch.setattr(store,'_audit',original)
    store.rollback(second['id'],'human:a')
    assert store.entries()[0]['value']=='first'
    assert store.get(second['id'])['value']=='second'
    assert len(detail(store,[SCOPE],proposal_id=second['id'])['versions'])==2


def test_rollback_refresh_failure_revokes_old_projection(store,monkeypatch):
    async def run():
        p=make(store,'one')
        p=store.apply(p['id'],'human:a',p['updated'])
        engine=Engine(store,{'sessions':[SCOPE]},None)
        await engine.refresh()
        async def broken():
            raise OSError('read unavailable')
        monkeypatch.setattr(engine,'refresh',broken)
        result=await engine.action(p['id'],'rollback','human:a',p['updated'])
        assert result['warning'] and result['state']=='rolled_back' and engine.cache_refresh_pending
        assert not any(e['active'] and e['proposal_id']==p['id'] for e in engine.cache)
        assert store.get(p['id'])['value']==p['value']
        assert store.get(p['id'])['state']=='rolled_back'
    asyncio.run(run())


def test_rollback_checks_stamp_inside_transaction(store):
    p=make(store,'one')
    p=store.apply(p['id'],'human:a',p['updated'])
    expected=p['updated']
    with store.transaction():
        p['note']='concurrent update'
        store._save(p)
    before=copy.deepcopy(store.entries())
    with pytest.raises(ValueError,match='changed'):
        store.rollback(p['id'],'human:a',expected)
    assert store.entries()==before and store.get(p['id'])['state']=='applied'


def test_archiving_pending_replacement_preserves_active_value_after_reopen(store):
    active=make(store,'item',value='current active value')
    store.apply(active['id'],'human:a',active['updated'])
    pending=make(store,'item',value='pending replacement')
    with store.transaction():
        pending['created']=1
        store.db.execute('UPDATE proposals SET created=1 WHERE id=?',(pending['id'],))
        store._save(pending)
    engine=Engine(store,{'sessions':[SCOPE]},None)
    before=copy.deepcopy(store.entries())
    govern(engine,SCOPE,[pending],'archive','human:a')
    assert store.entries()==before
    reopened=Store(store.path)
    try:
        d=detail(reopened,[SCOPE],proposal_id=pending['id'])
        assert d['entry']['value']=='current active value'
        assert d['proposal']['value']=='pending replacement'
        assert d['proposal']['before']['value']=='current active value'
        assert d['proposal']['evidence']==pending['evidence']
        assert len(d['versions'])==2
    finally:
        reopened.close()
