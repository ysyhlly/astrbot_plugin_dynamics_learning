import asyncio
import time
import uuid

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate,encode
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.core.workbench import govern,library,detail
from test_group_review import make,items,SCOPE,OTHER
from test_learning import event,raw


@pytest.fixture
def store(tmp_path):
    s=Store(tmp_path/'workbench.sqlite3')
    yield s
    s.close()


@pytest.mark.parametrize('action',['pause','cancel'])
def test_batch_hold_blocks_auto_worker_until_explicit_release(store,action):
    async def run():
        calls=[]
        async def model(*args):
            calls.append(1)
            return {'verdict':'approve','reason':'ok','risks':[]}
        p=make(store,'style',category='style',approved=False)
        engine=Engine(store,{'sessions':[SCOPE]},model)
        engine.current_personas[SCOPE]=p['persona']
        job=await engine.review_batches.prepare(SCOPE,items(p),'human:a')
        await engine.review_batches.control(job['id'],action,'human:a')
        await engine.apply_approved_profiles()
        await engine.review_pending_profiles()
        assert calls==[] and store.get(p['id'])['state']=='pending'
        assert (await engine.review_queue(view='held'))['total']==1
        if action=='pause':
            with pytest.raises(ValueError):
                govern(engine,SCOPE,[p],'release','human:a')
            await engine.review_batches.control(job['id'],'cancel','human:a')
        govern(engine,SCOPE,[p],'release','human:a')
        restored=Engine(store,{'sessions':[SCOPE]},model)
        restored.current_personas[SCOPE]=p['persona']
        assert not store.review_claim(p['id'])
        await restored.review_pending_profiles()
        assert calls==[1] and store.get(p['id'])['state']=='applied'
    asyncio.run(run())


def test_all_reviewed_categories_export_and_explain_exclusions(store):
    async def run():
        e=event()
        for c in ('jargon','relationship','state','decision_guidance'):
            data={**raw(),'category':c,'member':'' if c in {'jargon','decision_guidance'} else 'alice','key':'示例黑话' if c=='jargon' else c,'days':1}
            p=store.propose(candidate(data,[e],None),'human','fixture')
            store.apply(p['id'],'human:a',p['updated'])
        engine=Engine(store,{'sessions':[SCOPE]},None)
        await engine.refresh()
        before=store.db.total_changes
        args=dict(scope=SCOPE,persona=e['persona'],member='alice',baseline='base',host_version='host',query='示例黑话',trace=True)
        value=engine.context(**args)
        assert {r['category'] for r in value['reply_projection']['items']}=={'jargon','relationship','state','decision_guidance'}
        assert all(r['source_uids']==['alice'] and r['provenance_complete'] for r in value['reply_projection']['items'])
        no_keyword=engine.context(**{**args,'query':'你好'})
        assert next(r for r in no_keyword['candidates'] if r['category']=='jargon')['reason']
        assert store.db.total_changes==before
    asyncio.run(run())


def test_indexed_dedup_beyond_500_and_current_entry_provenance(store):
    p=make(store,'original')
    store.apply(p['id'],'human:a',p['updated'])
    with store.transaction():
        for i in range(550):
            row={**p,'id':uuid.uuid4().hex,'entry_id':str(i),'key':str(i),'created':time.time()+i}
            store.db.execute('INSERT INTO proposals VALUES(?,?,?,?,?,?,?)',(row['id'],row['scope'],row['persona'],row['state'],row['created'],row['updated'],encode(row)))
    assert make(store,'original',approved=False)['id']==p['id']
    assert store.entries()[0]['source_uids']==['alice']


def test_governance_archive_is_atomic_and_preserves_proofs(store):
    async def run():
        expired=make(store,'expired')
        ready=make(store,'ready','bob')
        with store.transaction():
            expired['created']=1
            store.db.execute('UPDATE proposals SET created=1 WHERE id=?',(expired['id'],))
            store._save(expired)
        engine=Engine(store,{'sessions':[SCOPE]},None)
        q=await engine.review_queue(view='stale',reason='expired')
        assert q['total']==1 and q['proposals'][0]['can_archive']
        with pytest.raises(ValueError):
            govern(engine,SCOPE,[expired,ready],'archive','human:a')
        assert store.get(expired['id'])['state']=='reviewed'
        proof=store.get(expired['id'])['evidence']
        assert govern(engine,SCOPE,[expired],'archive','human:a')['count']==1
        archived=store.get(expired['id'])
        assert archived['state']=='archived' and archived['evidence']==proof
        assert library(store,[SCOPE],state='archived')['total']==1
        assert (await engine.review_queue(view='all'))['total']==1
        with pytest.raises(ValueError):
            await engine.review(expired['id'])
    asyncio.run(run())


def test_version_center_rollback_latest_and_conflict_guards(store):
    first=make(store,'topic',value='first')
    store.apply(first['id'],'human:a',first['updated'])
    second=make(store,'topic',value='second')
    store.apply(second['id'],'human:a',second['updated'])
    assert not detail(store,[SCOPE],proposal_id=first['id'])['can_rollback']
    d=detail(store,[SCOPE],proposal_id=second['id'])
    assert d['can_rollback'] and len(d['versions'])==2 and d['timeline']
    assert d['proposal']['evidence']
    with pytest.raises(ValueError):
        store.rollback(first['id'],'human:a')
    store.rollback(second['id'],'human:a')
    assert detail(store,[SCOPE],proposal_id=first['id'])['can_rollback']
    assert library(store,[SCOPE],query='topic',state='rolled_back')['total']==1
    with pytest.raises(ValueError):
        detail(store,[OTHER],proposal_id=first['id'])


def test_queue_has_bounded_queries_and_sql_pagination(store,monkeypatch):
    async def run():
        p=make(store,'one',approved=False)
        with store.transaction():
            for i in range(1100):
                row={**p,'id':str(i),'entry_id':str(i),'key':str(i)}
                store.db.execute('INSERT INTO proposals VALUES(?,?,?,?,?,?,?)',(row['id'],row['scope'],row['persona'],row['state'],row['created'],row['updated'],encode(row)))
        engine=Engine(store,{'sessions':[SCOPE]},None)
        def forbidden(*args,**kwargs):
            raise AssertionError('per-row Python classification must not run')
        monkeypatch.setattr(engine,'can_llm_review',forbidden)
        monkeypatch.setattr(engine,'manual_review_status',forbidden)
        data=await engine.review_queue(view='all',page=22)
        assert data['total']==1101 and len(data['proposals'])==50
        assert all(r['can_llm_review'] for r in data['proposals'])
    asyncio.run(run())


def test_already_running_auto_review_cannot_race_into_new_batch(store):
    async def run():
        p=make(store,'style',category='style',approved=False)
        entered=asyncio.Event()
        release=asyncio.Event()
        async def model(*args):
            entered.set()
            await release.wait()
            return {'verdict':'approve','reason':'ok','risks':[]}
        engine=Engine(store,{'sessions':[SCOPE]},model)
        task=asyncio.create_task(engine.review(p['id']))
        await entered.wait()
        current=store.get(p['id'])
        with pytest.raises(ValueError,match='正在进行'):
            await engine.review_batches.prepare(SCOPE,items(current),'human:a')
        # A reservation appearing during a provider call must still invalidate the old writer.
        async with engine.lock:
            job=engine.review_batches.create(SCOPE,[current],'human:a')
        await engine.review_batches.control(job['id'],'pause','human:a')
        release.set()
        with pytest.raises(ValueError):
            await task
        assert store.get(p['id'])['state']=='pending' and not store.entries()
    asyncio.run(run())


def test_version_center_tracks_nonprofile_usage_without_prompt_text(store):
    p=make(store,'关系')
    store.apply(p['id'],'human:a',p['updated'])
    store.console.record_usage({'schema_version':1,'scope':SCOPE,'stage':'provider_kwargs_after_boundaries',
        'member':'alice','turn_id':'turn','material_hash':'hash','profiles':[],
        'learning':[{'source_id':p['id'],'category':'relationship','revision':1,'source_umo':SCOPE,'value':'不得保存的提示词'}]},[SCOPE])
    d=detail(store,[SCOPE],proposal_id=p['id'])
    assert len(d['usage'])==1
    assert '不得保存' not in store.db.execute('SELECT body FROM reply_usage').fetchone()[0]
    store.purge(SCOPE)
    assert store.db.execute('SELECT count(*) FROM reply_usage_refs').fetchone()[0]==0
