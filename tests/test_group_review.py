import asyncio
from types import SimpleNamespace

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from test_learning import event, raw

SCOPE='qq:GroupMessage:1'
OTHER='qq:GroupMessage:2'


@pytest.fixture
def store(tmp_path):
    value=Store(tmp_path/'group-review.sqlite3')
    yield value
    value.close()


def make(store,key,member='alice',scope=SCOPE,approved=True,persona='persona-a',category='relationship',value=None):
    e=event()
    e.update(scope=scope,member=member,persona=persona)
    data={**raw(),'category':category,'key':key,'member':member,'value':value or '经常一起讨论 '+key}
    p=store.propose(candidate(data,[e],None),'llm_then_human','test')
    if approved:
        p=store.review(p['id'],{'verdict':'approve','reason':'已核对','risks':[]},'llm:test',p['updated'])
    return p


def items(*rows):
    return [{'id':p['id'],'updated':p['updated']} for p in rows]


def test_global_search_filters_before_pagination_and_never_reads_disabled_scopes(store):
    async def run():
        for i in range(60):
            make(store,'唯一词条' if i==55 else f'词条{i}',member='bob' if i==55 else 'alice',approved=i%2==1)
        make(store,'另一个群',scope=OTHER)
        make(store,'禁用群',scope='qq:GroupMessage:disabled')
        rejected=make(store,'已拒绝')
        store.review(rejected['id'],{'verdict':'reject','reason':'no','risks':[]},'human:a',rejected['updated'])
        engine=Engine(store,{'sessions':[SCOPE,OTHER]},None)
        engine.social=SimpleNamespace(groups={SCOPE:{'users':{'bob':{'name':'小明'}}}})
        all_rows=await engine.review_queue(view='all')
        assert all_rows['total']==61 and len(all_rows['proposals'])==50
        second=await engine.review_queue(view='all',page=2)
        assert len(second['proposals'])==11
        result=await engine.review_queue(query='唯一词条',view='all')
        assert result['total']==1 and result['proposals'][0]['member']=='bob'
        assert result['proposals'][0]['evidence']==[] and result['proposals'][0]['evidence_count']==1
        assert (await engine.review_queue(query='小明',view='all'))['total']==1
        assert (await engine.review_queue(scope=SCOPE,state='pending',view='all'))['total']==30
        assert (await engine.review_queue(category='preference',view='all'))['total']==0
        assert (await engine.review_queue(view='ready'))['total']==31
        assert (await engine.review_queue(query="%' OR 1=1 --",view='all'))['total']==0
        clamped=await engine.review_queue(view='all',page=999)
        assert clamped['page']==2 and len(clamped['proposals'])==11
        with pytest.raises(ValueError):
            await engine.review_queue(scope='qq:GroupMessage:disabled')
        with pytest.raises(ValueError):
            await engine.review_queue(state='applied')
    asyncio.run(run())


def test_group_batch_crosses_members_but_preserves_identity_and_individual_rollback(store):
    async def run():
        first=make(store,'同名词条','alice')
        second=make(store,'同名词条','bob',persona='persona-b')
        engine=Engine(store,{'sessions':[SCOPE]},None)
        result=await engine.group_batch_action(SCOPE,items(first,second),'human:admin')
        assert result['count']==2
        assert {(e['member'],e['persona']) for e in store.entries()}=={('alice','persona-a'),('bob','persona-b')}
        p=store.get(first['id'])
        await engine.action(p['id'],'rollback','human:admin',p['updated'])
        assert [e['member'] for e in store.entries() if e['active']]==['bob']
    asyncio.run(run())


@pytest.mark.parametrize('problem',['foreign','stale','unapproved','duplicate','old_persona','auto_profile'])
def test_invalid_group_batch_applies_nothing(store,problem):
    async def run():
        first=make(store,'one')
        second=make(store,'two','bob',scope=OTHER if problem=='foreign' else SCOPE,
                    approved=problem!='unapproved',category='style' if problem=='auto_profile' else 'relationship')
        engine=Engine(store,{'sessions':[SCOPE,OTHER]},None)
        batch=items(first,second)
        if problem=='stale':
            batch[1]['updated']=0
        if problem=='duplicate':
            batch=items(first,first)
        if problem=='old_persona':
            engine.current_personas[SCOPE]='new-persona'
        with pytest.raises(ValueError):
            await engine.group_batch_action(SCOPE,batch,'human:admin')
        assert not store.entries()
        assert store.get(first['id'])['state']=='reviewed'
    asyncio.run(run())


def test_stale_entry_revision_rolls_back_entire_group_transaction(store):
    async def run():
        first=make(store,'one')
        second=make(store,'two','bob')
        competing=make(store,'two','bob',value='最近减少了往来')
        # Force an external revision after the proposal was prepared.
        store.apply(competing['id'],'human:admin',competing['updated'])
        engine=Engine(store,{'sessions':[SCOPE]},None)
        with pytest.raises(ValueError):
            await engine.group_batch_action(SCOPE,items(first,second),'human:admin')
        assert store.get(first['id'])['state']=='reviewed'
        assert len(store.entries())==1
    asyncio.run(run())


def test_group_batch_accepts_more_than_twenty_and_caps_explicit_selection(store):
    async def run():
        rows=[make(store,str(i),member='member'+str(i)) for i in range(21)]
        engine=Engine(store,{'sessions':[SCOPE]},None)
        with pytest.raises(ValueError):
            await engine.group_batch_action(SCOPE,items(*rows)*5,'human:admin')
        with pytest.raises(ValueError):
            await engine.group_batch_action(SCOPE,items(*rows),'llm:test')
        assert not store.entries()
        result=await engine.group_batch_action(SCOPE,items(*rows),'human:admin')
        assert result['count']==21 and len(store.entries())==21
    asyncio.run(run())
