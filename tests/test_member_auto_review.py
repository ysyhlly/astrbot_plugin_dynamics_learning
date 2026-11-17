import asyncio
import time

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate,event_record,MEMBER_PROFILES
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store
from test_learning import event,raw


@pytest.fixture
def store(tmp_path):
    value=Store(tmp_path/'members.sqlite3')
    yield value
    value.close()


def proposal(store,key='style',mode='llm_then_human',approved=True):
    e=event()
    data={k:v for k,v in e.items() if k not in {'id','received_at'}}
    data['kind']='observed'
    e=event_record(data)
    store.ingest(e)
    p=store.propose(candidate({**raw(),'key':key,'evidence_ids':[e['id']]},[e],None),mode,'test')
    if approved:
        p=store.review(p['id'],{'verdict':'approve','reason':'明确来源','risks':[]},'llm:review',p['updated'])
    return p


def test_existing_llm_approved_profile_applies_without_new_review_or_human(store):
    async def run():
        p=proposal(store)
        engine=Engine(store,{'sessions':[p['scope']]},None)
        await engine.refresh()
        result=await engine.apply_approved_profiles()
        assert result=={'applied':1,'stale':0,'awaiting_llm':0}
        applied=store.get(p['id'])
        assert applied['state']=='applied' and applied['mode']=='llm'
        assert applied['applied_by']=='system:llm-approved-profile'
        assert applied['reviews']==p['reviews']
        assert all(engine.mode(c)=='llm' for c in MEMBER_PROFILES)
        assert engine.mode('relationship')=='llm_then_human' and engine.mode('affection')=='auto'
        assert (await engine.apply_approved_profiles())['applied']==0
        await engine.action(applied['id'],'rollback','human:admin',applied['updated'])
        await engine.apply_approved_profiles()
        assert not store.entries()[0]['active']
    asyncio.run(run())


@pytest.mark.parametrize('verdict',['approve','defer','reject'])
def test_old_pending_profile_follows_llm_verdict_without_bypass(store,verdict):
    async def run():
        p=proposal(store,approved=False)
        async def model(role,*args):
            assert role=='review_provider'
            return {'verdict':verdict,'reason':'独立审查','risks':[]}
        engine=Engine(store,{'sessions':[p['scope']]},model)
        await engine.refresh()
        pending=await engine.apply_approved_profiles()
        assert pending['awaiting_llm']==1 and not store.entries()
        result=await engine.review(p['id'])
        assert result['state']=={'approve':'applied','defer':'pending','reject':'rejected'}[verdict]
        assert bool(store.entries())==(verdict=='approve')
    asyncio.run(run())


@pytest.mark.parametrize('change',['persona','expired','revision'])
def test_stale_approved_member_data_does_not_apply(store,change):
    async def run():
        p=proposal(store)
        engine=Engine(store,{'sessions':[p['scope']]},None)
        await engine.refresh()
        if change=='persona':
            engine.current_personas[p['scope']]='other'
        elif change=='expired':
            with store.db:
                p['created']=time.time()-100*86400
                store._save(p)
        else:
            e=store.events(p['scope'],p['persona'])[-1]
            other=store.propose(candidate({**raw(),'key':'style','value':'该成员喜欢详细解释','evidence_ids':[e['id']]},[e],None),'human','test')
            store.apply(other['id'],'human:admin',other['updated'])
            await engine.refresh()
        assert (await engine.apply_approved_profiles())['stale']==1
        assert store.get(p['id'])['state']=='rejected'
    asyncio.run(run())


def test_explicit_human_policy_and_rejected_records_are_preserved(store):
    async def run():
        p=proposal(store)
        rejected=proposal(store,key='rejected',approved=False)
        store.review(rejected['id'],{'verdict':'reject','reason':'人工拒绝','risks':[]},'human:admin',rejected['updated'])
        engine=Engine(store,{'sessions':[p['scope']],'style_review':'llm_then_human'},None)
        await engine.refresh()
        assert (await engine.apply_approved_profiles())['applied']==0
        assert store.get(p['id'])['state']=='reviewed'
        assert store.get(rejected['id'])['state']=='rejected'
        assert not store.entries()
    asyncio.run(run())
