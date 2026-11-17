import asyncio
import time

import pytest

from astrbot_plugin_dynamics_learning.core.contract import candidate, encode
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.store import Store, Conflict
from test_social import SCOPE, CHARACTER, evidence, legacy, character_key


@pytest.fixture
def store(tmp_path):
    store=Store(tmp_path/'auto.sqlite3')
    yield store
    store.close()


def add_event(store, mid):
    event={**evidence(mid), 'received_at':time.time(), 'timestamp':time.time()}
    store.ingest(event)
    return event


def raw(event, delta=2):
    return dict(category='affection',key='bot_affection',member='alice',value={'experience_delta':delta,'tone':.1},
                reason='真实互动',evidence_ids=[event['id']],days=30)


def test_auto_positive_negative_no_review_model_and_rollback(store):
    async def run():
        legacy(store)
        store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
        before=store.entries()[0]['value']['affection_exp']
        delta=2
        calls=[]
        async def model(role,prompt,data):
            calls.append(role)
            assert role=='analysis_provider', 'automatic affection must not call review model'
            return {'proposals':[raw(data['events'][-1],delta)]}
        engine=Engine(store,{'sessions':[SCOPE],'affection_review':'auto'},model)
        add_event(store,'first')
        await engine.refresh()
        await engine.analyze(SCOPE,'fingerprint')
        first=store.entries()[0]['value']['affection_exp']
        assert float(first)>float(before)
        assert store.proposals()[0]['state']=='applied' and store.proposals()[0]['reviews']==[]
        delta=-2
        add_event(store,'second')
        await engine.analyze(SCOPE,'fingerprint')
        second=store.proposals()[0]
        assert float(store.entries()[0]['value']['affection_exp'])==pytest.approx(float(before))
        assert second['applied_by']=='system:auto-affection'
        assert calls==['analysis_provider','analysis_provider']
        await engine.action(second['id'],'rollback','human:admin',second['updated'])
        assert store.entries()[0]['value']['affection_exp']==first
        await engine.apply_pending_affection()
        assert store.entries()[0]['value']['affection_exp']==first
        await engine.analyze(SCOPE,'fingerprint')
        assert store.entries()[0]['value']['affection_exp']==first
        assert store.get(second['id'])['state']=='rolled_back'
    asyncio.run(run())


def test_pending_migration_checks_revision_and_preserves_rejected_history(store):
    async def run():
        first=add_event(store,'first')
        second=add_event(store,'second')
        third=add_event(store,'third')
        p=store.propose(candidate(raw(first),[first],None),'llm_then_human','old')
        stale=store.propose(candidate(raw(second,3),[second],None),'human','old')
        rejected=store.propose(candidate(raw(third,4),[third],None),'human','old')
        store.review(rejected['id'],{'verdict':'reject','reason':'人工已拒绝','risks':[]},'human:admin',rejected['updated'])
        engine=Engine(store,{'sessions':[SCOPE],'affection_review':'auto'},None)
        await engine.refresh()
        assert await engine.apply_pending_affection()=={'applied':1,'stale':1}
        assert store.get(p['id'])['previous_mode']=='llm_then_human'
        assert store.get(p['id'])['reviews']==[]
        assert store.get(stale['id'])['state']=='rejected'
        assert store.get(rejected['id'])['reviews'][0]['actor']=='human:admin'
        value=store.entries()[0]['value']
        assert await engine.apply_pending_affection()=={'applied':0,'stale':0}
        assert store.entries()[0]['value']==value
    asyncio.run(run())


def test_auto_cannot_be_used_to_bypass_other_categories(store):
    event=add_event(store,'one')
    data=candidate({**raw(event),'category':'style','value':'该成员喜欢简短表达'},[event],None)
    with pytest.raises(ValueError,match='only supported'):
        store.propose(data,'auto','test')
    p=store.propose(data,'llm_then_human','test')
    with pytest.raises(Conflict,match='only supported'):
        store.apply(p['id'],'system:auto-affection',p['updated'],automatic_affection=True)
    engine=Engine(store,{'style_review':'auto'},None)
    with pytest.raises(ValueError,match='invalid review mode'):
        engine.mode('style')


@pytest.mark.parametrize('setting',[{'affection_enabled':False},{'sessions':[]},{'affection_review':'llm_then_human'}])
def test_disabled_or_reviewed_modes_do_not_apply_pending(store,setting):
    async def run():
        e=add_event(store,'one')
        p=store.propose(candidate(raw(e),[e],None),'llm_then_human','old')
        engine=Engine(store,{'sessions':[SCOPE],'affection_review':'auto',**setting},None)
        await engine.refresh()
        assert await engine.apply_pending_affection()=={'applied':0,'stale':0}
        assert store.get(p['id'])['state']=='pending' and not store.entries()
    asyncio.run(run())


def test_changed_persona_is_not_automatically_applied(store):
    async def run():
        e=add_event(store,'one')
        p=store.propose(candidate(raw(e),[e],None),'auto','test')
        engine=Engine(store,{'sessions':[SCOPE],'affection_review':'auto'},None)
        await engine.refresh()
        engine.current_personas[SCOPE]='new-role'
        assert await engine.apply_pending_affection()=={'applied':0,'stale':1}
        assert store.get(p['id'])['state']=='rejected' and not store.entries()
    asyncio.run(run())


def test_first_entry_rollback_zeroes_value_and_allows_subsequent_apply(store):
    async def run():
        add_event(store,'first')
        calls=[]
        async def model(role,prompt,data):
            calls.append(role)
            return {'proposals':[raw(data['events'][-1],2)]}
        store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
        engine=Engine(store,{'sessions':[SCOPE],'affection_review':'auto'},model)
        await engine.refresh()
        await engine.analyze(SCOPE,'fingerprint')
        applied=store.proposals()[0]
        assert applied['state']=='applied'
        assert float(store.entries()[0]['value']['affection_exp'])==pytest.approx(2.0)
        await engine.action(applied['id'],'rollback','human:admin',applied['updated'])
        rolled=store.entries()[0]
        assert not rolled['active']
        assert float(rolled['value']['affection_exp'])==0.0
        add_event(store,'second')
        await engine.analyze(SCOPE,'fingerprint')
        latest=store.proposals()[0]
        assert latest['state']=='applied'
        assert float(store.entries()[0]['value']['affection_exp'])==pytest.approx(2.0)
        assert store.entries()[0]['active']
    asyncio.run(run())


def test_expired_affection_entry_stays_active_baseline_beyond_30_days(store):
    async def run():
        legacy(store)
        store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
        add_event(store,'day1')
        async def model(role,prompt,data):
            return {'proposals':[raw(data['events'][-1],2)]}
        engine=Engine(store,{'sessions':[SCOPE],'affection_review':'auto'},model)
        await engine.refresh()
        await engine.analyze(SCOPE,'fingerprint')
        entry=store.entries()[0]
        expired=dict(entry,expires=time.time()-3600)
        store.db.execute('UPDATE entries SET body=? WHERE id=?',(encode(expired),entry['id']))
        store.db.commit()
        before=float(entry['value']['affection_exp'])
        add_event(store,'after-30-days')
        seen_current=[]
        async def model2(role,prompt,data):
            seen_current.extend(data['current'])
            return {'proposals':[raw(data['events'][-1],2)]}
        engine.call_model=model2
        await engine.refresh()
        result=await engine.analyze(SCOPE,'fingerprint')
        assert any(e['id']==entry['id'] for e in seen_current)
        assert result['proposals']==1
        assert store.proposals()[0]['state']=='applied'
        assert float(store.entries()[0]['value']['affection_exp'])==pytest.approx(before+2.0)
    asyncio.run(run())


def test_same_batch_multiple_affection_proposals_converge_without_stale_reject(store):
    async def run():
        legacy(store)
        store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
        before=float(store.entries()[0]['value']['affection_exp'])
        e1=add_event(store,'m1')
        e2=add_event(store,'m2')
        async def model(role,prompt,data):
            return {'proposals':[raw(e1,2),raw(e2,3)]}
        engine=Engine(store,{'sessions':[SCOPE],'affection_review':'auto'},model)
        await engine.refresh()
        result=await engine.analyze(SCOPE,'fingerprint')
        assert result['proposals']==1
        assert all(p['state']=='applied' for p in store.proposals())
        assert float(store.entries()[0]['value']['affection_exp'])==pytest.approx(before+5.0)
    asyncio.run(run())
