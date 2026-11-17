import asyncio
import time

from astrbot_plugin_dynamics_learning.core.contract import candidate
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.native import group_projection
from astrbot_plugin_dynamics_learning.core.social import Social,character_key
from astrbot_plugin_dynamics_learning.core.store import Store
from test_social import SCOPE,CHARACTER,evidence,legacy


def test_only_approved_current_member_profile_is_exported_and_revoke_is_immediate(tmp_path):
    async def run():
        store=Store(tmp_path/'profile.sqlite3')
        e={**evidence(),'received_at':time.time()}
        store.ingest(e)
        raw=dict(category='style',key='brief',member='alice',value='该成员常用短句',reason='明确来源',evidence_ids=[e['id']],days=30)
        p=store.propose(candidate(raw,[e],None),'human','test')
        engine=Engine(store,{'sessions':[SCOPE]},None)
        await engine.refresh()
        args=dict(scope=SCOPE,persona='fingerprint',member='alice',baseline='config',host_version='host')
        assert not engine.context(**args)['reply_projection']['items']
        p=store.apply(p['id'],'human:admin',p['updated'])
        await engine.refresh()
        projection=engine.context(**args)['reply_projection']
        assert projection['member']=='alice' and projection['items'][0]['source_id']==p['id']
        assert projection['items'][0]['source_uids']==['alice']
        assert not engine.context(**{**args,'member':'bob'})['reply_projection']['items']
        store.rollback(p['id'],'human:admin')
        await engine.refresh()
        assert not engine.context(**args)['reply_projection']['items']
        store.close()
    asyncio.run(run())


def test_private_reply_projection_preserves_grants_and_does_not_write(tmp_path):
    async def run():
        store=Store(tmp_path/'private.sqlite3')
        legacy(store)
        store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
        store.ingest({**evidence(),'received_at':time.time()})
        engine=Engine(store,{'sessions':[SCOPE],'private_shared_group_sources':[SCOPE]},None)
        await engine.refresh()
        engine.social=Social(engine)
        private={**CHARACTER,'umo':'account:FriendMessage:alice','instance_id':'private'}
        args=dict(scope=private['umo'],persona='private-fingerprint',member='alice',baseline='base',host_version='host',character_scope=private)
        before=store.db.total_changes
        projection=engine.context(**args)['reply_projection']
        assert projection['private_readonly'] and projection['items'][0]['source_umo']==SCOPE
        assert 'affection_exp' not in projection['items'][0]['value']
        assert len(engine.context(**args)['context'])<=5000
        assert not engine.context(**{**args,'member':'bob'})['reply_projection']['items']
        assert not engine.context(**{**args,'character_scope':{**private,'persona_id':'other'}})['reply_projection']['items']
        engine.config['private_shared_group_sources']=[]
        assert not engine.context(**args)['reply_projection']['items']
        assert store.db.total_changes==before
        store.close()
    asyncio.run(run())


def test_native_relationship_identity_matches_current_relationship_contract(tmp_path):
    async def run():
        store=Store(tmp_path/'relationship.sqlite3')
        legacy(store)
        store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
        store.ingest({**evidence(),'received_at':time.time()})
        engine=Engine(store,{'sessions':[SCOPE]},None)
        await engine.refresh()
        engine.social=Social(engine)
        row=group_projection(engine,CHARACTER,['alice'])[0]
        assert row['source_id'].startswith('affection:narrative:v1:')
        assert row['subject_id']=='alice' and row['scope']==CHARACTER
        assert row['source_umo']==SCOPE
        store.close()
    asyncio.run(run())
