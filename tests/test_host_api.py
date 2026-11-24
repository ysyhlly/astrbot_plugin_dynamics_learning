"""Run under the real AstrBot SDK in an isolated process, never against a live plugin instance."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("astrbot.api.star")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from astrbot_plugin_dynamics_learning.main import DynamicsLearning, StarTools


class Context:
    def __init__(self):
        self.registered_web_apis = []
        self.provider = None

    def register_web_api(self, *args):
        self.registered_web_apis.append(args)

    def get_config(self):
        return {"dashboard": {"username": "admin"}}

    def get_provider_by_id(self, pid):
        return self.provider


def test_sdk_lifecycle_and_scoped_unregister(tmp_path, monkeypatch):
    async def run():
        monkeypatch.setattr(StarTools, "get_data_dir", lambda *args: tmp_path)
        context = Context()
        plugin = DynamicsLearning(context, {"sessions": []})
        await plugin.initialize()
        assert len(context.registered_web_apis) == 31
        assert {row[0].split("/series_workspace/")[1] for row in context.registered_web_apis
                if "/series_workspace/" in row[0]} == {"open/<target>", "panel/<target>/<path:operation>"}
        assert plugin.learning_capabilities()["ready"]
        assert not plugin.ingest_dynamics_event({"scope": "unknown"})
        def replacement():
            return None
        context.registered_web_apis.append(("/replacement", replacement, ["GET"], ""))
        await plugin.terminate()
        assert len(context.registered_web_apis) == 1
        assert not plugin.learning_capabilities()["ready"]
    asyncio.run(run())


def test_unconfigured_model_and_wrong_provider_never_fall_back():
    async def run():
        ctx = Context()
        plugin = DynamicsLearning(ctx, {})
        with pytest.raises(ValueError):
            await plugin._model("review_provider", "instruction", {})
        plugin.config["review_provider"] = "decision"
        ctx.provider = SimpleNamespace(is_native_decision_provider=True, text_chat=lambda: None)
        with pytest.raises(ValueError):
            await plugin._model("review_provider", "instruction", {})
    asyncio.run(run())


def test_model_reload_invalidates_response():
    async def run():
        ctx = Context()
        plugin = DynamicsLearning(ctx, {"review_provider": "model"})
        async def generate(**kwargs):
            ctx.provider = object()
            return SimpleNamespace(completion_text='{}')
        ctx.provider = SimpleNamespace(text_chat=generate)
        with pytest.raises(ValueError, match="失效"):
            await plugin._model("review_provider", "instruction", {})
    asyncio.run(run())


@pytest.mark.parametrize('endpoint',['overview','review_queue','group_batch_action','group_batch_review','group_batch_reject','review_batch_jobs','review_batch_control','reply_preview','library','material_detail','governance_action','archive_stale'])
def test_api_rejects_unauthenticated_requests(monkeypatch,endpoint):
    async def run():
        plugin = DynamicsLearning(Context(), {})
        async def unauthorized():
            raise PermissionError()
        monkeypatch.setattr(plugin, "_authorized", unauthorized)
        response = await getattr(plugin,endpoint)()
        assert response.status_code == 403
        assert json.loads(response.body)["status"] == "error"
    asyncio.run(run())


def test_group_llm_route_returns_queued_and_exposes_background_progress(tmp_path,monkeypatch):
    from astrbot_plugin_dynamics_learning.core.store import Store
    from astrbot_plugin_dynamics_learning.core.engine import Engine
    from test_group_review import make,items,SCOPE
    async def run():
        store=Store(tmp_path/'route.sqlite3')
        entered=asyncio.Event()
        release=asyncio.Event()
        async def model(*args):
            entered.set()
            await release.wait()
            return {'verdict':'approve','reason':'ok','risks':[]}
        plugin=DynamicsLearning(Context(),{'sessions':[SCOPE],'review_provider':'test'})
        plugin.engine=Engine(store,plugin.config,model)
        p=make(store,'one',approved=False)
        body={'scope':SCOPE,'items':items(p)}
        async def serve(operation,fields=()):
            return await operation(body,'human:admin')
        monkeypatch.setattr(plugin,'_serve',serve)
        result=await plugin.group_batch_review()
        assert result['queued'] and result['count']==1
        await entered.wait()
        with pytest.raises(ValueError,match='已有任务'):
            await plugin.group_batch_review()
        body={'scope':SCOPE}
        progress=await plugin.review_batch_jobs()
        assert progress['jobs'][0]['status']=='running'
        release.set()
        await asyncio.gather(*plugin.tasks)
        assert not plugin.jobs
        assert (await plugin.review_batch_jobs())['jobs'][0]['status']=='completed'
        store.close()
    asyncio.run(run())


def test_job_control_route_pause_resume_schedules_only_remaining_items(tmp_path,monkeypatch):
    from astrbot_plugin_dynamics_learning.core.store import Store
    from astrbot_plugin_dynamics_learning.core.engine import Engine
    from test_group_review import make,items,SCOPE
    async def run():
        store=Store(tmp_path/'control-route.sqlite3')
        entered=asyncio.Event()
        release=asyncio.Event()
        calls=[]
        async def model(role,instruction,data):
            calls.append(data['proposal']['key'])
            if len(calls)==1:
                entered.set()
                await release.wait()
            return {'verdict':'approve','reason':'ok','risks':[]}
        plugin=DynamicsLearning(Context(),{'sessions':[SCOPE],'review_provider':'test'})
        plugin.engine=Engine(store,plugin.config,model)
        rows=[make(store,key,approved=False) for key in ('one','two')]
        body={'scope':SCOPE,'items':items(*rows)}
        async def serve(operation,fields=()):
            return await operation(body,'human:admin')
        monkeypatch.setattr(plugin,'_serve',serve)
        result=await plugin.group_batch_review()
        await entered.wait()
        body={'id':result['id'],'action':'pause'}
        assert (await plugin.review_batch_control())['status']=='pausing'
        release.set()
        await asyncio.gather(*plugin.tasks)
        assert plugin.engine.review_batches.get(result['id'])['status']=='paused'
        body={'id':result['id'],'action':'resume'}
        assert (await plugin.review_batch_control())['status']=='queued'
        with pytest.raises(ValueError):
            await plugin.review_batch_control()
        await asyncio.gather(*plugin.tasks)
        assert calls==['one','two']
        assert plugin.engine.review_batches.get(result['id'])['status']=='completed'
        store.close()
    asyncio.run(run())


def test_preview_uninitialized_group_returns_explained_unavailable(tmp_path,monkeypatch):
    from astrbot_plugin_dynamics_learning.core.store import Store
    from astrbot_plugin_dynamics_learning.core.engine import Engine
    async def run():
        scope='account:GroupMessage:uninitialized'
        plugin=DynamicsLearning(Context(),{'sessions':[scope]})
        store=Store(tmp_path/'preview-empty.sqlite3')
        plugin.engine=Engine(store,plugin.config,None)
        async def serve(operation,fields=()):
            return await operation({'scope':scope,'member':'alice','query':'你好'},'human:admin')
        monkeypatch.setattr(plugin,'_serve',serve)
        result=await plugin.reply_preview()
        assert not result['available'] and result['items']==[] and result['reason']
        assert not result['model_called'] and not result['message_sent']
        store.close()
    asyncio.run(run())


def test_preview_route_awaits_async_current_character_verification(tmp_path,monkeypatch):
    from astrbot_plugin_dynamics_learning.core.store import Store
    from astrbot_plugin_dynamics_learning.core.engine import Engine
    async def run():
        scope='account:GroupMessage:known'
        context=Context()
        calls=[]
        async def preview(*args):
            await asyncio.sleep(0)
            calls.append(args)
            return {'available':True,'character_verified':True,'items':[],'model_called':False,'message_sent':False}
        context.get_all_stars=lambda:[SimpleNamespace(name='cd',activated=True,star_cls=SimpleNamespace(preview_learning_material=preview))]
        plugin=DynamicsLearning(context,{'sessions':[scope]})
        store=Store(tmp_path/'preview-async.sqlite3')
        plugin.engine=Engine(store,plugin.config,None)
        plugin.engine.current_personas[scope]='fingerprint'
        async def serve(operation,fields=()):
            return await operation({'scope':scope,'member':'alice','query':'你好'},'human:admin')
        monkeypatch.setattr(plugin,'_serve',serve)
        result=await plugin.reply_preview()
        assert result['character_verified'] and calls==[(scope,'fingerprint','alice','你好')]
        store.close()
    asyncio.run(run())


def test_native_private_projection_hook_and_affection_permissions(tmp_path,monkeypatch):
    from test_social import SCOPE, CHARACTER, evidence, legacy, character_key
    from astrbot.core.provider.entities import ProviderRequest
    async def run():
        monkeypatch.setattr(StarTools,'get_data_dir',lambda *args:tmp_path)
        ctx=Context()
        rows=[]
        ctx.get_all_stars=lambda:rows
        plugin=DynamicsLearning(ctx,{'sessions':[SCOPE],'replace_selflearning':True,'private_shared_group_sources':[SCOPE]})
        await plugin.initialize()
        store=plugin.engine.store
        legacy(store)
        store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
        store.ingest({**evidence(),'received_at':1})
        await plugin.engine.refresh()
        private={**CHARACTER,'umo':'account:FriendMessage:alice','instance_id':'private'}
        async def character(scope,event):
            return {'available':True,'scope':private if 'FriendMessage' in scope['umo'] else CHARACTER}
        world=SimpleNamespace(get_character_context=character,get_dynamics_capabilities=lambda:{'schema_version':1,'enabled':True,'ready':True,'capabilities':{'character':1}})
        rows.append(SimpleNamespace(name='world',activated=True,star_cls=world))
        extra={}
        event=SimpleNamespace(unified_msg_origin=private['umo'],get_sender_id=lambda:'alice',get_message_str=lambda:'你好',
                              get_extra=lambda k,default=None:extra.get(k,default),set_extra=lambda k,v:extra.update({k:v}),is_admin=lambda:False)
        req=ProviderRequest(prompt='你好')
        before=store.db.total_changes
        await plugin.inject_social_context(event,req)
        assert len(req.extra_user_content_parts)==1 and SCOPE in req.extra_user_content_parts[0].text
        assert store.db.total_changes==before
        await plugin.inject_social_context(event,req)
        assert len(req.extra_user_content_parts)==1
        assert not json.loads(await plugin.query_affection(event,user_id='bob'))['available']
        assert not json.loads(await plugin.query_affection(event,mode='ranking'))['available']
        assert json.loads(await plugin.query_affection(event))['available']
        extra.clear()
        extra['_autonomous_blocking_context_v1']={'projection_restricted':True}
        req2=ProviderRequest(prompt='你好')
        await plugin.inject_social_context(event,req2)
        assert not req2.extra_user_content_parts
        extra.clear()
        extra['_narrative_managed_v1']=True
        await plugin.inject_social_context(event,req2)
        assert not req2.extra_user_content_parts  # Native narrative projection already owns this request.
        extra.clear()
        event._chat_dynamics_owned_request=True
        await plugin.inject_social_context(event,req2)
        assert not req2.extra_user_content_parts
        assert event._dynamics_learning_private_scope_v1['owner'] is plugin
        assert event._dynamics_learning_private_scope_v1['event'] is event
        rows.append(SimpleNamespace(name='astrbot_plugin_self_learning',activated=True,star_cls=object()))
        assert not plugin.get_dynamics_capabilities()['enabled']
        assert not (await plugin.observe_events([]))['available']
        await plugin.terminate()
    asyncio.run(run())
