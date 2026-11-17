import asyncio
import json
import time

import pytest

from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.social import Social
from astrbot_plugin_dynamics_learning.core.store import Store
from test_learning import proposal


@pytest.fixture
def store(tmp_path):
    s=Store(tmp_path/'console.sqlite3')
    yield s
    s.close()


def test_overview_is_summary_only_and_details_are_lazy(store):
    async def run():
        p=proposal(store)
        with store.db:
            p['evidence'][0]['text']='原文'*4000
            store._save(p)
        engine=Engine(store,{'sessions':[p['scope']]},None)
        data=await engine.overview()
        assert data['entries']==[] and data['entries_paged']
        assert data['proposals'][0]['evidence']==[] and data['proposals'][0]['evidence_count']==1
        assert '原文'*100 not in json.dumps(data,ensure_ascii=False)
        assert store.get(p['id'])['evidence'][0]['text']=='原文'*4000
    asyncio.run(run())


def test_compact_graph_search_preserves_global_metrics_without_refresh_wait(store):
    async def run():
        engine=Engine(store,{'sessions':['a:GroupMessage:1']},None)
        social=Social(engine)
        scope='a:GroupMessage:1'
        users={str(i):{'name':'成员'+str(i),'count':100-i,'large_unused':'x'*2000} for i in range(80)}
        edges=[{'a':str(i),'b':str(j),'score':50,'directed_hits':5,'a_name':users[str(i)]['name'],'b_name':users[str(j)]['name'],
                'raw_history':'x'*10000} for i in range(80) for j in range(i+1,80)]
        social.groups[scope]={'users':users,'all_relations':edges,'active_member_count':80,'relation_count':len(edges)}
        social.refreshed[scope]=time.monotonic()
        social.updated_at[scope]=time.time()
        data=await social.overview(scope,40)
        assert len(data['graph']['users'])==40 and data['graph']['total_members']==80
        assert data['graph']['relation_count']==3160
        assert len(data['graph']['all_relations'])<=300
        assert len(json.dumps(data).encode())<150000
        assert social.members(scope,'成员79')['members'][0]['id']=='79'
        focused=await social.overview(scope,20,'79')
        assert focused['graph']['focus']=='79' and '79' in focused['graph']['users']
        assert social.member_detail(scope,'79')['relations']
        with pytest.raises(ValueError):
            social.members('other:GroupMessage:1')
    asyncio.run(run())


def test_graph_refresh_coalesces_and_read_returns_cached_snapshot(store):
    async def run():
        engine=Engine(store,{'sessions':['a:GroupMessage:1']},None)
        social=Social(engine)
        release=asyncio.Event()
        calls=[]
        async def slow(*args,**kwargs):
            calls.append(1)
            await release.wait()
            return {'users':{},'all_relations':[]}
        social.graph.overview=slow
        first=await asyncio.wait_for(social.overview('a:GroupMessage:1'),.5)
        await asyncio.wait_for(social.overview('a:GroupMessage:1'),.5)
        assert first['refreshing'] and not first['available']
        await asyncio.sleep(.01)
        assert len(calls)==1
        release.set()
        await asyncio.gather(*list(social.refresh_tasks.values()))
        assert social.updated_at
    asyncio.run(run())


def test_tasks_report_failure_duration_and_later_success(store):
    first=store.console.task('group','analysis_provider')
    store.console.task_started(first)
    store.console.task_finished(first,'timeout','TimeoutError',time.time()+60)
    second=store.console.task('group','analysis_provider')
    store.console.task_started(second)
    store.console.task_finished(second,'succeeded')
    records=store.console.tasks()
    assert records[0]['status']=='succeeded'
    assert records[1]['error']=='TimeoutError' and records[1]['recovered']
    assert records[1]['elapsed_ms']>=0


def test_usage_requires_provider_proof_is_deduplicated_and_scope_isolated(store):
    raw={'schema_version':1,'scope':'a:GroupMessage:1','member':'alice','turn_id':'t','material_hash':'hash',
         'stage':'provider_kwargs_after_boundaries','profiles':[{'source_id':'p','revision':1,'category':'style','text':'不应保存'}],
         'memory_refs':['atom:1'],'omitted':[{'source_id':'p2','reason':'boundary_or_identity_filter'}],
         'rhythm':{'action':'reply','prompt':'不应保存'},'personality_locked':True}
    store.console.record_usage(raw,[raw['scope']])
    store.console.record_usage(raw,[raw['scope']])
    rows=store.console.usage(raw['scope'])
    assert rows['total']==1 and '不应保存' not in json.dumps(rows,ensure_ascii=False)
    assert not store.console.usage('other:GroupMessage:1')['records']
    with pytest.raises(ValueError):
        store.console.record_usage({**raw,'stage':'prepared_only'},[raw['scope']])
    store.purge(raw['scope'])
    assert not store.console.usage(raw['scope'])['records']
