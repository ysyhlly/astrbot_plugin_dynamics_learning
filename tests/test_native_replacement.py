import asyncio
import sqlite3
import time
from types import SimpleNamespace as NS

import pytest

from astrbot_plugin_dynamics_learning.core.contract import encode
from astrbot_plugin_dynamics_learning.core.engine import Engine
from astrbot_plugin_dynamics_learning.core.native import private_projection, replacement_allowed, role_provider
from astrbot_plugin_dynamics_learning.core.social import Social, character_key
from astrbot_plugin_dynamics_learning.core.store import Store
from astrbot_plugin_dynamics_learning.migration_replacement import plan, migrate, file_hash
from test_social import SCOPE, CHARACTER, evidence, legacy, affection_candidate


def test_native_receipt_dedup_scope_filter_and_conflict(tmp_path):
    store=Store(tmp_path/'native.sqlite3')
    event={'schema_version':1,'event_id':'received-1','type':'message.received','scope':CHARACTER,'actor_id':'alice','text':'原话'}
    assert store.native.observe([event],[SCOPE])==1
    assert store.native.observe([event],[SCOPE])==0
    with pytest.raises(ValueError,match='identity changed'):
        store.native.observe([{**event,'text':'篡改内容'}],[SCOPE])
    private={**event,'event_id':'private-1','scope':{**CHARACTER,'umo':'account:FriendMessage:alice'}}
    assert store.native.observe([private],['account:FriendMessage:alice'])==0
    assert store.native.observe([{**event,'event_id':'other-1'}],[])==0
    assert not store.entries()
    store.purge(SCOPE)
    assert store.db.execute('SELECT count(*) FROM native_receipts').fetchone()[0]==0
    store.close()


def test_replacement_and_role_selection_do_not_accept_duplicate_owners():
    legacy_plugin=NS(name='astrbot_plugin_self_learning',activated=True,star_cls=object())
    rows=[legacy_plugin]
    plugin=NS(config={'replace_selflearning':True},closed=False,context=NS(get_all_stars=lambda:rows))
    assert not replacement_allowed(plugin)
    legacy_plugin.activated=False
    assert replacement_allowed(plugin)
    world=NS(get_dynamics_capabilities=lambda:{'schema_version':1,'enabled':True,'ready':True,'capabilities':{'character':1}})
    rows.append(NS(name='native_world',activated=True,star_cls=world))
    assert role_provider(plugin.context,'character') is world
    rows.append(NS(name='other_world',activated=True,star_cls=NS(get_dynamics_capabilities=world.get_dynamics_capabilities)))
    assert role_provider(plugin.context,'character') is None


def test_private_group_projection_is_member_role_platform_scoped_and_read_only(tmp_path):
    async def run():
        store=Store(tmp_path/'private.sqlite3')
        legacy(store)
        store.bind_legacy_social(SCOPE,'fingerprint',character_key(CHARACTER))
        store.ingest({**evidence(),'received_at':time.time()})
        engine=Engine(store,{'sessions':[SCOPE],'private_shared_group_sources':[SCOPE]},None)
        await engine.refresh()
        engine.social=Social(engine)
        scope={'platform':'account','umo':'account:FriendMessage:alice','persona_id':'persona','instance_id':'private-instance'}
        changes=store.db.total_changes
        rows=private_projection(engine,scope,'alice')
        assert len(rows)==1 and rows[0]['source_umo']==SCOPE
        assert '125' in rows[0]['text'] and '不合并' in rows[0]['text']
        assert not private_projection(engine,scope,'bob')
        assert not private_projection(engine,{**scope,'persona_id':'other'},'alice')
        assert not private_projection(engine,{**scope,'platform':'foreign','umo':'foreign:FriendMessage:alice'},'alice')
        engine.config['private_shared_group_sources']=[]
        assert not private_projection(engine,scope,'alice')
        assert store.db.total_changes==changes
        store.close()
    asyncio.run(run())


def test_final_migration_preserves_native_updates_and_quarantines_unattributed_style(tmp_path):
    source=tmp_path/'old.sqlite3'
    target=tmp_path/'new.sqlite3'
    with sqlite3.connect(source) as db:
        db.executescript('''
        CREATE TABLE user_affections(id INTEGER PRIMARY KEY,group_id TEXT,user_id TEXT,affection_level REAL);
        CREATE TABLE jargon(id INTEGER PRIMARY KEY,chat_id TEXT,content TEXT,meaning TEXT,is_jargon INTEGER,is_global INTEGER,forget_stage INTEGER);
        CREATE TABLE expression_patterns(id INTEGER PRIMARY KEY,group_id TEXT,persona_id TEXT,user_id TEXT,situation TEXT,expression TEXT);
        CREATE TABLE style_learning_reviews(id INTEGER PRIMARY KEY,group_id TEXT,status TEXT);
        CREATE TABLE narrative_observations(scope_key TEXT,envelope TEXT);
        ''')
        key=character_key(CHARACTER)
        db.execute('INSERT INTO narrative_observations VALUES(?,?)',(key,encode({'scope':CHARACTER})))
        db.executemany('INSERT INTO user_affections VALUES(?,?,?,?)',[(1,key,'alice',180),(2,key,'bob',90),(3,'999','stranger',77)])
        db.execute('INSERT INTO jargon VALUES(1,?,?,?,?,?,?)',('123','群语','已确认含义',1,0,0))
        db.execute('INSERT INTO expression_patterns VALUES(1,?,?,?,?,?)',('123','persona',None,'情境','原始表达'))
        db.execute("INSERT INTO style_learning_reviews VALUES(1,'123','approved')")
    original=file_hash(source)
    store=Store(target)
    legacy(store)
    legacy(store,member='bob')
    store.bind_legacy_social(SCOPE,'fingerprint',key)
    store.ingest({**evidence(),'received_at':time.time()})
    p=affection_candidate(store)
    store.apply(p['id'],'human:admin',p['updated'])
    expected=next(e for e in store.entries() if e['member']=='alice')['value']
    store.close()
    preview=plan(source,[SCOPE])
    report=migrate(source,target,[SCOPE],preview['fingerprint'])
    assert report['native_updates_preserved']==1 and report['unknown_expression_auto_applied']==0
    assert report['history_by_category']['expression']==1
    assert file_hash(source)==original
    assert migrate(source,target,[SCOPE],preview['fingerprint'])['duplicate']
    store=Store(target)
    assert next(e for e in store.entries() if e['member']=='alice')['value']==expected
    assert next(e for e in store.entries() if e['member']=='bob')['value']['affection_level']==pytest.approx(90)
    assert not [e for e in store.entries() if e['category']=='expression_example']
    archive=store.native.history(SCOPE,'expression')
    assert archive['records'][0]['record']['original']['user_id'] is None
    assert archive['records'][0]['record']['status']=='historical_only'
    assert not store.native.history('foreign:GroupMessage:123')['records']
    store.close()
