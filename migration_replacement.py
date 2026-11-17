"""Final read-only SelfLearning handoff; history does not become approved facts."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .core.affection import state
from .core.contract import digest, encode, entry_key
from .core.social import character_key, group_id
from .core.store import Store
from .migration_social import connect


ARCHIVE_TABLES = {'expression_patterns':'expression','style_learning_reviews':'style_review','user_social_profiles':'profile',
    'conversation_goals':'legacy_goal','bot_mood':'legacy_mood','jargon':'jargon','user_affections':'affection',
    'persona_update_reviews':'legacy_persona_review','exemplar':'example'}


def plan(source, sessions):
    mapping={}
    for scope in sessions:
        gid=group_id(scope)
        if not gid or gid in mapping:
            raise ValueError('legacy group scope is ambiguous')
        mapping[gid]=scope
    scopes={}
    archive=[]
    records=[]
    with connect(source) as db:
        tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts={t:db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in tables if t.isidentifier()}
        if 'narrative_observations' in tables:
            for row in db.execute('SELECT scope_key,envelope FROM narrative_observations GROUP BY scope_key'):
                scope=json.loads(row['envelope']).get('scope',{})
                if scope.get('umo') in sessions and character_key(scope)==row['scope_key']:
                    mapping[row['scope_key']]=scope['umo']
                    scopes[row['scope_key']]=scope
        marks=','.join('?' for _ in mapping)
        for table,category in ARCHIVE_TABLES.items():
            if table not in tables:
                continue
            field='chat_id' if table=='jargon' else 'group_id'
            for row in db.execute('SELECT * FROM '+table+' WHERE '+field+' IN ('+marks+')',tuple(mapping)):
                value=dict(row)
                value.pop('embedding_json',None)  # Full source database remains in the backup.
                source_scope=str(row[field])
                scope=mapping[source_scope]
                archive.append(dict(id=digest([table,str(row['id'])]),scope=scope,category=category,table=table,
                    source_id=str(row['id']),body={'source_scope':source_scope,'original':value,
                    'status':'historical_only','note':'旧资料原文与原审查状态；未经新来源校验，不自动作为当前成员事实或角色设定。'}))
                if table=='user_affections' and source_scope!=str(row['user_id']):
                    records.append(dict(scope=scope,source_key=source_scope,category='affection',member=str(row['user_id']),
                        key='bot_affection',value=state(value),provenance={'plugin':'astrbot_plugin_self_learning','table':table,'row_id':row['id'],'source_scope':source_scope,'cutover':True}))
                elif table=='jargon' and row['is_jargon'] and not row['is_global'] and (row['forget_stage'] or 0)<3:
                    term=str(row['content'] or '').strip()
                    meaning=str(row['meaning'] or '').strip()
                    if term and len(term)<=64 and meaning:
                        records.append(dict(scope=scope,source_key=source_scope,category='jargon',member='',key=term,
                            value=meaning[:1000],provenance={'plugin':'astrbot_plugin_self_learning','table':table,'row_id':row['id'],'source_scope':source_scope,'cutover':True}))
    fingerprint=digest({'archive':archive,'records':records,'counts':counts,'scopes':scopes})
    return dict(fingerprint=fingerprint,archive=archive,records=records,scopes=scopes,counts=counts)


def migrate(source, destination, sessions, expected):
    if Path(source).resolve()==Path(destination).resolve():
        raise ValueError('source and destination must differ')
    data=plan(source,sessions)
    if data['fingerprint']!=expected:
        raise ValueError('source changed after preview')
    store=Store(destination)
    try:
        with store.lock,store.db:
            if store.db.execute('SELECT 1 FROM replacement_imports WHERE id=?',(expected,)).fetchone():
                return {'duplicate':True,'fingerprint':expected}
            for row in data['archive']:
                store.db.execute('INSERT OR REPLACE INTO legacy_archive VALUES(?,?,?,?,?,?)',
                    (row['id'],row['scope'],row['category'],row['table'],row['source_id'],encode(row['body'])))
            for row in data['records']:
                store.db.execute('INSERT OR REPLACE INTO legacy_social VALUES(?,?,?,?,?,?)',
                    (row['scope'],row['source_key'],row['category'],row['member'],row['key'],
                     encode({'value':row['value'],'provenance':row['provenance']})))
        store.native.bootstrap_scopes()
        bindings=store.native.scopes()
        active_personas=store.latest_personas()
        updated=0
        preserved=0
        # Bind missing members first, then refresh only untouched legacy values.
        for (scope,persona),native in bindings.items():
            if active_personas.get(scope)!=persona or scope not in sessions:
                continue
            identity=character_key(native)
            if not identity:
                continue
            store.bind_legacy_social(scope,persona,identity)
            sources=[r for r in data['records'] if r['scope']==scope]
            sources.sort(key=lambda r: r['source_key']==identity,reverse=True)
            seen=set()
            with store.lock,store.db:
                owner=store.db.execute('SELECT legacy_key FROM legacy_group_owners WHERE scope=?',(scope,)).fetchone()
                for row in sources:
                    if row['category']=='affection' and row['source_key']!=identity and not (row['source_key']==group_id(scope) and owner and owner[0]==identity):
                        continue
                    key=(row['category'],row['member'],row['key'])
                    if key in seen:
                        continue
                    seen.add(key)
                    eid=entry_key(scope,persona,row['member'],row['category'],row['key'])
                    old=store.db.execute('SELECT body FROM entries WHERE id=?',(eid,)).fetchone()
                    if not old:
                        continue
                    entry=json.loads(old[0])
                    owned=store.db.execute("SELECT 1 FROM proposals WHERE scope=? AND json_extract(body,'$.category')=? AND json_extract(body,'$.member')=? AND json_extract(body,'$.key')=? AND state IN ('applied','rolled_back') LIMIT 1",(scope,row['category'],row['member'],row['key'])).fetchone()
                    if owned or not entry.get('imported') or not entry['proposal_id'].startswith('legacy:'):
                        preserved+=1
                        continue
                    value=dict(row['value'],character_key=identity) if row['category']=='affection' else row['value']
                    entry.update(value=value,provenance=row['provenance'],revision=entry['revision']+1,state_updated_at=time.time())
                    entry['proposal_id']='legacy:'+digest([expected,eid])
                    store.db.execute('UPDATE entries SET body=? WHERE id=?',(encode(entry),eid))
                    store._audit(entry,'legacy_cutover_refresh','system:migration',{'source_table':row['provenance']['table'],'source_id':row['provenance']['row_id']})
                    updated+=1
        report={'fingerprint':expected,'source_counts':data['counts'],'history_records':len(data['archive']),
                'history_by_category':{c:sum(r['category']==c for r in data['archive']) for c in sorted(set(ARCHIVE_TABLES.values()))},
                'legacy_social_records':len(data['records']),'untouched_legacy_refreshed':updated,'native_updates_preserved':preserved,
                'imported_at':time.time(),'unknown_expression_auto_applied':0}
        with store.lock,store.db:
            store.db.execute('INSERT INTO replacement_imports VALUES(?,?,?)',(expected,time.time(),encode(report)))
        assert store.db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        return report
    finally:
        store.close()


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
