"""Read-only source migration. Preview first; import only explicitly selected full group scopes."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import time
from pathlib import Path

from .core.contract import encode
from .core.social import character_key, group_id
from .core.affection import state
from .core.store import Store
from .vendor.group_relations.store import RelationStore


@contextmanager
def connect(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def inspect_sources(learning_source, relation_source, sessions):
    sessions = sorted(set(sessions))
    by_group = {}
    for scope in sessions:
        gid = group_id(scope)
        if not gid:
            raise ValueError('migration accepts full group UMO only')
        if gid in by_group:
            raise ValueError('legacy bare group id is ambiguous between platform accounts')
        by_group[gid] = scope
    mapping = dict(by_group)
    records = []
    with connect(learning_source) as db:
        keys = {r[0] for r in db.execute("SELECT DISTINCT group_id FROM user_affections WHERE group_id LIKE 'narrative:v1:%'")}
        keys.update(r[0] for r in db.execute("SELECT DISTINCT chat_id FROM jargon WHERE chat_id LIKE 'narrative:v1:%'"))
        for key in sorted(keys):
            row = db.execute('SELECT envelope FROM narrative_observations WHERE scope_key=? ORDER BY timestamp DESC LIMIT 1',(key,)).fetchone()
            if not row:
                continue
            scope = json.loads(row[0]).get('scope',{})
            if scope.get('umo') in sessions and character_key(scope) == key:
                mapping[key] = scope['umo']
        placeholders = ','.join('?' for _ in mapping)
        if not placeholders:
            raise ValueError('no migration scopes')
        for row in db.execute('SELECT * FROM user_affections WHERE group_id IN ('+placeholders+')',tuple(mapping)):
            if row['group_id'] == row['user_id']:
                continue  # Legacy private-chat pseudo-groups are not group evidence.
            value = state(dict(row))
            records.append(dict(scope=mapping[row['group_id']], source_key=row['group_id'], category='affection',
                member=row['user_id'], key='bot_affection', value=value,
                provenance={'plugin':'astrbot_plugin_self_learning','table':'user_affections','row_id':row['id'],
                            'source_scope':row['group_id'],'updated_at':row['updated_at']}))
        for row in db.execute('SELECT * FROM jargon WHERE chat_id IN ('+placeholders+') AND is_jargon=1 AND coalesce(is_global,0)=0 AND coalesce(forget_stage,0)<3',tuple(mapping)):
            term, meaning = str(row['content'] or '').strip(), str(row['meaning'] or '').strip()
            if not term or len(term)>64 or not meaning:
                continue
            records.append(dict(scope=mapping[row['chat_id']], source_key=row['chat_id'], category='jargon',member='',
                key=term,value=meaning[:1000],provenance={'plugin':'astrbot_plugin_self_learning','table':'jargon',
                'row_id':row['id'],'source_scope':row['chat_id'],'count':row['count'],'updated_at':row['updated_at']}))
    graph = {}
    with connect(relation_source) as db:
        for table in ('users','messages','relations','relation_daily','member_daily','directed_events'):
            graph[table] = []
            for gid, scope in sorted(by_group.items()):
                for row in db.execute('SELECT * FROM '+table+' WHERE group_id=?',(gid,)):
                    record = dict(row)
                    record['group_id'] = scope
                    graph[table].append(record)
        graph['meta'] = [dict(r) for r in db.execute("SELECT * FROM meta WHERE key IN ('network_v2_started','last_daily_decay')")]
    fingerprint = hashlib.sha256(encode({'records':records,'graph':graph}).encode()).hexdigest()
    return {'fingerprint':fingerprint,'scopes':sessions,'records':records,'graph':graph}


def summary(preview):
    return {'fingerprint':preview['fingerprint'],'scopes':len(preview['scopes']),
            'affection_records':sum(r['category']=='affection' for r in preview['records']),
            'confirmed_jargon':sum(r['category']=='jargon' for r in preview['records']),
            'graph_rows':{k:len(v) for k,v in preview['graph'].items()}}


def migrate(learning_source, relation_source, target, sessions, expected_fingerprint):
    preview = inspect_sources(learning_source, relation_source, sessions)
    if preview['fingerprint'] != expected_fingerprint:
        raise ValueError('migration source changed after preview')
    target = Path(target)
    if target.resolve() in {Path(learning_source).resolve(),Path(relation_source).resolve()}:
        raise ValueError('source and destination must differ')
    store = Store(target)
    RelationStore(target)
    try:
        with store.lock, store.db:
            old = store.db.execute('SELECT body FROM social_imports WHERE import_key=?',(preview['fingerprint'],)).fetchone()
            if old:
                return {**json.loads(old[0]),'duplicate':True}
            # Import once into empty scope statistics. Never add a historical graph twice.
            for scope in preview['scopes']:
                if store.db.execute('SELECT 1 FROM users WHERE group_id=? LIMIT 1',(scope,)).fetchone():
                    raise ValueError('target graph scope is not empty; incremental source merge requires review')
            for record in preview['records']:
                store.db.execute('INSERT OR IGNORE INTO legacy_social VALUES(?,?,?,?,?,?)',
                    (record['scope'],record['source_key'],record['category'],record['member'],record['key'],
                     encode({'value':record['value'],'provenance':record['provenance']})))
            for table, rows in preview['graph'].items():
                for row in rows:
                    # messages/directed_events have independent autoincrement identities.
                    if table in ('messages','directed_events'):
                        row.pop('id',None)
                    columns = list(row)
                    statement = 'INSERT OR REPLACE' if table == 'meta' else 'INSERT'
                    store.db.execute(statement+' INTO '+table+'('+','.join(columns)+') VALUES('+','.join('?' for _ in columns)+')',
                                     tuple(row[k] for k in columns))
            result = summary(preview)
            result['imported_at'] = time.time()
            store.db.execute('INSERT INTO social_imports VALUES(?,?,?)',(preview['fingerprint'],time.time(),encode(result)))
            assert store.db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            return result
    finally:
        store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-learning',required=True)
    parser.add_argument('--relations',required=True)
    parser.add_argument('--target',required=True)
    parser.add_argument('--settings',required=True)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--expected-fingerprint')
    args = parser.parse_args()
    scopes = json.loads(Path(args.settings).read_text(encoding='utf-8-sig'))['sessions']
    if args.apply:
        result = migrate(args.self_learning,args.relations,args.target,scopes,args.expected_fingerprint)
    else:
        result = summary(inspect_sources(args.self_learning,args.relations,scopes))
    print(json.dumps(result,ensure_ascii=False))
