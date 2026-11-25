"""Native social ownership and read-only projections; no SelfLearning imports."""
from __future__ import annotations

import json
import time

from .contract import PERSONALITY_POLICY, digest, encode
from .social import character_key


def instances(context):
    getter = getattr(context, 'get_all_stars', None)
    rows = getter() if callable(getter) else []
    rows = rows.items() if isinstance(rows, dict) else (('', row) for row in rows or [])
    for key, row in rows:
        get = row.get if isinstance(row, dict) else lambda k, default=None: getattr(row, k, default)
        plugin = get('star_cls') or get('star') or get('plugin')
        if get('activated', True) and plugin is not None and not isinstance(plugin, type):
            yield str(get('name') or get('root_dir_name') or key), plugin


def role_provider(context, role, legacy_name=''):
    found = []
    for name, plugin in instances(context):
        getter = getattr(plugin, 'get_dynamics_capabilities', None)
        manifest = getter() if callable(getter) else {}
        declared = (isinstance(manifest, dict) and type(manifest.get('schema_version')) is int and manifest.get('schema_version') == 1
                    and manifest.get('enabled') is True and manifest.get('ready') is True and isinstance(manifest.get('capabilities'),dict)
                    and type(manifest['capabilities'].get(role)) is int and manifest['capabilities'].get(role) == 1)
        if (declared or name == legacy_name) and all(p is not plugin for p in found):
            found.append(plugin)
    return found[0] if len(found) == 1 else None


def replacement_allowed(plugin):
    return bool(plugin.config.get('replace_selflearning', False) and plugin.config.get('enabled', True) and
                not plugin.closed and not any(name == 'astrbot_plugin_self_learning' for name, _ in instances(plugin.context)))


class NativeStore:
    def __init__(self, store):
        self.store = store
        store.db.executescript('''
        CREATE TABLE IF NOT EXISTS native_scope_registry(scope TEXT,persona TEXT,body TEXT,PRIMARY KEY(scope,persona));
        CREATE TABLE IF NOT EXISTS native_receipts(event_id TEXT PRIMARY KEY,scope TEXT,created REAL,fingerprint TEXT,body TEXT);
        CREATE INDEX IF NOT EXISTS native_receipts_scope ON native_receipts(scope,created);
        CREATE INDEX IF NOT EXISTS native_receipts_created ON native_receipts(created);
        CREATE TABLE IF NOT EXISTS legacy_archive(id TEXT PRIMARY KEY,scope TEXT,category TEXT,source_table TEXT,source_id TEXT,body TEXT);
        CREATE INDEX IF NOT EXISTS legacy_archive_scope ON legacy_archive(scope,category);
        CREATE TABLE IF NOT EXISTS replacement_imports(id TEXT PRIMARY KEY,created REAL,body TEXT);
        ''')

    def register_scope(self, event):
        scope = event.get('context', {}).get('character_scope')
        if event.get('kind') == 'observed' and character_key(scope) and scope['umo'] == event['scope']:
            self.store.db.execute('INSERT OR REPLACE INTO native_scope_registry VALUES(?,?,?)',
                                 (event['scope'], event['persona'], encode(scope)))

    def scopes(self):
        with self.store.lock:
            return {(r[0], r[1]):json.loads(r[2]) for r in self.store.db.execute('SELECT scope,persona,body FROM native_scope_registry')}

    def bootstrap_scopes(self):
        with self.store.lock, self.store.db:
            for scope, persona in self.store.latest_personas().items():
                for e in reversed(self.store.events(scope, persona, 30)):
                    if e['kind'] == 'observed':
                        self.register_scope(e)
                        break

    def observe(self, events, allowed):
        accepted = 0
        with self.store.lock, self.store.db:
            for e in events:
                if not isinstance(e, dict) or e.get('schema_version') != 1 or e.get('type') not in {'message.received','delivery.success'}:
                    continue
                scope = e.get('scope', {})
                key = character_key(scope)
                if not key or scope['umo'] not in allowed:
                    continue  # Private input never writes group learning state.
                event_id = e.get('event_id')
                if not isinstance(event_id, str) or not 1 <= len(event_id) <= 256 or len(encode(e)) > 64000:
                    raise ValueError('invalid native event')
                fingerprint = digest(e)
                old = self.store.db.execute('SELECT fingerprint FROM native_receipts WHERE event_id=?',(event_id,)).fetchone()
                if old:
                    if old[0] != fingerprint:
                        raise ValueError('native event identity changed')
                    continue
                self.store.db.execute('INSERT INTO native_receipts VALUES(?,?,?,?,?)',
                    (event_id,scope['umo'],time.time(),fingerprint,''))
                accepted += 1
        return accepted

    def history(self, scope, category='', page=1):
        page = max(1, min(10000, int(page)))
        with self.store.lock:
            count = self.store.db.execute("SELECT count(*) FROM legacy_archive WHERE scope=? AND (?='' OR category=?)",(scope,category,category)).fetchone()[0]
            rows = self.store.db.execute("SELECT id,category,source_table,source_id,body FROM legacy_archive WHERE scope=? AND (?='' OR category=?) ORDER BY rowid DESC LIMIT 20 OFFSET ?",(scope,category,category,(page-1)*20)).fetchall()
            return {'total':count,'page':page,'page_size':20,'records':[dict(id=r[0],category=r[1],source_table=r[2],source_id=r[3],record=json.loads(r[4])) for r in rows]}


def current_persona(engine, scope):
    umo = scope.get('umo', '') if isinstance(scope, dict) else ''
    persona = engine.current_personas.get(umo)
    expected = character_key(scope)
    if not persona or not expected or not engine.allowed(umo):
        return None
    actual = engine.character_scopes.get((umo,persona))
    return persona if character_key(actual) == expected else None


def private_projection(engine, scope, member):
    """Read only this member from explicitly shared groups of the same character."""
    umo = scope.get('umo', '')
    parts = umo.rsplit(':',2)
    if len(parts) != 3 or parts[1] not in {'FriendMessage','PrivateMessage'} or parts[0] != scope.get('platform') or not scope.get('persona_id'):
        return []
    items = []
    for source in dict.fromkeys(engine.config.get('private_shared_group_sources', [])):
        source_parts = source.rsplit(':',2)
        if len(source_parts) != 3 or source_parts[0] != parts[0] or source_parts[1] != 'GroupMessage' or not engine.allowed(source):
            continue
        persona = engine.current_personas.get(source)
        origin = engine.character_scopes.get((source,persona), {})
        if origin.get('persona_id') != scope['persona_id']:
            continue
        affection = engine.social.current_affection(source,persona,member) if engine.social else None
        profile = engine.member_profile(source,member)
        data = {'source_umo':source,'member':member,'note':'授权群资料，只读、按来源分开，不合并好感数值，不代表私聊发生过这些互动。'}
        if affection:
            data['affection'] = affection['value']['affection_level']
            data['distance_requested'] = affection['value'].get('perceived_affection_distance') is True
        if profile['available']:
            data['observations'] = [row for row in profile['observations'] if row['category']!='expression_example'][:4]
        if 'affection' not in data and 'observations' not in data:
            continue
        text = PERSONALITY_POLICY+'\n'+encode(data)
        if sum(len(i['text']) for i in items)+len(text) > 4500:
            continue
        items.append({'text':text,'scope':dict(scope),'subject_id':member,'source_umo':source,
            'source_id':'shared-social:'+digest([source,persona,member,data]),'fact_kind':'subjective','unknown':True})
    return items


def group_projection(engine, scope, participants):
    persona = current_persona(engine, scope)
    if not persona:
        return []
    items=[]
    for member in dict.fromkeys(str(m) for m in participants[:8]):
        entry = engine.social.current_affection(scope['umo'],persona,member)
        if not entry:
            continue
        value=entry['value']
        content = {'member':member,'affection':round(value['affection_level'],4),'reference':100,'ceiling':None,
            'distance_requested':value.get('perceived_affection_distance') is True,
            'note':'角色对该成员的亲近倾向；不代表对方感受、信任或亲密许可。保持当前原始性格。'}
        items.append({'text':encode(content),'scope':dict(scope),'subject_id':member,'source_umo':scope['umo'],
            'source_id':'affection:'+character_key(scope)+':'+member+':'+str(entry['revision']),
            'occurred_at':entry.get('state_updated_at'),'fact_kind':'subjective','unknown':True})
    return items
