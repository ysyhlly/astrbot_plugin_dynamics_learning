"""Cached, approved member materials for a single reply; never changes persona."""
from __future__ import annotations

import time

from .contract import MEMBER_PROFILES, encode, policy_allowed
from .profiles import relevance


def item(entry, source_scope):
    authors=list(entry.get('source_uids',[])) or ([entry['member']] if entry['member'] else [])
    if entry['category']=='expression_example':
        authors.append(entry['value'].get('context_member',''))
    value=entry['value']
    if entry['category']=='affection':
        value={k:value[k] for k in ('affection_level','reference_affection','max_affection','perceived_affection_distance') if k in value}
    return {'source_id':entry['proposal_id'],'revision':entry['revision'],'category':entry['category'],
            'key':entry['key'],'value':value,'member':entry['member'],'source_umo':source_scope,
            'source_uids':sorted(set(a for a in authors if a)),'provenance_complete':entry.get('provenance_complete',bool(authors)),
            'expires':entry['expires']}


def member_items(engine, scope, persona, member, query='', budget=2600,trace=None):
    source_rows = engine._entries_for(scope, persona) if hasattr(engine, '_entries_for') else engine.cache
    now = time.time()
    rows=[e for e in source_rows if (e['scope'],e['persona'],e['member'])==(scope,persona,member)
          and e['category'] in MEMBER_PROFILES and e['active'] and e['expires']>now and policy_allowed(e)]
    order={'preference':0,'typing_habit':1,'style':2,'expression_example':3}
    rows.sort(key=lambda e:(order[e['category']],-relevance({'situation':encode(e['value'])},query),-e['revision']))
    result=[]
    total_len = 2  # len('[]')
    examples=0
    for entry in rows:
        reason=''
        if entry['category']=='expression_example':
            if not relevance(entry['value'],query):
                reason='与当前输入不相关'
            elif examples>=2:
                reason='原话样例数量预算'
        if len(result)>=8:
            reason='成员资料数量预算'
        value=item(entry,scope)
        encoded_len = len(encode(value))
        next_len = total_len + encoded_len + (1 if result else 0)
        if next_len>budget:
            reason='成员资料长度预算'
        if reason:
            if trace is not None:
                trace[entry['proposal_id']]=reason
            continue
        result.append(value)
        total_len = next_len
        examples+=entry['category']=='expression_example'
    return result


def private_material(engine, scope, persona, member, character_scope, query=''):
    result={'schema_version':1,'scope':scope,'persona':persona,'member':member,'items':[],'private_readonly':True}
    identity=character_scope if isinstance(character_scope,dict) else {}
    result['character_scope']={k:identity.get(k,'') for k in ('platform','umo','persona_id','instance_id')}
    parts=scope.rsplit(':',2)
    if (engine.closed or not engine.config.get('enabled',True) or len(parts)!=3 or parts[1] not in {'FriendMessage','PrivateMessage'} or
        identity.get('umo')!=scope or identity.get('platform')!=parts[0] or not identity.get('persona_id') or not identity.get('instance_id')):
        return result
    for source in dict.fromkeys(engine.config.get('private_shared_group_sources',[])):
        source_parts=source.rsplit(':',2)
        if len(source_parts)!=3 or source_parts[:2]!=[parts[0],'GroupMessage'] or not engine.allowed(source):
            continue
        source_persona=engine.current_personas.get(source)
        origin=engine.character_scopes.get((source,source_persona),{})
        if origin.get('persona_id')!=identity['persona_id']:
            continue
        additions=member_items(engine,source,source_persona,member,query,budget=1600)
        additions=[row for row in additions if row['category']!='expression_example']
        if engine.social:
            affection=engine.social.current_affection(source,source_persona,member)
            if affection:
                additions.insert(0,item(affection,source))
        for row in additions:
            row['private_readonly']=True
            if len(encode(result['items']+[row]))<=4000:
                result['items'].append(row)
    return result
