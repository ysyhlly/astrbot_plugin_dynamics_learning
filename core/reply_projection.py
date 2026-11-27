"""Cached, approved member materials for a single reply; never changes persona."""
from __future__ import annotations

import copy
import time

from .contract import MEMBER_PROFILES, PERSONALITY_POLICY, encode, policy_allowed
from .profiles import relevance

UNIFIED_REPLY_BUDGET = 6000
MIN_UNIFIED_REPLY_BUDGET = 1200
MAX_UNIFIED_REPLY_BUDGET = 8400
MAX_JARGON_PER_TURN = 4

_OTHER_PRIORITY = {
    "typing_habit": 0,
    "style": 1,
    "decision_guidance": 2,
    "relationship": 3,
    "state": 4,
    "participation": 5,
    "expression_example": 6,
    "social_snapshot": 7,
}


def item_priority(row: dict) -> tuple[int, int]:
    """Unified reply projection priority:
    0: 固定人格红线 (personality_policy)
    1: 成员边界/喜恶 (preference: boundary > dislike > like)
    2: 命中黑话 (jargon)
    3: 场景方法 (scene_pattern)
    4: 群体表达 (expression_style)
    5: 其他参考资料 (typing_habit / style / decision_guidance / relationship / state / ...)
    """
    cat = str(row.get("category") or "")
    if cat == "personality_policy":
        return (0, 0)
    if cat == "preference":
        val = row.get("value")
        stance = str(val.get("stance") or "") if isinstance(val, dict) else ""
        sub = 0 if stance == "boundary" else 1 if stance == "dislike" else 2
        return (1, sub)
    if cat == "jargon":
        return (2, 0)
    if cat == "scene_pattern":
        return (3, 0)
    if cat == "expression_style":
        return (4, 0)
    return (5, _OTHER_PRIORITY.get(cat, 9))


def _trim_channel_item(row: dict, remaining: int) -> dict | None:
    """Try fitting a scene_pattern or expression_style item by dropping optional examples first."""
    cat = row.get("category")
    if cat not in {"scene_pattern", "expression_style"}:
        return None
    val = row.get("value")
    if not isinstance(val, dict) or not val.get("examples"):
        return None
    trimmed = copy.deepcopy(row)
    trimmed["value"]["examples"] = []
    if cat == "expression_style" and isinstance(trimmed.get("selection"), dict):
        trimmed["selection"]["example_ids"] = []
    if len(encode(trimmed)) <= remaining:
        return trimmed
    return None


def prune_unified_channels(
    *,
    learning_items: list[dict],
    scene_items: list[dict] = (),
    expression_items: list[dict] = (),
    personality_policy: str = PERSONALITY_POLICY,
    budget: int = UNIFIED_REPLY_BUDGET,
) -> dict:
    """Enforce a single-turn total character budget across all three reply channels.

    Priority order:
        固定人格红线 > 成员边界/喜恶 > 命中黑话 > 场景方法 > 群体表达 > 其他
    """
    policy_text = str(personality_policy or "")
    policy_len = len(policy_text)
    effective_budget = max(policy_len, int(budget))
    used = policy_len
    tagged: list[tuple[tuple[int, int], int, str, dict]] = []
    idx = 0
    for channel, rows in (
        ("learning", learning_items),
        ("scene", scene_items),
        ("expression", expression_items),
    ):
        for row in rows or ():
            if not isinstance(row, dict):
                continue
            tagged.append((item_priority(row), idx, channel, row))
            idx += 1
    tagged.sort(key=lambda entry: (entry[0], entry[1]))

    kept: dict[str, list[tuple[int, dict]]] = {"learning": [], "scene": [], "expression": []}
    omitted: dict[str, str] = {}
    for _prio, orig_idx, channel, row in tagged:
        size = len(encode(row))
        sep = 1 if kept[channel] else 0
        if used + size + sep <= effective_budget:
            kept[channel].append((orig_idx, row))
            used += size + sep
            continue
        remaining = effective_budget - used - sep
        trimmed = _trim_channel_item(row, remaining) if remaining > 0 else None
        if trimmed is not None:
            kept[channel].append((orig_idx, trimmed))
            used += len(encode(trimmed)) + sep
            continue
        sid = str(row.get("source_id") or "")
        if sid:
            omitted[sid] = "统一回复总字符预算"

    return {
        "personality_policy": policy_text,
        "learning_items": [r for _, r in sorted(kept["learning"], key=lambda x: x[0])],
        "scene_items": [r for _, r in sorted(kept["scene"], key=lambda x: x[0])],
        "expression_items": [r for _, r in sorted(kept["expression"], key=lambda x: x[0])],
        "budget": effective_budget,
        "used_chars": used,
        "omitted": omitted,
    }


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
    stance_order={'boundary':0,'dislike':1,'like':2}
    rows.sort(
        key=lambda e: (
            order[e['category']],
            stance_order.get((e.get('value') or {}).get('stance', ''), 9) if e['category'] == 'preference' and isinstance(e.get('value'), dict) else 0,
            -relevance({'situation': encode(e['value'])}, query),
            -e['revision'],
        )
    )
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
    sources = (
        engine.effective_private_shared_group_sources()
        if hasattr(engine, 'effective_private_shared_group_sources')
        else list(dict.fromkeys(engine.config.get('private_shared_group_sources', [])))
    )
    for source in sources:
        source_parts=source.rsplit(':',2)
        if len(source_parts)!=3 or source_parts[:2]!=[parts[0],'GroupMessage'] or not engine.allowed(source):
            continue
        if hasattr(engine, 'is_privacy_muted') and engine.is_privacy_muted(source, member):
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
