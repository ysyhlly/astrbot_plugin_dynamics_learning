from __future__ import annotations

import hashlib
import json
import math
import re
import time

VERSION = 1
CATEGORIES = {"expression_example", "style", "typing_habit", "preference", "jargon", "affection", "relationship", "state", "decision_guidance", "participation"}
CATEGORIES |= {'scene_pattern', 'scene_adjustment', 'expression_style', 'expression_adjustment'}
MEMBER_PROFILES = {"expression_example", "style", "typing_habit", "preference"}
PERSONALITY_POLICY = "Bot 原始性格、身份、价值观和自身喜恶固定。成员资料只描述对应群友；可在原有人设内调整措辞和交流方式，不模仿或继承群友的性格、立场和喜恶。好感、短期状态和接话策略也不能改写自身性格。"


def policy_allowed(row):
    if row.get('category') in {'expression_style', 'expression_adjustment'}:
        return row.get('expression_policy') == 1 and row.get('member') == '' and ':GroupMessage:' in row.get('scope', '')
    if row.get('category') in {'scene_pattern', 'scene_adjustment'}:
        return row.get('scene_policy') == 1 and row.get('member') == '' and ':GroupMessage:' in row.get('scope', '')
    return row.get("category") in CATEGORIES and (
        row["category"] not in MEMBER_PROFILES or
        bool(row.get("member")) and row.get("profile_policy") == 1)
MODES = {"auto", "human", "llm", "llm_then_human"}
KINDS = {"observed", "decision", "outcome"}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def text(value, name, limit=256, empty=False):
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError("invalid " + name)
    return value


def number(value, name, low, high):
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("invalid " + name)
    return float(value)


def event_record(raw):
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or raw.get("schema_version") != VERSION or raw.get("kind") not in KINDS:
        raise ValueError("unsupported event")
    # A bounded JSON copy: never persist live host objects or arbitrary plugin state.
    serialized = encode(raw)
    if len(serialized.encode()) > 48_000:
        raise ValueError("event too large")
    row = json.loads(serialized)
    allowed = {"schema_version", "kind", "scope", "persona", "member", "turn_id", "message_id", "text",
               "host_version", "baseline", "baseline_threshold", "backend", "decision", "outcome",
               "context", "policy", "mandatory", "timestamp", "reply"}
    if set(row) - allowed:
        raise ValueError("unknown event fields")
    for key in ("scope", "persona", "member", "turn_id", "message_id", "host_version", "baseline"):
        text(row.get(key), key, 512)
    text(row.get("text", ""), "text", 4000, empty=True)
    number(row.get("baseline_threshold"), "baseline_threshold", 0, 100)
    if type(row.get("mandatory")) is not bool:
        raise ValueError("invalid mandatory")
    row["id"] = digest(row)
    row["received_at"] = time.time()
    return row


def candidate(raw, events, current):
    if not isinstance(raw, dict) or set(raw) - {"category", "key", "value", "member", "reason", "evidence_ids", "days"}:
        raise ValueError("invalid proposal fields")
    category = raw.get("category")
    if category not in CATEGORIES:
        raise ValueError("invalid category")
    ids = raw.get("evidence_ids")
    if not isinstance(ids, list) or not ids or len(ids) > (100 if category in {'typing_habit', 'scene_pattern', 'expression_style'} else 30) or any(not isinstance(i, str) for i in ids):
        raise ValueError("evidence required")
    by_id = {e["id"]: e for e in events}
    if any(i not in by_id for i in ids):
        raise ValueError("evidence not in supplied batch")
    evidence = [by_id[i] for i in dict.fromkeys(ids)]
    scope, persona = evidence[0]["scope"], evidence[0]["persona"]
    if any((e["scope"], e["persona"]) != (scope, persona) for e in evidence):
        raise ValueError("mixed scope or persona")
    member = text(raw.get("member", ""), "member", empty=True)
    if category in {"relationship", "state", "affection"} | MEMBER_PROFILES and not member:
        raise ValueError("member required")
    if member and category != 'expression_example' and any(e["member"] != member for e in evidence):
        raise ValueError("member evidence mismatch")
    key = text(raw.get("key"), "key", 80)
    value = raw.get("value")
    baseline, host_version = evidence[-1]["baseline"], evidence[-1]["host_version"]
    if category in {'expression_style', 'expression_adjustment'}:
        if member or ':GroupMessage:' not in scope:
            raise ValueError('expression style is group-only')
        from .expressions import validate_style
        from .scenes import validate_adjustment
        value = validate_style(value, evidence) if category == 'expression_style' else validate_adjustment(value, evidence)
        key = 'group_expression_style' if category == 'expression_style' else 'expression-penalty:' + digest([value['target_id'], value['target_revision']])[:32]
    elif category in {'scene_pattern', 'scene_adjustment'}:
        if member or ':GroupMessage:' not in scope:
            raise ValueError('scene patterns are group-only and never member profiles')
        from .scenes import validate_pattern, validate_adjustment
        value = validate_pattern(value, evidence) if category == 'scene_pattern' else validate_adjustment(value, evidence)
        key = 'scene:' + digest([value['scene'], value['method']])[:32] if category == 'scene_pattern' else 'penalty:' + digest([value['target_id'], value['target_revision']])[:32]
    elif category == 'expression_example':
        from .profiles import example_value, valid_pair
        if not isinstance(value, dict):
            raise ValueError('example requires source references')
        context, expression = by_id.get(value.get('context_id')), by_id.get(value.get('expression_id'))
        if not context or not expression or {e['id'] for e in evidence} != {context['id'], expression['id']} or expression['member'] != member or not valid_pair(context, expression):
            raise ValueError('invalid original conversation pair')
        expected = example_value(context, expression)
        if any(k not in expected or v != expected[k] for k, v in value.items()):
            raise ValueError('original quotes cannot be edited or invented')
        value = expected
        key = digest([member, context['text'], expression['text']])[:32]
    elif category == 'typing_habit' and isinstance(value, dict):
        from .profiles import statistics
        expected = statistics(evidence)
        if expected['sample_count'] < 10 or any(not e.get('context', {}).get('source_verified') for e in evidence):
            raise ValueError('statistics require ten distinct native messages')
        if any(k not in expected or v != expected[k] for k, v in value.items()):
            raise ValueError('statistics must match source messages')
        value, key = expected, 'observed_typing'
    elif category == 'preference':
        if not isinstance(value, dict) or set(value) != {'topic', 'stance', 'condition', 'quote', 'source_id'}:
            raise ValueError('preference requires topic, stance, condition, quote and source_id')
        if value['stance'] not in {'like', 'dislike', 'boundary'}:
            raise ValueError('invalid preference stance')
        for field, limit in [('topic', 80), ('condition', 160), ('quote', 400), ('source_id', 100)]:
            text(value[field], field, limit, empty=field == 'condition')
        source = next((e for e in evidence if e['id'] == value['source_id']), None)
        if not source or not source.get('context', {}).get('source_verified') or value['quote'] not in source['text']:
            raise ValueError('preference quote must match a native member message')
        key = 'pref:' + digest([value['topic'].strip().casefold(), value['condition'].strip().casefold()])[:32]
    elif category == "participation":
        if key != "reply_probability_threshold" or member:
            raise ValueError("only scope participation threshold is supported")
        value = number(value, "threshold", 0, 100)
        if any(e["baseline"] != baseline or e["host_version"] != host_version for e in evidence):
            raise ValueError("mixed decision baseline")
        if len({e["turn_id"] for e in evidence}) < 10:
            raise ValueError("participation requires at least ten distinct turns")
        previous = current.get("value") if current else evidence[-1]["baseline_threshold"]
        if abs(value - previous) > 5 or abs(value - evidence[-1]["baseline_threshold"]) > 10:
            raise ValueError("threshold adjustment exceeds bounded range")
    elif category == 'affection':
        from .affection import adjustment
        from .social import character_key
        if key != 'bot_affection':
            raise ValueError('affection key must be bot_affection')
        value = adjustment(value, current.get('value') if current else None)
        native_keys={character_key(e.get('context',{}).get('character_scope')) for e in evidence}
        native_keys.discard(None)
        if len(native_keys)>1:
            raise ValueError('affection evidence mixes character identities')
        identity=next(iter(native_keys),value.get('character_key'))
        if not identity:
            raise ValueError('affection requires a verified character identity')
        if value.get('character_key') not in (None,identity):
            raise ValueError('affection character changed')
        if any(e.get('context',{}).get('character_scope',{}).get('umo',scope)!=scope for e in evidence):
            raise ValueError('affection source scope mismatch')
        value['character_key']=identity
    else:
        value = text(value, "value", 1000)
        if category == 'jargon' and (member or len(key) > 64):
            raise ValueError('jargon must be a group term no longer than 64 characters')
    days = number(raw.get("days", 30), "days", 1, 90)
    if category == "state":
        days = min(days, 1)
    result = dict(scope=scope, persona=persona, member=member, category=category, key=key, value=value,
                reason=text(raw.get("reason"), "reason", 1500), evidence=evidence,
                baseline=baseline, host_version=host_version, days=days)
    if category in MEMBER_PROFILES:
        result['profile_policy'] = 1
    if category in {'scene_pattern', 'scene_adjustment'}:
        result['scene_policy'] = 1
    if category in {'expression_style', 'expression_adjustment'}:
        result['expression_policy'] = 1
    return result


def entry_key(scope, persona, member, category, key):
    return digest([scope, persona, member, category, key])


def review_result(raw):
    if not isinstance(raw, dict) or set(raw) != {"verdict", "reason", "risks"}:
        raise ValueError("invalid review result")
    if raw["verdict"] not in {"approve", "reject", "defer"}:
        raise ValueError("invalid review verdict")
    text(raw["reason"], "review reason", 2000)
    if not isinstance(raw["risks"], list) or len(raw["risks"]) > 10:
        raise ValueError("invalid risks")
    for risk in raw["risks"]:
        text(risk, "risk", 300)
    return raw


def llm_approved(proposal):
    latest = next((r for r in reversed(proposal["reviews"]) if r["actor"].startswith("llm:")), None)
    return latest is not None and latest["verdict"] == "approve"


def decode_model_object(content):
    """Accept a complete JSON object, optionally wrapped in one JSON Markdown fence."""
    if not isinstance(content, str) or len(content) > 24_000:
        raise ValueError("invalid model response size")
    payload = content.lstrip("\ufeff").strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", payload, re.DOTALL | re.IGNORECASE)
    if fenced:
        payload = fenced.group(1).strip()
    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        kind = "empty" if not payload else "fenced" if payload.startswith("```") else "non_json"
        raise ValueError(f"invalid model JSON: kind={kind}, chars={len(payload)}") from None
    if not isinstance(result, dict):
        raise ValueError("model response must be a JSON object")
    return result
