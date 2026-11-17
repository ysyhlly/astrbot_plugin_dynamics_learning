"""Read-only current/history selection; never combines or rewrites affection values."""
import json
from collections import defaultdict


def current_records(store, scope='', current_personas=None):
    """Return authoritative row IDs and presentation metadata before search/pagination.

    Stable identity includes the full group scope and character instance. Persona
    fingerprints identify text versions, not distinct relationships. Inactive
    current versions must not resurrect an older active fingerprint.
    """
    from .social import character_key

    rows = [json.loads(r[0]) for r in store.db.execute(
        "SELECT body FROM entries WHERE json_extract(body,'$.category')='affection' AND (?='' OR json_extract(body,'$.scope')=?) ORDER BY rowid",
        (scope, scope))]
    bindings = {}
    binding_order = {}
    for r in store.db.execute('SELECT scope,persona,legacy_key,rowid FROM social_bindings WHERE (?=\'\' OR scope=?)', (scope, scope)):
        bindings[(r[0], r[1])] = r[2]
        binding_order[(r[0], r[1])] = r[3]
    role_names = {}
    for r in store.db.execute('SELECT scope,persona,body FROM native_scope_registry WHERE (?=\'\' OR scope=?)', (scope, scope)):
        raw = json.loads(r[2])
        identity = character_key(raw)
        if identity:
            bindings[(r[0], r[1])] = identity
            role_names[(r[0], identity)] = raw.get('persona_id', '')
    seen = {}
    def observed(umo, persona):
        key = (umo, persona)
        if key not in seen:
            row = store.db.execute("SELECT created FROM events WHERE scope=? AND persona=? AND json_extract(body,'$.kind')='observed' ORDER BY created DESC LIMIT 1", key).fetchone()
            seen[key] = row[0] if row else 0
        return seen[key]
    current = {} if current_personas is None else dict(current_personas)
    aliases = defaultdict(set)
    for (umo, persona), identity in bindings.items():
        aliases[(umo, identity)].add(persona)
    grouped = defaultdict(list)
    metadata = {}
    insertion = {}
    for index, row in enumerate(rows):
        umo, persona, member = row['scope'], row['persona'], row['member']
        identity = row.get('value', {}).get('character_key')
        expected = bindings.get((umo, persona))
        valid = isinstance(identity, str) and identity.startswith('narrative:v1:')
        conflict = bool(expected and identity != expected)
        key = identity if valid and not conflict else 'unresolved-persona:' + persona
        group = (umo, key)
        aliases[group].add(persona)
        insertion[(umo, persona)] = index
        metadata[row['id']] = {'affection_current': False, 'affection_role_key': key,
            'affection_role_id': role_names.get(group, ''), 'affection_selection': 'identity_conflict' if conflict else 'unverified' if not valid else 'history'}
        if not conflict:
            grouped[(umo, key, member)].append(row)
    if current_personas is None:
        by_scope = defaultdict(set)
        for (umo, _), personas in aliases.items():
            by_scope[umo].update(personas)
        for umo, personas in by_scope.items():
            persona = max(personas, key=lambda p: (observed(umo, p), binding_order.get((umo, p), 0), p))
            if observed(umo, persona):
                current[umo] = persona
    chosen_personas = {}
    for group, personas in aliases.items():
        umo, identity = group
        if current.get(umo) in personas:
            chosen_personas[group] = (current[umo], 'current_persona')
        elif identity.startswith('unresolved-persona:') and current.get(umo):
            continue  # An unbound old fingerprint is not a verified additional role.
        else:
            persona = max(personas, key=lambda p: (observed(umo, p), binding_order.get((umo, p), 0), insertion.get((umo, p), -1), p))
            chosen_personas[group] = (persona, 'last_observed' if observed(umo, persona) else 'last_known')
    selected = set()
    for (umo, identity, _), versions in grouped.items():
        chosen = chosen_personas.get((umo, identity))
        for row in versions:
            metadata[row['id']]['affection_versions'] = len(versions)
        if not chosen:
            continue
        # Same current fingerprint and identity as the reply reader; take its first active row, never max(score).
        row = next((r for r in versions if r['persona'] == chosen[0] and r.get('active')), None)
        if row:
            selected.add(row['id'])
            metadata[row['id']].update(affection_current=True, affection_selection=chosen[1])
    return selected, metadata
