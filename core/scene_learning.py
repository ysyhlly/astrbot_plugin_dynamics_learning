"""Opt-in background scene learning and cached reply projection."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from .contract import candidate, digest, encode
from .scenes import ANALYZE_SCENES, DISAMBIGUATE, SCENE_CATEGORIES, episodes, scene_labels, select_patterns


class SceneLearning:
    def __init__(self, engine):
        self.engine = engine
        self.recent = {}

    def enabled(self, scope, persona=None):
        e = self.engine
        return (e.allowed(scope) and ':GroupMessage:' in scope and e.config.get('scene_learning_enabled', False)
                and (persona is None or e.current_personas.get(scope) == persona))

    def context(self, *, scope, persona, member, scene=None, frozen=None, query='', **_):
        result = {'schema_version': 1, 'scope': scope, 'persona': persona, 'member': member, 'items': [], 'versions': []}
        if not self.enabled(scope, persona):
            return result
        now = time.time()
        pair_entries = self.engine._entries_for(scope, persona) if hasattr(self.engine, '_entries_for') else self.engine.cache
        if frozen is not None:
            if not isinstance(frozen, list) or len(frozen) > 1:
                return result
            wanted = {(r.get('source_id'), r.get('revision')) for r in frozen if isinstance(r, dict)}
            selected = [row for row in pair_entries if row['scope'] == scope and row['persona'] == persona
                        and row['category'] == 'scene_pattern' and row.get('scene_policy') == 1 and not row['member']
                        and row['active'] and row['expires'] > now and (row['proposal_id'], row['revision']) in wanted]
        else:
            selected = select_patterns(pair_entries, scope, persona, scene, now, self.recent, query)
        for e in selected[:1]:
            value = {k: e['value'][k] for k in ('scene', 'method', 'conditions', 'avoid', 'examples', 'support')}
            item = {'source_id': e['proposal_id'], 'revision': e['revision'], 'member': '', 'category': 'scene_pattern',
                    'source_umo': scope, 'source_uids': e.get('source_uids', []),
                    'provenance_complete': e.get('provenance_complete', False), 'expires': e['expires'], 'value': value}
            if len(encode(item)) <= 3200:
                result['items'].append(item)
                result['versions'].append(e['proposal_id'])
        return result

    async def disambiguate(self, *, scope, persona, member, turn_id, text, context=None, **_):
        e = self.engine
        if not self.enabled(scope, persona) or not e.config.get('scene_disambiguation_enabled', False):
            return {}
        if not isinstance(text, str) or len(text) > 4000 or not isinstance(context, list) or len(encode(context)) > 8000:
            return {}
        token = e.scope_token(scope)
        provider = e.config.get('analysis_provider')
        # This is also capped by the host; queue time counts towards the five seconds.
        try:
            async with asyncio.timeout(5):
                result = await e.invoke_model('analysis_provider', DISAMBIGUATE,
                    {'scope': scope, 'member': member, 'text': text, 'context': context},
                    lambda: self.enabled(scope, persona) and token == e.scope_token(scope),
                    fast_lane=True)
            if token != e.scope_token(scope) or provider != e.config.get('analysis_provider') or not self.enabled(scope, persona):
                return {}
            if not isinstance(result, dict) or set(result) != set(scene_labels({})):
                return {}
            return scene_labels(result)
        except TimeoutError:
            raise
        except Exception:
            return {}

    def _sources(self, scope, persona):
        store = self.engine.store
        rows = store.profiles.sources(scope, persona)
        # Resolve missing explicit parents, at most five hops, within retained native sources.
        found = {e['message_id']: e for e in rows}
        for _ in range(5):
            missing = {e.get('context', {}).get('reply_to') for e in found.values()} - set(found) - {None, ''}
            if not missing:
                break
            additions = store.profiles.sources_for_events(scope, persona, [{'message_id': x} for x in missing])
            if not additions:
                break
            found.update(additions)
        group = self.engine.social.groups.get(scope, {}) if self.engine.social else {}
        names = [str(u.get('name', ''))[:128] for u in group.get('users', {}).values() if isinstance(u, dict) and u.get('name')][:2000]
        for row in found.values():
            # Keep only names actually present; never copy the whole group directory into each piece of evidence.
            present = [name for name in names if name in row['text']]
            row['context']['identity_tokens'] = list(dict.fromkeys([*row['context'].get('identity_tokens', []), *present]))
        return list(found.values())

    async def analyze(self, scope, persona):
        e = self.engine
        if not self.enabled(scope, persona):
            return
        async with e.operation('scene_analysis', scope):
            async with e.lock:
                if not self.enabled(scope, persona):
                    return
                token = e.scope_token(scope)
                self.recent = await asyncio.to_thread(e.store.scenes.recent)
            sources = await asyncio.to_thread(self._sources, scope, persona)
            pairs = episodes(sources)
            # Preserve the latest complete episodes within the model's input limit.
            while len(encode(pairs)) > 70000:
                pairs.pop(0)
            fingerprint = digest(pairs)
            def previous():
                with e.store.lock:
                    row = e.store.db.execute('SELECT fingerprint FROM scene_analysis WHERE scope=? AND persona=?', (scope, persona)).fetchone()
                    return row[0] if row else None
            if len(pairs) >= 3 and fingerprint != await asyncio.to_thread(previous):
                result = await e.invoke_model('analysis_provider', ANALYZE_SCENES, {'scope': scope, 'episodes': pairs},
                                             lambda: self.enabled(scope, persona) and token == e.scope_token(scope))
                if not isinstance(result, dict) or set(result) != {'patterns'} or not isinstance(result['patterns'], list) or len(result['patterns']) > 2:
                    raise ValueError('invalid scene analysis output')
                by_id = {p['id']: p for p in pairs}
                new = []
                for value in result['patterns']:
                    if not isinstance(value, dict) or not isinstance(value.get('episode_ids'), list):
                        continue
                    ids = {i for ep in value['episode_ids'] if isinstance(ep, str) and ep in by_id for i in by_id[ep]['source_ids']}
                    evidence = [s for s in sources if s['id'] in ids]
                    raw = {'category': 'scene_pattern', 'key': 'scene', 'member': '', 'value': value,
                           'reason': '跨成员和日期的独立对话；分别审查表达方法与具体内容。',
                           'evidence_ids': [s['id'] for s in evidence], 'days': 30}
                    try:
                        new.append(candidate(raw, evidence, None))
                    except (ValueError, TypeError, KeyError):
                        continue
                async with e.lock:
                    if not self.enabled(scope, persona) or token != e.scope_token(scope):
                        return
                    def commit():
                        with e.store.transaction():
                            for data in new:
                                e.store.propose(data, e.mode('scene_pattern'), 'llm:scene-analysis')
                            e.store.db.execute('INSERT OR REPLACE INTO scene_analysis VALUES(?,?,?)', (scope, persona, fingerprint))
                    await e.review_write(commit)
            await self.propose_adjustments(scope, persona, token)
            # Retry deferred/failed scene reviews on subsequent cycles, without bypassing claims.
            proposals = await asyncio.to_thread(e.store.proposals, scope, 2000)
            for p in [p for p in proposals if p['persona'] == persona and p['category'] in SCENE_CATEGORIES
                      and p['state'] in {'pending', 'reviewed'} and not e.store.review_claim(p['id'])][:8]:
                if not self.enabled(scope, persona) or token != e.scope_token(scope):
                    break
                if p['state'] == 'reviewed':
                    from .contract import llm_approved
                    if p['mode'] == 'llm' and e.mode(p['category']) == 'llm' and llm_approved(p):
                        await e.action(p['id'], 'approve', 'system:scene-review', p['updated'])
                else:
                    await e.review(p['id'])

    async def propose_adjustments(self, scope, persona, token, expression=False):
        e = self.engine
        facts = await asyncio.to_thread(e.store.scenes.adjustment_evidence, scope, persona)
        groups = defaultdict(list)
        for fact in facts:
            for ref in fact['context']['scene_feedback']['targets']:
                groups[tuple(ref)].append(fact)
        async with e.lock:
            enabled = e.expressions.enabled if expression else self.enabled
            if not enabled(scope, persona) or token != e.scope_token(scope):
                return
            for (pid, revision), evidence in list(groups.items())[:20]:
                if not any(r['proposal_id'] == pid and r['revision'] == revision and r['category'] == ('expression_style' if expression else 'scene_pattern') and r['active'] and r['expires'] > time.time() for r in e.cache):
                    continue
                category = 'expression_adjustment' if expression else 'scene_adjustment'
                raw = {'category': category, 'key': pid, 'member': '', 'days': 30,
                       'value': {'target_id': pid, 'target_revision': revision, 'penalty': .5, 'feedback_ids': [s['id'] for s in evidence[:20]]},
                       'reason': '两次独立回复收到明确纠正；仅建议降低关联模式优先级，不证明模型实际采用了它。',
                       'evidence_ids': [s['id'] for s in evidence[:20]]}
                try:
                    data = candidate(raw, evidence[:20], None)
                    await e.review_write(lambda: e.store.propose(data, e.mode(category), 'system:scene-feedback'))
                except (ValueError, TypeError):
                    continue
