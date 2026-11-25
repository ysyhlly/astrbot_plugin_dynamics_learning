"""Daily reviewed group expression and synchronous, frozen reply views."""
import asyncio
import time

from .contract import candidate, digest, encode, llm_approved
from .expressions import ANALYZE_STYLE, EXPRESSION_CATEGORIES, group_statistics, usage_mode, validate_style
from .model_projection import format_system_prompt, project_event_for_model


class ExpressionLearning:
    def __init__(self, engine):
        self.engine = engine
        self.recent = {}

    def enabled(self, scope, persona=None):
        e = self.engine
        return bool(e.allowed(scope) and ':GroupMessage:' in scope and e.config.get('expression_style_enabled', False)
                    and (persona is None or e.current_personas.get(scope) == persona))

    def context(self, *, scope, persona, member, scene=None, action='', query='', frozen=None, **_):
        result = {'schema_version': 1, 'scope': scope, 'persona': persona, 'member': member, 'items': [], 'versions': []}
        if not self.enabled(scope, persona):
            return result
        now = time.time()
        pair_entries = self.engine._entries_for(scope, persona) if hasattr(self.engine, '_entries_for') else self.engine.cache
        rows = [e for e in pair_entries if e['scope'] == scope and e['persona'] == persona and e['category'] == 'expression_style'
                and e.get('expression_policy') == 1 and not e['member'] and e['active'] and e['expires'] > now]
        if not rows:
            return result
        row = max(rows, key=lambda e: e['revision'])
        recent = self.recent.get((scope, persona), {})
        if recent.get('expires', 0) < now:
            recent = {}
        mode = usage_mode(scene, action, query)
        if frozen is not None:
            if not isinstance(frozen, dict) or frozen.get('source_id') != row['proposal_id'] or frozen.get('revision') != row['revision']:
                return result
            selection = frozen.get('selection', {})
            mode = selection.get('mode')
            if mode not in {'casual', 'playful', 'wording', 'closing'}:
                return result
            trait_ids, example_ids = selection.get('trait_ids', []), selection.get('example_ids', [])
            if not isinstance(trait_ids, list) or not isinstance(example_ids, list) or len(trait_ids) > 3 or len(example_ids) > 2:
                return result
            traits = [t for t in row['value']['traits'] if t['id'] in trait_ids]
            examples = [v for v in row['value']['examples'] if v['id'] in example_ids]
            if len(traits) != len(set(trait_ids)) or len(examples) != len(set(example_ids)):
                return result
        else:
            if mode == 'careful':
                return result
            penalized = any(e['category'] == 'expression_adjustment' and e['scope'] == scope and e['persona'] == persona
                and e['active'] and e['expires'] > now and e['value']['target_id'] == row['proposal_id']
                and e['value']['target_revision'] == row['revision'] for e in pair_entries)
            traits = []
            for t in row['value']['traits']:
                if t['kind'] in {'humor', 'teasing'} and mode != 'playful':
                    continue
                if mode == 'wording' and t['kind'] != 'punctuation':
                    continue
                if mode == 'closing' and t['kind'] not in {'length', 'ending'}:
                    continue
                if t.get('marker') in recent.get('markers', set()):
                    continue
                traits.append(t)
            basics = [t for t in traits if t['kind'] in {'length', 'fragment', 'punctuation'}]
            voices = [t for t in traits if t not in basics]
            if voices and mode in {'casual', 'playful'}:
                offset = int(digest([scope, persona, query])[:8], 16) % len(voices)
                voices = voices[offset:] + voices[:offset]
                traits = basics[:2] + voices[:1]
            # A reviewed penalty halves the practical adaptation budget (3 -> 1 traits, no examples).
            traits = traits[:1 if penalized else 3]
            unique_examples = {}
            for v in row['value']['examples']:
                if v['id'] not in recent.get('examples', set()) and not any(m in v['text'] for m in recent.get('markers', set())):
                    unique_examples.setdefault(v['text'], v)
            examples = [] if mode in {'closing', 'wording'} or penalized else list(unique_examples.values())[:2]
            trait_ids, example_ids = [t['id'] for t in traits], [v['id'] for v in examples]
        if not traits:
            return result
        total_budget = self.engine.reply_projection_budget() if hasattr(self.engine, 'reply_projection_budget') else 6000
        if frozen is None and hasattr(self.engine, '_turn_budget_remaining'):
            cap = min(3200, self.engine._turn_budget_remaining(scope, persona, member, query, include_scene=True))
        else:
            cap = min(3200, total_budget)
        value = {'mode': mode, 'guidance': '；'.join(t['guidance'] for t in traits)[:800],
                 'traits': [{'id': t['id'], 'kind': t['kind'], 'marker': t['marker']} for t in traits], 'examples': examples}
        item = {'source_id': row['proposal_id'], 'revision': row['revision'], 'member': '', 'category': 'expression_style',
                'source_umo': scope, 'source_uids': row.get('source_uids', []), 'provenance_complete': row.get('provenance_complete', False),
                'expires': row['expires'], 'value': value,
                'selection': {'mode': mode, 'trait_ids': trait_ids, 'example_ids': example_ids}}
        item_len = len(encode(item))
        if frozen is None and item_len > cap and examples:
            value = {**value, 'examples': []}
            item = {**item, 'value': value, 'selection': {'mode': mode, 'trait_ids': trait_ids, 'example_ids': []}}
            item_len = len(encode(item))
        if item_len <= cap:
            result['items'], result['versions'] = [item], [row['proposal_id']]
        return result

    async def analyze(self, scope, persona):
        e = self.engine
        if not self.enabled(scope, persona):
            return
        async with e.operation('expression_analysis', scope):
            async with e.lock:
                if not self.enabled(scope, persona):
                    return
                token = e.scope_token(scope)
                self.recent = await asyncio.to_thread(e.store.expressions.recent)
            await self.review_pending(scope, persona, token)
            sample, parents, frontier = await asyncio.to_thread(e.store.expressions.sample, scope, persona)
            stats = group_statistics(sample)
            if stats['messages'] < 30 or stats['members'] < 3 or stats['days'] < 2:
                return
            async with e.lock:
                if not self.enabled(scope, persona) or token != e.scope_token(scope):
                    return
                reserved_at = time.time()
                if not await e.review_write(lambda: e.store.expressions.reserve(scope, persona, frontier, reserved_at)):
                    return
            evidence = sample + [p for p in parents if p['id'] not in {s['id'] for s in sample}]
            current = [r['value'] for r in e.cache if r['scope'] == scope and r['persona'] == persona
                       and r['category'] == 'expression_style' and r['active'] and r['expires'] > time.time()][:1]
            feedback = await asyncio.to_thread(e.store.scenes.adjustment_evidence, scope, persona)
            def public_source(row):
                return project_event_for_model(row, include_scope=False)
            data = {'scope': scope, 'sample': [public_source(r) for r in sample], 'parents': [public_source(r) for r in parents],
                    'statistics': stats, 'current': current, 'feedback': [f['context']['scene_feedback']['kind'] for f in feedback[:10]]}
            submitted = [*evidence, *feedback[:10]]
            char_ctx = await asyncio.to_thread(e.character_context_for, scope, persona)
            instruction = format_system_prompt(ANALYZE_STYLE, char_ctx)
            try:
                output = await e.invoke_model('analysis_provider', instruction, data,
                    lambda: self.enabled(scope, persona) and token == e.scope_token(scope)
                    and not e.store.source_withdrawals.has_withdrawn(scope, submitted))
            finally:
                withdrawn = e.store.source_withdrawals.has_withdrawn(scope, submitted)
                if withdrawn:
                    await e.review_write(lambda: e.store.expressions.release_reservation(scope, persona, frontier, reserved_at))
            if withdrawn:
                return
            if not isinstance(output, dict) or set(output) != {'style'}:
                raise ValueError('invalid expression analysis response')
            if output['style'] is None:
                return
            if not isinstance(output['style'], dict):
                raise ValueError('invalid expression style object')
            value = {**output['style'], 'sample_ids': [r['id'] for r in sample]}
            # An unsupported catchphrase must not discard unrelated, well-supported formatting traits.
            offered_traits = value.get('traits')
            if not isinstance(offered_traits, list) or len(offered_traits) > 6:
                raise ValueError('invalid expression traits')
            accepted = []
            for trait in offered_traits:
                try:
                    validate_style({'sample_ids': value['sample_ids'], 'traits': [trait], 'example_ids': []}, evidence)
                    accepted.append(trait)
                except (ValueError, TypeError, KeyError):
                    continue
            value['traits'] = accepted
            offered_examples = value.get('example_ids', [])
            if not isinstance(offered_examples, list) or len(offered_examples) > 6:
                raise ValueError('invalid expression example list')
            allowed_examples = {r['id'] for r in sample if len(r['text']) <= 160}
            value['example_ids'] = list(dict.fromkeys(i for i in offered_examples if isinstance(i, str) and i in allowed_examples))
            basic = [t for t in accepted if t['kind'] in {'length', 'fragment', 'punctuation'}]
            variants = []
            if basic:
                variants.append({**value, 'traits': basic, 'example_ids': []})
            if len(basic) != len(accepted) or value['example_ids']:
                variants.append(value)
            if not variants:
                raise ValueError('no supported expression traits')
            # Each version receives its own independent review. Rejecting enrichment cannot revoke approved basics.
            reviewed = []
            for variant in variants:
                raw = {'category': 'expression_style', 'key': 'group_expression_style', 'member': '', 'value': variant,
                       'evidence_ids': [r['id'] for r in evidence], 'reason': f'按成员均衡抽样的群体写法；基础写法与口吻增强分开审查。过滤了 {len(offered_traits)-len(accepted)} 项缺少可核验证据的建议。', 'days': 30}
                proposal = candidate(raw, evidence, None)
                async with e.lock:
                    if not self.enabled(scope, persona) or token != e.scope_token(scope):
                        return
                    p = await e.review_write(
                        lambda: e.store.propose(proposal, e.mode('expression_style'), 'llm:expression-analysis')
                    )
                if p['state'] == 'pending':
                    try:
                        p = await e.review(p['id'], expected_updated=p['updated'], defer_apply=True, automatic=True)
                    except Exception as exc:
                        e.last_error = type(exc).__name__
                reviewed.append(p)
            if self.enabled(scope, persona) and token == e.scope_token(scope) and e.mode('expression_style') == 'llm':
                approved = [p for p in reviewed if p['state'] in {'reviewed', 'applied'} and llm_approved(p)]
                chosen = approved[-1] if approved else None
                if chosen and chosen['state'] == 'reviewed':
                    await e.action(chosen['id'], 'approve', 'system:expression-reviewed', chosen['updated'])
                if chosen:
                    for p in reviewed:
                        if p['id'] != chosen['id'] and p['state'] in {'pending', 'reviewed'}:
                            await e.action(p['id'], 'reject', 'system:expression-superseded', p['updated'], '本轮选择另一已审查版本；保留本候选审查记录，不再自动覆盖。')

    async def review_pending(self, scope, persona, token):
        e = self.engine
        await e.scenes.propose_adjustments(scope, persona, token, expression=True)
        rows = await asyncio.to_thread(
            e.store.pending_proposals_by_categories, scope, persona, EXPRESSION_CATEGORIES, 64
        )
        pending_ids = []
        for p in rows[:8]:
            if not self.enabled(scope, persona) or token != e.scope_token(scope):
                break
            current = next((r for r in e.cache if r['id'] == p['entry_id']), None)
            if p['category'] == 'expression_style' and (current['revision'] if current else 0) != p['base_revision']:
                await e.action(p['id'], 'reject', 'system:stale-expression', p['updated'], '已有较新的表达版本，等待新证据生成。')
                continue
            if p['state'] == 'reviewed':
                if p['mode'] == 'llm' and e.mode(p['category']) == 'llm' and llm_approved(p):
                    await e.action(p['id'], 'approve', 'system:expression-review', p['updated'])
            elif not e.automatic_review_waiting(p, time.time()):
                pending_ids.append(p['id'])
        if pending_ids and self.enabled(scope, persona) and token == e.scope_token(scope):
            await e.review_batch(
                pending_ids,
                valid=lambda: self.enabled(scope, persona) and token == e.scope_token(scope),
                automatic=True,
            )
