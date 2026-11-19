"""Member observations from native messages, never merged turns or bot replies."""
from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict

from .contract import digest, encode

EMOJI = re.compile('[\U0001f300-\U0001faff\u2600-\u27bf]|\\[[^\\[\\]\n]{1,12}\\]')


def native_message(event):
    message = getattr(event, 'message_obj', None)
    scope = str(getattr(event, 'unified_msg_origin', '') or '')
    member = str(event.get_sender_id() or '')
    source = str(getattr(message, 'message_id', '') or '')
    if ':GroupMessage:' not in scope or not member or not source:
        return None
    text = str(event.get_message_str() or '')
    # Retain excluded rows as timeline boundaries; never truncate a quote and call it exact.
    eligible = member != str(getattr(message, 'self_id', '')) and bool(text.strip()) and len(text) <= 1000 and not text.lstrip().startswith(('/', '#', '!', '！', '。', '[CQ:'))
    reply_to = ''
    mentions=[]
    identity_tokens = []
    name_getter = getattr(event, 'get_sender_name', None)
    if callable(name_getter):
        name = str(name_getter() or '')
        if name:
            identity_tokens.append(name[:128])
    for component in event.get_messages() or []:
        if type(component).__name__ == 'Reply':
            reply_to = str(getattr(component, 'id', '') or '')
        if type(component).__name__ == 'At':
            target=str(getattr(component,'qq','') or '')
            if target and target!=member:
                mentions.append(target)
            name = str(getattr(component, 'name', '') or '')
            if name:
                identity_tokens.append(name[:128])
    return dict(scope=scope, member=member, source_id=source, text=text[:1000],
                eligible=eligible, reply_to=reply_to, mentions=mentions[:16], identity_tokens=identity_tokens[:17], created=time.time())


def statistics(events):
    """Ratios describe observed message formatting, not personality or typing speed."""
    unique = {e['message_id']: e for e in events if e.get('context', {}).get('source_verified') and e.get('text', '').strip()}
    rows = sorted(unique.values(), key=lambda e: (e['timestamp'], e['message_id']))
    count = len(rows)
    if not count:
        return {'kind': 'observed_typing', 'sample_count': 0}
    texts = [e['text'] for e in rows]
    consecutive = sum(b['context'].get('previous_id') == a['message_id'] and 0 <= b['timestamp']-a['timestamp'] <= 90
                      for a, b in zip(rows, rows[1:]))
    def ratio(n):
        return round(n / count, 4)
    return dict(kind='observed_typing', sample_count=count, window_start=rows[0]['timestamp'], window_end=rows[-1]['timestamp'],
                mean_chars=round(sum(map(len, texts))/count, 2), short_rate=ratio(sum(len(t.strip()) <= 12 for t in texts)),
                multiline_rate=ratio(sum('\n' in t.strip() for t in texts)),
                punctuation_rate=ratio(sum(bool(re.search(r'[，。！？、；：,.!?;:…]', t)) for t in texts)),
                emoji_rate=ratio(sum(bool(EMOJI.search(t)) for t in texts)), consecutive_rate=ratio(consecutive),
                definitions='短句≤12字；连续补充=同一成员相邻消息且间隔≤90秒；表情为Unicode表情或方括号表情的近似识别。统计不代表喜恶或稳定性格。')


def example_value(context, expression):
    return dict(kind='expression_example', context_id=context['id'], expression_id=expression['id'],
                situation=context['text'], expression=expression['text'], context_member=context['member'],
                source_ids=[context['message_id'], expression['message_id']],
                link='reply' if expression['context'].get('reply_to') == context['message_id'] else 'adjacent')


def valid_pair(context, expression):
    if not all(e.get('context', {}).get('source_verified') for e in (context, expression)):
        return False
    if (context['scope'], context['persona']) != (expression['scope'], expression['persona']) or context['member'] == expression['member']:
        return False
    gap = expression['timestamp'] - context['timestamp']
    target = expression['context'].get('reply_to') or expression['context'].get('previous_id')
    return 0 <= gap <= 300 and target == context['message_id'] and context['text'] != expression['text']


def relevance(value, query):
    """Conservative lexical retrieval; unrelated original quotes are not injected."""
    def tokens(s):
        return set(re.findall(r'[a-z0-9_]{2,}', s.lower())) | {s[i:i+2] for i in range(len(s)-1)
            if all('\u4e00' <= c <= '\u9fff' for c in s[i:i+2])}
    return len(tokens(value['situation']) & tokens(query))


def useful_example(context, expression):
    text=expression['text'].strip()
    plain=re.sub(r'[\s，。！？、,.!?…~～]+','',text).casefold()
    if plain in {'嗯','哦','好','好的','好吧','嗯嗯','哈哈','哈哈哈','收到','谢谢','ok','yes','no'} or len(plain)<3:
        return False
    # These snippets are useful context, but poor reusable expression samples.
    return len(text)<=240 and len(context['text'])<=400 and not re.search(r'https?://|```|\b1[3-9]\d{9}\b|[\w.+-]+@[\w.-]+\.[a-z]{2,}',text,re.I)


class Profiles:
    def __init__(self, store):
        self.store = store
        store.db.executescript('''
        CREATE TABLE IF NOT EXISTS profile_raw(scope TEXT,source_id TEXT,created REAL,body TEXT,PRIMARY KEY(scope,source_id));
        CREATE INDEX IF NOT EXISTS profile_raw_time ON profile_raw(scope,created);
        CREATE TABLE IF NOT EXISTS profile_targets(scope TEXT,source_id TEXT,created REAL,body TEXT,PRIMARY KEY(scope,source_id));
        CREATE TABLE IF NOT EXISTS profile_messages(scope TEXT,persona TEXT,source_id TEXT,member TEXT,created REAL,body TEXT,
            PRIMARY KEY(scope,persona,source_id));
        CREATE INDEX IF NOT EXISTS profile_messages_scope ON profile_messages(scope,persona,created);
        ''')

    def _join(self, raw, target):
        if raw['member'] != target['member'] or not raw['eligible']:
            return
        evidence = {k: target[k] for k in ('scope', 'persona', 'member', 'baseline', 'host_version', 'baseline_threshold')}
        evidence.update(id='source:'+digest([raw['scope'], target['persona'], raw['source_id']]),
                        kind='observed', message_id=raw['source_id'], turn_id='source:'+raw['source_id'], text=raw['text'],
                        timestamp=raw['created'], received_at=raw['created'], mandatory=False,
                        context={'source_verified': True, 'reply_to': raw['reply_to'], 'previous_id': raw['previous_id'],
                                 'mentions':raw.get('mentions',[]),
                                 'identity_tokens': raw.get('identity_tokens', []),
                                 'character_scope': target.get('context', {}).get('character_scope', {})})
        self.store.db.execute('INSERT OR IGNORE INTO profile_messages VALUES(?,?,?,?,?,?)',
            (raw['scope'], target['persona'], raw['source_id'], raw['member'], raw['created'], encode(evidence)))

    def _observe_one(self, raw):
        old = self.store.db.execute('SELECT body FROM profile_raw WHERE scope=? AND source_id=?', (raw['scope'], raw['source_id'])).fetchone()
        if old:
            return  # Source receipt is immutable; repeated delivery does not increase statistics.
        previous = self.store.db.execute('SELECT source_id FROM profile_raw WHERE scope=? ORDER BY created DESC,rowid DESC LIMIT 1', (raw['scope'],)).fetchone()
        raw = {**raw, 'previous_id': previous[0] if previous else ''}
        self.store.db.execute('INSERT INTO profile_raw VALUES(?,?,?,?)', (raw['scope'], raw['source_id'], raw['created'], encode(raw)))
        target = self.store.db.execute('SELECT body FROM profile_targets WHERE scope=? AND source_id=?', (raw['scope'], raw['source_id'])).fetchone()
        if target:
            self._join(raw, json.loads(target[0]))

    def observe(self, raw):
        with self.store.transaction():
            self._observe_one(raw)

    def observe_batch(self, items):
        if not items:
            return
        with self.store.transaction():
            for raw in items:
                self._observe_one(raw)

    def bind(self, event):
        if event.get('kind') != 'observed' or ':GroupMessage:' not in event['scope']:
            return
        ids = event.get('context', {}).get('message_ids') or [event['message_id']]
        binding = {k: event[k] for k in ('scope', 'persona', 'member', 'baseline', 'host_version', 'baseline_threshold')}
        binding['context'] = {'character_scope':event.get('context', {}).get('character_scope', {})}
        with self.store.transaction():
            for source_id in dict.fromkeys(str(s) for s in ids[-16:] if s):
                self.store.db.execute('INSERT OR IGNORE INTO profile_targets VALUES(?,?,?,?)',
                    (event['scope'], source_id, event['received_at'], encode(binding)))
                target = json.loads(self.store.db.execute('SELECT body FROM profile_targets WHERE scope=? AND source_id=?', (event['scope'], source_id)).fetchone()[0])
                raw = self.store.db.execute('SELECT body FROM profile_raw WHERE scope=? AND source_id=?', (event['scope'], source_id)).fetchone()
                if raw:
                    self._join(json.loads(raw[0]), target)

    def sources(self, scope, persona, member='', days=30):
        with self.store.lock:
            rows = self.store.db.execute('''SELECT p.body FROM profile_messages p WHERE p.scope=? AND p.persona=?
                AND (?='' OR p.member=?) AND p.created>? AND NOT EXISTS (SELECT 1 FROM withdrawn_review_sources w
                WHERE w.scope=p.scope AND w.member=p.member AND w.message_id=p.source_id)
                ORDER BY p.created DESC,p.rowid DESC LIMIT 600''',
                (scope, persona, member, member, time.time()-days*86400)).fetchall()
            return [json.loads(r[0]) for r in reversed(rows)]

    def sources_for_events(self, scope, persona, events):
        ids=list(dict.fromkeys(str(mid) for e in events for mid in (e.get('context',{}).get('message_ids') or [e['message_id']]) if mid))
        with self.store.lock:
            result={}
            for offset in range(0,len(ids),400):
                batch=ids[offset:offset+400]
                rows=self.store.db.execute('''SELECT p.source_id,p.body FROM profile_messages p
                    WHERE p.scope=? AND p.persona=? AND NOT EXISTS (SELECT 1 FROM withdrawn_review_sources w
                    WHERE w.scope=p.scope AND w.member=p.member AND w.message_id=p.source_id)
                    AND p.source_id IN ('''+','.join('?' for _ in batch)+')', (scope,persona,*batch)).fetchall()
                result.update({r[0]:json.loads(r[1]) for r in rows})
            return result

    def bundle(self, scope, persona, member=''):
        rows = self.sources(scope, persona, member)[-100:] if member else self.sources(scope, persona)
        if member and rows:
            ids=[e['context'].get('reply_to') or e['context'].get('previous_id') for e in rows]
            contexts=self.sources_for_events(scope,persona,[{'context':{'message_ids':[i for i in ids if i]},'message_id':''}])
            rows=sorted({e['id']:e for e in [*rows,*contexts.values()]}.values(),key=lambda e:e['timestamp'])
        by_id = {e['message_id']: e for e in rows}
        grouped = defaultdict(list)
        examples = {}
        supports=defaultdict(list)
        for e in rows:
            if member and e['member'] != member:
                continue
            grouped[e['member']].append(e)
            context = by_id.get(e['context'].get('reply_to') or e['context'].get('previous_id'))
            if not context or not valid_pair(context, e) or not useful_example(context,e):
                continue
            quality=('明确引用回复' if e['context'].get('reply_to')==context['message_id'] else
                     '明确提及前一位成员' if context['member'] in e['context'].get('mentions',[]) else
                     '重复的同话题回应' if relevance({'situation':context['text']},e['text'])>=2 else '')
            if not quality:
                continue
            key = digest([e['member'], context['text'], e['text']])[:32]
            pair = examples.setdefault(key, {'key': key, 'member': e['member'], 'count': 0})
            pair.update(value=example_value(context, e), evidence=[context, e],quality=quality)
            supports[(e['member'],e['text'].strip().casefold())].append((e['timestamp'],e['message_id']))
            pair['count'] += 1
            age = max(0, time.time()-e['timestamp']) / 86400
            pair['score'] = round(pair['count'] * math.pow(0.5, age/15)*(3 if quality=='明确引用回复' else 2), 4)
        accepted=[]
        for pair in examples.values():
            if pair['quality']=='重复的同话题回应':
                support=supports[(pair['member'],pair['value']['expression'].strip().casefold())]
                if len({mid for _,mid in support})<2 or max(t for t,_ in support)-min(t for t,_ in support)<60:
                    continue
            accepted.append(pair)
        return {'examples': sorted(accepted, key=lambda p: p['score'], reverse=True)[:40],
                'statistics': {m: statistics(es[-100:]) for m, es in grouped.items()}, 'sources': rows}

    def proposals(self, scope, persona):
        bundle = self.bundle(scope, persona)
        output = []
        with self.store.lock:
            existing_examples = {
                r[0]
                for r in self.store.db.execute(
                    "SELECT json_extract(body,'$.key') FROM proposals WHERE scope=? AND persona=? AND json_extract(body,'$.category')='expression_example'",
                    (scope, persona),
                )
            }
            habits_by_member = {}
            for member_key, body in self.store.db.execute(
                "SELECT json_extract(body,'$.member'),body FROM proposals WHERE scope=? AND persona=? AND json_extract(body,'$.category')='typing_habit' AND json_extract(body,'$.key')='observed_typing' ORDER BY created DESC,rowid DESC LIMIT 200",
                (scope, persona),
            ):
                if member_key not in habits_by_member:
                    habits_by_member[member_key] = json.loads(body)
        for pair in bundle['examples']:
            if pair['key'] in existing_examples:
                continue
            output.append((dict(category='expression_example', key=pair['key'], member=pair['member'], value=pair['value'],
                reason=f"{pair['quality']}；窗口内该场景与表达出现 {pair['count']} 次。仍需审查语境和角色兼容性。",
                evidence_ids=[e['id'] for e in pair['evidence']], days=30), pair['evidence']))
            if len(output) >= 2:
                break
        for member, stats in sorted(bundle['statistics'].items(), key=lambda r: r[1]['sample_count'], reverse=True):
            if stats['sample_count'] < 10:
                continue
            previous = habits_by_member.get(member)
            sources = [e for e in bundle['sources'] if e['member'] == member][-60:]
            if previous and (previous['state'] in {'pending', 'reviewed'} or
                len({e['id'] for e in sources} - {e['id'] for e in previous['evidence']}) < 5):
                continue
            output.append((dict(category='typing_habit', key='observed_typing', member=member, value={'kind':'observed_typing'},
                reason='原始单条消息去重统计；只反映观察窗口内的写法，不推断喜恶、打字速度或人格。',
                evidence_ids=[e['id'] for e in sources], days=30), sources))
            break
        return output
