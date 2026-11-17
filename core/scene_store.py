"""Versioned selection/provider/delivery/feedback receipts, never adoption claims."""
from __future__ import annotations

import json
import re
import time
from collections import Counter

from .contract import digest, encode


class SceneStore:
    def __init__(self, store):
        self.store = store
        store.db.executescript('''
        CREATE TABLE IF NOT EXISTS scene_receipts(id TEXT PRIMARY KEY,scope TEXT,persona TEXT,turn_id TEXT,stage TEXT,created REAL,body TEXT);
        CREATE INDEX IF NOT EXISTS scene_receipts_turn ON scene_receipts(scope,persona,turn_id,stage);
        CREATE TABLE IF NOT EXISTS scene_deliveries(scope TEXT,persona TEXT,message_id TEXT,turn_id TEXT,member TEXT,created REAL,
            PRIMARY KEY(scope,persona,message_id));
        CREATE TABLE IF NOT EXISTS scene_feedback(id TEXT PRIMARY KEY,scope TEXT,persona TEXT,reply_id TEXT,created REAL,body TEXT);
        CREATE INDEX IF NOT EXISTS scene_feedback_scope ON scene_feedback(scope,persona,created);
        CREATE TABLE IF NOT EXISTS scene_analysis(scope TEXT,persona TEXT,fingerprint TEXT,PRIMARY KEY(scope,persona));
        ''')

    def receipt(self, raw):
        if raw.get('stage') not in {'selected', 'provider', 'disambiguation'}:
            raise ValueError('invalid scene receipt stage')
        if any(not isinstance(raw.get(k), str) or not 0 < len(raw[k]) <= 512 for k in ('scope', 'persona', 'turn_id', 'member')):
            raise ValueError('invalid scene receipt identity')
        clean = {k: raw[k] for k in ('scope', 'persona', 'turn_id', 'member', 'stage')}
        clean['refs'] = []
        for row in raw.get('refs', [])[:2]:
            if (isinstance(row, dict) and isinstance(row.get('source_id'), str) and len(row['source_id']) <= 128
                    and type(row.get('revision')) is int and row['revision'] > 0):
                clean['refs'].append({'source_id': row['source_id'], 'revision': row['revision']})
                if row.get('category') in {'scene_pattern', 'expression_style'}:
                    clean['refs'][-1]['category'] = row['category']
                if isinstance(row.get('example_ids'), list):
                    clean['refs'][-1]['example_ids'] = [s[:128] for s in row['example_ids'][:2] if isinstance(s, str)]
        if raw['stage'] == 'disambiguation':
            clean['status'] = raw.get('status') if raw.get('status') in {'succeeded', 'timeout', 'failed', 'skipped'} else 'failed'
            elapsed = raw.get('elapsed_ms', 0)
            clean['elapsed_ms'] = max(0, min(60000, int(elapsed))) if type(elapsed) in (int, float) else 0
        identity = digest(clean)
        with self.store.transaction():
            self.store.db.execute('INSERT OR IGNORE INTO scene_receipts VALUES(?,?,?,?,?,?,?)',
                                  (identity, clean['scope'], clean['persona'], clean['turn_id'], clean['stage'], time.time(), encode(clean)))

    def provider_receipt(self, usage):
        trace = usage.get('scene_trace')
        if usage.get('stage') != 'provider_kwargs_after_boundaries' or not isinstance(trace, dict):
            return
        if trace.get('scope') != usage.get('scope') or trace.get('member') != usage.get('member'):
            return
        refs = [{k: r.get(k) for k in ('source_id', 'revision', 'category', 'example_ids')} for r in usage.get('learning', [])
                if r.get('category') in {'scene_pattern', 'expression_style'}]
        self.receipt({**trace, 'stage': 'provider', 'refs': refs})

    def observe(self, event):
        with self.store.transaction():
            reply = event.get('reply', {})
            if reply.get('delivered') is True and reply.get('message_id'):
                self.store.db.execute('INSERT OR IGNORE INTO scene_deliveries VALUES(?,?,?,?,?,?)',
                    (event['scope'], event['persona'], str(reply['message_id']), event['turn_id'], event['member'], event['received_at']))
                # Late delivery callbacks may arrive after the human feedback event.
                for row in self.store.db.execute('SELECT body FROM events WHERE scope=? AND persona=? AND created>? ORDER BY created DESC LIMIT 160',
                        (event['scope'], event['persona'], event['received_at'] - 600)).fetchall():
                    self._feedback(json.loads(row[0]))
            else:
                self._feedback(event)

    def _feedback(self, event):
        if event.get('kind') not in {'observed', 'decision', 'outcome'}:
            return
        refs = {str(x) for x in event.get('context', {}).get('reply_to_ids', []) if x}
        if len(refs) != 1:
            return  # No proximity-based attribution.
        reply_id = next(iter(refs))
        row = self.store.db.execute('SELECT * FROM scene_deliveries WHERE scope=? AND persona=? AND message_id=?',
                                   (event['scope'], event['persona'], reply_id)).fetchone()
        if not row or row['member'] != event['member']:
            return
        if not -30 <= event.get('received_at', 0) - row['created'] <= 600:
            return
        content = event.get('text', '').strip()
        first_clause = re.split(r'[，,。！!\n]', content, maxsplit=1)[0].strip()
        # Conservative explicit corrections only; quotation / questions stay unknown.
        quoted = any(mark in content for mark in ('“', '”', '「', '」', '"', '?', '？'))
        from .expressions import expression_feedback
        correction = expression_feedback(content)
        if correction:
            kind = correction
        elif not quoted and first_clause in {'我是在开玩笑', '我是在开玩笑啊', '我开玩笑的', '这是反话', '我说的是反话'}:
            kind = 'joke_correction'
        elif not quoted and first_clause in {'不是这个意思', '你理解错了', '你没理解我的意思', '不是让你解释这个梗'}:
            kind = 'meaning_correction'
        else:
            scene = event.get('context', {}).get('interaction', {})
            from .scenes import scene_labels
            labels = scene_labels(scene)
            kind = ('joke_continuation' if labels['turn_update'] == 'continues_joke'
                    and labels['communication_intent'] == 'banter' and labels['expression_mode'] in {'joke', 'meme', 'callback'}
                    and len(content) >= 4 and not re.fullmatch(r'(?:哈+|[呵嘿]+|233+|666+)[!！。,.，~～\s]*', content) else 'unknown')
        identity = digest([event['scope'], event['persona'], event['message_id'], reply_id])
        fact = {'event_id': event['id'], 'message_id': event['message_id'], 'member': event['member'],
                'kind': kind, 'reply_turn': row['turn_id'], 'strength': 'weak' if kind == 'joke_continuation' else 'explicit' if kind != 'unknown' else 'unknown'}
        old = self.store.db.execute('SELECT body FROM scene_feedback WHERE id=?', (identity,)).fetchone()
        if old and json.loads(old[0])['kind'] != 'unknown':
            return
        self.store.db.execute('INSERT OR REPLACE INTO scene_feedback VALUES(?,?,?,?,?,?)',
                             (identity, event['scope'], event['persona'], reply_id, event['received_at'], encode(fact)))

    def adjustment_evidence(self, scope, persona):
        result = []
        with self.store.lock:
            for row in self.store.db.execute('SELECT body,reply_id FROM scene_feedback WHERE scope=? AND persona=? ORDER BY created DESC LIMIT 100', (scope, persona)):
                fact = json.loads(row[0])
                if fact['kind'] not in {'joke_correction', 'meaning_correction', 'too_formal', 'overexplaining', 'repeated_question', 'repeated_phrase'}:
                    continue
                targets = []
                for receipt in self.store.db.execute("SELECT body FROM scene_receipts WHERE scope=? AND persona=? AND turn_id=? AND stage='provider'",
                                                     (scope, persona, fact['reply_turn'])):
                    targets.extend([x['source_id'], x['revision']] for x in json.loads(receipt[0])['refs'])
                source = self.store.db.execute('SELECT body FROM events WHERE id=?', (fact['event_id'],)).fetchone()
                if source and targets:
                    event = json.loads(source[0])
                    event['context']['scene_feedback'] = {**fact, 'targets': targets, 'verified': True}
                    delivered = self.store.db.execute("SELECT json_extract(body,'$.reply.text') FROM events WHERE scope=? AND persona=? AND json_extract(body,'$.turn_id')=? AND json_extract(body,'$.reply.message_id')=? AND json_extract(body,'$.reply.delivered')=1 ORDER BY created LIMIT 1", (scope, persona, fact['reply_turn'], row[1])).fetchone()
                    if delivered and isinstance(delivered[0], str):
                        event['context']['scene_feedback']['delivered_bot_text'] = delivered[0][:4000]
                        event['context']['scene_feedback']['attribution'] = 'Bot实际已送达文本，仅用于反馈归因，不是群友表达训练来源'
                    result.append(event)
        return result

    def recent(self):
        with self.store.lock:
            counts = Counter()
            for row in self.store.db.execute("SELECT body FROM scene_receipts WHERE stage='provider' AND created>?", (time.time() - 3600,)):
                for ref in json.loads(row[0])['refs']:
                    counts[ref['source_id']] += 1
            return dict(counts)

    def summary(self):
        with self.store.lock:
            cutoff = time.time() - 86400
            stages = {}
            for stage in ('selected', 'provider'):
                stages[stage] = self.store.db.execute('SELECT count(*) FROM (SELECT DISTINCT scope,persona,turn_id FROM scene_receipts WHERE stage=? AND created>? AND json_array_length(body,\'$.refs\')>0)', (stage, cutoff)).fetchone()[0]
            linked = self.store.db.execute("SELECT count(*) FROM (SELECT DISTINCT d.scope,d.persona,d.turn_id FROM scene_deliveries d JOIN scene_receipts r ON r.scope=d.scope AND r.persona=d.persona AND r.turn_id=d.turn_id AND r.stage='provider' WHERE d.created>? AND json_array_length(r.body,'$.refs')>0)", (cutoff,)).fetchone()[0]
            feedback = self.store.db.execute("SELECT count(*) FROM scene_feedback f WHERE f.created>? AND json_extract(f.body,'$.kind')!='unknown' AND EXISTS (SELECT 1 FROM scene_receipts r WHERE r.scope=f.scope AND r.persona=f.persona AND r.turn_id=json_extract(f.body,'$.reply_turn') AND r.stage='provider' AND json_array_length(r.body,'$.refs')>0)", (cutoff,)).fetchone()[0]
            attempts = [json.loads(r[0]) for r in self.store.db.execute("SELECT body FROM scene_receipts WHERE stage='disambiguation' AND created>?", (cutoff,))]
            prepared = self.store.db.execute("SELECT count(*) FROM (SELECT DISTINCT scope,persona,turn_id FROM scene_receipts WHERE stage IN ('selected','disambiguation') AND created>?)", (cutoff,)).fetchone()[0]
            waits = sorted(r['elapsed_ms'] for r in attempts)
            learned = self.store.db.execute("SELECT count(*) FROM entries WHERE json_extract(body,'$.category')='scene_pattern' AND json_extract(body,'$.active')=1 AND json_extract(body,'$.expires')>?", (time.time(),)).fetchone()[0]
            return {'learned': learned, **stages, 'delivered': linked, 'feedback': feedback,
                    'attempts': len(attempts), 'timeouts': sum(r['status'] == 'timeout' for r in attempts),
                    'prepared_turns': prepared, 'attempt_rate': round(len(attempts) / prepared, 4) if prepared else None,
                    'wait_p95_ms': waits[max(0, (95 * len(waits) + 99) // 100 - 1)] if waits else 0,
                    'extra_wait_ms': sum(r['elapsed_ms'] for r in attempts), 'window': 'last_24h_except_learned'}

    def prune(self, days):
        with self.store.transaction():
            for table in ('scene_receipts', 'scene_deliveries', 'scene_feedback'):
                self.store.db.execute('DELETE FROM ' + table + ' WHERE created<?', (time.time() - days * 86400,))

    def purge(self, scope):
        for table in ('scene_receipts', 'scene_deliveries', 'scene_feedback', 'scene_analysis'):
            self.store.db.execute('DELETE FROM ' + table + ' WHERE scope=?', (scope,))
