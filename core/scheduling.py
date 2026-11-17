"""Durable per-turn frontiers: model latency cannot skip interleaved callbacks."""
from __future__ import annotations

import json
import time
import uuid

from .contract import encode


class Scheduling:
    def __init__(self, store):
        self.store = store
        store.db.executescript('''
        CREATE TABLE IF NOT EXISTS analysis_inbox(seq INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT UNIQUE,
            scope TEXT,persona TEXT,turn_id TEXT,created REAL,processed INTEGER NOT NULL DEFAULT 0);
        CREATE INDEX IF NOT EXISTS analysis_pending ON analysis_inbox(scope,persona,processed,seq);
        CREATE INDEX IF NOT EXISTS analysis_unprocessed ON analysis_inbox(scope,persona,created,turn_id,seq) WHERE processed=0;
        CREATE INDEX IF NOT EXISTS analysis_turn ON analysis_inbox(scope,persona,turn_id,seq);
        CREATE TABLE IF NOT EXISTS analysis_meta(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE IF NOT EXISTS analysis_runs(id TEXT PRIMARY KEY,scope TEXT,persona TEXT,created REAL,body TEXT);
        CREATE TABLE IF NOT EXISTS analysis_sources(event_id TEXT,scope TEXT,source_id TEXT,PRIMARY KEY(event_id,source_id));
        CREATE INDEX IF NOT EXISTS analysis_source_refs ON analysis_sources(scope,source_id);
        ''')
        with store.transaction():
            if not store.db.execute("SELECT 1 FROM analysis_meta WHERE key='seeded_v1'").fetchone():
                # Preserve the old completed boundary. It did not record per-turn
                # coverage, so do not invent a retrospective processing history.
                store.db.execute('''INSERT OR IGNORE INTO analysis_inbox(event_id,scope,persona,turn_id,created,processed)
                    SELECT e.id,e.scope,e.persona,json_extract(e.body,'$.turn_id'),e.created,
                      CASE WHEN e.created<=coalesce(c.cursor,0) THEN 1 ELSE 0 END
                    FROM events e LEFT JOIN checkpoints c ON e.scope=c.scope AND e.persona=c.persona
                    ORDER BY e.created,e.rowid''')
                store.db.execute("INSERT INTO analysis_meta VALUES('seeded_v1',?)",(str(time.time()),))
                for row in store.db.execute('SELECT e.body FROM events e JOIN analysis_inbox q ON q.event_id=e.id WHERE q.processed=0').fetchall():
                    self._sources(json.loads(row[0]))

    def _sources(self,event):
        ids=event.get('context',{}).get('message_ids') or [event['message_id']]
        for source in dict.fromkeys(str(s) for s in ids[-16:] if s):
            self.store.db.execute('INSERT OR IGNORE INTO analysis_sources VALUES(?,?,?)',(event['id'],event['scope'],source))

    def ingest(self, event):
        self.store.db.execute('INSERT OR IGNORE INTO analysis_inbox(event_id,scope,persona,turn_id,created) VALUES(?,?,?,?,?)',
            (event['id'],event['scope'],event['persona'],event['turn_id'],event['received_at']))
        self._sources(event)

    def batches(self, minimum=0):
        with self.store.lock:
            if minimum > 0:
                return [
                    dict(r)
                    for r in self.store.db.execute(
                        '''SELECT scope,persona,
                        count(DISTINCT turn_id) AS count,
                        count(*) AS callbacks,
                        min(created) AS oldest,
                        max(seq) AS cursor
                        FROM analysis_inbox
                        WHERE processed=0
                        GROUP BY scope,persona
                        HAVING count>=?
                        ORDER BY oldest''',
                        (minimum,),
                    )
                ]
            pending = {
                (r['scope'], r['persona']): dict(r)
                for r in self.store.db.execute(
                    '''SELECT scope,persona,
                    count(DISTINCT turn_id) AS count,
                    count(*) AS callbacks,
                    min(created) AS oldest,
                    max(seq) AS cursor
                    FROM analysis_inbox
                    WHERE processed=0
                    GROUP BY scope,persona'''
                )
            }
            all_pairs = self.store.db.execute(
                'SELECT scope,persona,max(seq) AS cursor FROM analysis_inbox GROUP BY scope,persona'
            ).fetchall()
            rows = []
            for r in all_pairs:
                row = pending.get((r['scope'], r['persona']))
                if row is not None:
                    row['cursor'] = max(row['cursor'], r['cursor'])
                    rows.append(row)
                else:
                    rows.append(
                        {
                            'scope': r['scope'],
                            'persona': r['persona'],
                            'count': 0,
                            'callbacks': 0,
                            'oldest': None,
                            'cursor': r['cursor'],
                        }
                    )
            rows.sort(key=lambda r: (r['oldest'] is None, r['oldest'] or 0))
            return rows

    def batch(self, scope, persona, limit=80):
        with self.store.lock:
            groups = self.store.db.execute(
                '''SELECT turn_id,min(seq) AS first,max(seq) AS frontier,count(*) AS callbacks
                FROM analysis_inbox WHERE scope=? AND persona=? AND processed=0
                GROUP BY turn_id ORDER BY first LIMIT ?''',
                (scope, persona, limit),
            ).fetchall()
            if not groups:
                return {'scope': scope, 'persona': persona, 'events': [], 'frontiers': []}
            seqs = [group['frontier'] for group in groups]
            placeholders = ','.join('?' for _ in seqs)
            rows_by_seq = {
                r[0]: r[1]
                for r in self.store.db.execute(
                    f'SELECT q.seq,e.body FROM analysis_inbox q JOIN events e ON q.event_id=e.id WHERE q.seq IN ({placeholders})',
                    seqs,
                )
            }
            events = []
            frontiers = []
            for group in groups:
                body = rows_by_seq.get(group['frontier'])
                if not body:
                    raise ValueError('pending evidence unavailable')
                events.append(json.loads(body))
                frontiers.append(dict(group))
            return {'scope': scope, 'persona': persona, 'events': events, 'frontiers': frontiers}

    def commit(self, batch, candidates, rejected):
        created=[]
        with self.store.transaction():
            for frontier in batch['frontiers']:
                found=self.store.db.execute('SELECT processed FROM analysis_inbox WHERE seq=? AND scope=? AND persona=? AND turn_id=?',
                    (frontier['frontier'],batch['scope'],batch['persona'],frontier['turn_id'])).fetchone()
                if not found or found[0]:
                    raise ValueError('analysis batch changed; retry')
            for data,mode in candidates:
                try:
                    created.append(self.store.propose(data,mode,'llm:analysis'))
                except ValueError:
                    rejected+=1
            callbacks=0
            for frontier in batch['frontiers']:
                callbacks+=self.store.db.execute('''UPDATE analysis_inbox SET processed=1
                    WHERE scope=? AND persona=? AND turn_id=? AND seq<=? AND processed=0''',
                    (batch['scope'],batch['persona'],frontier['turn_id'],frontier['frontier'])).rowcount
            record={'turns':len(batch['frontiers']),'callbacks':callbacks,'frontiers':batch['frontiers'],
                    'event_ids':[e['id'] for e in batch['events']],'proposals':[p['id'] for p in created],'invalid_candidates':rejected}
            self.store.db.execute('INSERT INTO analysis_runs VALUES(?,?,?,?,?)',
                (uuid.uuid4().hex,batch['scope'],batch['persona'],time.time(),encode(record)))
        return created,rejected,callbacks

    def recent(self):
        with self.store.lock:
            return [dict(scope=r[0],persona=r[1],created=r[2],**json.loads(r[3])) for r in self.store.db.execute(
                'SELECT scope,persona,created,body FROM analysis_runs ORDER BY created DESC LIMIT 30')]
