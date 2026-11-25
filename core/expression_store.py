"""Bounded expression scheduling and actual-output repetition observations."""
import json
import time

from .contract import encode
from .expressions import balanced_sample


class ExpressionStore:
    def __init__(self, store):
        self.store = store
        store.db.executescript('''
        CREATE TABLE IF NOT EXISTS expression_analysis(scope TEXT,persona TEXT,last_attempt REAL,frontier INTEGER,
            PRIMARY KEY(scope,persona));
        CREATE TABLE IF NOT EXISTS expression_deliveries(scope TEXT,persona TEXT,message_id TEXT,created REAL,markers TEXT,
            PRIMARY KEY(scope,persona,message_id));
        CREATE INDEX IF NOT EXISTS expression_delivery_time ON expression_deliveries(scope,persona,created);
        CREATE INDEX IF NOT EXISTS expression_deliveries_created ON expression_deliveries(created);
        CREATE TABLE IF NOT EXISTS expression_source_counts(scope TEXT,persona TEXT,counter INTEGER NOT NULL,
            PRIMARY KEY(scope,persona));
        CREATE TRIGGER IF NOT EXISTS expression_source_insert AFTER INSERT ON profile_messages BEGIN
            INSERT INTO expression_source_counts VALUES(NEW.scope,NEW.persona,1)
            ON CONFLICT(scope,persona) DO UPDATE SET counter=counter+1;
        END;
        ''')

    def sample(self, scope, persona):
        with self.store.lock:
            rows = self.store.db.execute('''SELECT body FROM (SELECT body,created,
                row_number() OVER(PARTITION BY member,cast(created/86400 AS INTEGER) ORDER BY created DESC,rowid DESC) AS n
                FROM profile_messages p WHERE scope=? AND persona=? AND created>? AND NOT EXISTS (
                SELECT 1 FROM withdrawn_review_sources w WHERE w.scope=p.scope AND w.member=p.member AND w.message_id=p.source_id
                )) WHERE n<=10 ORDER BY created DESC LIMIT 8000''',
                (scope, persona, time.time() - 30 * 86400)).fetchall()
            selected = balanced_sample([json.loads(r[0]) for r in rows])
            parents = {e.get('context', {}).get('reply_to') for e in selected} - {None, ''} - {e['message_id'] for e in selected}
            linked = self.store.profiles.sources_for_events(scope, persona, [{'message_id': i} for i in sorted(parents)[:20]])
            counter = self.store.db.execute('SELECT counter FROM expression_source_counts WHERE scope=? AND persona=?', (scope, persona)).fetchone()
            frontier = counter[0] if counter else 0
            return selected, list(linked.values()), frontier

    def reserve(self, scope, persona, frontier, now):
        with self.store.transaction():
            old = self.store.db.execute('SELECT last_attempt,frontier FROM expression_analysis WHERE scope=? AND persona=?', (scope, persona)).fetchone()
            if old:
                fresh = frontier - old[1]
                if now - old[0] < 86400 or fresh < 20:
                    return False
            self.store.db.execute('INSERT OR REPLACE INTO expression_analysis VALUES(?,?,?,?)', (scope, persona, now, frontier))
            return True

    def release_reservation(self, scope, persona, frontier, reserved_at):
        with self.store.transaction():
            self.store.db.execute('DELETE FROM expression_analysis WHERE scope=? AND persona=? AND frontier=? AND last_attempt=?',
                                  (scope, persona, frontier, reserved_at))

    def delivered(self, event, entries):
        reply = event.get('reply', {})
        if reply.get('delivered') is not True or not reply.get('message_id') or not isinstance(reply.get('text'), str):
            return
        markers = {t['marker'] for e in entries if e['scope'] == event['scope'] and e['persona'] == event['persona']
                   and e['category'] == 'expression_style' and e['active'] and e['expires'] > time.time()
                   for t in e['value']['traits'] if t.get('marker') and t['marker'] in reply['text']}
        with self.store.transaction():
            self.store.db.execute('INSERT OR IGNORE INTO expression_deliveries VALUES(?,?,?,?,?)',
                (event['scope'], event['persona'], str(reply['message_id']), event['received_at'], encode(sorted(markers))))

    def recent(self):
        result = {}
        cutoff = time.time() - 3600
        with self.store.lock:
            rows = self.store.db.execute('''SELECT scope,persona,markers,created FROM (
                SELECT *,row_number() OVER(PARTITION BY scope,persona ORDER BY created DESC) AS n
                FROM expression_deliveries WHERE created>?) WHERE n<=2''', (cutoff,))
            for row in rows:
                entry = result.setdefault((row[0], row[1]), {'markers': set(), 'examples': set(), 'expires': 0})
                entry['expires'] = max(entry['expires'], row[3] + 3600)
                entry['markers'].update(json.loads(row[2]))
            # Provider evidence only controls prompt diversity; it does not claim adoption.
            rows = self.store.db.execute('''SELECT scope,persona,body,created FROM (
                SELECT *,row_number() OVER(PARTITION BY scope,persona ORDER BY created DESC) AS n
                FROM scene_receipts WHERE stage='provider' AND created>?) WHERE n<=2''', (cutoff,))
            for row in rows:
                entry = result.setdefault((row[0], row[1]), {'markers': set(), 'examples': set(), 'expires': 0})
                entry['expires'] = max(entry['expires'], row[3] + 3600)
                for ref in json.loads(row[2]).get('refs', []):
                    entry['examples'].update(ref.get('example_ids', []))
        return result

    def purge(self, scope):
        for table in ('expression_analysis', 'expression_deliveries', 'expression_source_counts'):
            self.store.db.execute('DELETE FROM ' + table + ' WHERE scope=?', (scope,))

    def prune(self, days):
        with self.store.transaction():
            self.store.db.execute('DELETE FROM expression_deliveries WHERE created<?', (time.time() - days * 86400,))
