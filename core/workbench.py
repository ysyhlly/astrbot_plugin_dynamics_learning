"""Version history, stale-record governance and explicit release of review holds."""
from contextlib import closing
import json
import sqlite3
import time
from .contract import policy_allowed


def stale_reason(engine,p):
    if not policy_allowed(p):
        return 'policy_invalid'
    if engine.current_personas.get(p['scope'],p['persona'])!=p['persona']:
        return 'persona_changed'
    if p['created']+p['days']*86400<=time.time():
        return 'expired'
    with engine.store.lock:
        row=engine.store.db.execute("SELECT json_extract(body,'$.revision') FROM entries WHERE id=?",(p['entry_id'],)).fetchone()
    if (row[0] if row else 0)!=p['base_revision']:
        return 'revision_changed'
    if p['category']=='participation' and engine.latest.get((p['scope'],p['persona']))!=(p['baseline'],p['host_version']):
        return 'baseline_changed'
    return ''


def govern(engine,scope,rows,action,actor):
    store=engine.store
    with store.transaction():
        for original in rows:
            p=store.get(original['id'])
            if p['updated']!=original['updated'] or p['scope']!=scope or p['state'] not in {'pending','reviewed'}:
                raise ValueError('记录已变化，请刷新后重试')
            if action=='archive':
                reason=stale_reason(engine,p)
                if not reason:
                    raise ValueError('仅能归档已失效的提案')
                p.update(previous_state=p['state'],state='archived',archive_reason=reason,archived_at=time.time(),archived_by=actor)
                store.db.execute('DELETE FROM review_claims WHERE proposal=?',(p['id'],))
            elif action=='release':
                claim=store.db.execute('SELECT c.job,j.status FROM review_claims c LEFT JOIN review_batches j ON j.id=c.job WHERE c.proposal=?',(p['id'],)).fetchone()
                if not claim or claim[1] in {'queued','running','pausing','paused','cancelling'}:
                    raise ValueError('请先取消原批次，再将条目交回自动审查')
                reason='operator_released_review_hold'
                store.db.execute('DELETE FROM review_claims WHERE proposal=?',(p['id'],))
            else:
                raise ValueError('invalid governance action')
            store._save(p)
            store._audit(p,action,actor,{'reason':reason})
    return {'count':len(rows),'action':action}


def archive_stale_scope(engine, scope, actor):
    """Archive all stale pending/reviewed proposals within a single group scope."""
    result = engine.store.archive_stale_proposals(
        current_personas=dict(engine.current_personas),
        latest_baselines=dict(engine.latest),
        actor=actor,
        scope=scope,
        include_policy_invalid=True,
        include_revision_changed=True,
        require_known_baseline=False,
    )
    return {
        'scope': scope,
        'action': 'archive_stale',
        'count': result['count'],
        'archived': result['archived'],
        'reasons': result['reasons'],
    }


def library(store,scopes,scope='',state='',category='',query='',page=1):
    if scope and scope not in scopes or state not in ('','pending','reviewed','applied','rejected','rolled_back','archived'):
        raise ValueError('invalid history filter')
    page=max(1,min(10000,int(page)))
    condition="p.scope IN (SELECT value FROM json_each(?)) AND (?='' OR p.scope=?) AND (?='' OR p.state=?) AND (?='' OR json_extract(p.body,'$.category')=?) AND (?='' OR instr(lower(json_extract(p.body,'$.member')||' '||json_extract(p.body,'$.key')||' '||json_extract(p.body,'$.value')),lower(?))>0)"
    args=(json.dumps(scopes),scope,scope,state,state,category,category,query,query)
    with closing(sqlite3.connect(store.path)) as db:
        db.execute('PRAGMA busy_timeout=5000')
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        total=db.execute('SELECT count(*) FROM proposals p WHERE '+condition,args).fetchone()[0]
        page=min(page,max(1,(total+24)//25))
        records=db.execute("SELECT json_remove(p.body,'$.evidence'),json_array_length(p.body,'$.evidence') FROM proposals p WHERE "+condition+' ORDER BY p.created DESC,p.id LIMIT 25 OFFSET ?',(*args,(page-1)*25)).fetchall()
        rows=[]
        for body,evidence_count in records:
            p=json.loads(body)
            p['evidence']=[]
            row=store.console.summary(p)
            row.update(evidence_count=evidence_count,archive_reason=p.get('archive_reason',''))
            rows.append(row)
        return {'total':total,'page':page,'page_size':25,'records':rows}


def detail(store,scopes,proposal_id='',entry_id=''):
    with store.lock:
        p=store.get(proposal_id) if proposal_id and not proposal_id.startswith('legacy:') else None
        key=p['entry_id'] if p else entry_id
        record=store.db.execute('SELECT body FROM entries WHERE id=?',(key,)).fetchone()
        entry=json.loads(record[0]) if record else None
        subject=p or entry
        if not subject or subject['scope'] not in scopes:
            raise ValueError('资料不存在或会话已停用')
        history=[json.loads(r[0]) for r in store.db.execute("SELECT body FROM proposals WHERE json_extract(body,'$.entry_id')=? ORDER BY created DESC LIMIT 50",(key,))]
        audit=[{'action':r[0],'actor':r[1],'created':r[2],'body':json.loads(r[3]),'proposal_id':r[4]} for r in store.db.execute("SELECT a.action,a.actor,a.created,a.body,a.proposal FROM audit a LEFT JOIN proposals p ON p.id=a.proposal WHERE json_extract(p.body,'$.entry_id')=? OR a.proposal=? ORDER BY a.id DESC LIMIT 100",(key,key))]
        usage=[{'id':r[0],'created':r[1],'turn_id':r[2]} for r in store.db.execute("SELECT DISTINCT u.id,u.created,u.turn_id FROM proposals p JOIN reply_usage_refs f ON f.source_id=p.id JOIN reply_usage u ON u.id=f.receipt WHERE u.scope=? AND json_extract(p.body,'$.entry_id')=? ORDER BY u.created DESC LIMIT 20",(subject['scope'],key))]
        return {'proposal':p,'entry':entry,'versions':[store.console.summary(r) for r in history],
                'timeline':audit,'usage':usage,'claim':store.review_claim(p['id']) if p else None,
                'can_rollback':bool(p and entry and p['state']=='applied' and entry['revision']==p.get('applied_revision') and entry.get('proposal_id')==p['id']),
                'legacy_readonly':p is None}
