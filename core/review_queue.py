"""Indexed, paged administrative queries on a separate WAL reader connection."""
import json
import sqlite3
import time
from .contract import CATEGORIES,MEMBER_PROFILES

STALE={'policy_invalid','persona_changed','expired','revision_changed','baseline_changed'}
REASONS={'policy_invalid':'资料不符合当前固定人格规则','persona_changed':'角色资料已更新，请基于新回合重新生成提案',
         'expired':'提案证据有效期已过','revision_changed':'学习项版本已变化，请重新提案',
         'baseline_changed':'决策基线已变化或尚未确认','held':'由批量任务保留；自动审查不会接手',
         'automatic':'由系统审查或自动生效，无需人工批准','llm_required':'等待 LLM 审查通过','ready':''}

SQL='''WITH base AS (
 SELECT p.id,p.scope,p.persona,p.state,p.created,p.updated,
 json_extract(p.body,'$.days') AS days,json_extract(p.body,'$.profile_policy') AS profile_policy,
 json_extract(p.body,'$.baseline') AS baseline,json_extract(p.body,'$.host_version') AS host_version,
 json_extract(p.body,'$.category') AS category,json_extract(p.body,'$.member') AS member,
 json_extract(p.body,'$.mode') AS original_mode,json_extract(p.body,'$.entry_id') AS entry_id,
 json_extract(p.body,'$.auto_guard_reason') AS auto_guard_reason,
 json_type(p.body,'$.value') AS value_type,
 json_extract(p.body,'$.base_revision') AS base_revision,coalesce(json_extract(e.body,'$.revision'),0) AS current_revision,
 CASE
  WHEN json_extract(p.body,'$.category')='typing_habit' AND json_type(p.body,'$.value')!='object' AND m.value='llm' THEN 'llm_then_human'
  WHEN json_extract(p.body,'$.category')='affection' AND m.value='auto' AND coalesce(json_extract(p.body,'$.auto_guard_reason'),'')!='' THEN 'llm_then_human'
  ELSE m.value
 END AS mode,c.job AS claim,j.status AS job_status,s.value AS current_persona,
 b.baseline AS current_baseline,b.host AS current_host,
 coalesce((SELECT json_extract(r.value,'$.verdict')='approve' FROM json_each(p.body,'$.reviews') r
 WHERE json_extract(r.value,'$.actor') LIKE 'llm:%' ORDER BY CAST(r.key AS INTEGER) DESC LIMIT 1),0) AS approved
 FROM proposals p
 LEFT JOIN entries e ON e.id=json_extract(p.body,'$.entry_id')
 LEFT JOIN review_claims c ON c.proposal=p.id LEFT JOIN review_batches j ON j.id=c.job
 LEFT JOIN json_each(:modes) m ON m.key=json_extract(p.body,'$.category')
 LEFT JOIN json_each(:personas) s ON s.key=p.scope
 LEFT JOIN (SELECT json_extract(value,'$.scope') scope,json_extract(value,'$.persona') persona,
 json_extract(value,'$.baseline') baseline,json_extract(value,'$.host') host FROM json_each(:latest)) b ON b.scope=p.scope AND b.persona=p.persona
 WHERE p.state IN ('pending','reviewed') AND p.scope IN (SELECT value FROM json_each(:scopes))
 AND (:scope='' OR p.scope=:scope) AND (:category='' OR json_extract(p.body,'$.category')=:category)
 AND (:state='' OR p.state=:state)
 AND (:query='' OR instr(lower(p.scope||' '||coalesce(json_extract(p.body,'$.member'),'')||' '||
 coalesce(json_extract(p.body,'$.key'),'')||' '||coalesce(json_extract(p.body,'$.value'),'')||' '||
 coalesce(json_extract(p.body,'$.reason'),'')),lower(:query))>0 OR EXISTS
 (SELECT 1 FROM json_each(:names) n WHERE json_extract(n.value,'$.scope')=p.scope AND json_extract(n.value,'$.member')=json_extract(p.body,'$.member')))
), checked AS MATERIALIZED (
 SELECT *,CASE
 WHEN category NOT IN (SELECT value FROM json_each(:categories)) OR
 (category IN (SELECT value FROM json_each(:profiles)) AND (coalesce(member,'')='' OR coalesce(profile_policy,0)!=1)) THEN 'policy_invalid'
 WHEN current_persona IS NOT NULL AND current_persona!=persona THEN 'persona_changed'
 WHEN created+days*86400<=:now THEN 'expired'
 WHEN base_revision!=current_revision THEN 'revision_changed'
 WHEN category='participation' AND (current_baseline IS NULL OR current_host IS NULL OR current_baseline!=baseline OR current_host!=host_version) THEN 'baseline_changed'
 ELSE '' END AS stale_reason FROM base
), flags AS MATERIALIZED (
 SELECT *,CASE WHEN stale_reason!='' THEN stale_reason WHEN claim IS NOT NULL THEN 'held'
 WHEN mode='auto' OR original_mode='auto' OR (mode='llm' AND category IN (SELECT value FROM json_each(:profiles))) THEN 'automatic'
 WHEN NOT approved AND (mode!='human' OR original_mode IN ('llm','llm_then_human')) THEN 'llm_required' ELSE 'ready' END AS reason,
 (stale_reason='' AND claim IS NULL AND mode IN ('llm','llm_then_human') AND original_mode!='auto' AND NOT approved) AS can_llm
 FROM checked
) '''


def query_queue(store,scopes,modes,names,personas,latest,*,scope='',query='',category='',state='',view='ready',reason='',page=1):
    if scope and scope not in scopes:
        raise ValueError('scope not enabled')
    if category and category not in CATEGORIES or state not in ('','pending','reviewed') or view not in ('all','ready','stale','held'):
        raise ValueError('invalid pending filter')
    if reason and reason not in REASONS:
        raise ValueError('invalid reason')
    page=max(1,min(10000,int(page)))
    matches=[{'scope':s,'member':m} for (s,m),name in names.items() if query and query.casefold() in name.casefold()]
    args=dict(scopes=json.dumps(scopes),scope=scope,category=category,state=state,query=query,names=json.dumps(matches),
              modes=json.dumps(modes),personas=json.dumps(personas),latest=json.dumps(latest),
              categories=json.dumps(sorted(CATEGORIES)),profiles=json.dumps(sorted(MEMBER_PROFILES)),now=time.time(),reason=reason)
    restriction={'all':'1','ready':"reason='ready'",'stale':"stale_reason!=''",'held':"claim IS NOT NULL"}[view]
    restriction+=" AND (:reason='' OR reason=:reason)"
    db=sqlite3.connect(store.path.resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    try:
        db.execute('PRAGMA busy_timeout=5000')
        db.execute('BEGIN')
        # Classify once into a connection-local temp table. The production DB remains read-only.
        db.execute('CREATE TEMP TABLE queue_result AS '+SQL+'SELECT * FROM flags',args)
        groups=[dict(r) for r in db.execute("SELECT scope,count(*) AS count FROM proposals WHERE state IN ('pending','reviewed') AND scope IN (SELECT value FROM json_each(?)) GROUP BY scope ORDER BY min(created),scope",(args['scopes'],))]
        if not scope and not query and not category and not state:
            per_scope={r['scope']:dict(r) for r in db.execute("SELECT scope,coalesce(sum(reason='ready'),0) AS ready_count,coalesce(sum(stale_reason!=''),0) AS stale_count,coalesce(sum(claim IS NOT NULL),0) AS held_count FROM queue_result GROUP BY scope")}
            for g in groups:
                g.update(per_scope.get(g['scope'],{'ready_count':0,'stale_count':0,'held_count':0}))
        counts=db.execute("SELECT count(*) AS pending_total,coalesce(sum(reason='ready'),0) ready_total,coalesce(sum(stale_reason!=''),0) stale_total,coalesce(sum(claim IS NOT NULL),0) held_total,coalesce(sum("+restriction+") ,0) total FROM queue_result",args).fetchone()
        reasons={r[0]:r[1] for r in db.execute('SELECT reason,count(*) FROM queue_result GROUP BY reason',args)}
        page=min(page,max(1,(counts['total']+49)//50))
        args.update(offset=(page-1)*50)
        rows=db.execute("SELECT json_remove(p.body,'$.evidence') body,json_array_length(p.body,'$.evidence') evidence_count,reason,stale_reason,claim,job_status,can_llm FROM queue_result f JOIN proposals p ON p.id=f.id WHERE "+restriction+' ORDER BY f.created,f.id LIMIT 50 OFFSET :offset',args).fetchall()
        output=[]
        for r in rows:
            p=json.loads(r['body'])
            p['evidence']=[]
            summary=store.console.summary(p)
            summary.update(evidence_count=r['evidence_count'] or 0,can_approve=r['reason']=='ready',can_llm_review=bool(r['can_llm']),
                           can_reject=True,can_archive=bool(r['stale_reason']),can_release=bool(r['claim']) and r['job_status'] not in {'queued','running','pausing','paused','cancelling'},
                           reason_code=r['reason'],blocked_reason=REASONS[r['reason']],
                           display_name=names.get((p['scope'],p['member']),p['member'] or '全群'))
            output.append(summary)
        return {**dict(counts),'proposals':output,'page':page,'page_size':50,'groups':groups,'reasons':reasons}
    finally:
        db.close()
