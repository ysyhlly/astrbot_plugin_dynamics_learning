import {msg,tr,setText,feedback,getSharedScope,setSharedScope} from './series-ui.js';
import {installWorkbench} from './workbench.js?v=0.12.2';
let workbench=null,network=null,networkPromise=null,focusedReviewIndex=-1;const epochs={};function ticket(key){const n=epochs[key]=(epochs[key]||0)+1;return ()=>epochs[key]===n;}
const $ = s => document.querySelector(s);
const formatDate = (seconds) => new Date(seconds*1000).toLocaleString(document.documentElement.lang);
const formatTime = (seconds) => new Date(seconds*1000).toLocaleTimeString(document.documentElement.lang);
const SUB_SCOPE_SELECTORS=['#entries-scope','#social-scope','#profile-scope','#usage-scope','#preview-scope','#library-scope','#history-scope'];
window.DynamicsGetActiveScope=()=>($('#global-scope')?.value||'').trim();
const categories = {expression_style:"本群表达习惯",expression_adjustment:"表达反馈调整",scene_pattern:"群内接话场景",scene_adjustment:"接话反馈调整",expression_example:msg("原话样例"),style:msg("群友表达方式"),typing_habit:msg("群友打字习惯"),preference:msg("群友喜恶"),jargon:msg("群聊黑话"),affection:msg("数值好感度"),relationship:msg("社交关系"),state:msg("短期状态"),persona:msg("旧人格增量（已停用）"),decision_guidance:msg("决策指导"),participation:msg("接话阈值")};
const states = {pending:msg("待审查"),reviewed:msg("初审通过"),applied:msg("已应用"),rejected:msg("已拒绝"),rolled_back:msg("已回滚"),archived:msg("已归档")};
const modes = {auto:'自动生效（无需审查）',human:'人工审查',llm:'LLM 审查通过后自动生效',llm_then_human:'LLM 初审＋人工确认'};
let snapshot = {proposals:[],entries:[]}, editing = null;
let socialSnapshot=null;let activeSection=null;let entryPage=1,entryTotal=0,usagePage=1,usageTotal=0,memberPage=1,memberTotal=0;const socialPages={affection:1,jargon:1};let socialQuery='';
let reviewGroupKey=null,reviewPage=1,reviewTotal=0,reviewProposals=[],reviewEpoch=0;
let globalScope=null,scopeControlsReady=false;
const reviewSelected=new Set();
let pendingBatch=null;
let pendingBatchAction="approve",batchJobsEpoch=0;
let reviewView="ready";
let queueData=null,reviewBusy=false,reviewFailure='',reviewTimer;
const reviewSelectionRows=new Map();
const groupReview=()=>Boolean(snapshot.review_queue_version);
const batchActions=()=>Number(snapshot.review_queue_version)>=2;
const actionAllowed=(p,action)=>action==='archive'?p.can_archive===true:action==='release'?p.can_release===true:action==='approve'?canApprove(p):action==='review'?p.can_llm_review===true:p.can_reject===true;
const canSelect=p=>batchActions()?(p.can_approve||p.can_llm_review||p.can_reject):canApprove(p);
const batchLimit=()=>groupReview()?100:20;
function clearReviewSelection(){reviewSelected.clear();reviewSelectionRows.clear();}
function queueGroups(){const groups=new Map();for(const g of snapshot.review_groups||[]){if(!g.scope.includes(':GroupMessage:'))continue;groups.set(g.scope,(groups.get(g.scope)||0)+g.count);}return queueData?.groups||[...groups].map(([scope,count])=>({scope,count}));}
function reviewGroups(){return groupReview()?queueGroups():snapshot.review_groups||[];}
function reviewKey(g){return groupReview()?g.scope:groupKey(g);}
function updateReviewFilters(){clearReviewSelection();reviewPage=1;clearTimeout(reviewTimer);if(groupReview()){++reviewEpoch;reviewBusy=true;reviewProposals=[];renderReviewCards();reviewTimer=setTimeout(()=>fetchReviewGroup().then(renderReviewCards).catch(e=>note(e.message,'error')),250);}else renderReviewCards();}

const groupKey=p=>JSON.stringify([p.scope,p.persona,p.member||'']);
function canApprove(p){if(groupReview())return p.can_approve===true;const approved=[...p.reviews].reverse().find(r=>r.actor.startsWith('llm:'))?.verdict==='approve';const mode=snapshot.review_modes[p.category]||p.mode;return ['pending','reviewed'].includes(p.state)&&mode!=='auto'&&!(mode==='llm'&&['style','typing_habit','preference','expression_example'].includes(p.category))&&(p.mode==='human'||approved)&&(mode==='human'||approved);}
function shortScope(scope){return scope.replace(':GroupMessage:',' · 群 ').replace(':FriendMessage:',' · 私聊 ');}
function visibleReviews(){
 if(groupReview())return reviewProposals;
 const query=$('#review-query').value.trim().toLowerCase();
 return reviewProposals.filter(p=>(reviewView==='all'||canApprove(p))&&(!$('#state').value||p.state===$('#state').value)&&(!$('#review-category').value||p.category===$('#review-category').value)&&(!query||[p.key,p.member,p.reason,displayValue(p.value)].join(' ').toLowerCase().includes(query)))
  .sort((a,b)=>Number(canApprove(b))-Number(canApprove(a)));
}
function selectionStatus(){
 const eligible=visibleReviews().filter(canApprove), selected=reviewSelected.size;
 $('#preview-batch').textContent=selected?`批准所选（${selected}）`:`批准本页（${Math.min(batchLimit(),eligible.length)}）`;
 $('#preview-batch').disabled=reviewBusy||!reviewGroupKey||(!selected&&!eligible.length);
 $('#select-review-page').disabled=reviewBusy||!reviewGroupKey||!visibleReviews().some(canSelect);
 $('#select-review-page').textContent=batchActions()?'选择本页待审项':'选择本页可批准项';
 $('#select-review-page').hidden=!reviewGroupKey;
 $('.review-bottom').hidden=!reviewGroupKey;
 $('.review-bottom').dataset.hasSelection=String(selected>0);
 $('#clear-review-selection').hidden=!selected;
 $('#selection-note').textContent=selected?`已选择 ${selected} 项${groupReview()?'（可跨页）':''}`:`批准前核对变更，每次最多 ${batchLimit()} 项`;
 const candidates=selected?[...reviewSelectionRows.values()]:visibleReviews();
 for(const [action,id,label] of [['review','preview-batch-review','LLM 审查'],['reject','preview-batch-reject','拒绝'],['archive','preview-batch-archive','归档'],['release','preview-batch-release','交回自动']]){const b=$('#'+id),count=candidates.filter(p=>actionAllowed(p,action)).length;b.hidden=!batchActions()||(action==='archive'&&reviewView!=='stale')||(action==='release'&&reviewView!=='held');b.disabled=reviewBusy||!reviewGroupKey||!count;b.textContent=`${label}${selected?'所选':'本页'}（${count}）`;}
 if(batchActions()){const count=candidates.filter(canApprove).length;$('#preview-batch').disabled=reviewBusy||!reviewGroupKey||!count;$('#preview-batch').textContent=`批准${selected?'所选':'本页'}（${count}）`;}
 for(const c of document.querySelectorAll('.review-item'))c.classList.toggle('is-selected',reviewSelected.has(c.dataset.proposalId));
}
function renderReviewCards(){
 const groups=reviewGroups(),chooser=$('#review-group');chooser.replaceChildren();
 const recent=el('option','',groupReview()?'全部群 · 待审内容':'最近记录');recent.value='';chooser.append(recent);
 for(const group of groups){const option=el('option','',groupReview()?`${shortScope(group.scope)} (${group.count})`:`${shortScope(group.scope)} / ${group.display_name||group.member||'全群'} (${group.count})`);option.value=reviewKey(group);chooser.append(option);}if(groupReview()&&reviewGroupKey&&!groups.some(g=>g.scope===reviewGroupKey)){const o=el('option','',shortScope(reviewGroupKey)+' (0)');o.value=reviewGroupKey;chooser.append(o);}chooser.value=reviewGroupKey||'';
 const category=$('#review-category').value;$('#review-category').replaceChildren(...[['','全部类别'],...Object.entries(categories)].map(([key,label])=>{const option=el('option','',msg(label));option.value=key;return option;}));$('#review-category').value=category;
 const filters=Number(Boolean(category))+Number(Boolean($('#state').value))+Number(Boolean($('#review-reason').value));setText($('.review-filter-menu>summary'),filters?msg('筛选 · {0}',filters):msg('筛选'));
 const ready=reviewProposals.filter(canApprove).length;
 setText($('#review-ready'),msg(groupReview()?'可批准 {0}':'本页可批准 {0}',groupReview()?queueData?.ready_total??0:ready));
 setText($('#review-stale'),msg('失效治理 {0}',queueData?.stale_total??0));setText($('#review-held'),msg('人工保留 {0}',queueData?.held_total??0));
 setText($('#review-all'),msg(groupReview()?'全部待审 {0}':'本页全部 {0}',groupReview()?queueData?.pending_total??0:reviewProposals.length));
 for(const b of document.querySelectorAll('[data-review-view]'))b.setAttribute('aria-pressed',String(b.dataset.reviewView===reviewView));
 $('#review-summary').textContent=groupReview()?`${reviewGroupKey?shortScope(reviewGroupKey):'全部群'} · 匹配 ${reviewTotal} 项 · 第 ${reviewPage} / ${Math.max(1,Math.ceil(reviewTotal/50))} 页${!reviewGroupKey?' · 选择单个群可批量批准':''}`:reviewGroupKey?`本组 ${reviewTotal} 项 · 第 ${reviewPage} / ${Math.max(1,Math.ceil(reviewTotal/30))} 页`:`最近 ${reviewProposals.length} 条记录`;
 $('#review-prev').disabled=reviewBusy||(!groupReview()&&!reviewGroupKey)||reviewPage<=1;$('#review-next').disabled=reviewBusy||(!groupReview()&&!reviewGroupKey)||reviewPage*(groupReview()?50:30)>=reviewTotal;
 $('.review-pagination').hidden=(!groupReview()&&!reviewGroupKey)||reviewTotal<=(groupReview()?50:30);
 const index=groups.findIndex(g=>reviewKey(g)===reviewGroupKey);
 $('#review-next-group').disabled=!groups.length||(groups.length===1&&index===0);
 $('#review-next-group').hidden=groups.length<2;
 const cards=$('#cards');cards.replaceChildren();cards.setAttribute('aria-busy',String(reviewBusy));if(groupReview()&&(reviewBusy||reviewFailure)){cards.append(el('div','review-empty',reviewBusy?'正在查询全部待审内容…':reviewFailure));selectionStatus();return;}
 for(const p of visibleReviews()){
  const mode=snapshot.review_modes[p.category]||p.mode,ready=canApprove(p),pending=['pending','reviewed'].includes(p.state);
  const c=el('article','review-item'),top=el('div','review-item-head'),head=el('div','review-title');c.dataset.proposalId=p.id;c.tabIndex=0;
  if(reviewGroupKey&&canSelect(p)){const check=el('input');check.type='checkbox';check.dataset.proposal=p.id;check.checked=reviewSelected.has(p.id);check.setAttribute('aria-label','选择调整 '+p.key);check.onchange=()=>{if(check.checked&&reviewSelected.size>=batchLimit()){check.checked=false;note(`每次最多选择 ${batchLimit()} 条`);return;}if(check.checked){reviewSelected.add(p.id);reviewSelectionRows.set(p.id,p);}else{reviewSelected.delete(p.id);reviewSelectionRows.delete(p.id);}selectionStatus();};top.append(check);}
  head.append(el('span','review-category',categories[p.category]||p.category),el('h3','',p.key));top.append(head);
  const status=p.can_archive?'已失效':p.reason_code==='held'?'批次保留':ready?'待确认':p.state==='reviewed'&&mode==='llm'?'已通过 · 自动生效中':pending&&mode!=='human'?'等待 LLM 审查':states[p.state];
  top.append(el('span','badge',status));c.append(top);
  const body=el('div','review-item-body'),valWrap=el('div','review-val-col'),value=el('div','review-value',displayValue(p.value));valWrap.append(value);
  if(p.before?.active){const idiff=el('div','review-inline-diff');const oldText=String(displayValue(p.before.value)||'').replace(/\s+/g,' ').slice(0,110);const newText=String(displayValue(p.value)||'').replace(/\s+/g,' ').slice(0,110);idiff.append(el('div','diff-del','修改前：'+oldText),el('div','diff-add','修改后：'+newText));valWrap.append(idiff);}
  body.append(valWrap);
  const actions=el('div','review-row-actions');
  const evidence=proposalDetails(p);const original=button(msg('查看原文'),()=>{detail.open=true;evidence.open=true;evidence.scrollIntoView({block:'nearest'});});original.classList.add('series-open-source');actions.append(original);
  if(ready){const btnApp=button(msg("批准并应用"),()=>act(p,'approve'),'primary');btnApp.dataset.quickAction='approve';actions.append(btnApp);}
  if(pending){const btnRej=button(msg("拒绝"),()=>act(p,'reject'));btnRej.dataset.quickAction='reject';actions.append(btnRej);}
  if(p.state==='applied')actions.append(button(msg("回滚本次调整"),()=>act(p,'rollback'),'danger'));
  body.append(actions);c.append(body);
  const count=p.evidence_count??new Set((p.evidence||[]).map(e=>e.turn_id)).size;
  c.append(el('p','review-meta',`${groupReview()?shortScope(p.scope)+' · '+(p.display_name||p.member||'全群')+' · 人格 '+p.persona.slice(0,12):p.member||'全群'} · ${count} 条证据 · ${p.category==='affection'?'长期保存':p.days+' 天有效'}${mode==='llm'?' · LLM 审查通过后自动生效':''}`));
  if(groupReview()&&!ready&&p.blocked_reason)c.append(el('p','review-meta',p.blocked_reason));
  const detail=el('details','review-detail');detail.append(el('summary','',msg('查看依据与修改')));
  detail.append(el('p','',p.reason),el('p','scope',scopeLine(p)),el('p','muted',modes[mode]));
  const diff=el('div','diff');for(const [label,value] of [['调整前',p.before?.active?p.before.value:'尚无学习增量'],['调整后',p.value]]){const d=el('div');d.append(el('small','',label),el('span','',displayValue(value)));diff.append(d);}detail.append(diff);
  const reviews=el('div','review-verdicts');for(const r of p.reviews||[])reviews.append(el('p','muted',`${r.actor?.startsWith('llm:')?'LLM':'审查人'}：${{approve:'通过',reject:'拒绝',revise:'需修改'}[r.verdict]||r.verdict||''} · ${r.reason||'未提供说明'}`));detail.append(reviews,evidence);
  if(p.evaluation?.note)detail.append(el('p','muted',p.evaluation.note));
  if(p.observations)detail.append(details('应用前后观察',p.observations));
  if(p.applied_by)detail.append(el('p','muted',`应用人 ${p.applied_by}${p.rolled_back_by?' · 回滚人 '+p.rolled_back_by:''}`));
  const secondary=el('div','actions');
  if(pending){
   if(p.mode!=='auto'&&(p.mode!=='human'||mode!=='human'))secondary.append(button(msg("LLM 审查"),async()=>{await api('apiPost','review',{id:p.id});note('审查任务已开始，完成后刷新查看结果。');}));
   actions.append(button(msg("修改提案"),()=>{editing=p;$('#edit-value').value=p.category==='affection'?JSON.stringify({experience_delta:p.value.experience_delta,tone:p.value.tone??0},null,2):(typeof p.value==='object'?JSON.stringify(p.value,null,2):p.value);$('#edit-reason').value='';$('#edit').showModal();}));
  }
  detail.append(secondary);c.append(detail);cards.append(c);
 }
 if(!cards.children.length){const empty=el('div','review-empty');empty.append(el('strong','',reviewView==='ready'?(groupReview()?'当前筛选没有可批准的调整':'本页没有需要确认的调整'):'没有匹配的记录'),el('p','muted',reviewView==='ready'?(groupReview()?'已查询所选范围的全部待审内容，可切换全部待审或调整筛选。':'自动审查和自动生效的内容无需手动批准。可切换到本页全部，或继续查看下一页、下一组。'):'尝试清空搜索或筛选条件。'));cards.append(empty);}
 selectionStatus();
}

const displayValue=v=>{
 if(!v||typeof v!=='object')return v;
 if(v.kind==='expression_example')return `场景：${v.situation}\n成员原话：${v.expression}\n关联依据：${v.link==='reply'?'明确回复':'相邻消息，须核对语境'}`;
 if(v.kind==='observed_typing'){if(!v.sample_count)return '暂无可统计原始消息';const pct=k=>`${(100*v[k]).toFixed(1)}%`;return `样本 ${v.sample_count} 条 · 平均 ${v.mean_chars} 字\n短句 ${pct('short_rate')} · 多行 ${pct('multiline_rate')} · 标点 ${pct('punctuation_rate')}\n表情 ${pct('emoji_rate')} · 连续补充 ${pct('consecutive_rate')}\n${v.definitions}`;}
 if(v.stance)return `${{like:'喜欢',dislike:'反感',boundary:'交流边界'}[v.stance]}：${v.topic}\n条件：${v.condition||'未限定'}\n原话：${v.quote}`;
 if('affection_level' in v)return `好感 ${Number(v.affection_level||0).toFixed(4)} · 经验变化 ${v.experience_delta??'历史记录'}`;
 return JSON.stringify(v,null,2);
};
const el = (tag, cls, value) => {const e=document.createElement(tag);if(cls)e.className=cls;if(value!==undefined)setText(e,typeof value==='string'&&['badge','review-category'].includes(cls)?msg(value):value);return e;};
const note = (value,state='info') => {feedback($('#notice'),value,state);};
async function api(method,path,body){const bridge=window.AstrBotPluginPage;if(!bridge)throw Error('请从 AstrBot 插件详情打开此页面。');const r=await bridge[method](path,body||{});if(r.status==='error'||r.ok===false)throw Error(r.message||r.error||'请求失败');return r.data??r;}
function button(label,fn,cls=''){const b=el('button',cls,label);b.type='button';b.addEventListener('click',async()=>{if(b.disabled)return;b.disabled=true;b.setAttribute('aria-busy','true');try{await fn();}catch(e){note(e.message,'error');}finally{b.disabled=false;b.removeAttribute('aria-busy');}});return b;}
function details(label,value){const d=el('details');d.append(el('summary','',label),el('pre','',typeof value==='string'?value:JSON.stringify(value,null,2)));return d;}
function scopeLine(p){return `${p.scope} · 人格 ${p.persona.slice(0,12)} · ${p.member?'成员 '+p.member:'当前会话'}`;}
function evidenceBody(evidence){
 const box=el('div','series-evidence');
 const rows=Array.isArray(evidence)?evidence:[];
 for(const row of rows){if(typeof row?.text!=='string')continue;const card=el('article','series-evidence-item');card.append(el('small','scope',`${row.member||row.actor_id||'—'} · ${row.timestamp?formatDate(row.timestamp):row.message_id||row.turn_id||'—'}`),el('pre','series-original-text',row.text));box.append(card);}
 box.append(details(msg('完整证据结构'),evidence));return box;
}
function proposalDetails(p){
 const d=el('details','series-source-jump');d.append(el('summary','',msg('查看原始证据')));
 const body=el('div','series-evidence-body');d.append(body);
 if(!p.details_lazy){body.append(evidenceBody(p.evidence));return d;}
 body.append(el('p','muted',msg('展开后加载')));let loaded=false,loading=false;
 d.addEventListener('toggle',async()=>{if(!d.open||loaded||loading)return;loading=true;body.setAttribute('aria-busy','true');try{const full=await api('apiPost','proposal_detail',{id:p.id});body.replaceChildren(evidenceBody(full.evidence),details(msg('完整审查意见'),full.reviews),details(msg('应用前后观察'),full.observations));loaded=true;}catch(e){body.replaceChildren(el('p','',e.message));}finally{loading=false;body.removeAttribute('aria-busy');}});return d;
}
function renderOverviewStats() {
 const counts = snapshot.proposal_counts;
 const metrics = [
  ['待审查', counts ? (counts.pending||0)+(counts.reviewed||0) : snapshot.proposals.filter(p=>['pending','reviewed'].includes(p.state)).length],
  ['已应用', counts?.applied??snapshot.proposals.filter(p=>p.state==='applied').length],
  ['生效会话', snapshot.sessions.length],
  ['待分析回合', snapshot.batches.reduce((n,b)=>n+b.count,0)],
 ];
 $('#stats').replaceChildren(...metrics.map(metricCard));
}
function metricCard([label,value]) {
 const card = el('div','stat');
 card.append(el('strong','',value),el('span','',msg(label)));
 return card;
}
function render(){
 const previousScopes=scopeControlsReady?SUB_SCOPE_SELECTORS.map(selector=>[selector,$(selector)?.value]):[];
 workbench?.refreshScopes();
 renderOverviewStats();
 renderReviewCards();
 if(!snapshot.entries_paged)renderEntries(snapshot.entries.filter(e=>e.active));
 const scopeList=$('#scope-list');scopeList.replaceChildren();
 for(const s of snapshot.sessions){const c=el('article','card');c.append(el('h3','',s));for(const batch of snapshot.batches.filter(b=>b.scope===s)){const a=el('div','actions');a.append(el('span','muted',`人格 ${batch.persona.slice(0,12)} · ${batch.count} 个待分析回合`),button(msg("现在分析"),async()=>{await api('apiPost','analyze',{scope:s,persona:batch.persona});note('分析任务已开始，完成后刷新查看提案。');}));c.append(a);}c.append(button(msg("删除此会话学习数据"),async()=>{const confirmation=await window.DynamicsUI.prompt(msg('将删除此会话的事件、提案和学习内容。请输入完整会话标识确认：'),{danger:true});if(confirmation!==s)return;await api('apiPost','purge',{scope:s,confirm_scope:confirmation});await load();},'danger'));scopeList.append(c);}
 selectionStatus();
 if(!snapshot.sessions.length)scopeList.append(el('div','empty','尚未启用会话。请在插件配置的「生效会话」中填写完整 UMO。'));
 $('#modes').replaceChildren(...Object.entries(snapshot.review_modes).map(([k,v])=>el('p','muted',`${categories[k]}：${modes[v]}`)));
 renderHome();fillScope('#entries-scope',true);fillScope('#usage-scope');
 const selectedProfile=$('#profile-scope').value;$('#profile-scope').replaceChildren(...snapshot.sessions.filter(s=>s.includes(':GroupMessage:')).map(s=>{const o=el('option','',s);o.value=s;return o;}));if(snapshot.sessions.includes(selectedProfile))$('#profile-scope').value=selectedProfile;
 const historyScope=$('#history-scope').value;$('#history-scope').replaceChildren(...snapshot.sessions.map(s=>{const o=el('option','',s);o.value=s;return o;}));if(snapshot.sessions.includes(historyScope))$('#history-scope').value=historyScope;
 const previous=$('#social-scope').value;$('#social-scope').replaceChildren(...snapshot.sessions.filter(s=>s.includes(':GroupMessage:')).map(s=>{const o=el('option','',s);o.value=s;return o;}));if(snapshot.sessions.includes(previous))$('#social-scope').value=previous;
 const gSel=$('#global-scope');
 if(gSel){
  const cur=globalScope??getSharedScope()??'';
  gSel.replaceChildren();
  const allOpt=el('option','',msg('全部群'));allOpt.value='';gSel.append(allOpt);
  for(const s of snapshot.sessions){const o=el('option','',shortScope(s));o.value=s;gSel.append(o);}
  if(cur&&!snapshot.sessions.includes(cur)){const missing=el('option','',shortScope(cur)+' · '+msg('暂无资料'));missing.value=cur;gSel.append(missing);}gSel.value=cur;
 }
 for(const [selector,value] of previousScopes)if(value!==undefined)selectScope($(selector),value);
 scopeControlsReady=true;
}
function selectScope(node,scope){
 if(!node)return;
 if(![...node.options].some(o=>o.value===scope)){
  const option=el('option','',scope?shortScope(scope)+' · '+msg('暂无资料'):msg('请选择群聊'));
  option.value=scope;node.prepend(option);
 }
 node.value=scope;
}
function syncGlobalScopeToSubtabs(scope,triggerFetch=true){
 globalScope=scope||'';
 scopeControlsReady=true;
 selectScope($('#global-scope'),globalScope);
 if(getSharedScope()!==globalScope)setSharedScope(globalScope,{syncUrl:false,emit:false});
 for(const key of Object.keys(epochs))ticket(key);
 for(const sel of SUB_SCOPE_SELECTORS)selectScope($(sel),globalScope);
 if(groupReview())reviewGroupKey=globalScope;
 else{const group=reviewGroups().find(g=>g.scope===globalScope);reviewGroupKey=group?groupKey(group):'';}
 if(!triggerFetch)return;
 ++socialRequest;++memberRequest;++historyRequest;++reviewEpoch;++batchJobsEpoch;
 clearTimeout(jargonTimer);
 socialSnapshot=null;
 network?.setData({users:{},all_relations:[]});
 workbench?.invalidateScope();
 clearEntries();
 for(const selector of ['#profile-content','#member-results','#members-summary','#network-summary','#member-detail','#social-stats','#social-explanations','#usage-records','#usage-summary','#history-records','#history-summary','#batch-jobs'])$(selector)?.replaceChildren();
 for(const id of ['usage-prev','usage-next','history-prev','history-next','members-prev','members-next'])if($('#'+id))$('#'+id).disabled=true;
 for(const kind of ['affection','jargon'])clearSocialPage(kind);
 if(activeSection==='proposals'){reviewPage=1;fetchReviewGroup().then(renderReviewCards).catch(e=>note(e.message,'error'));}
 else if(activeSection==='entries'&&snapshot.entries_paged){loadEntries();}
 else if(activeSection==='social'){socialQuery='';socialPages.affection=socialPages.jargon=1;loadSocial();}
 else if(activeSection==='usage'){usagePage=1;loadUsage();}
 else if(activeSection==='history'){historyPage=1;loadHistory();}
 else if(activeSection==='library'){workbench?.loadLibrary();}
}
async function load(){
 try{
  snapshot=await api('apiGet','overview');
  if(activeSection===null)activate(snapshot.console_version?'home':'proposals',false);
  if(globalScope===null){
   if(groupReview())reviewView='all';
   const incoming=getSharedScope();
   if(incoming||new URLSearchParams(location.search).has('umo')){
    syncGlobalScopeToSubtabs(incoming,false);
   }else{
    globalScope='';
    const groups=reviewGroups();
    reviewGroupKey=groupReview()||!groups.length?'':groupKey(groups[0]);
   }
  }else if(!groupReview()&&reviewGroupKey&&!reviewGroups().some(g=>groupKey(g)===reviewGroupKey)){
   reviewGroupKey='';reviewPage=1;
  }
  if(activeSection==='proposals')await fetchReviewGroup();else reviewProposals=snapshot.proposals;
  render();
  note(snapshot.last_error?'最近任务：'+snapshot.last_error:snapshot.jobs.length?'正在处理：'+snapshot.jobs.length+' 个任务':`已连接 · 丢弃事件 ${snapshot.dropped} · 无效事件 ${snapshot.invalid}`);
 }catch(e){note(e.message,'error');}
}
async function act(p,action){await api('apiPost','action',{id:p.id,action,updated:p.updated});await load();}
$('#save-edit').onclick=async()=>{
 const save=$('#save-edit'),draft=editing;
 if(save.disabled||!draft)return;
 save.disabled=true;
 try{
  let value=$('#edit-value').value;
  if(draft.category==='participation')value=Number(value);
  if(typeof draft.value==='object')value=JSON.parse(value);
  const reason=$('#edit-reason').value;
  if(draft.manual_jargon)await api('apiPost','jargon_propose',{id:draft.id,revision:draft.revision,value,reason});
  else await api('apiPost','amend',{id:draft.id,updated:draft.updated,value,reason});
  if(editing===draft)$('#edit').close();
  await load();
 }catch(e){note(e.message,'error');}
 finally{save.disabled=false;}
};
$('#state').onchange=$('#review-category').onchange=$('#review-reason').onchange=updateReviewFilters;$('#refresh').onclick=load;
for(const b of document.querySelectorAll('[data-tab]'))b.onclick=()=>activate(b.dataset.tab);
function table(headers,rows){const wrapper=el('div','table-scroll');wrapper.tabIndex=0;wrapper.setAttribute('role','region');wrapper.setAttribute('aria-label',headers.join(' / '));const t=el('table');const head=el('tr');for(const label of headers)head.append(el('th','',label));t.append(head);for(const row of rows){const r=el('tr');for(const value of row){const td=el('td');if(value instanceof Node)td.append(value);else td.textContent=String(value??'');r.append(td);}t.append(r);}wrapper.append(t);return wrapper;}
function renderJargon(){const text=$('#jargon-search').value.toLowerCase();const rows=(socialSnapshot?.jargon||[]).filter(e=>(e.key+' '+e.value).toLowerCase().includes(text));$('#jargon-table').replaceChildren(table(['词语','含义','来源','维护'],rows.map(e=>[e.key,e.value,e.imported?'历史已确认词条':'审查后学习',button(msg("纠正含义"),()=>{editing={...e,manual_jargon:true};$('#edit-value').value=e.value;$('#edit-reason').value='';$('#edit').showModal();})])));}
function renderSocial(){const d=socialSnapshot,g=d.graph;$('#social-stats').replaceChildren(...[['观察成员',g.total_members??Object.keys(g.users).length],['近期活跃',g.active_member_count],['有效关系',g.relation_count],['互动圈',d.circle_count??d.circles.length]].map(([label,value])=>{const e=el('div','stat');e.append(el('strong','',value),el('span','',label));return e;}));
 network?.setData(g);
 $('#social-explanations').replaceChildren(details('群聊分布与统计说明',d.explanations));
 if(d.details_paged){loadSocialTable('affection');loadSocialTable('jargon');}
 else {renderAffections([...d.affections].sort((a,b)=>b.value.affection_level-a.value.affection_level),g.users);renderJargon();}
}
let socialRequest=0,memberRequest=0;
function ensureNetwork(){
 if(network)return Promise.resolve(network);
 if(!networkPromise)networkPromise=import('./network.js?v=0.8.0').then(({RelationNetwork})=>{
  network=new RelationNetwork($('#network'),async id=>{
   const box=$('#member-detail'),request=++memberRequest,scope=$('#social-scope').value;box.replaceChildren();
   if(!id){box.append(el('h3','','成员详情'),el('p','muted','点击节点或搜索成员，查看相关往来。'));return;}
   if(!socialSnapshot)return;
   if(socialSnapshot.details_paged){try{const d=await api('apiPost','member_detail',{scope:$('#social-scope').value,member:id});if(request!==memberRequest||scope!==$('#social-scope').value)return;box.append(el('h3','',d.name+' · '+d.member),table(['互动对象','互动强度','明确往来'],d.relations.map(e=>[e.a===id?e.b_name:e.a_name,Number(e.score).toFixed(1),e.directed_hits||0])),button(msg("查看学习资料"),()=>{$('#profile-scope').value=$('#social-scope').value;$('#profile-member').value=id;activate('profiles');$('#profile-load').click();}));}catch(e){if(request===memberRequest)note(e.message,'error');}return;}
   const g=socialSnapshot.graph,u=g.users[id];
   const pairs=g.all_relations.filter(e=>e.a===id||e.b===id).sort((a,b)=>b.score-a.score).slice(0,12);
   box.append(el('h3','',u.name||id),el('p','scope',id),el('p','muted',`近期 ${u.count||0} 条消息 · 显示最常往来的 ${pairs.length} 位伙伴`));
   if(pairs.length)box.append(table(['互动对象','互动强度','明确往来'],pairs.map(e=>[e.a===id?e.b_name:e.a_name,Number(e.score).toFixed(1),e.directed_hits||0])));
   else box.append(el('p','muted','尚无已观察到的往来'));
   box.append(button(msg("查看学习资料"),()=>{$('#profile-scope').value=$('#social-scope').value;$('#profile-member').value=id;document.querySelector('[data-tab="profiles"]').click();$('#profile-load').click();}));
  },(query)=>{if(!socialSnapshot?.details_paged)return false;socialQuery=query;loadSocial();return true;});
  return network;
 }).catch(e=>{networkPromise=null;throw e;});
 return networkPromise;
}
async function loadSocial(){const request=++socialRequest;try{
 const scope=$('#social-scope').value;if(!scope)return;
 $('#network-summary').textContent='正在读取当前群的关系…';
 const [data]=await Promise.all([api('apiPost','social',{scope,limit:Number($('#network-limit').value),query:socialQuery}),ensureNetwork()]);if(request!==socialRequest)return;
 socialSnapshot=data;renderSocial();if(data.details_paged){$('#network-summary').textContent+=(data.updated_at?' 数据更新于 '+formatTime(data.updated_at):' 缓存正在构建。')+(data.refreshing?' 后台更新中。':'');}
}catch(e){if(request===socialRequest)note(e.message,'error');}}
$('#social-scope').onchange=()=>{
 socialQuery='';socialPages.affection=socialPages.jargon=1;
 socialSnapshot=null;
 clearTimeout(jargonTimer);
 for(const kind of ['affection','jargon']){ticket('social-'+kind);clearSocialPage(kind);}
 loadSocial();
};
let jargonTimer;$('#jargon-search').oninput=()=>{if(!socialSnapshot?.details_paged){renderJargon();return;}clearTimeout(jargonTimer);jargonTimer=setTimeout(()=>{socialPages.jargon=1;loadSocialTable('jargon');},250);};
if(window.AstrBotPluginPage?.ready)await window.AstrBotPluginPage.ready();
await load();

$('#profile-load').onclick=async()=>{const current=ticket('profile');try{
 const scope=$('#profile-scope').value;let member=$('#profile-member').value.trim();if(!member)throw Error('请填写群友账号或昵称');if(snapshot.console_version){const found=await api('apiPost','members',{scope,query:member,page:1});if(!current())return;const exact=found.members.find(m=>m.id===member);if(exact)member=exact.id;else if(found.total===1)member=found.members[0].id;else{await searchMembers();note('请选择要查看的成员');return;}$('#profile-member').value=member;}
 const d=await api('apiPost','profile',{scope,member}),box=$('#profile-content');if(!current())return;box.replaceChildren();
 if(!d.available){box.append(el('p','',d.reason||'暂无资料'));return;}
 box.append(el('p','muted',d.note));
 for(const [title,key] of [['原话样例','examples'],['习惯统计','habits'],['喜恶摘要','preferences']]){
  const c=el('article','card');c.append(el('h3','',title+' · 已审查'));
  const rows=d.approved?.layers?.[key]||[];if(!rows.length)c.append(el('p','muted','暂无已批准资料'));
  for(const row of rows)c.append(el('pre','',displayValue(row.value)));
  box.append(c);
 }
 box.append(details('观察统计（尚未批准）',displayValue(d.observed_statistics)),details('候选原话（尚未批准）',d.candidate_examples));
}catch(e){if(current())note(e.message,'error');}};

let historyPage=1,historyCount=0,historyRequest=0;
async function loadHistory(){const request=++historyRequest;try{
 const scope=$('#history-scope').value;if(!scope)return;
 const data=await api('apiPost','history',{scope,category:$('#history-category').value,page:historyPage});if(request!==historyRequest)return;
 historyCount=data.total;$('#history-summary').textContent=`共 ${data.total} 条 · 第 ${data.page} 页 · 每页 ${data.page_size} 条`;
 $('#history-records').replaceChildren(...data.records.map(r=>{const c=el('article','card');c.append(el('h3','',`${r.category} · ${r.source_id}`),el('p','muted',r.record.note),details('查看历史原文与状态',r.record.original));return c;}));
 $('#history-prev').disabled=historyPage<=1;$('#history-next').disabled=historyPage*20>=historyCount;
}catch(e){note(e.message,'error');}}
$('#history-load').onclick=()=>{historyPage=1;loadHistory();};
$('#history-scope').onchange=$('#history-category').onchange=()=>{historyPage=1;loadHistory();};
$('#history-prev').onclick=()=>{if(historyPage>1){historyPage--;loadHistory();}};
$('#history-next').onclick=()=>{if(historyPage*20<historyCount){historyPage++;loadHistory();}};

async function fetchReviewGroup(preserveSelection=false){
 clearTimeout(reviewTimer);const epoch=++reviewEpoch;
 if(!preserveSelection)clearReviewSelection();
 if(groupReview()){
  if(reviewGroupKey&&(!reviewGroupKey.includes(':GroupMessage:')||!snapshot.sessions.includes(reviewGroupKey))){
   reviewBusy=false;reviewProposals=[];reviewTotal=0;queueData=null;reviewFailure=msg('所选会话未启用群间学习。');
   renderReviewCards();return;
  }
  reviewBusy=true;reviewFailure='';reviewProposals=[];renderReviewCards();
  try{const result=await api('apiPost','review_queue',{scope:reviewGroupKey||'',query:$('#review-query').value.trim(),category:$('#review-category').value,state:$('#state').value,reason:$('#review-reason').value,view:reviewView,page:reviewPage});
   if(epoch!==reviewEpoch)return;queueData=result;reviewProposals=result.proposals;reviewTotal=result.total;reviewPage=result.page;
   for(const p of result.proposals)if(reviewSelected.has(p.id)){if(canSelect(p)&&reviewSelectionRows.get(p.id)?.updated===p.updated)reviewSelectionRows.set(p.id,p);else{reviewSelected.delete(p.id);reviewSelectionRows.delete(p.id);}}
  }catch(e){if(epoch===reviewEpoch){reviewFailure='查询失败，请刷新重试：'+e.message;reviewProposals=[];clearReviewSelection();}throw e;}
  finally{if(epoch===reviewEpoch){reviewBusy=false;renderReviewCards();if(batchActions())loadBatchJobs();}}
  return;
 }
 if(!reviewGroupKey){reviewProposals=snapshot.proposals;reviewTotal=reviewProposals.length;return;}
 const [scope,persona,member]=JSON.parse(reviewGroupKey);const result=await api('apiPost','review_group',{scope,persona,member,page:reviewPage});if(epoch!==reviewEpoch)return;reviewProposals=result.proposals;reviewTotal=result.total;
}
$('#review-group').onchange=async()=>{try{reviewGroupKey=$('#review-group').value;reviewPage=1;reviewProposals=[];clearReviewSelection();await fetchReviewGroup();render();}catch(e){note(e.message,'error');}};
$('#review-prev').onclick=async()=>{try{if(reviewPage>1){reviewPage--;await fetchReviewGroup(groupReview());render();}}catch(e){note(e.message,'error');}};
$('#review-next').onclick=async()=>{try{if(reviewPage*(groupReview()?50:30)<reviewTotal){reviewPage++;await fetchReviewGroup(groupReview());render();}}catch(e){note(e.message,'error');}};
for(const b of document.querySelectorAll('[data-review-view]'))b.onclick=()=>{reviewView=b.dataset.reviewView;updateReviewFilters();};
$('#review-query').oninput=updateReviewFilters;
$('#review-reset').onclick=()=>{$('#state').value='';$('#review-category').value='';$('#review-query').value='';$('#review-reason').value='';updateReviewFilters();};
$('#review-next-group').onclick=async()=>{const groups=reviewGroups();if(!groups.length)return;const i=groups.findIndex(g=>reviewKey(g)===reviewGroupKey);$('#review-group').value=reviewKey(groups[(i+1)%groups.length]);await $('#review-group').onchange();};
$('#clear-review-selection').onclick=()=>{clearReviewSelection();renderReviewCards();};
$('#select-review-page').onclick=()=>{if(!groupReview())clearReviewSelection();for(const p of visibleReviews().filter(canSelect)){if(reviewSelected.size>=batchLimit())break;reviewSelected.add(p.id);reviewSelectionRows.set(p.id,p);}renderReviewCards();};
function previewBatch(action='approve'){
 if(reviewBusy)return;
 const selected=groupReview()?[...reviewSelectionRows.values()]:reviewProposals.filter(p=>reviewSelected.has(p.id));
 const candidates=reviewSelected.size?selected:visibleReviews();
 const rows=candidates.filter(p=>actionAllowed(p,action)).slice(0,batchLimit());
 if(!rows.length||!reviewGroupKey)return;
 pendingBatchAction=action;
 const items=rows.map(p=>({id:p.id,updated:p.updated}));
 if(groupReview()){
  if(rows.some(p=>p.scope!==reviewGroupKey)){note('请选择同群的调整');return;}
  pendingBatch={scope:reviewGroupKey,items};
  $('#batch-summary').textContent=`${shortScope(reviewGroupKey)} · ${new Set(rows.map(p=>p.member)).size} 位成员/全群对象 · ${rows.length} 条调整`;
 }else{const [scope,persona,member]=JSON.parse(reviewGroupKey);pendingBatch={scope,persona,member,items};$('#batch-summary').textContent=`${scope} · ${member?'成员 '+member:'本群整体'} · ${rows.length} 条调整`;}
 const labels={approve:['核对批量批准','确认批准并应用','逐项核对调整前后内容。提交时重新校验记录版本，任一项失效则整批不应用。'],
  review:['核对批量 LLM 审查','确认开始 LLM 审查','后台逐项审查，单项失败不撤销已完成项。成员资料通过后按现有规则自动生效，其他类别继续保留人工确认。'],
  archive:['核对失效归档','确认归档所选','只归档失效提案，保留证据、审查意见与历史；不删除或修改已生效资料。'],
  release:['核对交回自动审查','确认交回自动','取消批次所保留的条目将重新允许自动审查。成员资料通过后仍按原规则自动生效。'],
  reject:['核对批量拒绝','确认拒绝所选','只拒绝下面列出的待审提案，不撤回已生效资料。任一项版本已变化则整批不提交；理由保留在每项历史中。']}[action];
 $('#batch-title').textContent=labels[0];$('#commit-batch').textContent=labels[1];$('#batch-help').textContent=labels[2];
 $('#commit-batch').className=action==='reject'?'danger':'primary';
 const excluded=candidates.filter(p=>!rows.some(r=>r.id===p.id));
 $('#batch-excluded').hidden=!excluded.length;$('#batch-excluded').textContent=`本次跳过 ${excluded.length} 项（不符合此操作条件或超出单批上限）：${excluded.map(p=>(p.display_name||p.member||'全群')+' / '+p.key).join('；')}`;
 $('#batch-reason-label').hidden=action!=='reject';$('#batch-reason').value='';
 $('#batch-values').replaceChildren(...rows.map(p=>{const card=el('article','card');card.append(el('h3','',`${categories[p.category]} · ${p.key}`),el('p','scope',`${p.display_name||p.member||'全群'} · ${scopeLine(p)}`),el('pre','',`调整前：${p.before?.active?displayValue(p.before.value):'尚无记录'}\n\n调整后：${displayValue(p.value)}`),el('p','muted',p.reason));return card;}));
 $('#batch-message').textContent='';$('#batch-confirm').showModal();
}
$('#preview-batch').onclick=()=>previewBatch('approve');
$('#preview-batch-review').onclick=()=>previewBatch('review');
$('#preview-batch-reject').onclick=()=>previewBatch('reject');
$('#preview-batch-archive').onclick=()=>previewBatch('archive');$('#preview-batch-release').onclick=()=>previewBatch('release');
$('#cancel-batch').onclick=()=>$('#batch-confirm').close();
$('#commit-batch').onclick=async()=>{
 const b=$('#commit-batch');b.disabled=true;
 try{
  const action=pendingBatchAction,endpoint=['archive','release'].includes(action)?'governance_action':action==='review'?'group_batch_review':action==='reject'?'group_batch_reject':groupReview()?'group_batch_action':'batch_action';
  const body={...pendingBatch};if(['archive','release'].includes(action))body.action=action;if(action==='reject')body.reason=$('#batch-reason').value;
  const r=await api('apiPost',endpoint,body);$('#batch-confirm').close();clearReviewSelection();
  if(action==='review'){renderReviewCards();await loadBatchJobs();note(`已提交 ${r.count} 项 LLM 审查，可在任务进度查看逐项结果。`);}
  else{await load();note(action==='archive'?`已归档 ${r.count} 项失效提案，原始证据已保留。`:action==='release'?`已将 ${r.count} 项交回自动审查。`:action==='reject'?`已拒绝 ${r.count} 项，理由保留在各项审查历史中。`:`已应用 ${r.count} 条调整，每项均保留独立回滚记录。`);}
  selectionStatus();
 }catch(e){$('#batch-message').textContent=e.message;}finally{b.disabled=false;}
};

function fillScope(selector,all=false){const select=$(selector),old=select.value;select.replaceChildren();if(all){const o=el('option','',msg("全部群"));o.value='';select.append(o);}for(const s of snapshot.sessions){const o=el('option','',s);o.value=s;select.append(o);}if(snapshot.sessions.includes(old)||all&&old==='')select.value=old;}
function activate(section,fetch=true){activeSection=section;$('#stats').hidden=section!=='home';for(const s of ['home','proposals','entries','social','profiles','usage','history','scopes','preview','library'])$('#'+s).hidden=s!==section;for(const n of document.querySelectorAll('[data-tab]')){const selected=n.dataset.tab===section;n.classList.toggle('selected',selected);if(selected)n.setAttribute('aria-current','page');else n.removeAttribute('aria-current');}if(!fetch)return;if(section==='social')loadSocial();if(section==='entries'&&snapshot.entries_paged)loadEntries();if(section==='proposals')fetchReviewGroup().then(render).catch(e=>note(e.message,'error'));if(section==='usage')loadUsage();if(section==='library')workbench?.loadLibrary();}
function renderHome() {
 const pending = (snapshot.batches || []).filter(batch=>batch.count>0);
 const oldest = Math.min(...pending.map(batch=>batch.oldest||Date.now()/1000));
 const waitingMinutes = pending.length ? Math.max(0,Math.round((Date.now()/1000-oldest)/60)) : 0;
 const scene = snapshot.scene_learning || {};
 const expression = snapshot.expression_style || {};
 const summary = [
  ['24小时生效',snapshot.daily?.applied||0],
  ['24小时资料投影',snapshot.daily?.provider_uses||0],
  ['模型进行中',snapshot.scheduler?.active_models||0],
  ['最久等待(分钟)',waitingMinutes],
  ['表达习惯（有效）',expression.learned||0],
  ['接话模式（有效）',scene.learned||0],
  ['24h 检索进入候选',scene.selected||0],
  ['24h 进入模型输入',scene.provider||0],
  ['24h 关联已发回复',scene.delivered||0],
  ['24h 可归因反馈',scene.feedback||0],
 ];
 $('#home-summary').replaceChildren(...summary.map(metricCard));
 const status = msg('表达习惯{0} · 场景学习{1} · 额外判断 {2} 次 / {3} 个记录回合 · 占比 {4} · 超时 {5} 次 · 累计等待 {6}s。关联已发回复不代表模型采用或表达改善。',
  msg(expression.enabled?'已开启':'默认关闭'),msg(scene.enabled?'已开启':'默认关闭'),
  scene.attempts||0,scene.prepared_turns||0,scene.attempt_rate==null?'—':(scene.attempt_rate*100).toFixed(1)+'%',
  scene.timeouts||0,((scene.extra_wait_ms||0)/1000).toFixed(1));
 const nextCheck = snapshot.next_check_at ? formatTime(snapshot.next_check_at) : msg('等待调度');
 $('#backlog-list').replaceChildren(
  el('p','muted',status),
  el('p','muted',msg('采集队列 {0} · 模型排队 {1} · 下次检查 {2}',snapshot.queue,snapshot.scheduler?.waiting_models||0,nextCheck)),
  table(['群会话','待分析回合','原始回调'].map(key=>msg(key)),pending.map(batch=>[batch.scope,batch.count,batch.callbacks??batch.count]))
 );
 const statuses = {queued:'排队',running:'进行中',succeeded:'成功',failed:'失败',timeout:'超时',cancelled:'取消',interrupted:'重启中断'};
 const tasks = (snapshot.tasks||[]).map(task=>{
  const phase = tr(task.phase==='analysis_provider'?'分析':'审查');
  const recovery = task.recovered ? tr('后续调用成功') : task.retry ? tr('最早重试检查 {0}',formatTime(task.retry)) : '—';
  return [
   phase+' · '+(task.scope||tr('系统')),
   tr(statuses[task.status]||task.status)+(task.error?' · '+task.error:''),
   (task.queue_ms/1000).toFixed(1)+'s / '+(task.elapsed_ms===null?'—':(task.elapsed_ms/1000).toFixed(1)+'s'),
   formatTime(task.queued)+' · '+recovery,
  ];
 });
 $('#task-list').replaceChildren(table(['阶段 / 群','状态','排队 / 执行','时间与恢复'].map(key=>msg(key)),tasks));
}
window.addEventListener('series-languagechange',()=>{
 if (!snapshot.sessions) return;
 renderOverviewStats();
 renderHome();
});
function renderEntries(rows){$('#entry-list').replaceChildren(...rows.map(e=>{const c=el('article','card');c.append(el('h3','',categories[e.category]+' · '+e.key),el('p','scope',scopeLine(e)),el('p','',displayValue(e.value)),el('p','muted',`版本 ${e.revision} · ${e.active?'':'已撤回 · '}${e.category==='affection'?'稳定关系持久保存':formatDate(e.expires)}`));c.append(button(msg("查看来源与审查"),()=>workbench.openMaterial({proposal_id:e.proposal_id,entry_id:e.id})));return c;}));}
async function loadEntries(){const current=ticket('entries');try{const d=await api('apiPost','entries',{scope:$('#entries-scope').value,state:$('#entries-state').value,query:$('#entries-query').value,page:entryPage});if(!current())return;entryTotal=d.total;renderEntries(d.entries);$('#entries-summary').textContent=`${d.total} 条 · 第 ${d.page} 页`;$('#entries-prev').disabled=entryPage<=1;$('#entries-next').disabled=entryPage*25>=entryTotal;}catch(e){if(current())note(e.message,'error');}}
function clearEntries(){
 ticket('entries');
 entryPage=1;
 entryTotal=0;
 $('#entry-list').replaceChildren();
 $('#entries-summary').textContent='';
 $('#entries-prev').disabled=$('#entries-next').disabled=true;
}
$('#entries-load').onclick=()=>{entryPage=1;loadEntries();};$('#entries-prev').onclick=()=>{if(entryPage>1){entryPage--;loadEntries();}};$('#entries-next').onclick=()=>{if(entryPage*25<entryTotal){entryPage++;loadEntries();}};
function affectionVersion(entry){
 const version=el('span','',entry.persona.slice(0,12));
 if(entry.affection_current===true){
  version.append(' · ',el('span','',msg(entry.affection_selection==='current_persona'?'当前人设':'最近记录')));
 }else if(entry.affection_current===false){
  version.append(' · ',el('span','',msg('历史')));
 }
 if(entry.affection_role_id)setText(version,msg('角色：{0}',entry.affection_role_id),'title');
 return version;
}
function renderAffections(entries,users){
 $('#affection-table').replaceChildren(table([msg('成员'),msg('好感度'),msg('人设文本版本'),msg('来源')],entries.map(entry=>[
  users[entry.member]?.name||entry.member,
  Number(entry.value.affection_level).toFixed(4),
  affectionVersion(entry),
  el('span','',msg(entry.imported?'历史关系继承':'已生效调整')),
 ])));
}
function clearSocialPage(kind){
 $('#'+kind+'-table').replaceChildren();
 setText($('#'+kind+'-page'),'');
 $('#'+kind+'-prev').disabled=$('#'+kind+'-next').disabled=true;
}
async function loadSocialTable(kind){
 const current=ticket('social-'+kind),social=socialSnapshot;
 const scope=$('#social-scope').value;
 if(!scope||!social)return;
 const view=kind==='affection'?$('#affection-view').value:null;
 const request={scope,category:kind,state:view&&view!=='current'?'all':'active',query:kind==='jargon'?$('#jargon-search').value:'',page:socialPages[kind]};
 if(view)request.affection_view=view;
 try{
  const data=await api('apiPost','entries',request);
  if(!current()||social!==socialSnapshot||scope!==$('#social-scope').value||(view&&view!==$('#affection-view').value))return;
  social[kind==='affection'?'affections':'jargon']=data.entries;
  setText($('#'+kind+'-page'),msg('{0} 条 · 第 {1} 页',data.total,data.page));
  $('#'+kind+'-prev').disabled=data.page<=1;
  $('#'+kind+'-next').disabled=data.page*25>=data.total;
  if(kind==='jargon')renderJargon();
  else renderAffections(data.entries,social.graph.users);
 }catch(e){if(current())note(e.message,'error');}
}
$('#affection-view').onchange=()=>{
 socialPages.affection=1;
 clearSocialPage('affection');
 loadSocialTable('affection');
};
for(const kind of ['affection','jargon']){document.querySelector('#'+kind+'-prev').onclick=()=>{if(socialPages[kind]>1){socialPages[kind]--;loadSocialTable(kind);}};document.querySelector('#'+kind+'-next').onclick=()=>{socialPages[kind]++;loadSocialTable(kind);};}
async function searchMembers(){const current=ticket('members');try{const d=await api('apiPost','members',{scope:$('#profile-scope').value,query:$('#profile-member').value.trim(),page:memberPage});if(!current())return;memberTotal=d.total;$('#member-results').replaceChildren(...d.members.map(m=>button(`${m.name} · ${m.id} · ${m.messages} 条消息`,()=>{$('#profile-member').value=m.id;$('#profile-load').click();})));$('#members-summary').textContent=`${d.total} 位 · 第 ${d.page} 页`;$('#members-prev').disabled=memberPage<=1;$('#members-next').disabled=memberPage*20>=memberTotal;}catch(e){if(current())note(e.message,'error');}}
$('#profile-search').onclick=()=>{memberPage=1;searchMembers();};$('#members-prev').onclick=()=>{if(memberPage>1){memberPage--;searchMembers();}};$('#members-next').onclick=()=>{if(memberPage*20<memberTotal){memberPage++;searchMembers();}};
async function loadUsage(){const current=ticket('usage');try{const scope=$('#usage-scope').value;if(!scope)return;const d=await api('apiPost','usage',{scope,page:usagePage});if(!current())return;usageTotal=d.total;$('#usage-summary').textContent=`${d.total} 条 · 第 ${d.page} 页`;$('#usage-records').replaceChildren(...d.records.map(r=>{const c=el('article','card');c.append(el('h3','',`${r.member} · ${formatDate(r.created)}`),el('p','muted',`已送入模型参数 · 过往引用 ${r.memory_refs.length} · ${r.dialogue?'有连续对话锚点':'无连续对话锚点'} · 性格锁 ${r.personality_locked?'保留':'未知'}`),table(['实际提供的学习资料','版本'],[...r.profiles,...(r.learning||[])].map(p=>[`${categories[p.category]||p.category} · ${p.label||p.source_id}`,p.revision])),details('本轮节奏',r.rhythm),details('未进入组合层的候选（过滤或身份校验）',r.omitted),el('p','muted','未审查、过期或不相关的资料不会被提供为候选；这里不推断模型最终采用了哪句话。'));return c;}));$('#usage-prev').disabled=usagePage<=1;$('#usage-next').disabled=usagePage*20>=usageTotal;}catch(e){if(current())note(e.message,'error');}}
$('#usage-load').onclick=()=>{usagePage=1;loadUsage();};$('#usage-prev').onclick=()=>{if(usagePage>1){usagePage--;loadUsage();}};$('#usage-next').onclick=()=>{if(usagePage*20<usageTotal){usagePage++;loadUsage();}};
setInterval(()=>{if(activeSection==='home'&&!document.hidden)load();},20000);

for(const id of ['entries','profiles','usage','history','scopes']){const section=$('#'+id),help=section.querySelector(':scope > p.muted');if(help){const d=el('details','page-help');d.append(el('summary','','说明'),help);section.querySelector('h2').after(d);}}

async function loadBatchJobs(){
 if(!batchActions())return;
 const epoch=++batchJobsEpoch,scope=reviewGroupKey||'';
 if(scope&&(!scope.includes(':GroupMessage:')||!snapshot.sessions.includes(scope))){$('#batch-jobs').hidden=true;return;}
 const names={queued:'排队',running:'审查中',pausing:'暂停中 · 等待当前项结束',paused:'已暂停',cancelling:'取消中 · 等待在途请求结束',cancelled:'已取消',reviewed:'审查通过',applied:'通过已生效',rejected:'已拒绝',deferred:'需补充证据',skipped:'已跳过',failed:'失败',interrupted:'已中断',completed:'已完成',completed_with_issues:'已结束 · 含失败或跳过'};
 try{
  const result=await api('apiPost','review_batch_jobs',{scope});if(epoch!==batchJobsEpoch||scope!==(reviewGroupKey||''))return;
  const box=$('#batch-jobs'),openJobs=new Set([...box.querySelectorAll('details[open][data-job-id]')].map(d=>d.dataset.jobId));box.hidden=!result.jobs.length;box.replaceChildren();if(!result.jobs.length)return;
  const head=el('div','job-heading');head.append(el('h3','','批量 LLM 审查'),button(msg("更新待审列表"),()=>fetchReviewGroup().then(renderReviewCards)));box.append(head);
  for(const job of result.jobs){
   const retryable=job.items.filter(i=>i.status==='failed'||(i.status==='reviewed'&&i.auto_apply_pending)).length;const counts={};for(const item of job.items)counts[item.status]=(counts[item.status]||0)+1;
   const finished=job.items.filter(i=>!['queued','running','interrupted','cancelled'].includes(i.status)).length;
   const d=el('details');d.dataset.jobId=job.id;d.open=openJobs.has(job.id);
   d.append(el('summary','',`${shortScope(job.scope)} · ${job.parent_id?'重试批次 · ':''}${names[job.status]||job.status} · 已处理 ${finished}/${job.items.length}`));
   const controls=el('div','actions');
   if(['queued','running'].includes(job.status))controls.append(button(msg("暂停"),()=>controlBatchJob(job,'pause')));
   if(['paused','interrupted'].includes(job.status)&&job.items.some(i=>['queued','interrupted'].includes(i.status)))controls.append(button(msg("恢复未完成项"),()=>controlBatchJob(job,'resume')));
   if(['queued','running','pausing','paused','interrupted'].includes(job.status))controls.append(button(msg("取消剩余任务"),async()=>{if(await window.DynamicsUI.confirm(msg('取消此批未完成项？已完成的结果会保留，尚未提交的在途结果将丢弃。'),{danger:true}))return controlBatchJob(job,'cancel');},'danger'));
   if(!job.retries?.length&&!['queued','running','pausing','paused','cancelling'].includes(job.status)&&retryable)controls.append(button(`重试未完成项（${retryable}）`,()=>controlBatchJob(job,'retry')));
   d.append(el('p','muted',Object.entries(counts).map(([k,n])=>`${names[k]||k} ${n}`).join(' · ')),controls,table(['成员 / 内容','结果','说明'],job.items.map(i=>[`${i.member||'全群'} / ${i.key}`,i.auto_apply_pending?'通过 · 等待自动生效':names[i.status]||i.status,i.error||'—'])));
   if(['pausing','paused'].includes(job.status))d.append(el('p','muted','暂停保留当前项结果，恢复仅继续未完成项。'));
   if(job.status==='cancelling')d.append(el('p','muted','已停止启动后续项；已经发出的模型请求可能仍在运行，尚未提交的结果不会生效。'));
   if(job.retries?.length)d.append(el('p','muted','已创建重试批次；原始结果保留，请在最新重试任务中查看进度或继续重试。'));
   box.append(d);
  }
 }catch(e){if(epoch===batchJobsEpoch){$('#batch-jobs').hidden=false;$('#batch-jobs').textContent='任务进度暂时不可用：'+e.message;}}
}
async function controlBatchJob(job,action){
 const r=await api('apiPost','review_batch_control',{id:job.id,action});
 await loadBatchJobs();
 note(action==='retry'?`已提交重试（${r.count} 项）${r.retry_skipped?'，另有 '+r.retry_skipped+' 项因已变化而跳过':''}`:action==='pause'?'已请求暂停，当前项结束后停止。':action==='resume'?'已恢复未完成项，已完成结果不会重跑。':'已请求取消，已完成结果保留。');
}
setInterval(()=>{if(activeSection==='proposals'&&!document.hidden)loadBatchJobs();},5000);


for(const id of ['entries-scope','entries-state'])$('#'+id).onchange=()=>{clearEntries();return loadEntries();};
$('#entries-query').oninput=clearEntries;
$('#usage-scope').onchange=()=>{usagePage=1;loadUsage();};
$('#profile-member').addEventListener('input',()=>{ticket('members');ticket('profile');$('#member-results').replaceChildren();$('#profile-content').replaceChildren();});
$('#profile-scope').onchange=()=>{ticket('members');ticket('profile');memberPage=1;$('#member-results').replaceChildren();$('#profile-content').replaceChildren();};
if($('#global-scope'))$('#global-scope').onchange=()=>syncGlobalScopeToSubtabs($('#global-scope').value,true);
window.addEventListener('dynamics-scopechange',ev=>{const next=ev.detail?.umo||'';if(next!==globalScope)syncGlobalScopeToSubtabs(next);});
function focusReviewCard(nextIdx){
 const items=[...document.querySelectorAll('#cards .review-item')];
 if(!items.length){focusedReviewIndex=-1;return null;}
 focusedReviewIndex=Math.max(0,Math.min(items.length-1,nextIdx));
 items.forEach((n,i)=>n.classList.toggle('is-focused',i===focusedReviewIndex));
 const target=items[focusedReviewIndex];
 target.focus({preventScroll:true});
 target.scrollIntoView({block:'nearest'});
 return target;
}
document.addEventListener('keydown',ev=>{
 if(activeSection!=='proposals'||ev.metaKey||ev.ctrlKey||ev.altKey||document.querySelector('dialog[open]'))return;
 const tag=ev.target?.tagName?.toLowerCase()||'';
 const isCheck=tag==='input'&&ev.target.type==='checkbox';
 if((['input','textarea','select'].includes(tag)&&!isCheck)||ev.target?.isContentEditable)return;
 const items=[...document.querySelectorAll('#cards .review-item')];
 if(!items.length)return;
 const currentCard=ev.target?.closest?.('.review-item');
 if(currentCard){const idx=items.indexOf(currentCard);if(idx>=0)focusedReviewIndex=idx;}
 const key=ev.key.toLowerCase();
 if(key==='j'){ev.preventDefault();focusReviewCard(focusedReviewIndex<0?0:focusedReviewIndex+1);}
 else if(key==='k'){ev.preventDefault();focusReviewCard(focusedReviewIndex<=0?0:focusedReviewIndex-1);}
 else if(ev.key===' '&&ev.target?.classList?.contains('review-item')){ev.preventDefault();const cb=ev.target.querySelector('input[type="checkbox"][data-proposal]');if(cb&&!cb.disabled){cb.checked=!cb.checked;cb.dispatchEvent(new Event('change'));}}
 else if(key==='a'&&currentCard){const card=currentCard;const btn=card?.querySelector('[data-quick-action="approve"]');if(btn&&!btn.disabled){ev.preventDefault();btn.click();}}
 else if(key==='r'&&currentCard){const card=currentCard;const btn=card?.querySelector('[data-quick-action="reject"]');if(btn&&!btn.disabled){ev.preventDefault();btn.click();}}
 else if(key==='h'){const card=items[Math.max(0,focusedReviewIndex)];const det=card?.querySelector('details.review-detail');if(det){ev.preventDefault();det.open=!det.open;const src=card.querySelector('details.series-source-jump');if(det.open&&src)src.open=true;}}
});
workbench=installWorkbench({$,api,el,button,table,details,displayValue,categories,states,snapshot:()=>snapshot,note,reload:load});


