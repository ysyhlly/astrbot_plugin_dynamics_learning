import {msg,tr,setText,apply,feedback,confirmAction,getLanguage} from './series-ui.js';
export function installWorkbench({$,api,el,button,table,details,displayValue,categories,states,snapshot,note,reload}){
 let libraryPage=1,libraryTotal=0,libraryEpoch=0,previewEpoch=0,detailEpoch=0;
 function refreshScopes(){for(const id of ['library-scope','preview-scope']){const node=$('#'+id),old=node.value;node.replaceChildren();const placeholder=el('option','',msg(id==='library-scope'?'全部群':'请选择群聊'));placeholder.value='';node.append(placeholder);for(const scope of snapshot().sessions||[]){const o=el('option','',scope);o.value=scope;node.append(o);}if(old&&![...node.options].some(o=>o.value===old)){const o=el('option','',old+' · '+msg('暂无资料'));o.value=old;node.append(o);}node.value=old;}}
 function invalidateScope(){libraryPage=1;++libraryEpoch;++previewEpoch;for(const id of ['library-summary','library-records','preview-output'])$('#'+id)?.replaceChildren();for(const id of ['library-prev','library-next'])$('#'+id).disabled=true;}
 function clearLibrary(){
  libraryTotal=0;
  $('#library-summary').replaceChildren();
  $('#library-records').replaceChildren();
  $('#library-prev').disabled=$('#library-next').disabled=true;
 }
 async function loadLibrary(){const epoch=++libraryEpoch;clearLibrary();try{
  const d=await api('apiPost','library',{scope:$('#library-scope').value,state:$('#library-state').value,category:$('#library-category').value,query:$('#library-query').value,page:libraryPage});if(epoch!==libraryEpoch)return;
  libraryPage=d.page;libraryTotal=d.total;setText($('#library-summary'),msg('匹配 {0} 条 · 第 {1} 页',d.total,d.page));
  $('#library-records').replaceChildren(...d.records.map(p=>{const c=el('article','card');c.append(el('h3','',`${categories[p.category]||p.category} · ${p.key}`),el('p','muted',`${states[p.state]||p.state} · ${p.scope} · ${p.member||msg("全群")}`),el('p','',displayValue(p.value)),button(msg("查看版本与证据"),()=>openMaterial({proposal_id:p.id})));return c;}));
  if(!d.records.length)$('#library-records').append(el('p','empty',msg("没有匹配的版本记录")));
  $('#library-prev').disabled=d.page<=1;$('#library-next').disabled=d.page*25>=d.total;
 }catch(e){if(epoch===libraryEpoch)note(e.message);}}
 async function openMaterial(identity){const epoch=++detailEpoch,dialog=$('#material-detail');$('#material-body').replaceChildren(el('p','',msg("正在加载资料…")));if(!dialog.open)dialog.showModal();try{
  const d=await api('apiPost','material_detail',identity);if(epoch!==detailEpoch||!dialog.open)return;
  const p=d.proposal,e=d.entry,subject=p||e,box=$('#material-body');box.replaceChildren();
  box.append(el('h2','',`${categories[subject.category]||subject.category} · ${subject.key}`),el('p','scope',msg('{0} · 人格 {1}',msg('{0} · {1}',subject.scope,subject.member||msg('全群')),subject.persona)));
  if(e)box.append(el('h3','',msg("当前学习项")),el('pre','',displayValue(e.value)),el('p','muted',msg('版本 {0} · {1} · {2}',e.revision,!e.active?msg('已撤回'):e.category!=='affection'&&e.expires*1000<=Date.now()?msg('已过期'):msg('启用'),e.category==='affection'?msg('长期关系'):new Date(e.expires*1000).toLocaleString(getLanguage()))));
  if(p){box.append(el('h3','',msg("所选提案")),el('pre','',displayValue(p.value)),el('p','muted',`${states[p.state]||p.state} · ${p.archive_reason||''}`));
   box.append(details(msg("原始证据"),p.evidence),details(msg("完整审查意见"),p.reviews));
   if(d.can_rollback)box.append(button(msg("回滚本次调整"),async()=>{
    if(!await confirmAction(msg("确认回滚此版本？会恢复其调整前状态，后续版本存在时将拒绝操作。")))return;
    if(epoch!==detailEpoch||!dialog.open)return;
    const result=await api('apiPost','action',{id:p.id,updated:p.updated,action:'rollback'});
    if(epoch===detailEpoch&&dialog.open)await openMaterial(identity);
    await reload();
    note(result.warning||msg("回滚已提交，历史版本与证据已保留。"));
   },'danger'));
   else if(p.state==='applied')box.append(el('p','muted',msg("当前学习项已有后续版本，请先查看并处理最新版本。")));
  }else box.append(el('p','muted',msg("历史导入资料：可查看当前值，没有本插件提案可回滚。")));
  box.append(el('h3','',msg("版本记录")));
  for(const row of d.versions||[])box.append(button(`${states[row.state]||row.state} · ${new Date(row.created*1000).toLocaleString(getLanguage())} · ${row.id.slice(0,8)}`,()=>openMaterial({proposal_id:row.id})));
  box.append(el('h3','',msg("变更时间线")),table([msg("时间"),msg("操作"),msg("操作者")],(d.timeline||[]).map(r=>[new Date(r.created*1000).toLocaleString(getLanguage()),r.action,r.actor])),details(msg("时间线详情"),d.timeline||[]),el('h3','',msg("资料使用记录")),table([msg("时间"),msg("回合")],(d.usage||[]).map(r=>[new Date(r.created*1000).toLocaleString(getLanguage()),r.turn_id])));
  if(!d.usage?.length)box.append(el('p','muted',msg("暂无已送入模型的使用回执；这不代表模型效果结论。")));
 }catch(e){if(epoch===detailEpoch)$('#material-body').replaceChildren(el('p','',e.message));}}
 async function preview(){const epoch=++previewEpoch,scope=$('#preview-scope').value,member=$('#preview-member').value.trim(),query=$('#preview-query').value;const box=$('#preview-output');box.replaceChildren(el('p','',msg("正在读取当前材料与成员边界…")));try{
  const d=await api('apiPost','reply_preview',{scope,member,query});if(epoch!==previewEpoch)return;
  box.replaceChildren(el('p','',d.available?msg("预览完成：未调用模型，未发送消息。"):d.reason||msg("当前条件不足，无法确认最终材料")),el('p','muted',d.note||''));
  if(d.available&&d.reason)box.append(el('p','muted',msg('边界结果：{0}',d.reason)));
  const kept=new Set((d.items||[]).map(i=>i.source_id)),removed=new Map((d.excluded||[]).map(i=>[i.source_id,i.reason]));
  box.append(el('h3','',msg('经过当前边界过滤的材料 · {0}',(d.items||[]).length)));
  for(const row of d.items||[]){const c=el('article','card');c.append(el('h3','',`${categories[row.category]||row.category} · ${row.key}`),el('pre','',displayValue(row.value)),el('p','scope',msg('{0} · 版本 {1} · {2}',row.source_umo,row.revision??msg('快照'),row.source_id)));box.append(c);}
  const audit=el('details');audit.append(el('summary','',msg('候选与排除原因（{0} 项）',d.candidates?.length||0)),table([msg("类别 / 内容"),msg("阶段"),msg("原因")],(d.candidates||[]).map(r=>[`${categories[r.category]||r.category} · ${r.key}`,kept.has(r.source_id)?msg("当前可提供"):r.selected?msg("候选材料"):msg("未选入"),removed.get(r.source_id)||r.reason||(r.selected&&!d.available?msg("边界尚未确认"):'—')])));box.append(audit);
  box.append(details(msg("实际候选材料与过滤结果"),{offered:d.offered||[],items:d.items||[],excluded:d.excluded||[]}));
 }catch(e){if(epoch===previewEpoch)box.replaceChildren(el('p','',e.message));}}
 $('#material-close').onclick=()=>{++detailEpoch;$('#material-detail').close();};
 $('#material-detail').addEventListener('close',()=>{if(!$('#material-detail').open)++detailEpoch;});
 $('#library-load').onclick=()=>{libraryPage=1;loadLibrary();};
 for(const id of ['library-scope','library-state','library-category'])$('#'+id).onchange=()=>{libraryPage=1;loadLibrary();};
 $('#library-query').oninput=()=>{++libraryEpoch;libraryPage=1;clearLibrary();};
 $('#library-prev').onclick=()=>{if(libraryPage>1){libraryPage--;loadLibrary();}};$('#library-next').onclick=()=>{if(libraryPage*25<libraryTotal){libraryPage++;loadLibrary();}};
 $('#preview-run').onclick=preview;
 for(const id of ['preview-scope','preview-member','preview-query'])$('#'+id).addEventListener('input',()=>{++previewEpoch;$('#preview-output').replaceChildren();});
 for(const [key,value] of Object.entries(categories)){const o=el('option','',value);o.value=key;$('#library-category').append(o);}
 refreshScopes();
 return {refreshScopes,invalidateScope,loadLibrary,openMaterial};
}
