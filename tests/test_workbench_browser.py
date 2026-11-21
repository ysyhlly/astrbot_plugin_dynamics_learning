import json
import os
from pathlib import Path

import pytest
from test_browser import server,snapshot,playwright  # noqa: F401


@pytest.mark.parametrize('width',[1280,390])
def test_preview_library_archive_and_rollback_use_explicit_actions(server,width):  # noqa: F811
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':width,'height':1100})
        state=snapshot()
        state.update(console_version=1,entries_paged=True,review_queue_version=2,workbench_version=1)
        scope=state['sessions'][0]
        p={**state['proposals'][0],'state':'applied','applied_revision':1,'updated':12,'created':1700000000,'id':'applied','entry_id':'entry'}
        stale={**p,'id':'stale','state':'reviewed','updated':20,'can_approve':False,'can_llm_review':False,'can_reject':True,'can_archive':True,'reason_code':'persona_changed','blocked_reason':'角色版本变化'}
        entry={**p,'proposal_id':'applied','revision':1,'active':True,'expires':2000000000,'id':'entry'}
        state['proposals']=[]
        state['review_groups']=[dict(scope=scope,persona='demo',member='alice',count=1)]
        page.add_init_script('window.__state='+json.dumps(state)+';window.__p='+json.dumps(p)+';window.__stale='+json.dumps(stale)+';window.__entry='+json.dumps(entry)+';window.__calls=[];window.__archived=false;'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__state}),apiPost:async(path,body)=>{
          window.__calls.push({path,body});
          if(path==='library')return {data:{total:1,page:1,records:[window.__p]}};
          if(path==='entries')return {data:{total:1,page:1,entries:[window.__entry]}};
          if(path==='material_detail')return {data:{proposal:window.__p,entry:window.__entry,versions:[window.__p],timeline:[{action:'applied',actor:'human:admin',created:1700000000,body:{}}],usage:[],can_rollback:window.__p.state==='applied'}};
          if(path==='action'){window.__p.state='rolled_back';window.__entry.active=false;return {data:{}};}
          if(path==='reply_preview')return {data:{available:true,model_called:false,message_sent:false,note:'同真实边界过滤',items:[{source_id:'term',category:'jargon',key:'开团',value:'开始组队',source_umo:body.scope,revision:1}],offered:[],candidates:[{source_id:'term',category:'jargon',key:'开团',selected:true,reason:''},{source_id:'old',category:'preference',key:'旧喜恶',selected:false,reason:'已过期'}],excluded:[]}};
          if(path==='review_queue'){const total=window.__archived?0:1;return {data:{total,pending_total:total,ready_total:0,stale_total:total,held_total:0,page:1,page_size:50,proposals:total?[window.__stale]:[],groups:[{scope:window.__state.sessions[0],count:total}]}};}
          if(path==='review_batch_jobs')return {data:{jobs:[]}};
          if(path==='governance_action'){window.__archived=true;return {data:{count:1,action:body.action}};}
          return {data:{}};}};''')
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(server)
        page.locator('#nav-more-menu > summary').click()
        page.get_by_role('button',name='资料版本中心',exact=True).click()
        page.get_by_role('button',name='查看版本与证据',exact=True).click()
        page.get_by_role('button',name='回滚本次调整',exact=True).wait_for()
        assert not page.evaluate("window.__calls.some(c=>c.path==='action')")
        page.get_by_role('button',name='回滚本次调整',exact=True).click()
        page.locator('dialog.series-confirm[open]').get_by_role('button',name='取消',exact=True).click()
        assert not page.evaluate("window.__calls.some(c=>c.path==='action')")
        page.get_by_role('button',name='回滚本次调整',exact=True).click()
        page.locator('dialog.series-confirm[open]').get_by_role('button',name='继续',exact=True).click()
        page.wait_for_function("window.__calls.some(c=>c.path==='action')")
        assert page.evaluate("window.__calls.find(c=>c.path==='action').body.updated")==12
        page.get_by_role('button',name='关闭详情',exact=True).click()
        page.locator('#nav-more-menu > summary').click()
        page.get_by_role('button',name='回复材料预览',exact=True).click()
        page.get_by_label('预览成员账号',exact=True).fill('alice')
        page.locator('#preview-query').fill('一起开团吗')
        page.get_by_role('button',name='预览材料',exact=True).click()
        page.get_by_text('预览完成：未调用模型，未发送消息。',exact=True).wait_for()
        page.get_by_text('候选与排除原因（2 项）',exact=True).click()
        assert '开始组队' in page.locator('#preview-output').inner_text() and '已过期' in page.locator('#preview-output').inner_text()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            page.screenshot(path=str(Path(os.environ['DYNAMICS_SCREENSHOTS'])/f'preview-{width}.png'),full_page=True)
        page.get_by_role('button',name='调整审查',exact=True).click()
        page.locator('#review-stale').click()
        page.locator('#review-group').select_option(scope)
        page.get_by_role('button',name='归档本页（1）',exact=True).click()
        assert not page.evaluate("window.__calls.some(c=>c.path==='governance_action')")
        page.get_by_role('button',name='确认归档所选',exact=True).click()
        page.wait_for_function("window.__calls.some(c=>c.path==='governance_action')")
        body=page.evaluate("window.__calls.find(c=>c.path==='governance_action').body")
        assert body['scope']==scope and body['action']=='archive' and body['items']==[{'id':'stale','updated':20}]
        assert not errors
        browser.close()


def test_old_entries_response_cannot_replace_current_query(server):  # noqa: F811
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page()
        state=snapshot()
        state.update(console_version=1,entries_paged=True)
        entry={**state['proposals'][0],'id':'entry','proposal_id':'p','revision':1,'active':True,'expires':2000000000}
        page.add_init_script('window.__s='+json.dumps(state)+';window.__e='+json.dumps(entry)+';window.__calls=[];'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__s}),apiPost:async(path,body)=>{
          window.__calls.push({path,body});if(path==='entries'){if(body.query==='旧')await new Promise(r=>setTimeout(r,600));return {data:{total:1,page:1,entries:[{...window.__e,key:body.query||'默认'}]}};}return {data:{}};}};''')
        page.goto(server)
        page.get_by_role('button',name='已学内容',exact=True).click()
        page.locator('#entries-query').fill('旧')
        page.locator('#entries-load').click()
        page.wait_for_function("window.__calls.some(c=>c.body.query==='旧')")
        page.locator('#entries-query').fill('新')
        page.locator('#entries-load').click()
        page.locator('#entry-list').get_by_text('群友表达方式 · 新',exact=True).wait_for()
        page.wait_for_timeout(700)
        assert page.locator('#entry-list h3').inner_text()=='群友表达方式 · 新'
        browser.close()
