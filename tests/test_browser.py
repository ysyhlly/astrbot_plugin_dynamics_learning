import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]


def test_shared_theme_preference_survives_reload_and_host_theme(server):
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.add_init_script('window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:' + json.dumps(snapshot()) + '})};')
        page.goto(server)
        page.get_by_role('button', name='切换到夜间模式').click()
        assert page.evaluate("localStorage.getItem('chat_dynamics_ui')") == 'night'
        page.goto(server)
        assert page.locator('html').get_attribute('data-ui-theme') == 'night'
        page.evaluate("document.documentElement.dataset.theme='light'")
        assert page.evaluate("getComputedStyle(document.body).backgroundColor") == 'rgb(11, 20, 25)'
        page.evaluate("dispatchEvent(new StorageEvent('storage',{key:'chat_dynamics_ui',newValue:'day'}))")
        assert page.locator('html').get_attribute('data-ui-theme') == 'day'
        page.get_by_role('button', name='群友学习', exact=True).click()
        assert page.get_by_role('button', name='群友学习', exact=True).get_attribute('aria-current') == 'page'
        browser.close()


@pytest.fixture(scope="module")
def server():
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/plugin/page/bridge-sdk.js":
                # Tests install the bridge with page.add_init_script before loading the view.
                self.send_response(204)
                self.end_headers()
                return
            super().do_GET()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT / "pages" / "review"), **kwargs)

        def log_message(self, *_args):
            pass
    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{service.server_port}/index.html"
    service.shutdown()
    thread.join(2)


def snapshot():
    return dict(queue=0, dropped=0, invalid=0, last_error="", jobs=[], sessions=["demo:GroupMessage:example"],
        batches=[dict(scope="demo:GroupMessage:example", persona="demo-persona", count=24)], entries=[],
        review_modes={"style": "llm_then_human", "participation": "human"}, proposals=[dict(
        id="demo-proposal", category="style", key="先给结论", value="先简短回应重点；用户追问时再展开解释。",
        scope="demo:GroupMessage:example", persona="demo-persona", member="示例成员", state="reviewed", before=None,
        reason="对方在多次讨论中明确表示希望先看到结论，保留需要时继续解释的空间。", mode="llm_then_human",
        days=30, updated=1, evidence=[dict(turn_id="example-turn", text="能先告诉我结论吗？")],
        reviews=[dict(actor="llm:demo", verdict="approve", reason="与明确偏好一致，不影响角色核心设定。")],
        evaluation={"note": "示例数据 · 审查意见不是效果验证，应用后仍需观察真实互动。"})])


@pytest.mark.parametrize("width,theme", [(1280, "light"), (390, "light"), (320, "light"), (768, "light"), (1280, "dark")])
def test_review_ui_navigation_actions_and_no_overflow(server, width, theme):
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": width, "height": 1000}, color_scheme=theme)
        state = snapshot()
        context.add_init_script("window.__snapshot=" + json.dumps(state, ensure_ascii=False) + ";" + """
        window.__calls=[];
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({status:'ok',data:window.__snapshot}),
        apiPost:async(path,body)=>{window.__calls.push({path,body});return {status:'ok',data:{queued:true}}}};
        """)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(server + ("?ui=night" if theme == "dark" else ""))
        page.get_by_text("批准并应用", exact=True).wait_for()
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        directory = os.environ.get("DYNAMICS_SCREENSHOTS")
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(Path(directory) / f"review-{width}-{theme}.png"), full_page=True)
        page.get_by_text("批准并应用", exact=True).click()
        assert page.evaluate("window.__calls[0].body.updated") == 1
        page.get_by_text("查看依据与修改", exact=True).click()
        page.get_by_text("修改提案", exact=True).click()
        assert page.locator("#edit").is_visible()
        page.locator("#edit-value").fill("保留必要解释")
        page.locator("#edit-reason").fill("避免过度简短")
        page.get_by_text("保存新提案", exact=True).click()
        assert page.evaluate("window.__calls.some(x=>x.path==='amend')")
        page.get_by_role("button", name="会话与设置", exact=True).click()
        assert page.get_by_text("现在分析", exact=True).is_visible()
        page.get_by_text("现在分析", exact=True).click()
        assert page.evaluate("window.__calls.some(x=>x.path==='analyze')")
        assert not errors
        browser.close()


def test_evidence_is_rendered_as_text(server):
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        state = snapshot()
        state["proposals"][0]["value"] = '<img src=x onerror="window.pwned=true">'
        context.add_init_script("window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:" + json.dumps(state) + "})};")
        page = context.new_page()
        page.goto(server)
        page.get_by_text("批准并应用", exact=True).wait_for()
        assert page.locator("img").count() == 0
        assert not page.evaluate("!!window.pwned")
        browser.close()


@pytest.mark.parametrize('width',[1280,390])
def test_social_graph_slang_affection_and_member_details(server,width):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':width,'height':1000})
        state=snapshot()
        social={'scope':state['sessions'][0],'circles':[['alice','bob']], 'explanations':{'overview':['已观察两位成员']},
            'affections':[{'member':'alice','persona':'example-persona','value':{'affection_level':125.25},'imported':True}],
            'jargon':[{'key':'老六','value':'本群对出其不意操作的戏称','imported':True}],
            'graph':{'users':{'alice':{'name':'Alice','count':20},'bob':{'name':'Bob','count':10}},
              'active_member_count':2,'relation_count':1,'all_relations':[{'a':'alice','b':'bob','a_name':'Alice','b_name':'Bob',
                'score':25,'directed_hits':5,'affinity':20,'affinity_confidence':.8}]}}
        context.add_init_script('window.__snapshot='+json.dumps(state)+';window.__social='+json.dumps(social)+';'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),
        apiPost:async(path)=>({data:path==='social'?window.__social:{queued:true}})};''')
        page=context.new_page()
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(server)
        page.get_by_role('button',name='群友关系',exact=True).click()
        page.locator('#network circle').first.wait_for()
        assert page.locator('#network circle').count()==2
        assert page.locator('#network line').count()==1
        assert '125.2500' in page.locator('#affection-table').inner_text()
        page.locator('#network [aria-label="Alice"]').click()
        assert 'Bob' in page.locator('#member-detail').inner_text()
        page.locator('#jargon-search').fill('不存在')
        assert '老六' not in page.locator('#jargon-table').inner_text()
        page.locator('#jargon-search').fill('老六')
        assert '出其不意' in page.locator('#jargon-table').inner_text()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            directory=Path(os.environ['DYNAMICS_SCREENSHOTS'])
            directory.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(directory/f'social-{width}.png'),full_page=True)
        assert not errors
        browser.close()


@pytest.mark.parametrize('width',[1280,390])
def test_three_layer_view_and_structured_values(server,width):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':width,'height':1000})
        state=snapshot()
        value={'kind':'expression_example','situation':'游戏失败了','expression':'<img src=x onerror="window.pwned=true">', 'link':'reply'}
        stats={'kind':'observed_typing','sample_count':20,'mean_chars':8,'short_rate':.8,'multiline_rate':.1,
               'punctuation_rate':.4,'emoji_rate':.2,'consecutive_rate':.3,'definitions':'仅限观察窗口'}
        pref={'topic':'剧透','stance':'dislike','condition':'观看前','quote':'我不喜欢剧透'}
        data={'available':True,'note':'候选尚未批准','observed_statistics':stats,'candidate_examples':[],
              'approved':{'layers':{key:[{'value':v}] for key,v in [('examples',value),('habits',stats),('preferences',pref)]}}}
        context.add_init_script('window.__snapshot='+json.dumps(state)+';window.__profile='+json.dumps(data)+';'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),
        apiPost:async(path,body)=>{window.__request={path,body};return {data:window.__profile}}};''')
        page=context.new_page()
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(server)
        page.get_by_role('button',name='群友学习',exact=True).click()
        page.get_by_label('群友账号',exact=True).fill('alice')
        page.get_by_role('button',name='查看三层资料',exact=True).click()
        page.get_by_text('原话样例 · 已审查',exact=True).wait_for()
        assert '样本 20 条' in page.locator('#profile-content').inner_text()
        assert '反感：剧透' in page.locator('#profile-content').inner_text()
        assert page.evaluate('window.__request.path') == 'profile'
        assert page.evaluate('window.__request.body.member') == 'alice'
        assert page.locator('img').count() == 0 and not page.evaluate('!!window.pwned')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            directory=Path(os.environ['DYNAMICS_SCREENSHOTS'])
            directory.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(directory/f'profiles-{width}.png'),full_page=True)
        assert not errors
        browser.close()


@pytest.mark.parametrize('width,theme',[(1440,'light'),(390,'light'),(1280,'dark')])
def test_dense_relationship_layout_focus_resize_and_drag(server,width,theme):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':width,'height':1100},color_scheme=theme)
        state=snapshot()
        users={str(i):{'name':f'群友昵称很长的成员{i:02d}','count':400-i*3} for i in range(80)}
        edges=[{'a':str(i),'b':str(j),'a_name':users[str(i)]['name'],'b_name':users[str(j)]['name'],
                'score':80-abs(i-j),'directed_hits':10,'affinity':30,'affinity_confidence':.8}
               for i in range(80) for j in range(i+1,min(80,i+8))]
        data={'scope':state['sessions'][0],'circles':[list(users)],'explanations':{'overview':['测试数据']},
              'affections':[],'jargon':[],'graph':{'users':users,'all_relations':edges,'relation_count':len(edges),'active_member_count':80}}
        context.add_init_script('window.__snapshot='+json.dumps(state)+';window.__social='+json.dumps(data)+';'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),
        apiPost:async()=>({data:window.__social})};''')
        page=context.new_page()
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(server)
        page.get_by_role('button',name='群友关系',exact=True).click()
        page.locator('#network [data-member]').first.wait_for()
        page.get_by_label('图中成员数量',exact=True).select_option('40')
        assert page.locator('#network [data-member]').count()==40
        collisions=page.evaluate('''()=>{
          const nodes=[...document.querySelectorAll('#network [data-member]')].map(e=>e.getBoundingClientRect());let overlaps=0;
          for(let i=0;i<nodes.length;i++)for(let j=0;j<i;j++){
            const a=nodes[i],b=nodes[j];if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1)overlaps++;
          }return overlaps;
        }''')
        assert collisions==0
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert page.locator('#network line').count()<120
        assert page.locator('#network [data-member="0"] text').text_content().endswith('…')
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            directory=Path(os.environ['DYNAMICS_SCREENSHOTS'])
            directory.mkdir(parents=True,exist_ok=True)
            page.locator('#social').screenshot(path=str(directory/f'network-{width}-{theme}.png'))
        page.get_by_label('查找群友',exact=True).fill('79')
        page.get_by_role('button',name='定位成员',exact=True).click()
        assert page.locator('#network [data-member="79"]').get_attribute('aria-pressed')=='true'
        assert '成员79' in page.locator('#member-detail').inner_text()
        assert page.locator('#network line[opacity="0.07"]').count()>0
        node=page.locator('#network [data-member="79"] circle')
        node.scroll_into_view_if_needed()
        box=node.bounding_box()
        previous=page.locator('#network [data-member="79"]').get_attribute('transform')
        page.mouse.move(box['x']+box['width']/2,box['y']+box['height']/2)
        page.mouse.down()
        page.mouse.move(box['x']+box['width']/2+30,box['y']+box['height']/2+25,steps=5)
        page.mouse.up()
        assert page.locator('#network [data-member="79"]').get_attribute('transform')!=previous
        page.get_by_role('button',name='恢复全图',exact=True).click()
        assert page.locator('#network [aria-pressed="true"]').count()==0
        assert page.locator('#network [data-member="79"]').count()==0
        assert not errors
        browser.close()


def test_history_archive_is_scoped_paginated_and_rendered_as_text(server):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page()
        state=snapshot()
        archive={'total':23,'page':1,'page_size':20,'records':[{'id':'historical','category':'expression','source_id':'42',
                 'record':{'note':'历史资料，不自动生效','original':{'expression':'<img src=x onerror="window.pwned=true">','user_id':None}}}]}
        page.add_init_script('window.__snapshot='+json.dumps(state)+';window.__archive='+json.dumps(archive)+';'+'''
        window.__calls=[];window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),
        apiPost:async(path,body)=>{window.__calls.push({path,body});return {data:{...window.__archive,page:body.page}}}};''')
        page.goto(server)
        page.get_by_role('button',name='历史资料',exact=True).click()
        page.get_by_role('button',name='查看历史',exact=True).click()
        page.get_by_text('历史资料，不自动生效',exact=True).wait_for()
        assert page.evaluate('window.__calls[0].body.scope')==state['sessions'][0]
        assert page.locator('img').count()==0
        page.get_by_role('button',name='下一页',exact=True).click()
        page.wait_for_function('window.__calls.length===2')
        assert page.evaluate('window.__calls[1].body.page')==2
        browser.close()


@pytest.mark.parametrize('width',[1280,390])
def test_review_workbench_requires_explicit_batch_confirmation(server,width):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':width,'height':1100})
        state=snapshot()
        first={**state['proposals'][0],'id':'first','member':'alice','key':'先给结论'}
        second={**first,'id':'second','key':'不剧透','value':'该成员不喜欢提前知道剧情。','updated':2}
        state['proposals']=[first,second]
        state['review_groups']=[dict(scope=first['scope'],persona=first['persona'],member='alice',display_name='Alice',count=2,reviewed=2),
                                dict(scope=first['scope'],persona=first['persona'],member='bob',display_name='Bob',count=1,reviewed=0)]
        state['proposal_counts']={'reviewed':2,'pending':1}
        page.add_init_script('window.__snapshot='+json.dumps(state)+';window.__calls=[];'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),apiPost:async(path,body)=>{
        window.__calls.push({path,body});if(path==='review_group')return {data:{total:2,page:1,page_size:30,proposals:window.__snapshot.proposals}};
        if(path==='batch_action')return {data:{count:body.items.length,applied:body.items.map(x=>x.id)}};return {data:{}};}};''')
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(server)
        page.get_by_role('button',name='选择本页可批准项',exact=True).wait_for()
        assert page.locator('input[data-proposal]:checked').count()==0
        page.get_by_role('button',name='选择本页可批准项',exact=True).click()
        assert page.locator('input[data-proposal]:checked').count()==2
        page.get_by_role('button',name='批准所选（2）',exact=True).click()
        assert page.locator('#batch-confirm').is_visible()
        assert 'alice' in page.locator('#batch-summary').inner_text()
        assert '不剧透' in page.locator('#batch-values').inner_text()
        assert not page.evaluate("window.__calls.some(x=>x.path==='batch_action')")
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            directory=Path(os.environ['DYNAMICS_SCREENSHOTS'])
            directory.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(directory/f'workbench-{width}.png'),full_page=True)
        page.get_by_role('button',name='确认批准并应用',exact=True).click()
        page.wait_for_function("window.__calls.some(x=>x.path==='batch_action')")
        body=page.evaluate("window.__calls.find(x=>x.path==='batch_action').body")
        assert body['member']=='alice' and body['scope']==first['scope']
        assert body['items']==[{'id':'first','updated':1},{'id':'second','updated':2}]
        assert not errors
        browser.close()


def test_llm_member_mode_does_not_show_human_approval_button(server):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page()
        state=snapshot()
        state['review_modes']['style']='llm'
        state['proposals'][0]['mode']='llm'
        page.add_init_script('window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:'+json.dumps(state)+'})};')
        page.goto(server)
        page.locator('#review-all').click()
        page.get_by_text('已通过 · 自动生效中',exact=True).wait_for()
        assert page.get_by_role('button',name='批准并应用',exact=True).count()==0
        assert 'LLM 审查通过后自动生效' in page.locator('#cards').inner_text()
        browser.close()


@pytest.mark.parametrize('width', [1280, 390])
def test_compact_inbox_direct_batch_filters_and_hidden_evidence(server, width):
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': width, 'height': 1000})
        state = snapshot()
        original = state['proposals'][0]
        rows = [{**original, 'id': f'p{i}', 'key': title, 'member': 'alice', 'category': 'jargon',
                 'value': value, 'details_lazy': True, 'evidence_count': 4, 'evidence': []}
                for i, (title, value) in enumerate([
                    ('开团', '本群通常指开始组队，需结合游戏讨论判断。'),
                    ('先给结论', '先用一句话回答问题，需要时再补充细节。'),
                    ('避免剧透', '对方尚未看完的剧情，不主动透露后续发展。')])]
        state['review_modes']['jargon'] = 'llm_then_human'
        state['review_modes']['style'] = 'llm'
        state['proposals'] = rows + [{**original, 'id': 'auto', 'mode': 'llm', 'member': 'alice'},
                                    {**rows[0], 'id': 'pending', 'state': 'pending', 'reviews': []}]
        state['review_groups'] = [dict(scope=original['scope'], persona=original['persona'], member='alice',
                                      display_name='Alice', count=5, reviewed=4)]
        page.add_init_script('window.__snapshot=' + json.dumps(state) + ';window.__calls=[];' + '''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),apiPost:async(path,body)=>{
          window.__calls.push({path,body});if(path==='review_group')return {data:{total:5,page:1,proposals:window.__snapshot.proposals}};
          if(path==='batch_action')return {data:{count:body.items.length}};
          if(path==='proposal_detail')return {data:{evidence:[{text:'示例原话'}],reviews:[],observations:{}}};
          return {data:{}};}};''')
        page.goto(server)
        page.get_by_role('button', name='批准本页（3）', exact=True).wait_for()
        assert page.locator('.review-item').count() == 3
        assert page.locator('.review-detail[open]').count() == 0
        assert not page.evaluate("window.__calls.some(c=>c.path==='proposal_detail'||c.path==='batch_action')")
        assert page.locator('input[data-proposal]:checked').count() == 0
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            page.screenshot(path=str(Path(os.environ['DYNAMICS_SCREENSHOTS']) / f'inbox-{width}.png'), full_page=True)
        page.get_by_role('button', name='批准本页（3）', exact=True).click()
        assert page.locator('#batch-confirm').is_visible()
        assert '开团' in page.locator('#batch-values').inner_text()
        assert '避免剧透' in page.locator('#batch-values').inner_text()
        assert not page.evaluate("window.__calls.some(c=>c.path==='batch_action')")
        page.get_by_role('button', name='取消', exact=True).click()
        page.locator('input[data-proposal]').first.check()
        page.get_by_label('搜索全部待审内容').fill('开团')
        assert page.locator('input[data-proposal]:checked').count() == 0
        assert page.locator('.review-item').count() == 1
        page.get_by_role('button', name='批准本页（1）', exact=True).click()
        page.get_by_role('button', name='确认批准并应用', exact=True).click()
        page.wait_for_function("window.__calls.some(c=>c.path==='batch_action')")
        assert page.evaluate("window.__calls.find(c=>c.path==='batch_action').body.items.map(i=>i.id)") == ['p0']
        page.get_by_label('搜索全部待审内容').fill('')
        page.locator('#review-all').click()
        assert page.locator('.review-item').count() == 5
        assert page.get_by_text('已通过 · 自动生效中', exact=True).is_visible()
        browser.close()


@pytest.mark.parametrize('width', [1280, 390])
def test_group_queue_cross_page_members_global_search_and_late_response(server, width):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':width,'height':1000})
        state=snapshot()
        state.update(console_version=1,review_queue_version=1,entries_paged=True)
        scope=state['sessions'][0]
        other='demo:GroupMessage:other'
        state['sessions'].append(other)
        original=state['proposals'][0]
        rows=[{**original,'id':f'p{i}','key':f'学习项{i}','member':'alice' if i<50 else 'bob',
               'display_name':'Alice' if i<50 else 'Bob','can_approve':True,'category':'relationship',
               'value':'远处内容' if i==54 else f'经常一起讨论话题{i}'} for i in range(55)]
        rows.append({**rows[0],'id':'foreign','scope':other})
        state['review_groups']=[dict(scope=scope,persona='demo-persona',member='alice',count=50),
                                dict(scope=scope,persona='demo-persona',member='bob',count=5),
                                dict(scope=other,persona='demo-persona',member='alice',count=1)]
        page.add_init_script('window.__snapshot='+json.dumps(state)+';window.__rows='+json.dumps(rows)+';window.__calls=[];'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),apiPost:async(path,body)=>{
          window.__calls.push({path,body});
          if(path==='review_queue'){
            if(body.query==='旧查询')await new Promise(r=>setTimeout(r,700));
            const matched=window.__rows.filter(p=>(!body.scope||p.scope===body.scope)&&(!body.category||p.category===body.category)&&(!body.state||p.state===body.state)&&(!body.query||body.query==='旧查询'||(p.value+p.member+p.display_name).includes(body.query)));
            const selected=body.view==='ready'?matched.filter(p=>p.can_approve):matched;
            const page=Math.min(body.page,Math.max(1,Math.ceil(selected.length/50)));
            return {data:{total:selected.length,ready_total:matched.filter(p=>p.can_approve).length,pending_total:matched.length,
              page,page_size:50,proposals:selected.slice((page-1)*50,page*50),groups:[{scope:window.__snapshot.sessions[0],count:55},{scope:window.__snapshot.sessions[1],count:1}]}};
          }
          if(path==='group_batch_action')return {data:{count:body.items.length}};
          return {data:{}};}};''')
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(server)
        page.get_by_role('button',name='调整审查',exact=True).click()
        page.locator('.review-item').first.wait_for()
        assert page.locator('#review-group option').count()==3
        page.locator('#review-group').select_option(scope)
        page.wait_for_function("document.querySelector('#review-summary').textContent.includes('匹配 55') && document.querySelector('#cards').getAttribute('aria-busy')==='false'")
        page.locator('input[data-proposal="p0"]').check()
        page.locator('#review-next').click()
        page.locator('input[data-proposal="p50"]').check()
        page.get_by_role('button',name='批准所选（2）',exact=True).click()
        assert 'Alice' in page.locator('#batch-values').inner_text() and 'Bob' in page.locator('#batch-values').inner_text()
        assert not page.evaluate("window.__calls.some(c=>c.path==='group_batch_action')")
        page.get_by_role('button',name='确认批准并应用',exact=True).click()
        page.wait_for_function("window.__calls.some(c=>c.path==='group_batch_action')")
        sent=page.evaluate("window.__calls.find(c=>c.path==='group_batch_action').body")
        assert set(sent)=={'scope','items'} and sent['scope']==scope
        assert [r['id'] for r in sent['items']]==['p0','p50']
        page.get_by_label('搜索全部待审内容').fill('远处内容')
        page.wait_for_function("document.querySelector('#review-summary').textContent.includes('匹配 1 项') && document.querySelector('.review-value')?.textContent==='远处内容'")
        assert page.locator('.review-item').count()==1
        assert page.locator('input[data-proposal]:checked').count()==0
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.locator('.review-filter-menu>summary').click()
        page.get_by_label('审查类别',exact=True).select_option('preference')
        page.get_by_text('没有匹配的记录',exact=True).wait_for()
        assert page.evaluate("window.__calls.some(c=>c.path==='review_queue'&&c.body.category==='preference'&&c.body.query==='远处内容'&&c.body.page===1)")
        page.get_by_role('button',name='清空筛选',exact=True).click()
        page.locator('.review-filter-menu>summary').click()
        page.get_by_label('搜索全部待审内容').fill('旧查询')
        page.wait_for_function("window.__calls.some(c=>c.path==='review_queue'&&c.body.query==='旧查询')")
        page.get_by_label('搜索全部待审内容').fill('远处内容')
        page.wait_for_function("document.querySelector('#cards').getAttribute('aria-busy')==='false' && document.querySelectorAll('.review-item').length===1")
        page.wait_for_timeout(800)
        assert page.locator('.review-item').count()==1
        assert page.locator('.review-value').inner_text()=='远处内容'
        assert not errors
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            page.screenshot(path=str(Path(os.environ['DYNAMICS_SCREENSHOTS'])/f'group-review-{width}.png'),full_page=True)
        browser.close()


@pytest.mark.parametrize('width',[1280,390])
def test_batch_llm_and_rejection_share_cross_page_selection_and_confirmation(server,width):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':width,'height':1000})
        state=snapshot()
        state.update(console_version=1,review_queue_version=2,entries_paged=True)
        scope=state['sessions'][0]
        original=state['proposals'][0]
        rows=[{**original,'id':f'p{i}','member':'alice' if i<50 else 'bob','key':f'待审项{i}',
               'state':'reviewed' if i==0 else 'pending','can_approve':i==0,'can_llm_review':i!=0,'can_reject':True}
              for i in range(51)]
        state['review_groups']=[dict(scope=scope,persona='demo-persona',member='alice',count=51)]
        page.add_init_script('window.__snapshot='+json.dumps(state)+';window.__rows='+json.dumps(rows)+';window.__calls=[];window.__jobs=[];window.__rejectConflict=true;'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),apiPost:async(path,body)=>{
          window.__calls.push({path,body});
          if(path==='review_queue')return {data:{total:51,ready_total:1,pending_total:51,page:body.page,page_size:50,groups:[{scope:window.__snapshot.sessions[0],count:51}],proposals:window.__rows.slice((body.page-1)*50,body.page*50)}};
          if(path==='review_batch_jobs')return {data:{jobs:window.__jobs}};
          if(path==='group_batch_review'){window.__jobs=[{id:'job',scope:body.scope,status:'completed',items:body.items.map(i=>({...i,member:'bob',key:'待审项50',status:'reviewed',error:''}))}];return {data:{queued:true,id:'job',count:body.items.length}};}
          if(path==='group_batch_reject')return window.__rejectConflict?{status:'error',message:'所选记录已变化'}:{data:{count:body.items.length}};
          return {data:{}};}};''')
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(server)
        page.get_by_role('button',name='调整审查',exact=True).click()
        page.locator('.review-item').first.wait_for()
        page.locator('#review-group').select_option(scope)
        page.locator('input[data-proposal="p0"]').check()
        page.locator('#review-next').click()
        page.locator('input[data-proposal="p50"]').check()
        page.get_by_role('button',name='LLM 审查所选（1）',exact=True).click()
        assert '跳过 1 项' in page.locator('#batch-excluded').inner_text()
        assert '待审项0' in page.locator('#batch-excluded').inner_text()
        assert not page.evaluate("window.__calls.some(c=>c.path==='group_batch_review')")
        page.get_by_role('button',name='取消',exact=True).click()
        page.get_by_role('button',name='LLM 审查所选（1）',exact=True).click()
        page.get_by_role('button',name='确认开始 LLM 审查',exact=True).click()
        page.wait_for_function("window.__calls.some(c=>c.path==='group_batch_review')")
        assert page.evaluate("window.__calls.find(c=>c.path==='group_batch_review').body.items.map(i=>i.id)")==['p50']
        page.locator('#batch-jobs summary').wait_for()
        assert '已完成' in page.locator('#batch-jobs').inner_text()
        assert page.locator('input[data-proposal]:checked').count()==0
        page.locator('#review-prev').click()
        page.locator('input[data-proposal="p0"]').check()
        page.locator('#review-next').click()
        page.locator('input[data-proposal="p50"]').check()
        page.get_by_role('button',name='拒绝所选（2）',exact=True).click()
        page.locator('#batch-reason').fill('证据不足，等待更多互动')
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            page.screenshot(path=str(Path(os.environ['DYNAMICS_SCREENSHOTS'])/f'batch-reject-{width}.png'),full_page=True)
        page.get_by_role('button',name='确认拒绝所选',exact=True).click()
        page.get_by_text('所选记录已变化',exact=True).wait_for()
        assert page.locator('#batch-confirm').is_visible()
        body=page.evaluate("window.__calls.find(c=>c.path==='group_batch_reject').body")
        assert [i['id'] for i in body['items']]==['p0','p50'] and body['scope']==scope
        assert body['reason']=='证据不足，等待更多互动'
        page.evaluate('window.__rejectConflict=false')
        page.get_by_role('button',name='确认拒绝所选',exact=True).click()
        page.wait_for_function("!document.querySelector('#batch-confirm').open")
        assert not errors
        browser.close()


@pytest.mark.parametrize('width',[1280,390])
def test_job_pause_resume_cancel_and_failed_retry_controls(server,width):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':width,'height':1000})
        state=snapshot()
        state.update(console_version=1,review_queue_version=2,entries_paged=True)
        scope=state['sessions'][0]
        jobs=[dict(id='active',scope=scope,status='running',items=[dict(id='done',member='alice',key='已完成项',status='reviewed',error=''),dict(id='working',member='bob',key='未完成项',status='running',error='')]),
              dict(id='failure',scope=scope,status='completed_with_issues',items=[dict(id='ok',member='alice',key='原成功项',status='reviewed',error=''),dict(id='bad',member='bob',key='失败项',status='failed',error='TimeoutError')])]
        page.add_init_script('window.__snapshot='+json.dumps(state)+';window.__jobs='+json.dumps(jobs)+';window.__calls=[];'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__snapshot}),apiPost:async(path,body)=>{
          window.__calls.push({path,body});
          if(path==='review_queue')return {data:{total:0,ready_total:0,pending_total:0,page:1,page_size:50,proposals:[],groups:[]}};
          if(path==='review_batch_jobs')return {data:{jobs:window.__jobs}};
          if(path==='review_batch_control'){
            const job=window.__jobs.find(j=>j.id===body.id);
            if(body.action==='pause'){job.status='paused';job.items[1].status='queued';}
            if(body.action==='resume'){job.status='running';job.items[1].status='running';}
            if(body.action==='cancel'){job.status='cancelled';job.items[1].status='cancelled';}
            if(body.action==='retry'){job.retries=['child'];const child={id:'child',parent_id:job.id,scope:job.scope,status:'queued',items:job.items.filter(i=>i.status==='failed').map(i=>({...i,status:'queued'}))};window.__jobs.unshift(child);return {data:{id:'child',status:'queued',count:1,retry_skipped:0}};}
            return {data:{id:job.id,status:job.status,count:job.items.length}};
          }
          return {data:{}};}};''')
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(server)
        page.get_by_role('button',name='调整审查',exact=True).click()
        active=page.locator('details[data-job-id="active"]')
        active.locator('summary').click()
        active.get_by_role('button',name='暂停',exact=True).click()
        active.get_by_role('button',name='恢复未完成项',exact=True).wait_for()
        assert active.get_attribute('open') is not None
        assert '已完成项' in active.inner_text()
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            page.screenshot(path=str(Path(os.environ['DYNAMICS_SCREENSHOTS'])/f'job-controls-{width}.png'),full_page=True)
        active.get_by_role('button',name='恢复未完成项',exact=True).click()
        active.get_by_role('button',name='暂停',exact=True).wait_for()
        active.get_by_role('button',name='取消剩余任务',exact=True).click()
        page.locator('dialog[open]').get_by_role('button',name='取消',exact=True).click()
        assert not page.evaluate("window.__calls.some(c=>c.path==='review_batch_control'&&c.body.action==='cancel')")
        active.get_by_role('button',name='取消剩余任务',exact=True).click()
        page.locator('dialog[open]').get_by_role('button',name='继续',exact=True).click()
        page.wait_for_function("window.__jobs.find(j=>j.id==='active').status==='cancelled'")
        assert page.evaluate("window.__jobs.find(j=>j.id==='active').items[0].status")=='reviewed'
        failed=page.locator('details[data-job-id="failure"]')
        failed.locator('summary').click()
        failed.get_by_role('button',name='重试未完成项（1）',exact=True).click()
        page.locator('details[data-job-id="child"]').wait_for()
        assert page.evaluate("window.__jobs.find(j=>j.id==='child').items.map(i=>i.id)")==['bad']
        assert page.evaluate("window.__jobs.find(j=>j.id==='failure').items.map(i=>i.status)")==['reviewed','failed']
        assert failed.get_by_role('button',name='重试未完成项（1）',exact=True).count()==0
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors
        browser.close()


@pytest.mark.parametrize('width',[1280,390])
def test_console_home_search_paged_entries_and_provider_use(server,width):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':width,'height':1100})
        state=snapshot()
        state.update(console_version=1,entries_paged=True,daily={'applied':166,'provider_uses':4},
            tasks=[dict(phase='analysis_provider',scope=state['sessions'][0],status='timeout',error='TimeoutError',queued=1700000000,
                        queue_ms=10,elapsed_ms=60000,retry=1700000600,recovered=True)],
            scheduler={'active_models':0,'waiting_models':0},next_check_at=1700000600)
        usage={'total':1,'page':1,'page_size':20,'records':[{'member':'alice','created':1700000000,'profiles':[{'category':'preference','label':'交流长度','revision':'2'}],
            'memory_refs':['atom:1'],'dialogue':True,'personality_locked':True,'rhythm':{'mode':'listen'},'omitted':[]}]}
        page.add_init_script('window.__state='+json.dumps(state)+';window.__usage='+json.dumps(usage)+';window.__calls=[];'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__state}),apiPost:async(path,body)=>{
          window.__calls.push({path,body});if(path==='members')return {data:{total:1,page:1,members:[{id:'alice',name:'Alice',messages:10}]}};
          if(path==='profile')return {data:{available:true,note:'资料',approved:{layers:{}},observed_statistics:{},candidate_examples:[]}};
          if(path==='usage')return {data:window.__usage};if(path==='entries')return {data:{total:0,page:1,entries:[]}};
          return {data:{}};}};''')
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(server)
        page.get_by_text('后续调用成功',exact=False).wait_for()
        assert page.locator('#home').is_visible() and not page.locator('#proposals').is_visible()
        assert not page.evaluate("window.__calls.some(c=>c.path==='entries'||c.path==='social')")
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        if os.environ.get('DYNAMICS_SCREENSHOTS'):
            directory=Path(os.environ['DYNAMICS_SCREENSHOTS'])
            directory.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(directory/f'console-{width}.png'),full_page=True)
        page.get_by_role('button',name='已学内容',exact=True).click()
        page.wait_for_function("window.__calls.some(c=>c.path==='entries')")
        page.get_by_role('button',name='群友学习',exact=True).click()
        page.get_by_label('群友账号',exact=True).fill('Alice')
        page.get_by_role('button',name='查找群友',exact=True).click()
        page.get_by_role('button',name='Alice · alice · 10 条消息',exact=True).click()
        page.wait_for_function("window.__calls.some(c=>c.path==='profile')")
        assert page.evaluate("window.__calls.find(c=>c.path==='profile').body.member")=='alice'
        page.get_by_role('button',name='资料使用',exact=True).click()
        page.locator('#usage-records').get_by_text('喜恶',exact=False).first.wait_for()
        assert '交流长度' in page.locator('#usage-records').inner_text()
        assert '已送入模型参数' in page.locator('#usage-records').inner_text()
        assert not errors
        browser.close()


def test_all_groups_grouped_batch_debounce_edit_limit_and_guidance(server):
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1280,'height':1000})
        state=snapshot()
        s1='demo:GroupMessage:g1'
        s2='demo:GroupMessage:g2'
        state.update(
            console_version=1,review_queue_version=2,entries_paged=True,sessions=[s1,s2],
            legacy_kv_v1={'present':True,'count':3},
            provider_readiness={'analysis_provider':{'configured':False,'ready':False,'provider_id':''},
                                'review_provider':{'configured':True,'ready':True,'provider_id':'r1'}},
            review_groups=[dict(scope=s1,persona='p',member='a',count=1),dict(scope=s2,persona='p',member='b',count=1)],
        )
        base=state['proposals'][0]
        rows=[
            {**base,'id':'r1','scope':s1,'key':'群1词条','can_approve':True,'can_llm_review':False,'can_reject':True,'updated':10},
            {**base,'id':'r2','scope':s2,'key':'群2词条','can_approve':True,'can_llm_review':False,'can_reject':True,'updated':20},
        ]
        page.add_init_script('window.__state='+json.dumps(state)+';window.__rows='+json.dumps(rows)+';window.__calls=[];'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({data:window.__state}),apiPost:async(path,body)=>{
          window.__calls.push({path,body});
          if(path==='review_queue'){
            if(body.query==='慢')await new Promise(r=>setTimeout(r,200));
            return {data:{total:window.__rows.length,pending_total:2,ready_total:2,stale_total:0,held_total:0,page:1,page_size:50,
              groups:[{scope:window.__state.sessions[0],count:1,ready_count:1,stale_count:0},{scope:window.__state.sessions[1],count:1,ready_count:1,stale_count:0}],
              proposals:window.__rows}};
          }
          if(path==='review_batch_jobs')return {data:{jobs:[]}};
          if(path==='group_batch_action')return {data:{count:body.items.length}};
          return {data:{}};}};''')
        errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(server)
        assert 'learning_*_v1' in page.locator('#backlog-list').inner_text()
        assert '未配置分析模型' in page.locator('#backlog-list').inner_text()
        assert page.locator('#edit-value').get_attribute('maxlength') is None
        long_text='甲'*1200
        page.get_by_role('button',name='调整审查',exact=True).click()
        page.locator('.review-item').first.wait_for()
        assert '可批 1' in page.locator('#review-group').inner_text()
        page.locator('#review-query').fill('慢')
        assert page.locator('.review-item').count()==2
        page.wait_for_function("window.__calls.some(c=>c.path==='review_queue'&&c.body.query==='慢')")
        page.locator('.review-item').first.get_by_text('查看依据与修改',exact=True).click()
        page.locator('.review-item').first.get_by_text('修改提案',exact=True).click()
        page.locator('#edit-value').fill(long_text)
        assert len(page.locator('#edit-value').input_value())==1200
        page.locator('#edit button[value="cancel"]').click()
        page.wait_for_function("!document.querySelector('#edit').open")
        page.locator('input[data-proposal="r1"]').check()
        page.locator('input[data-proposal="r2"]').check()
        page.get_by_role('button',name='批准所选（2）',exact=True).click()
        assert '按群分组提交' in page.locator('#batch-summary').inner_text()
        page.get_by_role('button',name='确认批准并应用',exact=True).click()
        page.wait_for_function("window.__calls.filter(c=>c.path==='group_batch_action').length===2")
        sent=page.evaluate("window.__calls.filter(c=>c.path==='group_batch_action').map(c=>c.body)")
        assert [b['scope'] for b in sent]==[s1,s2]
        assert not errors
        browser.close()
