import json
import pytest
from test_browser import server,snapshot,playwright  # noqa: F401


@pytest.mark.parametrize('width',[1280,390])
def test_presence_buttons_and_safe_navigation_fallback(server,width):
    state=snapshot()
    data={'legacy_conflict':False,'plugins':[
        dict(id='astrbot_plugin_dynamics_learning',label='群间学习',state='ready',page='review',role='学习',current=True,can_open=True),
        dict(id='astrbot_plugin_chat_dynamics',label='Chat Dynamics',state='enabled',page='app',role='回复',current=False,can_open=True),
        dict(id='astrbot_plugin_dynamics_world',label='群间世界',state='inactive',page='world',role='角色',current=False,can_open=True),
        dict(id='astrbot_plugin_dynamics_memory',label='群间记忆',state='missing',page='dashboard',role='记忆',current=False,can_open=False)]}
    with playwright.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':width,'height':1000})
        page.add_init_script('window.__snapshot='+json.dumps(state)+';window.__presence='+json.dumps(data)+';window.__nav=[];'+'''
        window.AstrBotPluginPage={ready:async()=>{},apiGet:async(path)=>({data:path==='series_presence'?window.__presence:window.__snapshot}),
        navigateToPlugin:async(...args)=>window.__nav.push(args)};''')
        page.goto(server)
        root=page.locator('#dynamics-series-panel')
        root.get_by_role('button').filter(has_text='对话主控').wait_for()
        assert root.get_by_role('button').filter(has_text='群间记忆').is_disabled()
        assert root.get_by_role('button').filter(has_text='群间学习').is_disabled()
        assert root.get_by_role('button').filter(has_text='返回群间小屋').is_enabled()
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.evaluate('window.ChatDynamicsBeforeNavigate=async()=>false')
        root.get_by_role('button').filter(has_text='对话主控').click()
        assert page.evaluate('window.__nav.length')==0
        page.evaluate('window.ChatDynamicsBeforeNavigate=async()=>true')
        root.get_by_role('button').filter(has_text='对话主控').click()
        assert page.evaluate('window.__nav[0].slice(0,2)')==['astrbot_plugin_chat_dynamics','app']
        page.evaluate('delete window.AstrBotPluginPage.navigateToPlugin')
        root.get_by_role('button').filter(has_text='对话主控').click()
        assert '宿主未提供跨插件导航' in root.inner_text()
        browser.close()
