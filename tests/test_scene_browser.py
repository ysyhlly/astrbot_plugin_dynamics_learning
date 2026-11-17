import json
import os
from pathlib import Path

import pytest
from test_browser import playwright, snapshot
from test_browser import server as server


@pytest.mark.parametrize('width', [390, 1280])
def test_scene_funnel_is_visible_without_claiming_adoption(server, width):
    state = snapshot()
    state['console_version'] = 1
    state['expression_style'] = {'enabled': True, 'learned': 1}
    state['scene_learning'] = {'enabled': False, 'learned': 2, 'selected': 7, 'provider': 6, 'delivered': 5,
                               'feedback': 1, 'attempts': 3, 'timeouts': 2, 'extra_wait_ms': 12000}
    with playwright.sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': width, 'height': 1000})
        page.add_init_script('window.AstrBotPluginPage={ready:async()=>{},apiGet:async()=>({status:"ok",data:' + json.dumps(state) + '})};')
        page.goto(server)
        page.get_by_text('接话模式（有效）', exact=True).wait_for()
        assert page.get_by_text('表达习惯（有效）', exact=True).is_visible()
        assert page.get_by_text('24h 进入模型输入', exact=True).is_visible()
        assert page.get_by_text('24h 关联已发回复', exact=True).is_visible()
        assert page.get_by_text('24h 可归因反馈', exact=True).is_visible()
        assert '不代表模型采用或表达改善' in page.locator('#backlog-list').inner_text()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        output = os.environ.get('SCENE_SCREENSHOT_DIR')
        if output:
            page.screenshot(path=str(Path(output) / f'scene-ui-{width}.png'), full_page=True)
        browser.close()
