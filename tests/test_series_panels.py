import asyncio
import json
from types import SimpleNamespace as NS

from astrbot_plugin_dynamics_learning import series_panels as panels


def test_presence_distinguishes_installation_runtime_and_failed_manifest(tmp_path):
    for identity,_,page,_ in panels.CATALOG:
        if page:
            path=tmp_path/identity/'pages'/page
            path.mkdir(parents=True)
            (path/'index.html').write_text('test')
    ready=NS(get_dynamics_capabilities=lambda:{'schema_version':1,'ready':True,'enabled':True})
    inactive=NS(get_dynamics_capabilities=lambda:{'schema_version':1,'ready':False,'enabled':False})
    def failed():
        raise RuntimeError('do not expose confidential config')
    context=NS(get_all_stars=lambda:[
        NS(name='astrbot_plugin_chat_dynamics',activated=True,star_cls=NS(),version='v1'),
        NS(name='astrbot_plugin_dynamics_learning',activated=True,star_cls=ready),
        NS(name='astrbot_plugin_dynamics_world',activated=True,star_cls=inactive),
        NS(name='astrbot_plugin_dynamics_memory',activated=False,star_cls=ready),
        NS(name='astrbot_plugin_autonomous_blocking',activated=True,star_cls=NS(get_dynamics_capabilities=failed))])
    data=panels.presence(context,'astrbot_plugin_dynamics_learning',tmp_path)
    rows={r['id']:r for r in data['plugins']}
    assert rows['astrbot_plugin_chat_dynamics']['state']=='enabled'
    assert rows['astrbot_plugin_dynamics_learning']['ready'] is True
    assert rows['astrbot_plugin_dynamics_world']['state']=='inactive' and rows['astrbot_plugin_dynamics_world']['can_open']
    assert rows['astrbot_plugin_dynamics_memory']['state']=='disabled' and not rows['astrbot_plugin_dynamics_memory']['can_open']
    assert rows['astrbot_plugin_self_learning']['state']=='missing'
    assert rows['astrbot_plugin_autonomous_blocking']['error']=='RuntimeError'
    assert 'confidential' not in json.dumps(data)
    context.get_all_stars=lambda:[]
    assert panels.presence(context,'x',tmp_path)['plugins'][0]['state']=='unloaded'


def test_presence_does_not_execute_async_capabilities(tmp_path):
    called=[]
    async def caps():
        called.append(1)
    context=NS(get_all_stars=lambda:[NS(name='astrbot_plugin_dynamics_learning',activated=True,star_cls=NS(get_dynamics_capabilities=caps))])
    data=panels.presence(context,'x',tmp_path)
    assert data['plugins'][1]['state']=='enabled' and called==[]


def test_status_requires_admin_and_owned_routes_unregister(monkeypatch):
    import astrbot.dashboard.api.auth as auth
    async def denied(raw):
        raise PermissionError()
    monkeypatch.setattr(auth,'require_dashboard_user',denied)
    monkeypatch.setattr(panels,'request',NS(_request=object()))
    context=NS(registered_web_apis=[],get_config=lambda:{'dashboard':{'username':'admin'}})
    context.register_web_api=lambda *args:context.registered_web_apis.append(args)
    owner=NS(context=context)
    panels.register_series_panels(owner,'test')
    first=owner._series_panels
    response=asyncio.run(first.status())
    assert response.status_code==403
    panels.register_series_panels(owner,'test')
    expected={"/test/series_presence", "/test/series_workspace/open/<target>",
              "/test/series_workspace/panel/<target>/<path:operation>"}
    assert len(context.registered_web_apis)==len(expected)
    assert {row[0] for row in context.registered_web_apis}==expected
    first.close()
    assert {row[0] for row in context.registered_web_apis}==expected
    owner._series_panels.close()
    assert not context.registered_web_apis
