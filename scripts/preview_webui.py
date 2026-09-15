"""Local UI preview with synthetic data and an in-memory plugin, never host data.

Run: python scripts/preview_webui.py --port 8766
"""
from __future__ import annotations

import argparse
import asyncio
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent), str(ROOT)]
from tests.conftest import _install_astrbot_double  # noqa: E402

_install_astrbot_double()
from astrbot.api.star import Context  # noqa: E402
from astrbot_plugin_dynamics_learning.main import DynamicsLearningPlugin  # noqa: E402
from astrbot_plugin_dynamics_learning.core.shadow import evaluate_shadow  # noqa: E402
from tests.factories import annotated_sessions, export_payload  # noqa: E402
from scripts.smoke import attach_shadow  # noqa: E402

BRIDGE = """
window.AstrBotPluginPage = {
  ready: async () => {},
  apiGet: (endpoint, params) => fetch('/preview-api/' + endpoint + '?' + new URLSearchParams(params || {})).then(r => r.json()),
  apiPost: (endpoint, body) => fetch('/preview-api/' + endpoint, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r => r.json())
};
"""


async def prepare():
    plugin = DynamicsLearningPlugin(Context(), {})
    rows = attach_shadow(annotated_sessions(sessions=8, per_session=30, start=time.time() - 7200))
    await plugin.ingest(source="export", payload=export_payload(rows))
    await plugin.run_analysis()
    return plugin, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    loop = asyncio.new_event_loop()
    plugin, rows = loop.run_until_complete(prepare())
    threading.Thread(target=loop.run_forever, daemon=True).start()

    async def dispatch(endpoint, params, body):
        if endpoint == "scope":
            return await plugin.scope_payload(params.get("id", ""))
        if endpoint == "samples":
            return await plugin.samples_payload(page=int(params.get("page", 1)),
                                                page_size=int(params.get("page_size", 50)),
                                                scope=params.get("scope", ""), task=params.get("task", ""))
        if endpoint == "policy":
            return await plugin.update_policy(body.get("version", ""), body.get("action", ""))
        if endpoint == "ingest":
            return await plugin.ingest(source="export", payload=export_payload(rows))
        if endpoint == "analyze":
            return await plugin.run_analysis()
        if endpoint == "shadow":
            return evaluate_shadow(await plugin.load_samples())
        if endpoint == "annotation_window":
            return {"sessions": [], "hint": "预览不读取真实宿主窗口。"}
        if endpoint in {"review", "reply_review"}:
            return {"state": "unavailable", "reason": "本地预览未连接模型。确定性统计仍可查看。"}
        methods = {"overview": plugin.overview_payload, "quality": plugin.quality_payload,
                   "scopes": plugin.scopes_payload, "report": plugin.report_payload,
                   "policies": plugin.policies_payload, "attribution": plugin.attribution_payload}
        if endpoint == "export":
            return {"preview": True, "samples": [row.as_dict() for row in await plugin.load_samples()]}
        if endpoint not in methods:
            raise ValueError("Unknown preview endpoint")
        return await methods[endpoint]()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **kwargs):
            super().__init__(*handler_args, directory=str(ROOT / "pages" / "learning"), **kwargs)

        def respond(self, data, content_type="application/json", status=200):
            payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/plugin/page/bridge-sdk.js":
                return self.respond(BRIDGE.encode(), "application/javascript")
            if parsed.path.startswith("/preview-api/"):
                return self.api(parsed)
            if parsed.path in {"/", "/index.html"}:
                text = (ROOT / "pages/learning/index.html").read_text(encoding="utf-8")
                theme = "dark" if parse_qs(parsed.query).get("theme") == ["dark"] else "light"
                text = text.replace('<html lang="zh-CN">', f'<html lang="zh-CN" data-theme="{theme}">')
                text = text.replace("</body>", '<div style="position:fixed;bottom:4px;right:12px;font-size:11px;opacity:.7;pointer-events:none">本地预览 · 合成数据</div></body>')
                return self.respond(text.encode(), "text/html; charset=utf-8")
            return super().do_GET()

        def do_POST(self):
            return self.api(urlparse(self.path))

        def api(self, parsed):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                params = {key: value[0] for key, value in parse_qs(parsed.query).items()}
                future = asyncio.run_coroutine_threadsafe(dispatch(parsed.path.rsplit("/", 1)[-1], params, body), loop)
                return self.respond({"status": "ok", "data": future.result(timeout=120)})
            except Exception as exc:
                return self.respond({"status": "error", "message": str(exc)}, status=500)

    print(f"Synthetic preview: http://127.0.0.1:{args.port}", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    finally:
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
