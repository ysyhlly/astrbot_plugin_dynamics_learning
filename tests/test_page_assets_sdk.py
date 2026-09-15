"""Exercise real host asset rewriting outside the unit-test SDK doubles.

Set DYNAMICS_LEARNING_SDK_PYTHON to a Python executable with AstrBot installed.
The subprocess uses a temporary working directory; it does not start a bot.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


_CHECK = r'''
import importlib.util
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, unquote

if importlib.util.find_spec("astrbot") is None:
    sys.exit(77)

from astrbot.dashboard.services.plugin_page_service import PluginPageService

class Assets(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.urls = []
        self.theme = None
        self.color_scheme = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and "src" in attrs:
            self.urls.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.urls.append(attrs["href"])
        if tag == "html":
            self.theme = attrs.get("data-theme")
        if tag == "meta" and attrs.get("name") == "color-scheme":
            self.color_scheme = attrs.get("content")

page = Path(sys.argv[1])
source = page.read_text(encoding="utf-8")
original = Assets(source)
assert original.urls, "The page must declare its assets"
service = PluginPageService(plugin_manager=None)
prefix = "/api/plugin/page/content/astrbot_plugin_dynamics_learning/learning/"
bridge = "/api/plugin/page/bridge-sdk.js"

for theme in ("dark", "light"):
    rendered = Assets(service.rewrite_plugin_page_html(
        source, "astrbot_plugin_dynamics_learning", "learning", "index.html",
        theme=theme, extra_query_params={"asset_token": "test-only-token"},
    ))
    assert rendered.theme == theme, (theme, rendered.theme)
    assert rendered.color_scheme == theme, (theme, rendered.color_scheme)
    expected = []
    for url in original.urls:
        if urlsplit(url).path == bridge:
            expected.append(bridge)
        else:
            asset = service.resolve_referenced_asset_path("index.html", url)
            assert (page.parent / asset).is_file(), f"Missing declared asset: {asset}"
            expected.append(prefix + asset)
    if bridge not in expected:
        expected.append(bridge)
    assert [unquote(urlsplit(url).path) for url in rendered.urls] == expected
    for url in rendered.urls:
        assert parse_qs(urlsplit(url).query).get("asset_token") == ["test-only-token"], url
'''


def test_real_sdk_rewrites_page_assets_and_initial_theme(tmp_path: Path) -> None:
    """Every declared file keeps its load order, token and host initial theme."""
    python = os.environ.get("DYNAMICS_LEARNING_SDK_PYTHON", sys.executable)
    page = Path(__file__).resolve().parents[1] / "pages" / "learning" / "index.html"
    result = subprocess.run(
        [python, "-c", _CHECK, str(page)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("Real AstrBot SDK unavailable; set DYNAMICS_LEARNING_SDK_PYTHON")
    assert result.returncode == 0, result.stdout + result.stderr
