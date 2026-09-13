"""Pytest configuration and explicit AstrBot SDK doubles.

The unit tests must run without a live AstrBot, so a minimal double is installed
only when the real SDK is missing. Integration against the real host is the
host's own concern; here the goal is that every core module is exercised
deterministically.
"""
from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = TEST_DIR.parent
WORKSPACE_ROOT = PLUGIN_DIR.parent

# The workspace root comes first so the plugin is imported as the package
# `astrbot_plugin_dynamics_learning`, which is how AstrBot loads it too. The
# plugin directory stays on the path so `tests` still resolves as a package.
_sys_paths = [path for path in sys.path if path not in {str(PLUGIN_DIR), str(WORKSPACE_ROOT)}]
sys.path[:] = [str(WORKSPACE_ROOT), str(PLUGIN_DIR), *_sys_paths]


def _install_astrbot_double() -> None:
    """Install the double unconditionally, replacing any real SDK.

    A real `astrbot.core.star.Star` binds its KV store to the running bot's
    shared-preferences database, which unit tests must not touch or require.
    Nothing under test needs the real host: the contract with ChatDynamics is
    exercised through explicit preference rows instead.
    """
    if os.environ.get("DYNAMICS_LEARNING_REAL_SDK") == "1":  # pragma: no cover
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    star = types.ModuleType("astrbot.api.star")
    web = types.ModuleType("astrbot.api.web")

    class Star:
        def __init__(self, context=None, config=None):
            self.context = context
            self.config = config
            self._kv: dict = {}

        async def put_kv_data(self, key, value):
            self._kv[key] = value

        async def get_kv_data(self, key, default=None):
            return self._kv.get(key, default)

        async def delete_kv_data(self, key):
            self._kv.pop(key, None)

    class Context:
        def __init__(self):
            self.routes: list[tuple] = []

        def register_web_api(self, route, handler, methods, desc):
            self.routes.append((route, handler, tuple(methods), desc))

    class Request:
        args: dict = {}
        query: dict = {}
        username = "pytest"
        content_length = None

        async def json(self, default=None):
            return default if default is not None else {}

    def register(*_args, **_kwargs):
        return lambda target: target

    def json_response(data=None, *, status_code=200, headers=None):
        payload = dict(data) if isinstance(data, dict) else {"data": data}
        if status_code != 200:
            payload["status_code"] = status_code
        if headers:
            payload["headers"] = headers
        return payload

    def error_response(message, *, status_code=400, data=None, headers=None):
        return json_response(
            {"status": "error", "message": message, "data": data, "ok": False, "error": message},
            status_code=status_code, headers=headers)

    api.logger = logging.getLogger("astrbot-test")
    star.Context = Context
    star.Star = Star
    star.register = register
    web.json_response = json_response
    web.error_response = error_response
    web.request = Request()

    sys.modules.update({
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.star": star,
        "astrbot.api.web": web,
    })


_install_astrbot_double()


@pytest.fixture
def fake_context():
    from astrbot.api.star import Context

    return Context()


@pytest.fixture
def plugin(fake_context):
    """A real plugin instance backed by the SDK double's in-memory KV store."""
    from astrbot_plugin_dynamics_learning.main import DynamicsLearningPlugin

    return DynamicsLearningPlugin(fake_context, {})
