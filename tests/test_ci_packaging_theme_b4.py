"""Batch 4 regression tests: theme.js null-safety, offline series_panels import, packaging hygiene & CI gates."""
from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from astrbot_plugin_dynamics_learning import series_panels
from astrbot_plugin_dynamics_learning.tools.package_release import (
    INSTALL_BLACKLIST_FILES,
    SOURCE_ONLY,
    build,
    verify_install_archive,
)

ROOT = Path(__file__).resolve().parents[1]


def _node_path() -> str | None:
    return shutil.which("node")


def test_theme_js_syntax_and_null_safety_via_node():
    node = _node_path()
    if not node:
        pytest.skip("node not available in PATH")

    theme_js = ROOT / "pages" / "review" / "theme.js"
    check = subprocess.run([node, "--check", str(theme_js)], capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stderr

    # Run real Node.js VM tests covering missing DOM nodes, missing label, sandboxed storage,
    # already-loaded readyState, ?ui / ?theme query params, and storage sync events.
    harness = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

const source = fs.readFileSync(process.argv[1], 'utf8');

function makeElement(id) {
  const attrs = {};
  return {
    id,
    attrs,
    textContent: '',
    onclick: null,
    setAttribute(k, v) { attrs[k] = String(v); },
    getAttribute(k) { return attrs[k]; },
  };
}

function runScenario(opts) {
  const elements = {};
  if (opts.hasToggle) elements['theme-toggle'] = makeElement('theme-toggle');
  if (opts.hasLabel) elements['theme-label'] = makeElement('theme-label');

  const docListeners = {};
  const winListeners = {};
  const storageMap = new Map(Object.entries(opts.storage || {}));
  let replacedUrl = null;

  const doc = {
    readyState: opts.readyState || 'loading',
    documentElement: opts.noRoot ? null : { dataset: opts.noDataset ? null : {} },
    getElementById(id) { return elements[id] || null; },
    addEventListener(type, fn) {
      docListeners[type] = docListeners[type] || [];
      docListeners[type].push(fn);
    },
  };
  const win = {
    addEventListener(type, fn) {
      winListeners[type] = winListeners[type] || [];
      winListeners[type].push(fn);
    },
  };
  const localStorage = opts.throwStorage
    ? {
        getItem() { throw new Error('SecurityError'); },
        setItem() { throw new Error('SecurityError'); },
      }
    : {
        getItem(k) { return storageMap.has(k) ? storageMap.get(k) : null; },
        setItem(k, v) { storageMap.set(k, String(v)); },
      };
  const location = {
    search: opts.search || '',
    href: 'https://example.test/review' + (opts.search || ''),
  };
  const history = {
    state: null,
    replaceState(_state, _title, url) {
      if (opts.throwHistory) throw new Error('HistoryBlocked');
      replacedUrl = String(url);
    },
  };

  const sandbox = {
    document: doc,
    window: win,
    localStorage,
    location,
    history,
    URL,
    URLSearchParams,
  };
  vm.runInNewContext(source, sandbox);

  return {
    doc,
    elements,
    storageMap,
    getReplacedUrl: () => replacedUrl,
    fireDomReady() {
      for (const fn of docListeners['DOMContentLoaded'] || []) fn();
    },
    fireStorage(ev) {
      for (const fn of winListeners['storage'] || []) fn(ev);
    },
  };
}

// 1. Missing both #theme-toggle and #theme-label must NOT throw on init, DOMContentLoaded, or storage event.
{
  const s = runScenario({ hasToggle: false, hasLabel: false });
  assert.strictEqual(s.doc.documentElement.dataset.uiTheme, 'day');
  s.fireDomReady();
  s.fireStorage({ key: 'chat_dynamics_ui', newValue: 'night' });
  assert.strictEqual(s.doc.documentElement.dataset.uiTheme, 'night');
  s.fireStorage(null); // null event guard
}

// 2. Present #theme-toggle WITHOUT #theme-label must NOT throw and still toggle cleanly.
{
  const s = runScenario({ hasToggle: true, hasLabel: false });
  s.fireDomReady();
  const toggle = s.elements['theme-toggle'];
  assert.strictEqual(toggle.getAttribute('aria-pressed'), 'false');
  assert.strictEqual(typeof toggle.onclick, 'function');
  toggle.onclick();
  assert.strictEqual(s.doc.documentElement.dataset.uiTheme, 'night');
  assert.strictEqual(toggle.getAttribute('aria-pressed'), 'true');
  assert.strictEqual(s.storageMap.get('chat_dynamics_ui'), 'night');
}

// 3. Full DOM with readyState='complete' binds immediately without waiting for DOMContentLoaded.
{
  const s = runScenario({ hasToggle: true, hasLabel: true, readyState: 'complete' });
  const toggle = s.elements['theme-toggle'];
  const label = s.elements['theme-label'];
  assert.strictEqual(label.textContent, '日间模式');
  assert.strictEqual(typeof toggle.onclick, 'function');
  toggle.onclick();
  assert.strictEqual(s.doc.documentElement.dataset.uiTheme, 'night');
  assert.strictEqual(label.textContent, '夜间模式');
  assert.strictEqual(toggle.getAttribute('aria-label'), '切换到日间模式');
  assert.ok(s.getReplacedUrl().includes('ui=night'));
  toggle.onclick();
  assert.strictEqual(s.doc.documentElement.dataset.uiTheme, 'day');
  assert.strictEqual(label.textContent, '日间模式');
}

// 4. Query param precedence (?ui overrides saved; ?theme=dark fallback works when ?ui absent).
{
  const s1 = runScenario({ hasToggle: true, hasLabel: true, storage: { chat_dynamics_ui: 'day' }, search: '?ui=night' });
  assert.strictEqual(s1.doc.documentElement.dataset.uiTheme, 'night');
  const s2 = runScenario({ hasToggle: true, hasLabel: true, storage: { chat_dynamics_ui: 'day' }, search: '?theme=dark' });
  assert.strictEqual(s2.doc.documentElement.dataset.uiTheme, 'night');
}

// 5. Sandboxed localStorage / history throwing and missing documentElement.dataset must not throw.
{
  const s = runScenario({ hasToggle: true, hasLabel: true, throwStorage: true, throwHistory: true, readyState: 'complete' });
  s.elements['theme-toggle'].onclick();
  assert.strictEqual(s.doc.documentElement.dataset.uiTheme, 'night');

  const sNoRoot = runScenario({ hasToggle: false, hasLabel: false, noRoot: true });
  sNoRoot.fireDomReady();
}
console.log('THEME_JS_OK');
"""
    res = subprocess.run([node, "-e", harness, str(theme_js)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"stderr={res.stderr}\nstdout={res.stdout}"
    assert "THEME_JS_OK" in res.stdout


def test_series_panels_imports_cleanly_offline(tmp_path):
    assert hasattr(series_panels, "presence")
    assert hasattr(series_panels, "require_series_admin")
    assert hasattr(series_panels, "series_llm_call")
    # Confirm presence() executes offline with empty context
    out = series_panels.presence(None, "astrbot_plugin_dynamics_learning", tmp_path)
    assert isinstance(out, dict)
    assert any(row["id"] == "astrbot_plugin_dynamics_learning" for row in out.get("plugins", []))


def test_package_release_single_top_level_dir_and_dev_blacklist(tmp_path):
    records = build(ROOT, tmp_path)
    install_rec = next(r for r in records if r["kind"] == "install")
    install_zip = tmp_path / install_rec["file"]

    verified = verify_install_archive(install_zip, "astrbot_plugin_dynamics_learning")
    assert verified["plugin"] == "astrbot_plugin_dynamics_learning"
    assert verified["files"] > 20

    with zipfile.ZipFile(install_zip) as zf:
        names = zf.namelist()
        top_dirs = {n.split("/", 1)[0] for n in names}
        assert top_dirs == {"astrbot_plugin_dynamics_learning"}
        rels = {n.split("/", 1)[1] for n in names}
        assert "main.py" in rels
        assert "metadata.yaml" in rels
        assert "_conf_schema.json" in rels
        assert "pages/review/theme.js" in rels
        assert "series_panels.py" in rels

        # Verify zero dev/CI/build blacklist files leaked
        for rel in rels:
            parts = rel.split("/")
            assert parts[0] not in SOURCE_ONLY, f"Leaked source-only dir: {rel}"
            assert parts[-1] not in INSTALL_BLACKLIST_FILES, f"Leaked dev file: {rel}"
            assert "__pycache__" not in parts
            assert not rel.endswith((".pyc", ".sqlite3", ".db"))

    # Negative checks: verify_install_archive rejects bad top-level dirs and dev files
    for label, bad_entry in (
        ("stray-root-file", "README.md"),
        ("wrong-top-dir", "wrong_plugin/main.py"),
        ("github-workflow", "astrbot_plugin_dynamics_learning/.github/workflows/ci.yml"),
        ("package-lock", "astrbot_plugin_dynamics_learning/package-lock.json"),
        ("gitignore", "astrbot_plugin_dynamics_learning/.gitignore"),
        ("build-release", "astrbot_plugin_dynamics_learning/build_release.py"),
    ):
        bad_zip = tmp_path / f"bad-{label}.zip"
        with zipfile.ZipFile(install_zip) as src, zipfile.ZipFile(bad_zip, "w") as dst:
            for item in src.infolist():
                dst.writestr(item, src.read(item.filename))
            dst.writestr(bad_entry, b"leaked")
        with pytest.raises(ValueError):
            verify_install_archive(bad_zip, "astrbot_plugin_dynamics_learning")


def test_ci_workflow_enforces_offline_pytest_and_packaging_gates():
    ci_path = ROOT / ".github" / "workflows" / "ci.yml"
    raw = ci_path.read_text(encoding="utf-8")
    assert "offline-and-packaging:" in raw
    assert "lint-and-unit:" in raw
    assert "series_panels" in raw
    assert "tools/package_release.py" in raw
    assert "verify_install_archive" in raw
    assert "node --check" in raw
    assert "python -m ruff check ." in raw
    assert "python -m pytest tests" in raw

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None
    if yaml is not None:
        parsed = yaml.safe_load(raw)
        assert isinstance(parsed, dict)
        assert "offline-and-packaging" in parsed["jobs"]
        assert "lint-and-unit" in parsed["jobs"]
        # Offline job must NOT check out AstrBotDevs/AstrBot
        offline_steps = parsed["jobs"]["offline-and-packaging"]["steps"]
        assert not any(
            (step.get("with") or {}).get("repository") == "AstrBotDevs/AstrBot"
            for step in offline_steps
        )
