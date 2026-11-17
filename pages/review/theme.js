// Share Chat Dynamics' browser preference without depending on its private API.
(() => {
  const key = 'chat_dynamics_ui';
  const valid = value => value === 'day' || value === 'night';
  function paint(value) {
    const night = value === 'night';
    document.documentElement.dataset.uiTheme = night ? 'night' : 'day';
    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.setAttribute('aria-pressed', String(night));
      toggle.setAttribute('aria-label', night ? '切换到日间模式' : '切换到夜间模式');
      document.getElementById('theme-label').textContent = night ? '夜间模式' : '日间模式';
    }
  }
  let saved;
  try { saved = localStorage.getItem(key); } catch { /* Sandboxed storage can be unavailable. */ }
  const query = new URLSearchParams(location.search).get('ui');
  paint(valid(query) ? query : valid(saved) ? saved : 'day');
  document.addEventListener('DOMContentLoaded', () => {
    paint(document.documentElement.dataset.uiTheme);
    document.getElementById('theme-toggle').onclick = () => {
      const next = document.documentElement.dataset.uiTheme === 'night' ? 'day' : 'night';
      paint(next);
      try { localStorage.setItem(key, next); } catch { /* Current page remains usable. */ }
      try { const url = new URL(location.href); url.searchParams.set('ui', next); history.replaceState(history.state, '', url); } catch { /* Sandboxed history. */ }
    };
  });
  window.addEventListener('storage', event => {
    if (event.key === key && valid(event.newValue)) paint(event.newValue);
  });
})();
