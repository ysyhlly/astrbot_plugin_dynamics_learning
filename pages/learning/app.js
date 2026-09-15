/** App coordination. View renderers and bridge transport live in separate files. */
const RESOURCE_VIEWS = {
  overview: { hosts: ["overview"], render: data => { state.overview = data; renderOverview(data); } },
  quality: { hosts: ["quality", "datasetGate"], render: data => { state.quality = data; renderQuality(data); updateHeader(); } },
  scopes: { hosts: ["scopes"], render: data => { state.scopes = data; renderScopes(data); } },
  report: { hosts: ["window", "errors", "recs", "candidates", "tuning", "eval"], render: data => {
    state.report = data.report;
    for (const render of [renderWindow, renderErrors, renderRecommendations, renderCandidates, renderTuning, renderEvaluation]) render(state.report);
  } },
  attribution: { hosts: ["attribution"], render: renderAttribution },
  shadow: { hosts: ["shadow", "shadowOperational"], render: renderShadow },
  policies: { hosts: ["policies"], render: renderPolicies },
  annotationWindow: { hosts: ["annotationWindow"], render: data => { state.annotationWindow = data; renderAnnotationWindow(data); } },
};
const VIEW_RESOURCES = {
  workspace: ["scopes"], data: ["annotationWindow"],
  evaluation: ["report", "attribution"], shadow: ["shadow"], policies: ["report", "policies"],
};

function connectionStatus() {
  const active = ["overview", "quality", ...VIEW_RESOURCES[state.view]];
  const failed = active.filter(key => resources[key]?.error);
  const loading = active.some(key => resources[key]?.pending);
  $("linkLamp").classList.toggle("on", !failed.length && !loading);
  $("linkLabel").textContent = failed.length ? "部分数据未更新" : loading ? "更新中…" : "已连接";
}

function loadResource(key, { force = false } = {}) {
  const config = RESOURCE_VIEWS[key];
  const entry = resources[key] || (resources[key] = {});
  if (entry.pending) {
    if (entry.pendingEpoch !== (entry.epoch || 0)) {
      return entry.pending.catch(() => {}).then(() => loadResource(key, { force: true }));
    }
    return entry.pending;
  }
  if (entry.loaded && !force && !entry.error && !entry.stale) return Promise.resolve(entry.value);
  config.hosts.forEach(id => $(id)?.setAttribute("aria-busy", "true"));
  const epoch = entry.epoch || 0;
  entry.pendingEpoch = epoch;
  entry.pending = call(ENDPOINTS[key]).then(data => {
    if (epoch !== (entry.epoch || 0)) return data;
    config.render(data);
    entry.value = data; entry.loaded = true; entry.stale = false; entry.error = null; entry.updatedAt = Date.now();
    return data;
  }).catch(error => {
    if (epoch !== (entry.epoch || 0)) return;
    entry.error = error;
    if (!entry.loaded) config.hosts.forEach(id => { if ($(id)) $(id).innerHTML = ""; });
    config.hosts.forEach(id => sectionError(id, key, error));
    throw error;
  }).finally(() => {
    entry.pending = null;
    config.hosts.forEach(id => $(id)?.setAttribute("aria-busy", "false"));
    connectionStatus();
  });
  connectionStatus();
  return entry.pending;
}

async function loadView({ force = false } = {}) {
  const results = await Promise.allSettled(["overview", "quality", ...VIEW_RESOURCES[state.view]]
    .map(key => loadResource(key, { force })));
  const failed = results.filter(result => result.status === "rejected");
  if (failed.length) throw new Error(`${failed.length} 个区域未能更新，可以在对应区域重试。`);
}

let refreshPending = null;
function refresh() {
  if (refreshPending) return refreshPending;
  refreshPending = (async () => {
    const jobs = [loadView({ force: true })];
    if (state.view === "workspace" && state.selectedScope) {
      jobs.push(loadScope(state.selectedScope, { force: true }));
      if (state.detailTab === "samples") jobs.push(loadSamples(state.samplesPage, state.samplesTask));
    }
    const results = await Promise.allSettled(jobs);
    const failure = results.find(result => result.status === "rejected");
    if (failure) throw failure.reason;
  })().finally(() => { refreshPending = null; });
  return refreshPending;
}

function routeValue(view, scope = state.selectedScope, tab = state.detailTab) {
  const params = new URLSearchParams();
  if (scope) params.set("scope", scope);
  if (tab === "samples") params.set("tab", tab);
  return `#${view}${params.size ? "?" + params.toString() : ""}`;
}

function navigate(view, { scope = state.selectedScope, tab = state.detailTab } = {}) {
  const hash = routeValue(view, scope, tab);
  if (window.location.hash !== hash) window.history.pushState(null, "", hash);
  applyRoute();
}

function setView(view) {
  state.view = Object.hasOwn(VIEW_META, view) ? view : "workspace";
  document.querySelectorAll("[data-page]").forEach(panel => { panel.hidden = panel.dataset.page !== state.view; });
  document.querySelectorAll("[data-view-btn]").forEach(button => {
    const active = button.dataset.viewBtn === state.view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("pageTitle").textContent = VIEW_META[state.view][0];
  $("pageDescription").textContent = VIEW_META[state.view][1];
  $("scopeBadge").textContent = state.view === "workspace" ? "按会话查看" : "全局数据";
}

function applyRoute() {
  const [requested, query = ""] = window.location.hash.slice(1).split("?");
  const params = new URLSearchParams(query);
  const scope = params.get("scope") || "";
  const validScope = /^[0-9a-f]{64}$/.test(scope) ? scope : "";
  setView(requested);
  setDetailTab(params.get("tab"));
  if (!validScope) clearScope();
  const tasks = [loadView()];
  if (validScope && state.view === "workspace") {
    tasks.push(loadScope(validScope));
    if (state.detailTab === "samples" && (state.samplesScope !== validScope || !state.samples)) {
      tasks.push(loadSamples(1, state.samplesTask));
    }
  } else if (validScope !== state.selectedScope) {
    clearScope();
    // Remember the route selection without claiming a profile has loaded.
    state.selectedScope = validScope;
  }
  Promise.allSettled(tasks).then(results => {
    if (results.some(result => result.status === "rejected")) notice("部分内容未加载，请在对应区域重试。", "error");
  });
}

async function withBusy(button, label, task) {
  if (button.disabled) return;
  const original = button.textContent;
  button.disabled = true; button.textContent = label;
  try { await task(); }
  catch (error) { notice(String(error.message || error), "error"); }
  finally {
    button.disabled = false; button.textContent = original;
    if (button.id === "nextAction") updateHeader();
    if (button.id === "btnSamplesPrev") button.disabled = state.samplesPage <= 1;
    if (button.id === "btnSamplesNext") button.disabled = state.samplesPage * 50 >= state.samplesTotal;
  }
}

async function ingestData() {
  const result = await call(ENDPOINTS.ingest, { method: "POST", body: { source: "host" } });
  for (const entry of Object.values(resources)) { entry.stale = true; entry.epoch = (entry.epoch || 0) + 1; }
  state.samples = null; state.samplesScope = "";
  if (result.ok === false) throw new Error("导入失败，请检查数据来源。");
  await refresh();
  notice(`已导入 ${result.annotations} 条标注，生成 ${result.imported_samples} 条样本。`, "ok");
}
async function analyzeData() {
  await call(ENDPOINTS.analyze, { method: "POST", body: { with_evaluation: true } });
  for (const entry of Object.values(resources)) { entry.stale = true; entry.epoch = (entry.epoch || 0) + 1; }
  await refresh();
  notice("分析完成，可前往评测查看结果。", "ok");
}

function bind() {
  document.querySelector(".skip-link").addEventListener("click", event => {
    event.preventDefault(); $("main").focus();
  });
  const on = (id, label, action) => $(id).addEventListener("click", () => withBusy($(id), label, action));
  on("btnRefresh", "更新中…", refresh);
  on("btnIngest", "导入中…", ingestData);
  on("btnAnalyze", "分析中…", analyzeData);
  on("nextAction", "处理中…", async () => {
    const action = $("nextAction").dataset.next;
    if (action === "ingest") await ingestData();
    else if (action === "analyze") await analyzeData();
    else navigate(action);
  });
  on("btnExport", "导出中…", async () => {
    downloadJson(await call(ENDPOINTS.export), "dynamics_learning.json"); notice("数据已导出。", "ok");
  });
  on("btnScopes", "更新中…", () => loadScopes({ force: true }));
  on("btnSamples", "读取中…", () => loadSamples(1, $("taskFilter").value));
  on("btnSamplesPrev", "读取中…", () => loadSamples(state.samplesPage - 1, state.samplesTask));
  on("btnSamplesNext", "读取中…", () => loadSamples(state.samplesPage + 1, state.samplesTask));
  $("taskFilter").addEventListener("change", () => loadSamples(1, $("taskFilter").value).catch(error => notice(error.message, "error")));
  on("btnAnnotationWindow", "更新中…", () => loadResource("annotationWindow", { force: true }));
  on("btnExportWindow", "导出中…", exportAnnotationWindow);
  on("btnReview", "解读中…", () => loadReview());
  on("btnReviewRegenerate", "重新解读中…", () => loadReview({ refresh: true }));
  on("btnReplyReview", "复盘中…", () => loadReplyReview());
  on("btnReplyRegenerate", "重新复盘中…", () => loadReplyReview({ refresh: true }));
  $("scopeSearch").addEventListener("input", event => { state.scopeSearch = event.target.value; renderScopes(state.scopes); });
  $("scopeFilter").addEventListener("change", event => { state.scopeFilter = event.target.value; renderScopes(state.scopes); });
  $("scopeBack").addEventListener("click", () => {
    const previous = state.selectedScope;
    navigate("workspace", { scope: "" });
    const item = document.querySelector(`[data-scope="${previous}"]`);
    (item || $("scopeSearch")).focus();
  });
  document.querySelectorAll("[data-view-btn]").forEach(button => button.addEventListener("click", () => {
    navigate(button.dataset.viewBtn); $("pageTitle").focus({ preventScroll: true });
  }));
  document.querySelector(".main-nav").addEventListener("keydown", event => {
    const tabs = [...document.querySelectorAll("[data-view-btn]")];
    const index = tabs.indexOf(document.activeElement);
    if (index < 0 || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
      : (index + (["ArrowRight", "ArrowDown"].includes(event.key) ? 1 : -1) + tabs.length) % tabs.length;
    navigate(tabs[next].dataset.viewBtn); tabs[next].focus();
  });
  document.querySelectorAll("[data-detail-tab]").forEach(button => {
    button.addEventListener("click", () => navigate("workspace", { tab: button.dataset.detailTab }));
    button.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const tab = event.key === "Home" ? "profile" : event.key === "End" ? "samples"
        : state.detailTab === "profile" ? "samples" : "profile";
      document.querySelector(`[data-detail-tab="${tab}"]`).focus();
      navigate("workspace", { tab });
    });
  });
  document.addEventListener("click", event => {
    const target = event.target.closest?.("button");
    if (!target) return;
    if (target.dataset.scope) {
      navigate("workspace", { scope: target.dataset.scope }); $("scopeTitle").focus({ preventScroll: true }); return;
    }
    if (target.hasAttribute("data-clear-filters")) {
      state.scopeSearch = ""; state.scopeFilter = "all";
      $("scopeSearch").value = ""; $("scopeFilter").value = "all"; renderScopes(state.scopes); return;
    }
    if (target.dataset.go) { navigate(target.dataset.go); return; }
    if (target.hasAttribute("data-open-samples")) { navigate("workspace", { tab: "samples" }); return; }
    if (target.dataset.retry) {
      const key = target.dataset.retry;
      withBusy(target, "重试中…", () => key === "scope" ? loadScope(state.selectedScope, { force: true })
        : key === "samples" ? loadSamples(state.samplesPage, state.samplesTask)
        : key === "review" ? loadReview() : key === "replyReview" ? loadReplyReview()
        : loadResource(key, { force: true })); return;
    }
    if (target.dataset.reviewRefresh) { withBusy(target, "解读中…", () => loadReview({ refresh: true })); return; }
    const { policy: version, action } = target.dataset;
    if (version && action) {
      withBusy(target, "处理中…", async () => {
        await call(ENDPOINTS.policy, { method: "POST", body: { version, action } });
        await Promise.all([loadResource("policies", { force: true }), loadResource("overview", { force: true })]);
        notice(`策略 ${version} 的状态已更新；是否在线采用由 ChatDynamics 决定。`, "ok");
      }); return;
    }
    if (target.dataset.accept) { navigate("policies"); notice("请审阅策略证据后，使用该版本允许的操作。"); }
  });
  window.addEventListener("popstate", applyRoute);
  window.addEventListener("hashchange", applyRoute);
}

async function main() {
  bind();
  setView("workspace"); clearScope();
  const api = await waitForBridge();
  if (api && typeof api.ready === "function") {
    try { await api.ready(); } catch (_) { /* Requests show actionable failures. */ }
  }
  applyRoute();
}
main();
