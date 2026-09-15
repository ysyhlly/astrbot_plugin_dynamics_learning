/** Session selection, profile, and evidence share one scope identity. */
function renderOverview(data) {
  const dataset = data.dataset || {};
  $("overview").innerHTML = [
    statCard("被检查样本", dataset.samples ?? "—", "来自人工标注，非全部消息"),
    statCard("会话", dataset.sessions ?? "—", "选择一个会话开始排查"),
    statCard("策略记录", data.policies ?? "—", "采纳仅记录在学习层"),
  ].join("");
  if (data.version) $("footVersion").textContent = `Dynamics Learning ${data.version}`;
  updateHeader();
}

function updateHeader() {
  const data = state.overview;
  const button = $("nextAction");
  if (!data) {
    $("workspaceStatus").textContent = "正在读取学习状态…";
    button.disabled = true;
    return;
  }
  const gate = state.quality?.dataset_gate;
  button.disabled = false;
  if (!data.dataset?.samples) {
    button.textContent = "导入标注"; button.dataset.next = "ingest";
  } else if (gate?.ok === false) {
    button.textContent = "查看缺失项"; button.dataset.next = "data";
  } else if (!data.has_report) {
    button.textContent = "运行分析"; button.dataset.next = "analyze";
  } else {
    button.textContent = "查看评测"; button.dataset.next = "evaluation";
  }
  $("workspaceStatus").textContent = `上次导入 ${moment(data.last_ingest_at)} · 上次分析 ${moment(data.last_analysis_at)}`;
}

function renderScopes(data) {
  const all = data?.rows || [];
  const query = state.scopeSearch.trim().toLowerCase();
  const rows = all.filter(row => String(row.scope_label || "").toLowerCase().includes(query))
    .filter(row => state.scopeFilter === "problems" ? (row.dominant_errors || []).length > 0
      : state.scopeFilter === "insufficient" ? row.confidence === "insufficient" : true);
  $("scopeCount").textContent = `${rows.length} / ${all.length} 个会话`;
  $("scopes").innerHTML = rows.length ? rows.map(row => `<button type="button"
    class="scope-item ${state.selectedScope === row.scope_hash ? "active" : ""}"
    data-scope="${esc(row.scope_hash)}" aria-pressed="${state.selectedScope === row.scope_hash}">
    <span class="scope-item-head"><strong>会话 ${esc(row.scope_label)}</strong><span class="scope-samples">${esc(row.samples)} 条</span></span>
    <span class="scope-item-problem">${esc((row.dominant_labels || []).join(" · ") || "暂无明确的主要问题")}</span>
    <span class="scope-item-meta">${esc(row.confidence_label || "证据状态未知")} · ${esc(row.annotation_days)} 个标注日</span>
  </button>`).join("") : `<div class="empty"><strong>${all.length ? "没有匹配的会话" : "从一批标注开始"}</strong>
    <p>${all.length ? "尝试其他标识或筛选条件。" : "导入 ChatDynamics 的人工标注后，会话与问题将在这里出现。"}</p>
    ${all.length ? '<button class="btn" data-clear-filters>清除筛选</button>' : '<button class="btn" data-go="data">前往数据与复盘</button>'}</div>`;
}

function setDetailTab(tab) {
  state.detailTab = tab === "samples" ? "samples" : "profile";
  for (const name of ["profile", "samples"]) {
    $("detail-" + name).hidden = state.detailTab !== name;
  }
  document.querySelectorAll("[data-detail-tab]").forEach(button => {
    const active = button.dataset.detailTab === state.detailTab;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    button.classList.toggle("active", active);
  });
}

let scopeRequest = 0;
let samplesRequest = 0;
function clearScope() {
  ++scopeRequest; ++samplesRequest;
  state.selectedScope = ""; state.scopeDetail = null; state.samples = null;
  state.samplesScope = ""; state.samplesPage = 1; state.samplesTotal = 0;
  $("panel-workspace").classList.remove("has-selection");
  $("scopeTitle").textContent = "选择一个会话";
  $("scopeSubtitle").textContent = "画像和样本证据会在这里呈现。";
  $("scopeDetail").innerHTML = '<div class="empty"><strong>从左侧选择会话</strong><p>先看主要问题，再查看支持它的样本。</p></div>';
  $("samples").innerHTML = '<p class="empty">先选择会话，再查看样本证据。</p>';
  $("samplesPage").textContent = "未选择会话";
  $("btnSamplesPrev").disabled = true; $("btnSamplesNext").disabled = true;
  if (state.scopes) renderScopes(state.scopes);
}

async function loadScope(scopeHash, { force = false } = {}) {
  if (!/^[0-9a-f]{64}$/.test(scopeHash)) throw new Error("会话标识无效，请重新选择会话。");
  const changed = state.selectedScope !== scopeHash;
  if (!changed && state.scopeDetail && !force) return;
  const request = ++scopeRequest;
  state.selectedScope = scopeHash;
  $("panel-workspace").classList.add("has-selection");
  $("scopeTitle").textContent = `会话 ${scopeHash.slice(0, 12)}`;
  if (changed) {
    ++samplesRequest;
    state.scopeDetail = null; state.samples = null; state.samplesScope = "";
    state.samplesPage = 1; state.samplesTotal = 0;
    $("scopeDetail").innerHTML = '<p class="empty">正在读取会话画像…</p>';
    $("samples").innerHTML = '<p class="empty">打开样本证据，查看这个会话的标注。</p>';
    $("samplesPage").textContent = "尚未加载";
    $("btnSamplesPrev").disabled = true; $("btnSamplesNext").disabled = true;
  }
  $("scopeSubtitle").textContent = "当前会话 · 仅统计被人工检查的样本";
  if (state.scopes) renderScopes(state.scopes);
  try {
    const payload = await call(ENDPOINTS.scope, { params: { id: scopeHash } });
    if (request !== scopeRequest || state.selectedScope !== scopeHash) return;
    state.scopeDetail = payload;
    renderScopeDetail(payload);
  } catch (error) {
    if (request !== scopeRequest) return;
    if (!state.scopeDetail) $("scopeDetail").innerHTML = "";
    sectionError("scopeDetail", "scope", error);
    throw error;
  }
}

async function loadScopes({ force = false } = {}) {
  return loadResource("scopes", { force });
}

async function loadSamples(page = 1, task = state.samplesTask) {
  const scope = state.selectedScope;
  if (!scope) { clearScope(); return; }
  const request = ++samplesRequest;
  $("btnSamplesPrev").disabled = true; $("btnSamplesNext").disabled = true;
  $("samples").setAttribute("aria-busy", "true");
  // A different filter is a different dataset: never retain the old rows under it.
  if (state.samplesTask !== task || state.samplesScope !== scope) {
    state.samples = null; state.samplesPage = 1; state.samplesTotal = 0;
    $("samples").innerHTML = '<p class="empty">正在读取样本…</p>';
  }
  state.samplesTask = task;
  try {
    const data = await call(ENDPOINTS.samples, { params: { task, page, page_size: 50, scope } });
    if (request !== samplesRequest || scope !== state.selectedScope) return;
    state.samplesPage = data.page || page; state.samplesTotal = data.total || 0;
    state.samplesScope = scope; state.samples = data;
    renderSamples(data);
  } catch (error) {
    if (request !== samplesRequest) return;
    if (!state.samples) $("samples").innerHTML = "";
    sectionError("samples", "samples", error);
    throw error;
  } finally {
    if (request === samplesRequest) {
      $("samples").setAttribute("aria-busy", "false");
      $("samplesPage").textContent = `第 ${state.samplesPage} / ${Math.max(1, Math.ceil(state.samplesTotal / 50))} 页 · ${state.samplesTotal} 条`;
      $("btnSamplesPrev").disabled = state.samplesPage <= 1;
      $("btnSamplesNext").disabled = state.samplesPage * 50 >= state.samplesTotal;
    }
  }
}

function renderSamples(data) {
  const rows = data?.rows || [];
  $("samples").innerHTML = rows.length ? `<table><thead><tr><th>任务 / 消息</th><th>预测</th><th>人工标注</th>
    <th>结果</th><th>证据</th></tr></thead><tbody>${rows.map(row => `<tr>
    <td>${esc(TASK_LABEL[row.task] || row.task)}<div class="sub">样本 ${esc(String(row.sample_id || "").slice(0, 8) || "—")}</div></td>
    <td>${esc(sampleLabel(row.task, row.predicted))}</td><td>${esc(sampleLabel(row.task, row.expected))}</td>
    <td><span class="tag ${row.correct === true ? "ok" : row.correct === false ? "bad" : ""}">${row.correct === true ? "一致" : row.correct === false ? "不一致" : "未知"}</span></td>
    <td><details><summary>查看证据</summary><p>${esc(errorLabel(row.error_type))}</p>
      <p>置信度 ${num(row.confidence)}</p><p>${(row.codes || []).map(esc).join(" · ") || "未记录证据码"}</p>
      <p class="sub">消息标识 ${esc(row.msg_id || "—")} · 样本不含正文。</p></details></td></tr>`).join("")}</tbody></table>`
    : '<p class="empty">这个会话在当前任务下没有样本，试试其他任务。</p>';
}

function sampleLabel(task, value) {
  const labels = task === "recipient" ? { bot: "机器人", other: "其他对象" }
    : task === "reply_admission" || task === "reply_outcome" ? { reply: "回复", silent: "不回复" } : {};
  return Object.hasOwn(labels, value) ? labels[value] : value || "未知";
}

function deltaBar(delta) {
  if (delta === null || delta === undefined) return "";
  const width = Math.min(100, Math.abs(delta) * 100 * 3);
  const cls = delta >= 0 ? "bad" : "ok";
  return `<span class="bar-track"><span class="bar-fill ${cls}" style="width:${width.toFixed(1)}%"></span></span>`;
}

function renderScopeDetail(payload) {
  const host = $("scopeDetail");
  if (!payload || !payload.profile) {
    host.innerHTML = '<p class="empty">没有这个会话的画像。</p>';
    return;
  }
  const profile = payload.profile;
  const deltas = (payload.deltas || []).filter((row) => row.support > 0);
  const rows = deltas.map((row) => `<tr>
      <td>${esc(row.label)}</td>
      <td class="num">${pct(row.raw_rate)}<div class="sub">${esc(row.count)}/${esc(row.support)}</div></td>
      <td class="num">${row.smoothed_rate === null ? "—" : pct(row.smoothed_rate)}</td>
      <td class="num">${row.loo_rate === null ? "—" : pct(row.loo_rate)}<div class="sub">${esc(row.loo_count)}/${esc(row.loo_support)}</div></td>
      <td class="num">${row.delta_pp === null ? "—" : (row.delta_pp >= 0 ? "+" : "") + row.delta_pp + "pp"}
        ${deltaBar(row.delta)}${row.comparable ? "" : `<div class="sub">${esc(row.reason)}</div>`}</td>
    </tr>`).join("");
  const diagnosis = payload.diagnosis || {};
  host.innerHTML = `<article class="rec diagnostic"><h3>${esc((payload.dominant_labels || []).join(" · ") || "暂无明确的主要问题")}</h3>
      <p>${esc(profile.confidence_reason || "当前证据不足以判断。")}</p>
      <button class="btn" data-open-samples>检查样本证据</button></article><div class="grid">
      ${statCard("被检查样本", profile.labelled_samples, `${profile.labelled_sessions} 个会话 · ${profile.annotation_days} 个标注日`)}
      ${statCard("证据充分度", profile.confidence_label, "指标来自被人工检查的样本")}
      ${statCard("候选覆盖", pct(((profile.candidate_metrics || {}).candidate_recall || {}).coverage),
        "记录候选集的话题标注占比")}
    </div><details><summary>话题候选的具体指标</summary><div class="grid">
      ${statCard("Recall@3", pct(((profile.candidate_metrics || {}).candidate_recall || {}).recall_at_3),
        "正确话题进入前 3 的比例")}
      ${statCard("选中准确率", pct(((profile.candidate_metrics || {}).selection_accuracy || {}).accuracy),
        "仅统计正确话题已进候选集的样本")}
    </div></details>
    <div class="table-host"><table><thead><tr>
      <th>问题</th><th class="num">本会话（原始）</th><th class="num">平滑后</th>
      <th class="num">其余会话</th><th class="num">偏差</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    <article class="rec ${diagnosis.code === "candidate_generation" || diagnosis.code === "ranking_or_scoring" ? "actionable" : "diagnostic"}">
      <h3>候选链诊断：${esc(diagnosis.label || "—")}</h3>
      <p class="rationale">${esc(diagnosis.detail || "")}</p>
      ${diagnosis.recommended_target ? `<div class="meta"><span class="tag">优化目标 ${esc(diagnosis.recommended_target)}</span></div>` : ""}
    </article>
    <details><summary>完整诊断依据</summary><ul class="bullets">${(payload.diagnostics || []).map((line) => `<li>${esc(line)}</li>`).join("")}</ul></details>
    <p class="hint">${esc(payload.note || "")}</p>`;
}

