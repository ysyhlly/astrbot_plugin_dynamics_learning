/**
 * Dynamics Learning console page.
 *
 * Talks to this plugin's own endpoints only. Nothing here can change a
 * ChatDynamics setting: "accept" writes a policy record inside this plugin.
 */
const ENDPOINTS = {
  overview: "overview",
  samples: "samples",
  quality: "quality",
  attribution: "attribution",
  shadow: "shadow",
  scopes: "scopes",
  scope: "scope",
  ingest: "ingest",
  analyze: "analyze",
  report: "report",
  policies: "policies",
  policy: "policy",
  export: "export",
};

// Contract-health statuses. "unsupported" is deliberately its own look: it says
// the corpus cannot answer this question, which is a finding, not a failure.
const CAP_STATUS_CLASS = { ok: "ok", warning: "warn", unsupported: "bad", insufficient: "" };

// Two reply layers, because they are two questions: admission is "should this
// have entered the reply flow" (the router's cut), outcome is "did anything
// actually go out". A turn suppressed by 作息 is correct on the first and
// negative on the second, and one label for both hid exactly that.
const TASK_LABEL = {
  recipient: "对话对象",
  topic: "话题",
  reply: "回复准入（旧）",
  reply_admission: "回复准入",
  reply_outcome: "最终发送",
};
const VERDICT_LABEL = {
  accepted: { text: "可采纳", cls: "ok" },
  rejected: { text: "拒绝", cls: "bad" },
  insufficient: { text: "样本不足", cls: "warn" },
};
const CONFIDENCE_LABEL = {
  insufficient: "样本不足",
  low: "置信度低",
  moderate: "置信度中",
};

const state = { overview: null, report: null, quality: null, attribution: null, scopes: null, view: "overview" };

function $(id) {
  return document.getElementById(id);
}

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function num(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function pct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? (n * 100).toFixed(1) + "%" : "—";
}

function notice(message, kind = "") {
  const el = $("notice");
  if (!message) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.className = "notice" + (kind ? " " + kind : "");
  el.textContent = message;
}

function bridge() {
  return window.AstrBotPluginPage || null;
}

async function call(endpoint, { method = "GET", params, body } = {}) {
  const api = bridge();
  if (!api) throw new Error("Plugin Page bridge 不可用");
  const raw = method === "POST"
    ? await api.apiPost(endpoint, body || {})
    : await api.apiGet(endpoint, params || {});
  const payload = typeof raw === "string" ? JSON.parse(raw) : raw;
  if (!payload || typeof payload !== "object") throw new Error("空响应");
  if (payload.status === "error" || payload.ok === false) {
    throw new Error(payload.error || payload.message || "请求失败");
  }
  return payload.data !== undefined ? payload.data : payload;
}

function statCard(key, value, note) {
  return `<div class="stat"><div class="k">${esc(key)}</div><div class="v">${esc(value)}</div>
    <div class="n">${esc(note || "")}</div></div>`;
}

function renderOverview(data) {
  const dataset = data.dataset || {};
  const tasks = dataset.tasks || {};
  const url = (data.config || {}).source_plugin_id || "";
  $("overview").innerHTML = [
    statCard("样本", dataset.samples ?? 0, "分任务：" +
      Object.entries(tasks).map(([k, v]) => `${TASK_LABEL[k] || k} ${v}`).join(" / ")),
    statCard("会话", dataset.sessions ?? 0, "按会话分片存储"),
    statCard("策略记录", data.policies ?? 0, "采纳不会改动本体配置"),
    statCard("数据来源", url || "未配置", data.diagnostics && data.diagnostics.available === false
      ? "共享首选项不可用" : "只读共享首选项"),
  ].join("");

  const diag = data.diagnostics || {};
  if (diag.available === false) {
    notice("读取 ChatDynamics 共享首选项失败：" + esc(diag.error || "未知原因"), "error");
  } else if (diag.records !== undefined) {
    notice(`上次导入：${diag.records} 条标注，${diag.runtime_sessions || 0} 个会话，`
      + `无法归属 ${diag.unknown_sessions || 0} 条，格式异常 ${diag.malformed || 0} 条。`);
  }
}

function renderWindow(report) {
  const host = $("window");
  if (!report) {
    host.innerHTML = '<p class="empty">还没有分析结果，先点「运行分析」。</p>';
    return;
  }
  const rows = ["recipient", "topic", "reply_admission", "reply_outcome"].map((task) => {
    const row = (report.overview || {})[task] || {};
    const label = TASK_LABEL[task] || task;
    return statCard(label, row.total ? pct(row.accuracy) : "—", `标注 ${row.total ?? 0} 条`);
  });
  host.innerHTML = rows.join("");
  const window = report.overview || {};
  if (!window.samples_in_window && window.samples_total) {
    host.insertAdjacentHTML("beforeend",
      `<p class="empty">窗口内没有样本：${esc(window.samples_total)} 条标注都早于 `
      + `${esc(window.window_days)} 天。误差统计仍按全量样本计算。</p>`);
  }
}

function renderErrors(report) {
  const host = $("errors");
  const errors = (report && report.errors) || {};
  const entries = Object.entries(errors);
  if (!entries.length) {
    host.innerHTML = '<p class="empty">没有错误样本，或者还没有标注。</p>';
    return;
  }
  const rows = [];
  for (const [task, counts] of entries) {
    for (const [kind, count] of Object.entries(counts)) {
      rows.push(`<tr><td>${esc(TASK_LABEL[task] || task)}</td><td>${esc(kind)}</td>
        <td class="num">${esc(count)}</td></tr>`);
    }
  }
  host.innerHTML = `<table><thead><tr><th>任务</th><th>错误类型</th><th class="num">数量</th></tr></thead>
    <tbody>${rows.join("")}</tbody></table>`;
}

function renderRecommendations(report) {
  const host = $("recs");
  const rows = (report && report.recommendations) || [];
  if (!rows.length) {
    host.innerHTML = '<p class="empty">本批样本还没有产生建议。</p>';
    return;
  }
  host.innerHTML = rows.map((row) => {
    const kind = row.kind === "config_param" ? "actionable" : "diagnostic";
    const tag = row.kind === "config_param"
      ? `<span class="tag ${row.actionable ? "ok" : "warn"}">${esc(CONFIDENCE_LABEL[row.confidence] || row.confidence)}</span>`
      : '<span class="tag warn">工程诊断</span>';
    const verdict = (row.evidence || {}).evaluation_verdict;
    const verdictTag = verdict
      ? `<span class="tag ${(VERDICT_LABEL[verdict] || {}).cls || ""}">评测：${esc((VERDICT_LABEL[verdict] || {}).text || verdict)}</span>`
      : "";
    const actions = row.kind === "config_param" && row.actionable
      ? `<div class="actions">
           <button class="btn small" data-accept="${esc(row.param)}" data-after="${esc(row.after)}">采纳</button>
           <button class="btn small" data-ignore="${esc(row.param)}">忽略</button>
         </div>`
      : "";
    return `<article class="rec ${kind}">
      <h3>${esc(row.title)}</h3>
      <div class="detail">${esc(row.detail)}</div>
      <p class="rationale">${esc(row.rationale || "")}</p>
      <div class="meta">${tag}${verdictTag}
        <span class="tag">样本 ${esc(row.samples ?? 0)}</span>
        ${row.param ? `<span class="tag">${esc(row.param)}</span>` : ""}
      </div>
      ${actions}
    </article>`;
  }).join("");
}

const TUNE_DECISION_CLASS = {
  strong_promote: "ok",
  promote: "ok",
  candidate: "warn",
  needs_review: "warn",
  reject: "bad",
  rollback: "bad",
  no_change: "",
  insufficient: "",
};

function money(value) {
  return value === null || value === undefined ? "—" : num(value, 4);
}

function ratio(value) {
  return value === null || value === undefined ? "—" : (value >= 0 ? "+" : "") + pct(value);
}

function renderCandidates(report) {
  const host = $("candidates");
  const topic = (report && report.topic) || {};
  const metrics = topic.candidate_metrics || {};
  const recall = metrics.candidate_recall;
  const selection = metrics.selection_accuracy;
  const attribution = metrics.attribution;
  if (!recall || !recall.recorded) {
    host.innerHTML = `<p class="hint">候选召回：本批标注里没有记录话题候选集`
      + `（routing.topic_candidates），无法区分「候选生成没召回」和「召回了但排序选错」。</p>`;
    return;
  }
  const counts = (attribution || {}).counts || {};
  const labels = (attribution || {}).labels || {};
  const rows = Object.entries(counts)
    .filter(([kind]) => kind !== "correct")
    .map(([kind, count]) => `<tr><td>${esc(labels[kind] || kind)}</td>
      <td class="num">${esc(count)}</td>
      <td class="num">${pct((attribution.rates || {})[kind])}</td></tr>`).join("");
  host.innerHTML = `<article class="rec diagnostic">
    <h3>话题候选归因</h3>
    <div class="grid">
      ${statCard("候选覆盖", pct(recall.coverage), `${recall.recorded}/${recall.total} 条标注记录了候选集`)}
      ${statCard("Recall@1", pct(recall.recall_at_1), "正确话题出现在第 1 位")}
      ${statCard("Recall@3", pct(recall.recall_at_3), "正确话题出现在前 3 位")}
      ${statCard("选中准确率", pct(selection && selection.accuracy),
        `${(selection && selection.eligible) || 0} 条正确话题在候选集内`)}
    </div>
    <p class="rationale">
      「选中准确率」只在正确话题进入候选集的样本上计算——所以它只反映打分与阈值，
      「Recall@K」只反映候选生成。两者一起看才能判断该改 embedding 还是该改阈值。
    </p>
    ${rows ? `<div class="table-host"><table><thead><tr><th>归因</th><th class="num">条数</th>
      <th class="num">占比</th></tr></thead><tbody>${rows}</tbody></table></div>` : ""}
  </article>`;
}

// ---- error attribution chain -------------------------------------------
//
// One row per bucket, and one bucket per message. The table is a review order,
// not a causal claim: it answers "which layer do I open first", and a message
// that failed two layers still appears once — the second failure is listed
// under 「同时出错」 rather than inflating the counts.

const BUCKET_CLASS = {
  ok: "ok",
  recipient_error: "bad",
  topic_candidate_miss: "bad",
  topic_ranking_error: "warn",
  participation_error: "warn",
  gate_suppression: "",
  generation_failure: "",
  delivery_failure: "",
  unattributable: "",
};

function renderAttribution(data) {
  const host = $("attribution");
  if (!data || !data.messages) {
    host.innerHTML = '<p class="empty">还没有可归因的消息，先导入标注。</p>';
    return;
  }
  const counts = data.counts || {};
  const rates = data.rates || {};
  const labels = data.labels || {};
  const actions = data.actions || {};
  const rows = Object.keys(labels).map((bucket) => {
    const count = counts[bucket] ?? 0;
    return `<tr>
      <td>` + esc(labels[bucket]) + `<div class="sub">` + esc(bucket) + `</div></td>
      <td><span class="tag ` + (BUCKET_CLASS[bucket] || "") + `">` + esc(count) + `</span></td>
      <td class="num">` + (count ? pct(rates[bucket]) : "—") + `</td>
      <td>` + esc(actions[bucket] || "") + `</td>
    </tr>`;
  }).join("");
  const coverage = data.coverage || {};
  const also = Object.entries(data.also_failed || {}).map(([bucket, count]) =>
    esc(labels[bucket] || bucket) + " ×" + esc(count)).join("、");
  const reasons = Object.entries(data.suppression_reasons || {}).map(([reason, count]) =>
    esc(reason) + " ×" + esc(count)).join("、");
  const examples = (data.examples || []).slice(0, 5).map((row) => `<li>
      <span class="tag ` + (BUCKET_CLASS[row.bucket] || "") + `">` + esc(row.bucket_label) + `</span>
      ` + esc(row.reason || "—") + `</li>`).join("");
  host.innerHTML = `<div class="grid">
      ` + statCard("可归因消息", data.messages,
        "链路覆盖：话题 " + (coverage.topic ?? 0) + " · 回复 " + (coverage.admission ?? 0)
        + " · 最终结果 " + (coverage.outcome ?? 0)) + `
      ` + statCard("模型错误", data.model_errors,
        "收件人 / 候选生成 / 排序 / 参与准入，占 " + pct(data.model_error_rate)) + `
      ` + statCard("系统事件", data.system_events,
        "门禁 / 生成 / 发送，占 " + pct(data.system_event_rate)) + `
      ` + statCard("没有最终结果记录", data.outcome_unavailable,
        "schema 2 记录：只能归因到路由层") + `
    </div>
    <div class="table-host"><table><thead><tr>
      <th>归因</th><th>条数</th><th class="num">占比</th><th>该改哪一层</th>
    </tr></thead><tbody>` + rows + `</tbody></table></div>
    ` + (also ? `<p class="hint">同时出错（未计入上面的条数）：` + also + `</p>` : "") + `
    ` + (reasons ? `<p class="hint">压制原因：` + reasons + `</p>` : "") + `
    ` + (examples ? `<ul class="bullets">` + examples + `</ul>` : "") + `
    <p class="hint">` + (data.notes || []).map(esc).join("<br />") + `</p>`;
}

// ---- shadow A/B ---------------------------------------------------------
//
// The table is a 2x2 over the labelled turns: whether each decision was right.
// The two off-diagonal cells are the disagreement subset, and that is the only
// place the policy's effect exists — the diagonal is where the two decisions
// agreed and therefore where nothing about the policy can be learned.

const SHADOW_CELLS = [
  { key: "both_correct", label: "两边都对", note: "两个判定一致，与策略无关" },
  { key: "both_wrong", label: "两边都错", note: "两个判定一致，与策略无关" },
  { key: "baseline_only", label: "只有 baseline 对", note: "策略的代价" },
  { key: "shadow_only", label: "只有 shadow 对", note: "策略的收益" },
];

function renderOperationalShadow(data) {
  const host = $("shadowOperational");
  if (data && data.source_status === "unavailable") {
    host.innerHTML = '<p class="empty">读取 shadow telemetry 失败，请稍后重试；此状态不表示零比较。</p>';
    return;
  }
  if (!data || !data.available) {
    host.innerHTML = '<p class="empty">尚未收到独立 shadow telemetry 快照。</p>';
    return;
  }
  const window = data.window || {};
  const timestamp = (value) => typeof value === "number" && Number.isFinite(value)
    ? new Date(value * 1000).toLocaleString() : "—";
  const distribution = (title, rows, bucket = false) => `<h4>${esc(title)}</h4>
    <div class="table-host"><table><thead><tr><th>分组</th><th>比较</th><th>分歧</th><th>分歧率</th></tr></thead><tbody>`
    + (rows || []).map((row) => `<tr><td>${esc(bucket ? row.policy_id + " / " + row.host_version : row.key)}</td>
      <td>${esc(row.comparisons)}</td><td>${esc(row.disagreements)}</td><td>${pct(row.disagreement_rate)}</td></tr>`).join("")
    + `</tbody></table></div>`;
  host.innerHTML = `<div class="grid">`
    + statCard("真实比较", data.comparisons, "保留窗口内去重、校验后的独立记录")
    + statCard("真实分歧", data.disagreements, "分歧率 " + pct(data.disagreement_rate))
    + `</div><p class="hint">此分母仅覆盖保留窗口，不代表历史总流量，也不作为已标注训练样本。
      保留 ${esc(window.retention_seconds)} 秒，最多 ${esc(window.max_records)} 条。
      当前记录范围 ${esc(timestamp(window.from))} 至 ${esc(timestamp(window.to))}；快照更新 ${esc(timestamp(data.updated_at))}。
      重复 ${esc(data.duplicates)} 条，无效 ${esc(data.invalid)} 条，过期 ${esc(data.expired)} 条。</p>`
    + distribution("策略 / 本体版本", data.buckets, true)
    + distribution("准入原因", data.by_reason)
    + `<details><summary>会话与活跃时段分布</summary>`
    + distribution("匿名会话", data.by_session)
    + distribution("判定时段（UTC）", data.by_hour) + `</details>`;
}

function renderShadow(data) {
  renderOperationalShadow(data && data.operational);
  const host = $("shadow");
  if (!data || !data.rows) {
    host.innerHTML = '<p class="empty">还没有可评估的带标签 shadow 样本；真实比较覆盖见上方。</p>';
    return;
  }
  const table = data.table || {};
  const labelled = table.labelled ?? 0;
  const rows = SHADOW_CELLS.map((cell) => `<tr>
      <td>` + esc(cell.label) + `<div class="sub">` + esc(cell.key) + `</div></td>
      <td class="num">` + esc(table[cell.key] ?? 0) + `</td>
      <td>` + esc(cell.note) + `</td>
    </tr>`).join("");
  const gate = data.gate || {};
  const GATE_CLASS = { ok: "ok", warn: "warn", block: "bad" };
  const checks = (gate.checks || []).map((row) => `<tr>
      <td>` + esc(row.name) + `</td>
      <td><span class="tag ` + (GATE_CLASS[row.status] || "") + `">` + esc(row.status) + `</span></td>
      <td>` + esc(row.detail) + `</td>
    </tr>`).join("");
  const net = table.net_gain ?? 0;
  host.innerHTML = `<div class="grid">
      ` + statCard("记录 shadow 的消息", data.rows,
        "其中带人工标签 " + labelled + " 条；只有带标签的才能判对错") + `
      ` + statCard("策略产生分歧", table.changed ?? 0,
        "占比 " + pct(table.changed_rate) + "；其余 " + esc(table.same ?? 0) + " 条两个判定一致") + `
      ` + statCard("净收益", (net >= 0 ? "+" : "") + net,
        "分歧里 shadow 多赢 " + esc(table.shadow_only ?? 0) + " 条、多输 "
        + esc(table.baseline_only ?? 0) + " 条") + `
      ` + statCard("总体准确率", pct(table.overall_shadow_accuracy),
        "baseline " + pct(table.overall_baseline_accuracy) + "（被那 "
        + pct(table.changed_rate) + " 的分歧稀释）") + `
    </div>
    <div class="table-host"><table><thead><tr>
      <th>配对结果</th><th class="num">条数</th><th>含义</th>
    </tr></thead><tbody>` + rows + `</tbody></table></div>
    <p class="hint">分歧子集上的准确率：baseline ` + pct(table.subset_baseline_accuracy)
      + ` → shadow ` + pct(table.subset_shadow_accuracy) + `；
      95% 区间 ` + esc((data.interval || {}).lower === null || (data.interval || {}).lower === undefined
        ? "—"
        : "[" + num(data.interval.lower, 4) + ", " + num(data.interval.upper, 4) + "]") + `
      ` + ((data.interval || {}).crosses_zero ? "（跨 0）" : "") + `。
      覆盖 ` + esc(data.sessions ?? 0) + ` 个会话、` + esc(data.active_hours ?? 0)
      + ` 个活跃时段。</p>
    <article class="rec ` + (gate.ok ? "actionable" : "diagnostic") + `">
      <h3>进入 active 的门槛：` + (gate.ok ? "已通过" : "未通过") + `</h3>
      <div class="table-host"><table><thead><tr><th>检查</th><th>状态</th><th>说明</th></tr></thead>
        <tbody>` + checks + `</tbody></table></div>
    </article>
    <p class="hint">` + (data.notes || []).map(esc).join("<br />") + `</p>`;
}

function renderTuning(report) {
  const host = $("tuning");
  const runs = (report && report.tuning) || [];
  if (!runs.length) {
    host.innerHTML = '<p class="empty">还没有迭代结果。</p>';
    return;
  }
  host.innerHTML = runs.map((run) => {
    const cls = TUNE_DECISION_CLASS[run.decision] || "";
    const driftRows = (run.drift || []).map((row) =>
      `${esc(row.label || row.param)} ${num(row.baseline)}→${num(row.value)} (${ratio(row.delta_ratio)})`
    ).join("<br />") || "—";
    const steps = (run.steps || []).map((step) => `<tr>
      <td>${esc(step.index)}</td>
      <td>${(step.changes || []).map((m) => `${esc(m.param)} ${ratio(m.delta_ratio)}`).join("<br />") || "—"}</td>
      <td class="num">${money(step.step_delta)}</td>
      <td class="num">${money(step.cumulative_delta)}</td>
      <td>${esc(step.target_error || "—")}</td>
      <td class="num">${ratio(step.target_error_cumulative)}</td>
      <td>${step.safe ? '<span class="tag ok">通过</span>' : '<span class="tag bad">未通过</span>'}
        ${step.advanced ? '<span class="tag">继续</span>' : ""}
        ${(step.clamped || []).length ? '<span class="tag warn">触顶</span>' : ""}
        ${(step.guard_failures || []).length ? '<span class="tag bad">回退</span>' : ""}</td>
      <td>${esc((step.reasons || []).join("；"))}</td>
    </tr>`).join("");
    return `<article class="rec ${run.promoted ? "actionable" : "diagnostic"}">
      <h3>${esc(TASK_LABEL[run.task] || run.task)} ·
        <span class="tag ${cls}">${esc(run.decision_label)}</span>
        ${run.high_confidence ? '<span class="tag ok">高置信度</span>' : ""}</h3>
      <p class="rationale">${esc(run.stop_reason)}
        · 采纳 ${esc(run.adopted_steps)} 步 · 主指标 ${esc(run.primary_metric)}</p>
      <div class="meta">
        <span class="tag">单步 ≤ ${pct(run.rules.step_delta_ratio)}</span>
        <span class="tag">累计 ≤ ${pct(run.rules.cumulative_delta_ratio)}</span>
        <span class="tag">最多 ${esc(run.rules.max_steps)} 步</span>
        <span class="tag">采纳线 ${pct(run.rules.promote_improvement)} / 强采纳 ${pct(run.rules.strong_improvement)}</span>
      </div>
      <div class="detail">累计漂移：${driftRows}</div>
      ${steps ? `<div class="table-host"><table><thead><tr>
        <th>步</th><th>本步参数</th><th class="num">边际</th><th class="num">累计</th>
        <th>目标错误</th><th class="num">错误相对</th><th>判定</th><th>说明</th>
      </tr></thead><tbody>${steps}</tbody></table></div>` : ""}
    </article>`;
  }).join("");
}

function renderEvaluation(report) {
  const host = $("eval");
  const evaluation = report && report.evaluation;
  if (!evaluation) {
    host.innerHTML = '<p class="empty">还没有评测结果。</p>';
    return;
  }
  const verdict = VERDICT_LABEL[evaluation.verdict] || { text: evaluation.verdict, cls: "" };
  const head = `<div class="rec ${evaluation.verdict === "accepted" ? "actionable" : "diagnostic"}">
    <h3>结论：${esc(verdict.text)}</h3>
    <p class="rationale">${(evaluation.reasons || []).map(esc).join("<br />")}</p>
    <div class="meta">
      <span class="tag">训练 ${esc((evaluation.split || {}).train_samples ?? 0)} 条 /
        ${esc((evaluation.split || {}).train_sessions ?? 0)} 会话</span>
      <span class="tag">留出 ${esc((evaluation.split || {}).holdout_samples ?? 0)} 条 /
        ${esc((evaluation.split || {}).holdout_sessions ?? 0)} 会话</span>
    </div>
  </div>`;

  const tasks = Object.values(evaluation.tasks || {});
  const table = tasks.length ? `<table><thead><tr>
      <th>任务</th><th>指标</th><th class="num">baseline</th><th class="num">candidate</th><th class="num">变化</th>
      <th>95% 置信区间（按会话重采样）</th>
    </tr></thead><tbody>${tasks.map((task) => Object.entries(task.deltas || {}).map(([metric, row], index) => `
      <tr>
        <td>${index === 0 ? esc(TASK_LABEL[task.task] || task.task) + ` (${task.holdout})` : ""}</td>
        <td>${esc(metric)}${metric === task.primary_metric ? " ★" : ""}</td>
        <td class="num">${num(row.before)}</td>
        <td class="num">${num(row.after)}</td>
        <td class="num">${row.delta === null ? "—" : (row.delta >= 0 ? "+" : "") + num(row.delta)}</td>
        <td>${metric === task.primary_metric
          ? esc(task.interval || "—") + ((task.bootstrap || {}).crosses_zero
            ? ` <span class="tag bad">跨 0</span>` : "")
          : ""}</td>
      </tr>`).join("")).join("")}</tbody></table>` : '<p class="empty">没有可评测的任务。</p>';

  const candidate = evaluation.candidate;
  const deltas = candidate && candidate.deltas && candidate.deltas.length
    ? `<table><thead><tr><th>参数</th><th class="num">当前</th><th class="num">候选</th><th class="num">变化</th></tr></thead>
       <tbody>${candidate.deltas.map((row) => `<tr><td>${esc(row.label || row.param)}</td>
         <td class="num">${num(row.before)}</td><td class="num">${num(row.after)}</td>
         <td class="num">${row.delta >= 0 ? "+" : ""}${num(row.delta)}</td></tr>`).join("")}</tbody></table>`
    : '<p class="empty">本次评测没有选出新的参数。</p>';

  const candidateBlock = candidate
    ? `<h3 class="hint">候选参数 ${esc(candidate.version)}</h3>${deltas}`
    : "";

  // A fitted ambient scorer is measured but never gated: adopting it would
  // require ChatDynamics to expose a scorer replacement, which no configuration
  // key expresses today. It is shown as evidence for that gap, not as a change.
  const learned = Object.entries(evaluation.tasks || {})
    .filter(([, task]) => task.learned_scorer && task.learned_scorer.accuracy !== undefined);
  const learnedBlock = learned.length
    ? `<h3 class="hint">环境层学习评分（仅供诊断，无法通过配置导出）</h3>
       <table><thead><tr><th>任务</th><th class="num">记录评分准确率</th>
       <th class="num">学习评分准确率</th><th class="num">学习评分 F1</th><th>前提</th></tr></thead>
       <tbody>${learned.map(([name, task]) => `<tr>
         <td>${esc(TASK_LABEL[name] || name)}</td>
         <td class="num">${num((task.baseline || {}).accuracy)}</td>
         <td class="num">${num(task.learned_scorer.accuracy)}</td>
         <td class="num">${num(task.learned_scorer.f1)}</td>
         <td>${esc(task.learned_scorer.requires_host_support || "—")}</td>
       </tr>`).join("")}</tbody></table>`
    : "";

  const gate = (report && report.promotion) || {};
  const forward = report && report.forward;
  const gateBlock = `<article class="rec ${gate.verdict === "accepted" ? "actionable" : "diagnostic"}">
    <h3>采纳门槛：${esc((VERDICT_LABEL[gate.verdict] || {}).text || gate.verdict || "—")}</h3>
    <p class="rationale">${(gate.reasons || []).map(esc).join("<br />")}</p>
    <div class="meta">
      <span class="tag">会话留出集：${esc((VERDICT_LABEL[gate.session_verdict] || {}).text || gate.session_verdict || "—")}</span>
      <span class="tag">前向验证：${esc((VERDICT_LABEL[gate.forward_verdict] || {}).text || gate.forward_verdict || "未运行")}</span>
      ${gate.interval ? `<span class="tag">${esc(gate.interval)}</span>` : ""}
    </div>
    <p class="hint">会话留出集回答「换到没见过的会话还成立吗」，前向验证回答「换到更晚的数据还成立吗」；
    区间跨 0 时不下可采纳结论。</p>
    ${forward ? `<p class="hint">前向切分：训练 ${forward.split.train_samples} 条 /
      留出 ${forward.split.holdout_samples} 条（按标注时间，同一会话可能跨边界）</p>` : ""}
  </article>`;

  const strata = tasks
    .map((task) => ({ task: task.task, rows: ((task.strata || {}).groups || []).filter((row) => row.eligible) }))
    .filter((entry) => entry.rows.length);
  const strataBlock = strata.length
    ? `<h3 class="hint">按会话分层（只诊断，不产生本地策略）</h3>`
      + strata.map((entry) => `<table><thead><tr><th>${esc(TASK_LABEL[entry.task] || entry.task)}</th>
        <th class="num">支撑</th><th class="num">baseline</th><th class="num">candidate</th><th class="num">变化</th></tr></thead>
        <tbody>${entry.rows.slice(0, 8).map((row) => `<tr>
          <td>${esc(row.group.slice(0, 12))}</td>
          <td class="num">${esc(row.support)}</td>
          <td class="num">${num(row.baseline)}</td>
          <td class="num">${num(row.candidate)}</td>
          <td class="num">${(row.delta >= 0 ? "+" : "") + num(row.delta)}</td>
        </tr>`).join("")}</tbody></table>`).join("")
    : "";

  host.innerHTML = head + gateBlock + table + candidateBlock + strataBlock + learnedBlock;
}

// The state machine's actions. "采纳" moves a record to promoted, which is the
// only state /published offers to ChatDynamics — and even then the host decides
// whether to read it.
const POLICY_ACTIONS = [
  { action: "validate", label: "标记已验证" },
  { action: "shadow", label: "影子观察" },
  { action: "accept", label: "采纳" },
  { action: "ignore", label: "忽略" },
  { action: "rollback", label: "回滚" },
];

const POLICY_STATUS_CLASS = {
  proposed: "",
  validated: "warn",
  shadow: "warn",
  promoted: "ok",
  superseded: "",
  rolled_back: "bad",
  rejected: "bad",
};

function policyEvidence(row) {
  const holdout = row.holdout_result || {};
  const parts = [];
  if (holdout.primary_metric) {
    const delta = holdout.cumulative_delta;
    parts.push("留出集 " + esc(holdout.primary_metric) + " "
      + (delta === null || delta === undefined ? "—" : money(delta)));
  }
  if (row.target_error) parts.push("目标错误 " + esc(row.target_error));
  if (row.forward_result && Object.keys(row.forward_result).length) {
    parts.push("已做前向验证");
  } else {
    parts.push("未做前向验证");
  }
  if ((row.collateral_regressions || []).length) {
    parts.push("附带变化 " + esc(row.collateral_regressions.join("、")));
  }
  return parts.join("<br />");
}

function renderPolicies(data) {
  const host = $("policies");
  const rows = (data && data.rows) || [];
  if (!rows.length) {
    host.innerHTML = '<p class="empty">还没有策略记录。</p>';
    return;
  }
  const published = (data.published || []).length;
  host.innerHTML = `` + statCard("策略记录", data.total ?? rows.length,
      "状态分布：" + Object.entries(data.status_counts || {})
        .map(([key, value]) => key + " " + value).join(" / ")) + `
    ` + statCard("已发布给本体", published,
      "只有 promoted 会出现在 /published；是否采用由 ChatDynamics 决定") + `
    <table><thead><tr><th>版本</th><th>状态</th><th>来源</th><th>参数变化</th>
      <th>验证结果</th><th>操作</th></tr></thead>
    <tbody>${rows.map((row) => {
      const cls = POLICY_STATUS_CLASS[row.status] || "";
      return `<tr>
      <td>${esc(row.version)}</td>
      <td><span class="tag ${cls}">${esc(row.status_label || row.status)}</span></td>
      <td>${esc(row.source)}</td>
      <td>${(row.deltas || []).map((d) => `${esc(d.param)} ${num(d.before)}→${num(d.after)}`).join("<br />") || "—"}</td>
      <td class="sub">${policyEvidence(row)}</td>
      <td>${POLICY_ACTIONS.map((item) => `<button class="btn small" data-policy="${esc(row.version)}" data-action="${item.action}">${item.label}</button>`).join("")}</td>
    </tr>`;
    }).join("")}</tbody></table>`;
}

function renderSamples(data) {
  const host = $("samples");
  const rows = (data && data.rows) || [];
  if (!rows.length) {
    host.innerHTML = '<p class="empty">没有样本。</p>';
    return;
  }
  host.innerHTML = `<p class="hint">共 ${esc(data.total)} 条，第 ${esc(data.page)} 页。</p>
    <table><thead><tr><th>任务</th><th>会话</th><th>消息</th><th>预测</th><th>标注</th>
    <th>判定</th><th class="num">置信度</th><th>错误类型</th></tr></thead>
    <tbody>${rows.map((row) => `<tr>
      <td>${esc(TASK_LABEL[row.task] || row.task)}</td>
      <td>${esc(row.session)}</td>
      <td>${esc(row.msg_id)}</td>
      <td>${esc(row.predicted || "(未归属)")}</td>
      <td>${esc(row.expected || "(未归属)")}</td>
      <td><span class="tag ${row.correct ? "ok" : "bad"}">${row.correct ? "对" : "错"}</span></td>
      <td class="num">${num(row.confidence)}</td>
      <td>${esc(row.error_type)}</td>
    </tr>`).join("")}</tbody></table>`;
}


// ---- data contract health ----------------------------------------------

function contractLine(contract) {
  if (!contract) return "还没有导入记录：契约面（本体到底写了什么字段）暂无数据。";
  const candidates = contract.topic_candidates || {};
  const total = (candidates.missing || 0) + (candidates.empty || 0) + (candidates.nonempty || 0);
  return [
    `标注 ${contract.annotations_kept}/${contract.annotations_seen} 条（格式异常 ${contract.malformed}，`
      + `无法归属 ${contract.unknown_session}，截断 ${contract.truncated}）`,
    `decision_trace 缺失 ${contract.decision_trace_absent} 条`,
    `候选集字段 缺失 ${candidates.missing || 0}/${total}`,
  ].join(" · ");
}

function traceLine(trace) {
  // Which trace schema the host writes, and which ones this reader knows. The
  // capability matrix below can only say "no data yet" or "not in the record";
  // only this line can say "the host does not write that field at all".
  if (!trace) return "—";
  const observed = trace.observed || {};
  const keys = Object.keys(observed).sort();
  const rendered = keys.length
    ? keys.map((key) => `${key}×${observed[key]}`).join("、")
    : "还没有读到轨迹";
  const supported = (trace.supported || []).join(" / ");
  const unreadable = (trace.unreadable || []).length
    ? `；本插件读不了：${trace.unreadable.join("、")}`
    : "";
  return `本体写了 ${rendered}（读取端支持 ${supported}，最新 ${trace.latest}）${unreadable}`;
}

function renderQuality(data) {
  const host = $("quality");
  if (!data) {
    host.innerHTML = '<p class="empty">还没有数据契约状态。</p>';
    return;
  }
  const dataset = data.dataset || {};
  const capabilities = data.capabilities || {};
  const found = Object.values(capabilities);
  const needing = found.filter((row) => row.status !== "ok").length;
  const rows = found.map((row) => {
    const cls = CAP_STATUS_CLASS[row.status] || "";
    const reasons = (row.reasons || []).map(esc).join("<br />");
    return `<tr>
      <td>${esc(row.name)}<div class="sub">${esc(row.definition || "")}</div></td>
      <td><span class="tag ${cls}">${esc(row.status_label || row.status)}</span></td>
      <td class="num">${row.coverage === null || row.coverage === undefined ? "—" : pct(row.coverage)}</td>
      <td class="num">${esc(row.eligible)}/${esc(row.total)}</td>
      <td>${reasons || "—"}</td>
    </tr>`;
  }).join("");
  const blocked = (data.blocked || []).map((line) => `<li>${esc(line)}</li>`).join("");
  // The pre-learning gate. It is published with quality, not with the report, so
  // a reader can see why a corpus will produce no policy BEFORE running one.
  const gate = data.dataset_gate || {};
  const GATE_CLASS = { ok: "ok", warn: "warn", block: "bad" };
  const gateRows = (gate.checks || []).map((row) => `<tr>
      <td>${esc(row.name)}<div class="sub">${row.blocking ? "阻塞项" : "限定条件"}</div></td>
      <td><span class="tag ${GATE_CLASS[row.status] || ""}">${esc(row.status)}</span></td>
      <td>${esc(row.detail)}</td>
    </tr>`).join("");
  const gateBlock = (gate.checks || []).length
    ? `<article class="rec">
        <h3>数据门槛：${gate.ok ? "通过" : "未通过，本次不出策略"}</h3>
        <p class="rationale">${esc(gate.summary || "")}</p>
        <div class="table-host"><table><thead><tr><th>检查</th><th>状态</th><th>说明</th></tr></thead>
          <tbody>${gateRows}</tbody></table></div>
      </article>`
    : "";
  const findings = (data.contract_findings || []).map((line) => `<li>${esc(line)}</li>`).join("");
  host.innerHTML = `<div class="grid">
      ${statCard("样本", dataset.samples ?? 0, `会话 ${dataset.sessions ?? 0} · 作用域 ${dataset.scopes ?? 0}（${dataset.scope_level || "session"}）`)}
      ${statCard("需要补数据的能力", needing, "状态不是「正常」的能力数（不支持 / 警告 / 样本不足）")}
      ${statCard("解析异常轨迹", dataset.degraded_traces ?? 0, "契约降级的样本条数")}
    </div>
    <div class="table-host"><table><thead><tr>
      <th>能力</th><th>状态</th><th class="num">覆盖</th><th class="num">可用/合计</th><th>说明</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    ${gateBlock}
    ${blocked ? `<div class="rec diagnostic"><h3>当前不支持的分析</h3><ul class="bullets">${blocked}</ul></div>` : ""}
    <p class="hint">契约面：${esc(contractLine(data.contract))}</p>
    <p class="hint">轨迹 schema：${esc(traceLine(data.trace))}</p>
    ${findings ? `<ul class="bullets">${findings}</ul>` : ""}
    <p class="hint">${(data.notes || []).map(esc).join("<br />")}</p>`;
}

// ---- scope review profile ----------------------------------------------

function deltaBar(delta) {
  if (delta === null || delta === undefined) return "";
  const width = Math.min(100, Math.abs(delta) * 100 * 3);
  const cls = delta >= 0 ? "bad" : "ok";
  return `<span class="bar-track"><span class="bar-fill ${cls}" style="width:${width.toFixed(1)}%"></span></span>`;
}

function renderScopes(data) {
  const host = $("scopes");
  const rows = (data && data.rows) || [];
  if (!rows.length) {
    host.innerHTML = '<p class="empty">还没有标注样本，先导入并标注。</p>';
    return;
  }
  host.innerHTML = `<table><thead><tr>
      <th>会话</th><th class="num">被检查样本</th><th class="num">标注日</th><th>置信度</th>
      <th>主要问题</th><th>候选链诊断</th><th></th>
    </tr></thead><tbody>${rows.map((row) => `<tr>
      <td>${esc(row.scope_label)}</td>
      <td class="num">${esc(row.samples)}</td>
      <td class="num">${esc(row.annotation_days)}</td>
      <td>${esc(row.confidence_label)}</td>
      <td>${(row.dominant_labels || []).map(esc).join("、") || "—"}</td>
      <td>${esc(row.diagnosis_label || "—")}</td>
      <td><button class="btn small" data-scope="${esc(row.scope_hash)}">查看</button></td>
    </tr>`).join("")}</tbody></table>
    <p class="hint">${(data.notes || []).map(esc).join("<br />")}</p>`;
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
  host.innerHTML = `<div class="grid">
      ${statCard("被检查样本", profile.labelled_samples, `${profile.labelled_sessions} 个会话 · ${profile.annotation_days} 个标注日`)}
      ${statCard("置信度", profile.confidence_label, profile.confidence_reason)}
      ${statCard("候选覆盖", pct(((profile.candidate_metrics || {}).candidate_recall || {}).coverage),
        "记录候选集的话题标注占比")}
      ${statCard("Recall@3", pct(((profile.candidate_metrics || {}).candidate_recall || {}).recall_at_3),
        "正确话题进入前 3 的比例")}
      ${statCard("选中准确率", pct(((profile.candidate_metrics || {}).selection_accuracy || {}).accuracy),
        "仅统计正确话题已进候选集的样本")}
    </div>
    <div class="table-host"><table><thead><tr>
      <th>问题</th><th class="num">本会话（原始）</th><th class="num">平滑后</th>
      <th class="num">其余会话</th><th class="num">偏差</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    <article class="rec ${diagnosis.code === "candidate_generation" || diagnosis.code === "ranking_or_scoring" ? "actionable" : "diagnostic"}">
      <h3>候选链诊断：${esc(diagnosis.label || "—")}</h3>
      <p class="rationale">${esc(diagnosis.detail || "")}</p>
      ${diagnosis.recommended_target ? `<div class="meta"><span class="tag">优化目标 ${esc(diagnosis.recommended_target)}</span></div>` : ""}
    </article>
    <ul class="bullets">${(payload.diagnostics || []).map((line) => `<li>${esc(line)}</li>`).join("")}</ul>
    <p class="hint">${esc(payload.note || "")}</p>`;
}

async function loadScope(scopeHash) {
  const button = document.querySelector(`[data-scope="${scopeHash}"]`);
  const target = button instanceof HTMLElement ? button : $("btnScopes");
  await withBusy(target, "…", async () => {
    renderScopeDetail(await call(ENDPOINTS.scope, { params: { id: scopeHash } }));
  });
}

async function loadScopes() {
  const [listing, quality] = await Promise.all([
    call(ENDPOINTS.scopes),
    call(ENDPOINTS.quality),
  ]);
  state.scopes = listing;
  state.quality = quality;
  renderScopes(listing);
  return listing;
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("[data-view]").forEach((section) => {
    section.hidden = section.dataset.view !== view;
  });
  document.querySelectorAll("[data-view-btn]").forEach((button) => {
    const active = button.dataset.viewBtn === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "true" : "false");
  });
}

async function refresh() {
  try {
    const [overview, report, policies, quality, attribution, shadow] = await Promise.all([
      call(ENDPOINTS.overview),
      call(ENDPOINTS.report),
      call(ENDPOINTS.policies),
      call(ENDPOINTS.quality),
      call(ENDPOINTS.attribution),
      call(ENDPOINTS.shadow),
    ]);
    state.overview = overview;
    state.report = report.report;
    state.quality = quality;
    state.attribution = attribution;
    $("linkLamp").classList.add("on");
    $("linkLabel").textContent = "已连接";
    renderOverview(overview);
    renderAttribution(attribution);
    renderShadow(shadow);
    renderWindow(state.report);
    renderErrors(state.report);
    renderRecommendations(state.report);
    renderCandidates(state.report);
    renderTuning(state.report);
    renderEvaluation(state.report);
    renderPolicies(policies);
    renderQuality(quality);
  } catch (error) {
    $("linkLamp").classList.remove("on");
    $("linkLabel").textContent = "未连接";
    notice(String(error.message || error), "error");
  }
}

async function withBusy(button, label, task) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = label;
  try {
    await task();
  } catch (error) {
    notice(String(error.message || error), "error");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function bind() {
  $("btnRefresh").addEventListener("click", () => withBusy($("btnRefresh"), "刷新中…", async () => {
    await refresh();
    notice("");
  }));

  $("btnIngest").addEventListener("click", () => withBusy($("btnIngest"), "导入中…", async () => {
    const result = await call(ENDPOINTS.ingest, { method: "POST", body: { source: "host" } });
    if (result.ok === false) {
      notice("读取 ChatDynamics 共享首选项失败："
        + esc((result.diagnostics || {}).error || "未知原因"), "error");
    } else {
      notice(`导入完成：${result.annotations} 条标注 → ${result.imported_samples} 条学习样本，`
        + `覆盖 ${result.sessions} 个会话，累计 ${result.stored_samples} 条。`, "ok");
    }
    await refresh();
  }));

  $("btnAnalyze").addEventListener("click", () => withBusy($("btnAnalyze"), "分析中…", async () => {
    await call(ENDPOINTS.analyze, { method: "POST", body: { with_evaluation: true } });
    notice("分析完成。", "ok");
    await refresh();
  }));

  $("btnExport").addEventListener("click", () => withBusy($("btnExport"), "导出中…", async () => {
    const payload = await call(ENDPOINTS.export);
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "dynamics_learning.json";
    anchor.click();
    URL.revokeObjectURL(url);
    notice("已导出 JSON。", "ok");
  }));

  $("btnSamples").addEventListener("click", () => withBusy($("btnSamples"), "加载中…", async () => {
    const task = $("taskFilter").value;
    renderSamples(await call(ENDPOINTS.samples, { params: { task, page: 1, page_size: 50 } }));
  }));

  document.querySelectorAll("[data-view-btn]").forEach((button) => {
    button.addEventListener("click", async () => {
      const view = button.dataset.viewBtn || "overview";
      setView(view);
      // Loaded on first visit rather than on every refresh: the scope list is a
      // second view, not part of the overview a reader opens the page for.
      if (view === "scopes" && !state.scopes) {
        try {
          await loadScopes();
        } catch (error) {
          notice(String(error.message || error), "error");
        }
      }
    });
  });

  $("btnScopes").addEventListener("click", () => withBusy($("btnScopes"), "加载中…", async () => {
    const listing = await loadScopes();
    notice(`已加载 ${listing.total} 个会话画像。`, "ok");
  }));

  document.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.dataset.scope) {
      loadScope(target.dataset.scope);
      return;
    }
    const version = target.dataset.policy;
    const action = target.dataset.action;
    if (version && action) {
      withBusy(target, "…", async () => {
        await call(ENDPOINTS.policy, { method: "POST", body: { version, action } });
        notice(`策略 ${version} 已标记为 ${action}。ChatDynamics 配置未发生任何变化。`, "ok");
        await refresh();
      });
      return;
    }
    // A recommendation card has no policy version yet; accepting it only
    // records the intent, so it is reported as such rather than as an applied
    // configuration change.
    if (target.dataset.accept) {
      notice("参数建议需要先通过离线评测；请在下方策略版本中采纳已通过评测的候选。", "");
    }
  });
}

async function main() {
  bind();
  setView(state.view);
  const api = bridge();
  if (api && typeof api.ready === "function") {
    try {
      await api.ready();
    } catch (error) {
      /* the page still renders; refresh() reports the real failure */
    }
  }
  await refresh();
}

main();
