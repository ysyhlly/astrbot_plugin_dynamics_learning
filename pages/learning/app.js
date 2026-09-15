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
  review: "review",
  replyReview: "reply_review",
  annotationWindow: "annotation_window",
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

const state = {
  overview: null, report: null, quality: null, attribution: null, scopes: null,
  review: null, replyReview: null, annotationWindow: null, view: "overview",
  // The sample browser pages server-side; the page only ever holds one page.
  samplesPage: 1, samplesTotal: 0, samplesTask: "",
};

// "下载一个 JSON" is one behaviour, so it is one function: the anchor is
// attached before it is clicked (a detached anchor is unreliable outside
// Chrome) and the object URL is released on a later turn, not synchronously
// under the click that is still starting the download.
function downloadJson(payload, filename) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

function stamp() {
  return new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
}

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
  const n = value == null || value === "" ? NaN : Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function pct(value) {
  const n = value == null || value === "" ? NaN : Number(value);
  return Number.isFinite(n) ? (n * 100).toFixed(1) + "%" : "—";
}

// One status line for the whole page. It takes **plain text** — several call
// sites used to hand it an esc() result, which then landed in textContent and
// rendered as &amp; / &lt;. Escaping belongs at the innerHTML boundary only.
function notice(message, kind = "") {
  const el = $("notice");
  if (!message) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.className = "notice" + (kind ? " " + kind : "");
  el.textContent = message;
  // A message the reader cannot dismiss is a message that stays wrong forever.
  const close = document.createElement("button");
  close.type = "button";
  close.className = "notice-close";
  close.setAttribute("aria-label", "关闭提示");
  close.textContent = "×";
  close.addEventListener("click", () => notice(""));
  el.appendChild(close);
}

// The page is a report, not an editor: a failed section must not blank the
// other twelve. Each host is rendered independently and a failure lands in the
// host it belongs to, where the reader is already looking.
function sectionError(hostId, label, error) {
  const host = $(hostId);
  if (host) {
    host.innerHTML = '<p class="notice error">' + esc(label) + "失败：" + esc(String(error.message || error)) + "</p>";
  }
}

function bridge() {
  return window.AstrBotPluginPage || null;
}

// AstrBot appends its page bridge after the page own scripts when the page does
// not load it itself, so at first paint the API may simply not be there yet.
// Waiting a moment is the difference between "the host is down" and "the host
// SDK landed 20 ms after we asked".
async function waitForBridge(timeoutMs = 4000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const api = bridge();
    if (api) return api;
    if (Date.now() >= deadline) return null;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

async function call(endpoint, { method = "GET", params, body, timeoutMs = 120000 } = {}) {
  const api = bridge();
  if (!api) {
    throw new Error("没有找到 AstrBot 插件页桥接（window.AstrBotPluginPage）："
      + "请从 AstrBot 面板的插件页入口打开本页，不要直接打开 index.html 文件。");
  }
  let timer;
  let raw;
  try {
    raw = await Promise.race([
      Promise.resolve().then(() => method === "POST"
        ? api.apiPost(endpoint, body || {}) : api.apiGet(endpoint, params || {})),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("请求超时；服务端可能仍在处理，请刷新检查结果。")), timeoutMs); }),
    ]);
  } finally { clearTimeout(timer); }
  const payload = typeof raw === "string" ? JSON.parse(raw) : raw;
  if (!payload || typeof payload !== "object") throw new Error("空响应");
  if (payload.status === "error" || payload.ok === false) {
    throw new Error(payload.error || payload.message || "请求失败");
  }
  return payload.data !== undefined ? payload.data : payload;
}

function moment(epoch) {
  const value = epoch == null || epoch === "" ? NaN : Number(epoch);
  if (!Number.isFinite(value) || value <= 0) return "还没有";
  return new Date(value * 1000).toLocaleString();
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
    // Freshness was computed by the API from the start and never shown; a report
    // with no "as of when" is a report the reader has to guess about.
    statCard("上次导入", moment(data.last_ingest_at), "共享首选项被读取的时间"),
    statCard("上次分析", moment(data.last_analysis_at), data.has_report ? "报告已生成" : "还没有运行过"),
  ].join("");

  // The console version comes from the API rather than a literal in the
  // markup: three copies of the version number is three chances to disagree.
  const foot = $("footVersion");
  if (foot && data.version) {
    foot.textContent = `Dynamics Learning ${data.version} · Shadow Learning：只学习、只分析、只推荐。`;
  }
  return diagnosticLine(data);
}

// What the last import did, as a sentence. Returned rather than written, so the
// caller decides whether it outranks whatever the reader just asked for.
function diagnosticLine(data) {
  const diag = (data && data.diagnostics) || {};
  if (diag.available === false) {
    return { text: "读取 ChatDynamics 共享首选项失败：" + (diag.error || "未知原因"), kind: "error" };
  }
  if (diag.records !== undefined) {
    return {
      text: `上次导入：${diag.records} 条标注，${diag.runtime_sessions || 0} 个会话，`
        + `无法归属 ${diag.unknown_sessions || 0} 条，格式异常 ${diag.malformed || 0} 条。`,
      kind: "",
    };
  }
  return null;
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

// The learner names its errors in snake_case tokens. The attribution card above
// has carried Chinese labels for the same vocabulary all along; this table used
// to print the tokens raw, so one page described one mistake in two languages.
const ERROR_LABEL = {
  correct: "正确",
  missed_bot: "漏认收件人（该认成机器人）",
  false_bot: "误认收件人（不该认成机器人）",
  wrong_recipient: "收件人判错",
  missed_reply: "漏回（该回没回）",
  premature_reply: "抢话（不该回却回了）",
  wrong_reply: "回复判错",
  wrong_topic: "话题归属错误",
  topic_miss: "候选生成缺失",
  topic_rank: "候选排序错误",
  undelivered_reply: "该发出但没发出",
  unsolicited_reply: "不该发出却发出了",
  unknown: "未知",
};

function errorLabel(kind) {
  return ERROR_LABEL[kind] || kind;
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
  let total = 0;
  for (const [task, counts] of entries) {
    for (const [kind, count] of Object.entries(counts)) {
      total += Number(count) || 0;
      rows.push(`<tr><td>${esc(TASK_LABEL[task] || task)}</td>
        <td>${esc(errorLabel(kind))}<div class="sub">${esc(kind)}</div></td>
        <td class="num">${esc(count)}</td></tr>`);
    }
  }
  host.innerHTML = `<p class="hint">共 ${total} 条错误样本。错误类型的中文是本页的读法，下面是记录里的原始值。</p>
    <table><thead><tr><th>任务</th><th>错误类型</th><th class="num">数量</th></tr></thead>
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
    // A recommendation has a parameter but no policy version, and accepting one
    // is a state-machine move the store performs on a version. So the card does
    // not offer "采纳" — it offered one anyway until now, and the click only
    // ever produced a notice explaining that it could not. The honest control
    // is the one that takes the reader to where adopting actually happens.
    const actions = row.kind === "config_param" && row.actionable
      ? `<div class="actions">
           <a class="btn small" href="#policiesCard" data-goto-policies="1">去「策略版本」采纳</a>
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
  if (value === null || value === undefined) return "—";
  const n = value == null || value === "" ? NaN : Number(value);
  if (!Number.isFinite(n)) return "—";  // pct() renders "—"; a sign in front of it is noise
  return (n >= 0 ? "+" : "") + pct(n);
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
      <td>${POLICY_ACTIONS.filter(item => (row.available_actions || []).includes(item.action)).map((item) => `<button class="btn small" data-policy="${esc(row.version)}" data-action="${item.action}">${item.label}</button>`).join("")}</td>
    </tr>`;
    }).join("")}</tbody></table>`;
}

let samplesRequest = 0;
async function loadSamples(page, task) {
  const request = ++samplesRequest;
  $("btnSamplesPrev").disabled = true;
  $("btnSamplesNext").disabled = true;
  try {
    const data = await call(ENDPOINTS.samples, { params: { task, page, page_size: 50 } });
    if (request !== samplesRequest) return;
    state.samplesPage = data.page || page;
    state.samplesTotal = data.total || 0;
    state.samplesTask = task;
    renderSamples(data);
  } finally {
    if (request === samplesRequest) {
      $("samplesPage").textContent = `第 ${state.samplesPage} 页 / 共 ${Math.max(1, Math.ceil(state.samplesTotal / 50))} 页`;
      $("btnSamplesPrev").disabled = state.samplesPage <= 1;
      $("btnSamplesNext").disabled = state.samplesPage * 50 >= state.samplesTotal;
    }
  }
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

// ---- the model-written contract review --------------------------------
//
// The matrix below is the input, not the answer: what a reader opens is the
// model's reading of it. Every number in the reply was checked against the
// digest it was given (the numbers rendered here come from that digest, not
// from the reply), rows the model skipped are marked as the raw verdict, and
// the deterministic table is folded in underneath rather than hidden.

const VERDICT_CLASS = { empty: "", usable: "ok", partial: "warn", blocked: "bad" };

function setQualityRawOpen(open) {
  const details = $("qualityRaw");
  if (details) details.open = open;
}

function reviewTags(row) {
  const tags = [];
  if (row.source === "deterministic") tags.push('<span class="tag">原始判定</span>');
  if (row.source === "model" && row.agrees === false) tags.push('<span class="tag warn">模型改判</span>');
  return tags.length ? " " + tags.join(" ") : "";
}

function renderReview(data) {
  const host = $("review");
  if (!host) return;
  const review = data && data.review;
  if (!review) {
    setQualityRawOpen(true);
    const reason = (data && data.reason) || "模型解读不可用。";
    host.innerHTML = `<p class="hint">${esc(reason)}</p>`;
    return;
  }
  const rows = (review.rows || []).map((row) => {
    const cls = CAP_STATUS_CLASS[row.status] || "";
    const detail = [row.explanation, row.next_action ? `下一步：${row.next_action}` : ""]
      .filter(Boolean).map(esc).join("<br />");
    return `<tr>
      <td>${esc(row.label || row.capability)}<div class="sub">${esc(row.capability)}</div></td>
      <td><span class="tag ${cls}">${esc(row.status_label || row.status)}</span>${reviewTags(row)}</td>
      <td class="num">${row.coverage === null || row.coverage === undefined ? "—" : pct(row.coverage)}</td>
      <td class="num">${esc(row.eligible)}/${esc(row.total)}</td>
      <td>${detail || "—"}</td>
    </tr>`;
  }).join("");
  const actions = (review.actions || []).map((line) => `<li>${esc(line)}</li>`).join("");
  const caveats = (review.caveats || []).map((line) => `<li>${esc(line)}</li>`).join("");
  const numbers = (review.unverified_numbers || []).length
    ? `<p class="notice error">模型引用了摘要里没有的数字：${esc(review.unverified_numbers.join("、"))}
       —— 这些数字没有展示，请以右侧计数为准。</p>`
    : "";
  const invented = (review.invented_capabilities || []).length
    ? `<p class="hint">模型写了摘要里没有的能力名，已忽略：${esc(review.invented_capabilities.join("、"))}</p>`
    : "";
  const stamp = typeof review.generated_at === "number"
    ? new Date(review.generated_at * 1000).toLocaleString()
    : (data.generated_at ? new Date(data.generated_at * 1000).toLocaleString() : "");

  host.innerHTML = `<article class="rec">
      <h3>模型解读${review.verdict_label
        ? ` · <span class="tag ${VERDICT_CLASS[review.verdict] || ""}">${esc(review.verdict_label)}</span>`
        : ""}</h3>
      <p class="rationale">${esc(review.headline || "")}</p>
      ${numbers}
      ${invented}
      <div class="table-host"><table><thead><tr>
        <th>能力</th><th>状态</th><th class="num">覆盖</th><th class="num">可用/合计</th><th>说明</th>
      </tr></thead><tbody>${rows}</tbody></table></div>
      ${actions ? `<h4>下一步</h4><ul class="bullets">${actions}</ul>` : ""}
      ${caveats ? `<h4>前提</h4><ul class="bullets">${caveats}</ul>` : ""}
      <p class="hint">
        由 ${esc(review.model || data.provider_id || "当前模型")} 生成于 ${esc(stamp || "—")}${data.state === "cached" ? "（缓存）" : ""}；
        计数与覆盖来自本插件，模型只决定这些行怎么读。
        <button type="button" class="btn small" data-review-refresh="1">重新解读</button>
      </p>
    </article>`;
  setQualityRawOpen(false);
}

const sectionRequests = { review: 0, replyReview: 0, annotationWindow: 0 };
async function loadReview({ refresh = false, quiet = false } = {}) {
  const request = ++sectionRequests.review;
  const host = $("review");
  if (!host) return;
  if (!quiet) host.innerHTML = '<p class="empty">正在让模型解读这份数据契约…</p>';
  try {
    const data = await call(ENDPOINTS.review, { params: refresh ? { refresh: 1 } : {} });
    if (request !== sectionRequests.review) return;
    state.review = data;
    renderReview(data);
    if (!data.review) throw new Error(data.reason || "模型解读暂不可用");
  } catch (error) {
    if (request !== sectionRequests.review) return;
    setQualityRawOpen(true);
    host.innerHTML = `<p class="hint">模型解读失败：${esc(error.message || error)}；下面是本插件自己的判定。</p>`;
    throw error;
  }
}

// ---- per-message reply post-mortem -------------------------------------
//
// The model is asked one question about each message and is deliberately not
// shown the human label or what the host decided: it sees the text, and its
// judgement is then placed beside both. Where the three disagree is the only
// place a reader learns something they did not already know. Nothing here is
// stored: the text makes one trip and the answer lives in the plugin process.

const HUMAN_LABEL = { true: "该回", false: "不该回" };

function humanCell(value) {
  if (value === null || value === undefined) return `<span class="sub">没有标注</span>`;
  return `<span class="tag ${value ? "ok" : ""}">${esc(HUMAN_LABEL[String(value)] || "—")}</span>`;
}

function hostCell(row) {
  const level = row.host_level ? `<span class="tag">${esc(row.host_level)}</span>` : `<span class="sub">未记录</span>`;
  const outcome = row.outcome || {};
  const actual = row.outcome_recorded
    ? `<span class="tag ${outcome.delivered ? "ok" : "bad"}">${esc(outcome.value_label || outcome.value || "—")}</span>`
      + (outcome.suppression_reason ? `<div class="sub">${esc(outcome.suppression_reason)}</div>` : "")
    : `<span class="sub">结果未记录</span>`;
  return `${level}<div class="sub">实际：${actual}</div>`;
}

function modelCell(row) {
  if (!row.decided) {
    return `<span class="tag">未判断</span><div class="sub">${esc(row.model_reason || "模型没有给出布尔判断")}</div>`;
  }
  // The reason is the only part of a judgement a reader can argue with, so it
  // sits next to the call rather than behind a tooltip.
  return `<span class="tag ${row.model_should_reply ? "ok" : ""}">${esc(HUMAN_LABEL[String(row.model_should_reply)])}</span>`
    + `<div class="sub">置信度 ${esc(row.model_confidence)}</div>`
    + (row.model_reason ? `<div class="sub">${esc(row.model_reason)}</div>` : "");
}

function renderReplyReview(data) {
  const host = $("replyReview");
  if (!host) return;
  const review = data && data.review;
  if (!review) {
    const stats = (data && data.stats) || {};
    const detail = stats.selected ? `（选中 ${esc(stats.selected)} 条，其中带正文 ${esc(stats.with_text || 0)} 条）` : "";
    host.innerHTML = `<p class="hint">${esc((data && data.reason) || "还没有复盘结果。")}${detail}</p>`;
    return;
  }
  const counts = review.counts || {};
  const rows = (review.rows || []).map((row) => `<tr>
      <td>${row.has_text ? esc(row.text) : `<span class="sub">（没有正文）</span>`}
        <div class="sub">${esc(row.msg_id)}${row.mentions_bot ? " · 提到了机器人" : ""}</div></td>
      <td>${humanCell(row.human_expected_reply)}</td>
      <td>${hostCell(row)}</td>
      <td>${modelCell(row)}</td>
      <td><span class="tag ${row.verdict === "missed" || row.verdict === "over_replied" ? "warn" : ""}">${esc(row.verdict_label || "")}</span>
        ${row.vs_human !== "unknown" ? `<span class="tag ${row.vs_human === "disagree" ? "warn" : ""}">${esc(row.vs_human_label)}</span>` : ""}</td>
    </tr>`).join("");
  const patterns = (review.patterns || []).map((line) => `<li>${esc(line)}</li>`).join("");
  host.innerHTML = `<div class="grid">
      ${statCard("模型判断", counts.decided ?? 0, `未判断 ${counts.undecided ?? 0} 条`)}
      ${statCard("与人工一致", `${counts.human_agree ?? 0}/${counts.with_human ?? 0}`, `不一致 ${counts.human_disagree ?? 0} 条`)}
      ${statCard("模型认为漏回", counts.missed ?? 0, `认为多回 ${counts.over_replied ?? 0} 条`)}
    </div>
    <p class="rationale">${esc(review.summary || "")}</p>
    <div class="table-host"><table><thead><tr>
      <th>消息</th><th>人工标注</th><th>本体 / 实际</th><th>模型判断</th><th>对照</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    ${patterns ? `<h4>反复出现的模式</h4><ul class="bullets">${patterns}</ul>` : ""}
    <p class="hint">由 ${esc(review.model || (data && data.provider_id) || "当前模型")} 复盘子
      ${esc(typeof review.generated_at === "number" ? new Date(review.generated_at * 1000).toLocaleString() : "—")}
      ${data.state === "cached" ? "（直接复用进程内结果）" : ""}；
      本插件不保存正文，复盘结果也不落盘。${esc((data && data.text_policy) || "")}</p>`;
}

async function loadReplyReview({ refresh = false } = {}) {
  const request = ++sectionRequests.replyReview;
  const host = $("replyReview");
  if (!host) return;
  host.innerHTML = `<p class="empty">正在让模型逐条复盘…</p>`;
  try {
    const data = await call(ENDPOINTS.replyReview, { params: refresh ? { refresh: 1 } : {} });
    if (request !== sectionRequests.replyReview) return;
    state.replyReview = data;
    renderReplyReview(data);
    if (!data.review) throw new Error(data.reason || "回复复盘暂不可用");
  } catch (error) {
    if (request !== sectionRequests.replyReview) return;
    host.innerHTML = `<p class="hint">回复复盘失败：${esc(error.message || error)}</p>`;
    throw error;
  }
}
// ---- the annotation window ---------------------------------------------
//
// Not a model call and not a judgement: this is the deadline. The host keeps a
// bounded message graph per session, so the messages a human can still label
// are the ones inside it, and the useful thing to say about them is how long
// that stays true — the annotation records themselves are permanent, and they
// are what keeps a message text.

function humanDuration(seconds) {
  const value = seconds == null || seconds === "" ? NaN : Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "—";
  if (value < 90) return Math.round(value) + " 秒";
  if (value < 5400) return Math.round(value / 60) + " 分钟";
  return (value / 3600).toFixed(1) + " 小时";
}

function wallMoment(epoch) {
  const value = epoch == null || epoch === "" ? NaN : Number(epoch);
  if (!Number.isFinite(value)) return "—";
  return new Date(value * 1000).toLocaleString();
}

function windowDeadline(seconds) {
  const value = seconds == null || seconds === "" ? NaN : Number(seconds);
  if (!Number.isFinite(value)) return "—";
  if (value <= 0) return "已超期，随时清理";
  return "约 " + humanDuration(value) + " 后";
}

function renderAnnotationWindow(data) {
  const host = $("annotationWindow");
  if (!host) return;
  const rows = (data && data.sessions) || [];
  const totals = (data && data.totals) || {};
  const limits = (data && data.limits) || {};
  if (!rows.length) {
    host.innerHTML = `<p class="empty">${esc((data && data.hint) || "还没有可标注的窗口。")}</p>`;
    return;
  }
  const table = rows.map((row) => `<tr>
      <td>${esc(row.session)}<div class="sub">${esc(row.session_hash)}</div></td>
      <td class="num">${esc(row.messages)}</td>
      <td class="num">${esc(row.with_text)}</td>
      <td class="num">${esc(row.unlabelled)}</td>
      <td class="num">${esc(humanDuration(row.span_seconds))}</td>
      <td class="num">${esc(windowDeadline(row.expires_in_seconds))}</td>
      <td>${esc(wallMoment(row.oldest_wall))}</td>
    </tr>`).join("");
  const retention = `${limits.max_nodes ?? "—"} 条 / ${humanDuration(limits.ttl_seconds)}`;
  host.innerHTML = `<div class="grid">
      ${statCard("窗口内消息", totals.messages ?? 0, `带正文 ${totals.with_text ?? 0} 条`)}
      ${statCard("还没标注", totals.unlabelled ?? 0, `已标注 ${totals.annotated ?? 0} 条`)}
      ${statCard("本体保留规则", retention, limits.reported_by_host ? "由本体快照报出" : "本体未报出，保留规则未知")}
    </div>
    <p class="rationale">${esc(data.hint || "")}</p>
    <div class="table-host"><table><thead><tr>
      <th>会话</th><th class="num">窗口内</th><th class="num">带正文</th><th class="num">未标注</th>
      <th class="num">跨度</th><th class="num">最旧的还剩</th><th>最旧一条的时间</th>
    </tr></thead><tbody>${table}</tbody></table></div>
    <p class="hint">${esc(data.text_policy || "")}</p>`;
}

async function loadAnnotationWindow() {
  const request = ++sectionRequests.annotationWindow;
  const host = $("annotationWindow");
  if (!host) return;
  try {
    const data = await call(ENDPOINTS.annotationWindow);
    if (request !== sectionRequests.annotationWindow) return;
    state.annotationWindow = data;
    renderAnnotationWindow(data);
  } catch (error) {
    if (request !== sectionRequests.annotationWindow) return;
    host.innerHTML = `<p class="hint">读取待标注窗口失败：${esc(error.message || error)}</p>`;
    throw error;
  }
}

async function exportAnnotationWindow(button) {
  const data = await call(ENDPOINTS.annotationWindow, { params: { export: 1 } });
  downloadJson(data, `chat_dynamics_annotation_window_${stamp()}.json`);
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
  document.querySelectorAll('[role="tabpanel"]').forEach(panel => { panel.hidden = panel.id !== `panel-${view}`; });
  document.querySelectorAll("[data-view]").forEach((section) => {
    section.hidden = section.dataset.view !== view;
  });
  document.querySelectorAll("[data-view-btn]").forEach((button) => {
    const active = button.dataset.viewBtn === view;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
}

let refreshPending = null;
function refresh() {
  if (refreshPending) return refreshPending;
  refreshPending = refreshSections().finally(() => { refreshPending = null; });
  return refreshPending;
}
async function refreshSections() {
  const sections = [
    [ENDPOINTS.overview, ["overview"], data => { state.overview = data; renderOverview(data); }],
    [ENDPOINTS.report, ["window", "errors", "recommendations", "candidates", "tuning", "eval"], data => {
      state.report = data.report;
      for (const render of [renderWindow, renderErrors, renderRecommendations, renderCandidates, renderTuning, renderEvaluation]) render(state.report);
    }],
    [ENDPOINTS.policies, ["policies"], renderPolicies],
    [ENDPOINTS.quality, ["quality"], data => { state.quality = data; renderQuality(data); }],
    [ENDPOINTS.attribution, ["attribution"], data => { state.attribution = data; renderAttribution(data); }],
    [ENDPOINTS.shadow, ["shadow"], renderShadow],
  ];
  const results = await Promise.allSettled(sections.map(async ([endpoint, hosts, render]) => {
    try { render(await call(endpoint)); }
    catch (error) { hosts.forEach(host => sectionError(host, endpoint, error)); throw error; }
  }));
  const failed = results.filter(result => result.status === "rejected");
  $("linkLamp").classList.toggle("on", failed.length === 0);
  $("linkLabel").textContent = failed.length ? "部分加载失败" : "已连接";
  loadAnnotationWindow().catch(error => notice(error.message, "error"));
  loadReview().catch(error => notice(error.message, "error"));
  if (failed.length) throw new Error(`${failed.length} 个区块加载失败：${failed.map(result => result.reason.message).join("；")}`);
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
  document.querySelector('[role="tablist"]').addEventListener("keydown", event => {
    const tabs = [...document.querySelectorAll("[data-view-btn]")];
    const index = tabs.indexOf(document.activeElement);
    if (index < 0 || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
      : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    tabs[next].focus(); tabs[next].click();
  });
  $("btnRefresh").addEventListener("click", () => withBusy($("btnRefresh"), "刷新中…", async () => {
    await refresh();
    notice("");
  }));

  $("btnIngest").addEventListener("click", () => withBusy($("btnIngest"), "导入中…", async () => {
    const result = await call(ENDPOINTS.ingest, { method: "POST", body: { source: "host" } });
    if (result.ok === false) {
      notice("读取 ChatDynamics 共享首选项失败："
        + ((result.diagnostics || {}).error || "未知原因"), "error");
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
    downloadJson(payload, "dynamics_learning.json");
    notice("已导出 JSON。", "ok");
  }));

  $("btnSamples").addEventListener("click", () => withBusy($("btnSamples"), "加载中…", async () => {
    await loadSamples(1, $("taskFilter").value);
  }));

  for (const [id, offset] of [["btnSamplesPrev", -1], ["btnSamplesNext", 1]]) {
    $(id).addEventListener("click", () => loadSamples(state.samplesPage + offset, state.samplesTask).catch(error => notice(error.message, "error")));
  }
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

  $("btnAnnotationWindow").addEventListener("click", () => withBusy($("btnAnnotationWindow"), "读取中…", async () => {
    await loadAnnotationWindow();
    notice("已读取本体当前的标注窗口。", "ok");
  }));

  $("btnExportWindow").addEventListener("click", () => withBusy($("btnExportWindow"), "导出中…", async () => {
    await exportAnnotationWindow();
    notice("窗口已导出到你本机；本插件没有保存正文。", "ok");
  }));

  $("btnReplyReview").addEventListener("click", () => withBusy($("btnReplyReview"), "复盘中…", async () => {
    await loadReplyReview();
    notice("复盘完成；正文没有被写进本插件的存储。", "ok");
  }));

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
    if (target.dataset.reviewRefresh) {
      withBusy(target, "解读中…", async () => {
        await loadReview({ refresh: true, quiet: true });
        notice("已重新解读当前数据契约。", "ok");
      });
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
  const api = await waitForBridge();
  if (api && typeof api.ready === "function") {
    try {
      await api.ready();
    } catch (error) {
      /* the page still renders; refresh() reports the real failure */
    }
  }
  await refresh().catch(error => notice(error.message, "error"));
}

main();
