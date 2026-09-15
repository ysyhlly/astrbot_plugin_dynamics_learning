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

function shadowExperiments(data) {
  const experiments = data.experiments || [];
  const rows = experiments.map(experiment => {
    const identity = experiment.identity || {}, gate = experiment.gate || {};
    const table = experiment.table || {};
    return `<tr><td>${esc(identity.experiment_id || "—")}<div class="sub">策略 ${esc(identity.policy_id || "—")} · 宿主 ${esc(identity.host_version || "—")}</div>
      <details><summary>候选与基线身份</summary><p>候选 ${esc(identity.candidate_hash || "—")}</p><p>基线 ${esc(identity.baseline_hash || "—")}</p></details></td>
      <td class="num">${esc(table.labelled ?? 0)}</td><td class="num">${esc(table.changed ?? 0)}</td><td class="num">${esc(table.net_gain ?? 0)}</td>
      <td><span class="tag ${gate.ok ? "ok" : "bad"}">${gate.ok ? "已通过" : "未通过"}</span><div class="sub">${(gate.checks || []).filter(check => check.status !== "ok").map(check => esc(check.detail || check.name)).join("<br />")}</div></td></tr>`;
  }).join("");
  return `<h3>独立实验结果</h3><p class="hint">每项仅使用同一实验、候选、宿主版本、基线和策略的记录。汇总统计仅供诊断；身份不完整的旧记录 ${esc(data.legacy_rows ?? 0)} 条不能用于晋级。</p>`
    + (rows ? `<div class="table-host"><table><thead><tr><th>实验身份</th><th>已标注</th><th>分歧</th><th>净收益</th><th>晋级门槛</th></tr></thead><tbody>${rows}</tbody></table></div>`
      : '<p class="empty">尚无身份完整的独立实验。</p>');
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
  host.innerHTML = shadowExperiments(data) + `<h3>记录总览</h3><div class="grid">
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

