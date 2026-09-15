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
  host.innerHTML = `<div class="grid">${statCard("策略记录", data.total ?? rows.length,
      "状态分布：" + Object.entries(data.status_counts || {})
        .map(([key, value]) => key + " " + value).join(" / "))}
    ${statCard("已采纳策略", published, "采纳后可供 ChatDynamics 读取，实际采用情况由宿主决定")}</div>
    ${rows.map((row) => {
      const cls = POLICY_STATUS_CLASS[row.status] || "";
      const history = (row.status_history || []).map((entry) => {
        const time = entry.at ? new Date(Number(entry.at) * 1000).toLocaleString() : "时间未知";
        return `<li><strong>${esc(entry.from)} → ${esc(entry.to)}</strong>
          <span class="sub">${esc(time)}</span>${entry.reason ? `<p>${esc(entry.reason)}</p>` : ""}</li>`;
      }).join("");
      const actions = POLICY_ACTIONS.filter(item => (row.available_actions || []).includes(item.action));
      return `<article class="rec">
        <h3>${esc(row.version)} <span class="tag ${cls}">${esc(row.status_label || row.status)}</span></h3>
        <p class="sub">来源：${esc(row.source || "未知")}</p>
        <div class="detail">${(row.deltas || []).map((d) => `${esc(d.param)} ${num(d.before)} → ${num(d.after)}`).join("<br />") || "没有参数变化"}</div>
        <details><summary>查看验证证据与状态历史</summary>
          <p class="rationale">${policyEvidence(row)}</p>
          <h4>状态历史</h4>${history ? `<ol>${history}</ol>` : '<p class="empty">尚无状态变更记录。</p>'}
        </details>
        <div class="actions">${actions.map((item) => `<button class="btn small" data-policy="${esc(row.version)}" data-action="${item.action}">${item.label}</button>`).join("") || '<span class="sub">当前没有可用操作</span>'}</div>
      </article>`;
    }).join("")}`;
}
