function renderWindow(report) {
  const host = $("window");
  if (!report) {
    host.innerHTML = '<p class="empty">还没有分析结果，先点「运行分析」。</p>';
    return;
  }
  const rows = ["recipient", "topic", "reply_admission", "reply_outcome"].map((task) => {
    const row = (report.overview || {})[task] || {};
    const label = TASK_LABEL[task] || task;
    return statCard(label, row.total ? pct(row.accuracy) : "—", `标注 ${row.total ?? "—"} 条`);
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
      <details><summary>建议依据</summary><p class="rationale">${esc(row.rationale || "暂无补充依据")}</p></details>
      <div class="meta">${tag}${verdictTag}
        <span class="tag">样本 ${esc(row.samples ?? "—")}</span>
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
        `${selection?.eligible ?? "—"} 条正确话题在候选集内`)}
    </div>
    <details><summary>如何理解候选指标</summary><p class="rationale">
      「选中准确率」只在正确话题进入候选集的样本上计算——所以它只反映打分与阈值，
      「Recall@K」只反映候选生成。两者一起看才能判断该改 embedding 还是该改阈值。
    </p></details>
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

function renderTuning(report) {
  const host = $("tuning");
  const runs = (report && report.tuning) || [];
  if (!runs.length) {
    host.innerHTML = '<p class="empty">还没有迭代结果。</p>';
    return;
  }
  host.innerHTML = runs.map((run) => {
    const cls = TUNE_DECISION_CLASS[run.decision] || "";
    const rules = run.rules || {};
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
      <details><summary>查看调参规则与逐步记录</summary><div class="meta">
        <span class="tag">单步 ≤ ${pct(rules.step_delta_ratio)}</span>
        <span class="tag">累计 ≤ ${pct(rules.cumulative_delta_ratio)}</span>
        <span class="tag">最多 ${esc(rules.max_steps)} 步</span>
        <span class="tag">采纳线 ${pct(rules.promote_improvement)} / 强采纳 ${pct(rules.strong_improvement)}</span>
      </div>
      <div class="detail">累计漂移：${driftRows}</div>
      ${steps ? `<div class="table-host"><table><thead><tr>
        <th>步</th><th>本步参数</th><th class="num">边际</th><th class="num">累计</th>
        <th>目标错误</th><th class="num">错误相对</th><th>判定</th><th>说明</th>
      </tr></thead><tbody>${steps}</tbody></table></div>` : ""}</details>
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
  const gate = (report && report.promotion) || {};
  const hasPromotion = Boolean(gate.verdict);
  const conclusion = hasPromotion ? gate : evaluation;
  const verdict = VERDICT_LABEL[conclusion.verdict] || { text: conclusion.verdict, cls: "" };
  const head = `<div class="rec ${conclusion.verdict === "accepted" ? "actionable" : "diagnostic"}">
    <h3>${hasPromotion ? "综合采纳结论" : "会话留出评测"}：<span class="tag ${verdict.cls}">${esc(verdict.text || "未知")}</span></h3>
    <p class="rationale">${(conclusion.reasons || []).map(esc).join("<br />")}</p>
    <div class="meta">
      <span class="tag">训练 ${esc((evaluation.split || {}).train_samples ?? "—")} 条 /
        ${esc((evaluation.split || {}).train_sessions ?? "—")} 会话</span>
      <span class="tag">留出 ${esc((evaluation.split || {}).holdout_samples ?? "—")} 条 /
        ${esc((evaluation.split || {}).holdout_sessions ?? "—")} 会话</span>
    </div>
  </div>`;

  const tasks = Object.values(evaluation.tasks || {});
  const table = tasks.length ? `<table><thead><tr>
      <th>任务</th><th>指标</th><th class="num">基线</th><th class="num">候选</th><th class="num">变化</th>
      <th>95% 置信区间（按会话重采样）</th>
    </tr></thead><tbody>${tasks.map((task) => Object.entries(task.deltas || {}).map(([metric, row], index) => `
      <tr>
        <td>${index === 0 ? esc(TASK_LABEL[task.task] || task.task) + `<div class="sub">有效 ${task.holdout != null && task.unreplayable != null ? num(Math.max(0, task.holdout - task.unreplayable), 0) : "—"} / 留出 ${num(task.holdout, 0)} · 不可回放 ${num(task.unreplayable, 0)}</div>` : ""}</td>
        <td>${esc(metric)}${metric === task.primary_metric ? " ★" : ""}</td>
        <td class="num">${num(row.before)}</td>
        <td class="num">${num(row.after)}</td>
        <td class="num">${row.delta == null ? "—" : (row.delta >= 0 ? "+" : "") + num(row.delta)}</td>
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
         <td class="num">${row.delta == null ? "—" : (row.delta >= 0 ? "+" : "") + num(row.delta)}</td></tr>`).join("")}</tbody></table>`
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

  const forward = report && report.forward;
  const gateBlock = `<article class="rec diagnostic">
    <h3>验证证据</h3>
    <p class="rationale">会话留出集结果：${esc((VERDICT_LABEL[evaluation.verdict] || {}).text || evaluation.verdict || "—")}</p>
    <p class="rationale">${(evaluation.reasons || []).map(esc).join("<br />")}</p>
    <div class="meta">
      <span class="tag">会话留出集：${esc((VERDICT_LABEL[gate.session_verdict] || {}).text || gate.session_verdict || "—")}</span>
      <span class="tag">前向验证：${esc((VERDICT_LABEL[gate.forward_verdict] || {}).text || gate.forward_verdict || "未运行")}</span>
      ${gate.interval ? `<span class="tag">${esc(gate.interval)}</span>` : ""}
    </div>
    <details><summary>门槛与切分说明</summary><p class="hint">会话留出集回答「换到没见过的会话还成立吗」，前向验证回答「换到更晚的数据还成立吗」；
    区间跨 0 时不下可采纳结论。</p>
    ${forward ? `<p class="hint">前向切分：训练 ${esc(forward.split?.train_samples ?? "—")} 条 /
      留出 ${esc(forward.split?.holdout_samples ?? "—")} 条（按标注时间，同一会话可能跨边界）</p>` : ""}</details>
  </article>`;

  const strata = tasks
    .map((task) => ({ task: task.task, rows: ((task.strata || {}).groups || []).filter((row) => row.eligible) }))
    .filter((entry) => entry.rows.length);
  const strataBlock = strata.length
    ? `<h3 class="hint">按会话分层（只诊断，不产生本地策略）</h3>`
      + strata.map((entry) => `<table><thead><tr><th>${esc(TASK_LABEL[entry.task] || entry.task)}</th>
        <th class="num">支撑</th><th class="num">基线</th><th class="num">候选</th><th class="num">变化</th></tr></thead>
        <tbody>${entry.rows.slice(0, 8).map((row) => `<tr>
          <td>${esc(row.group.slice(0, 12))}</td>
          <td class="num">${esc(row.support)}</td>
          <td class="num">${num(row.baseline)}</td>
          <td class="num">${num(row.candidate)}</td>
          <td class="num">${row.delta == null ? "—" : (row.delta >= 0 ? "+" : "") + num(row.delta)}</td>
        </tr>`).join("")}</tbody></table>`).join("")
    : "";

  host.innerHTML = head + gateBlock + '<h3>基线与候选比较</h3><p class="hint">有效样本 = 留出样本 − 不可回放样本；缺失数据以 — 显示。</p><div class="table-host">' + table + '</div>'
    + (candidateBlock ? '<section class="rec">' + candidateBlock + '</section>' : '')
    + (strataBlock || learnedBlock ? '<details><summary>查看分层诊断与学习评分</summary><div class="table-host">' + strataBlock + learnedBlock + '</div></details>' : '');
}

// The state machine's actions. "采纳" moves a record to promoted, which is the
// only state /published offers to ChatDynamics — and even then the host decides
// whether to read it.
