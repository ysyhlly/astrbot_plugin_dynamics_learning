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
  if (!quiet && !state.review) host.innerHTML = '<p class="empty">正在让模型解读这份数据契约…</p>';
  try {
    const data = await call(ENDPOINTS.review, { params: refresh ? { refresh: 1 } : {} });
    if (request !== sectionRequests.review) return;
    if (!data.review) throw new Error(data.reason || "模型解读暂不可用");
    state.review = data;
    renderReview(data);
  } catch (error) {
    if (request !== sectionRequests.review) return;
    setQualityRawOpen(true);
    if (!state.review) host.innerHTML = "";
    sectionError("review", "review", error);
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
  const stages = row.decision_stages || {};
  const persona = stages.persona || {};
  const gate = stages.gate || {};
  const personaLabel = { ignore: "沉默", acknowledge: "简短回应", clarify: "澄清", reply: "回复", close: "收尾" };
  const stageDetails = (persona.action ? `<div class="sub">角色决定：${esc(personaLabel[persona.action] || persona.action)}</div>` : "")
    + (gate.allowed === false ? `<div class="sub">发送约束：未通过 ${esc(gate.reason_code || "")}</div>`
      : gate.evaluated === true && gate.allowed === true ? `<div class="sub">发送约束：通过${gate.length_hint === "brief" ? "（短回应）" : ""}</div>` : "");
  const actual = row.outcome_recorded
    ? `<span class="tag ${outcome.delivered ? "ok" : "bad"}">${esc(outcome.value_label || outcome.value || "—")}</span>`
      + (outcome.suppression_reason ? `<div class="sub">${esc(outcome.suppression_reason)}</div>` : "")
    : `<span class="sub">结果未记录</span>`;
  return `${level}${stageDetails}<div class="sub">实际：${actual}</div>`;
}

function modelCell(row) {
  if (!row.decided) {
    return `<span class="tag">未判断</span><div class="sub">${esc(row.model_reason || "正文、上下文或有效置信度不足")}</div>`;
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
        <div class="sub">${esc(row.session)} · ${esc(row.msg_id)}${row.mentions_bot ? " · 提到了机器人" : ""}</div></td>
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
      ${statCard("未发送但模型倾向回应", counts.reply_preference ?? 0, `已发送但模型倾向不回 ${counts.over_replied ?? 0} 条`)}
    </div>
    <p class="hint">模型意见不等于规则错误。门禁、角色选择、生成和发送需分别核查；未记录发送结果时仅对照规则准入。复盘结果不作为训练标签。</p>
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
  if (!state.replyReview) host.innerHTML = `<p class="empty">正在让模型逐条复盘…</p>`;
  try {
    const data = await call(ENDPOINTS.replyReview, { params: refresh ? { refresh: 1 } : {} });
    if (request !== sectionRequests.replyReview) return;
    if (!data.review) throw new Error(data.reason || "回复复盘暂不可用");
    state.replyReview = data;
    renderReplyReview(data);
  } catch (error) {
    if (request !== sectionRequests.replyReview) return;
    if (!state.replyReview) host.innerHTML = "";
    sectionError("replyReview", "replyReview", error);
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
    sectionError("annotationWindow", "annotationWindow", error);
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
        ${(gate.checks || []).filter(row => row.status === "block").map(row => `<p class="notice error">${esc(row.detail)}</p>`).join("")}
        <details><summary>查看 ${(gate.checks || []).length} 项检查</summary>
        <div class="table-host"><table><thead><tr><th>检查</th><th>状态</th><th>说明</th></tr></thead>
          <tbody>${gateRows}</tbody></table></div></details>
      </article>`
    : "";
  const findings = (data.contract_findings || []).map((line) => `<li>${esc(line)}</li>`).join("");
  $("datasetGate").innerHTML = gateBlock || '<p class="empty">数据门槛尚不可用。</p>';
  host.innerHTML = `<div class="grid">
      ${statCard("样本", dataset.samples ?? 0, `会话 ${dataset.sessions ?? 0} · 作用域 ${dataset.scopes ?? 0}（${dataset.scope_level || "session"}）`)}
      ${statCard("需要补数据的能力", needing, "状态不是「正常」的能力数（不支持 / 警告 / 样本不足）")}
      ${statCard("解析异常轨迹", dataset.degraded_traces ?? 0, "契约降级的样本条数")}
    </div>
    <div class="table-host"><table><thead><tr>
      <th>能力</th><th>状态</th><th class="num">覆盖</th><th class="num">可用/合计</th><th>说明</th>
    </tr></thead><tbody>${rows}</tbody></table></div>
    ${blocked ? `<div class="rec diagnostic"><h3>当前不支持的分析</h3><ul class="bullets">${blocked}</ul></div>` : ""}
    <p class="hint">契约面：${esc(contractLine(data.contract))}</p>
    <p class="hint">轨迹 schema：${esc(traceLine(data.trace))}</p>
    ${findings ? `<ul class="bullets">${findings}</ul>` : ""}
    <p class="hint">${(data.notes || []).map(esc).join("<br />")}</p>`;
}

// ---- scope review profile ----------------------------------------------

