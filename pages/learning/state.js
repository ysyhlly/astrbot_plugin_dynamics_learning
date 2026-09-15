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
  review: null, replyReview: null, annotationWindow: null, view: "workspace",
  selectedScope: "", detailTab: "profile", scopeDetail: null,
  scopeSearch: "", scopeFilter: "all", samplesScope: "", samples: null,
  // The sample browser pages server-side; the page only ever holds one page.
  samplesPage: 1, samplesTotal: 0, samplesTask: "",
};

const VIEW_META = {
  workspace: ["会话工作台", "从一个会话开始，沿着证据找到问题。"],
  data: ["数据与复盘", "了解这批数据能回答什么，再决定补充什么。"],
  evaluation: ["评测", "比较候选与基线，检查收益和证据。"],
  shadow: ["影子观察", "分别查看真实流量和人工标注的比较结果。"],
  policies: ["策略", "审阅参数变化与证据，管理策略状态。"],
};
const resources = {};

// "下载一个 JSON" is one behaviour, so it is one function: the anchor is
// attached before it is clicked (a detached anchor is unreliable outside
// Chrome) and the object URL is released on a later turn, not synchronously
// under the click that is still starting the download.
