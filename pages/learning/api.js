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

