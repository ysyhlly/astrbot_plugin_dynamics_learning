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
    host.querySelector?.("[data-section-error]")?.remove();
    const content = `<div class="notice error" data-section-error role="status">${esc(String(error.message || error))}
      <button class="btn small" data-retry="${esc(label)}">重试</button>
      <span class="sub">请重试；已有结果的时间不会自动更新。</span></div>`;
    if (host.insertAdjacentHTML) host.insertAdjacentHTML("afterbegin", content);
    else host.innerHTML = content + host.innerHTML;
  }
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

