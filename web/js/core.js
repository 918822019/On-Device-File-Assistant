/* 端上文件助手 Web UI —— 原生 JS，无构建步骤。
 * 同源部署（FastAPI /web 静态托管）时用相对路径；
 * 也可用 ?api=http://host:port 指定后端（需后端允许跨域）。
 */
"use strict";

export const API_BASE = new URLSearchParams(location.search).get("api") || "";

// 后端下发的限制值。ES module 的 live binding 加上所有模块都裸引 ./core.js，
// 保证这是唯一一份实例：search.js 写入后 chat.js 立刻可见。
// 初值与后端 FILE_MEMORY_MAX_QUERY_LEN 的默认值一致，这样 index-status 尚未
// 返回（或后端不可达）时输入框也有正确上限。
export const limits = { maxQueryLen: 120 };

// ---------------------------------------------------------------- 基础工具
export const $ = (sel) => document.querySelector(sel);

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

export function toast(msg, kind = "") {
  const host = $("#toast-host");
  const el = document.createElement("div");
  el.className = `toast ${kind ? "toast-" + kind : ""}`;
  el.textContent = msg;
  host.appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 8000 : 4000);
}

export async function api(path, body) {
  const resp = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const trace = resp.headers.get("x-trace-id") || "";
  let data;
  try { data = await resp.json(); } catch { data = null; }
  if (!resp.ok) {
    const err = data && data.error;
    const detail = err ? `${err.code}: ${err.message}` : `HTTP ${resp.status}`;
    const e = new Error(detail + (trace ? ` (trace ${trace.slice(0, 8)})` : ""));
    e.status = resp.status;
    throw e;
  }
  return data;
}

export function fmtTime(iso) {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 16);
}

export function busy(btn, on, textOn) {
  if (!btn) return;
  btn.disabled = on;
  if (textOn !== undefined) {
    if (on) { btn.dataset.oldText = btn.textContent; btn.textContent = textOn; }
    else if (btn.dataset.oldText) { btn.textContent = btn.dataset.oldText; }
  }
}

// ---------------------------------------------------------------- 健康状态
async function checkHealth() {
  const dot = $("#health-dot"), text = $("#health-text");
  const online = API_BASE ? `在线 · ${API_BASE}` : `在线 · ${location.origin}`;
  try {
    const resp = await fetch(API_BASE + "/health", { signal: AbortSignal.timeout(5000) });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error("bad payload");
    // degraded 时后端仍返回 200 + ok（部署门禁契约），但有服务未装配成功。
    // 典型：embedding_runtime 加载失败后，搜索会静默退化为纯关键词匹配，
    // 结果看起来正常却少了语义召回 —— 必须在 UI 上露出来。
    const services = data.services || {};
    const names = Object.keys(services);
    const down = names.filter((k) => !services[k]);
    if (down.length) {
      dot.className = "dot dot-warn";
      text.textContent =
        down.length === names.length ? "降级 · 服务未装配" : `降级 · 缺 ${down.join("/")}`;
      dot.title = `以下服务未就绪：${down.join(", ")}`;
    } else {
      dot.className = "dot dot-ok";
      text.textContent = online;
      dot.title = "服务状态";
    }
  } catch {
    dot.className = "dot dot-err";
    text.textContent = "服务不可达";
    dot.title = "服务状态";
  }
}
checkHealth();
setInterval(checkHealth, 30000);

