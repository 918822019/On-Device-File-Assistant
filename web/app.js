/* 端上文件助手 Web UI —— 原生 JS，无构建步骤。
 * 同源部署（FastAPI /web 静态托管）时用相对路径；
 * 也可用 ?api=http://host:port 指定后端（需后端允许跨域）。
 */
"use strict";

const API_BASE = new URLSearchParams(location.search).get("api") || "";

// ---------------------------------------------------------------- 基础工具
const $ = (sel) => document.querySelector(sel);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function toast(msg, kind = "") {
  const host = $("#toast-host");
  const el = document.createElement("div");
  el.className = `toast ${kind ? "toast-" + kind : ""}`;
  el.textContent = msg;
  host.appendChild(el);
  setTimeout(() => el.remove(), kind === "err" ? 8000 : 4000);
}

async function api(path, body) {
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

function fmtTime(iso) {
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 16);
}

function busy(btn, on, textOn) {
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
  try {
    const resp = await fetch(API_BASE + "/health", { signal: AbortSignal.timeout(5000) });
    const data = await resp.json();
    if (resp.ok && data.ok) {
      dot.className = "dot dot-ok";
      text.textContent = API_BASE ? `在线 · ${API_BASE}` : `在线 · ${location.origin}`;
      return;
    }
    throw new Error("bad payload");
  } catch {
    dot.className = "dot dot-err";
    text.textContent = "服务不可达";
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ---------------------------------------------------------------- Tabs
$("#tabs").addEventListener("click", (ev) => {
  const btn = ev.target.closest(".tab");
  if (!btn) return;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === btn));
  document.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
});

// ================================================================ 文件搜索
const searchState = { sessionId: null, candidates: [], selectedFileId: null };

function stateBadge(state) {
  const map = {
    resolved: ["badge-ok", "已命中 resolved"],
    needs_clarification: ["badge-warn", "需要澄清 needs_clarification"],
    not_found: ["badge-err", "未找到 not_found"],
  };
  const [cls, label] = map[state] || ["badge-muted", state || "unknown"];
  return `<span class="badge ${cls}">${esc(label)}</span>`;
}

function candidateCard(c, idx) {
  const isSelected = searchState.selectedFileId === c.file_id;
  const clues = [
    ...(c.visual_hints || []).map((h) => `<span class="chip">🎨 ${esc(h)}</span>`),
    ...(c.matched_clues || []).map((h) => `<span class="chip chip-accent">🧵 ${esc(h)}</span>`),
  ].join("");
  const others = searchState.candidates.filter((o) => o.file_id !== c.file_id);
  const peerSelect = others.length
    ? `<select data-peer-for="${esc(c.file_id)}">
         ${others.map((o) => `<option value="${esc(o.file_id)}">对比: ${esc(o.title)}</option>`).join("")}
       </select>`
    : "";
  return `
  <div class="candidate ${isSelected ? "selected" : ""}" data-file-id="${esc(c.file_id)}">
    <div class="cand-head">
      <span class="cand-title">${esc(c.title)}</span>
      <span class="cand-score">${(c.score ?? 0).toFixed(3)}</span>
    </div>
    <div class="scorebar"><div style="width:${Math.round((c.score ?? 0) * 100)}%"></div></div>
    <div class="cand-meta">
      <span>来源: ${esc(c.source_app || "—")}</span>
      <span>类型: ${esc(c.doc_type || "—")}</span>
      <span>时间: ${esc(fmtTime(c.captured_at))}</span>
      <span class="muted" title="file_id">${esc(c.file_id.slice(0, 14))}…</span>
    </div>
    ${c.evidence ? `<div class="cand-evidence">证据: ${esc(c.evidence)}</div>` : ""}
    ${c.preview ? `<div class="cand-preview">${esc(c.preview)}</div>` : ""}
    <div>${clues}</div>
    <div class="cand-actions">
      <button class="btn btn-sm" data-act="select">✅ 确认就是它</button>
      <button class="btn btn-sm" data-act="open">📂 打开</button>
      <button class="btn btn-sm" data-act="share">📤 分享</button>
      ${peerSelect}<button class="btn btn-sm" data-act="compare" ${others.length ? "" : "disabled"}>🆚 对比</button>
      <button class="btn btn-sm" data-act="annotate">📝 备注</button>
      <button class="btn btn-sm" data-act="archive">🗄 归档</button>
    </div>
  </div>`;
}

function renderSearchResponse(data, opts = {}) {
  searchState.sessionId = data.session_id;
  searchState.candidates = data.candidates || [];
  if (data.selected_file_id) searchState.selectedFileId = data.selected_file_id;
  // 后端 clarify 收敛到 resolved 时可能不回填 selected_file_id，兜底取首个候选
  if (data.state === "resolved" && !data.selected_file_id && searchState.candidates.length) {
    searchState.selectedFileId = searchState.candidates[0].file_id;
  }

  $("#search-session").classList.remove("hidden");
  $("#state-badge").outerHTML = stateBadge(data.state).replace("<span", '<span id="state-badge"');
  $("#session-query").textContent = `「${data.query}」 · session ${String(data.session_id).slice(0, 8)}…`;

  const q = $("#clarify-question");
  if (data.state === "needs_clarification" && data.question) {
    q.textContent = data.question;
    q.classList.remove("hidden");
  } else {
    q.classList.add("hidden");
  }

  $("#candidates").innerHTML = searchState.candidates.length
    ? searchState.candidates.map(candidateCard).join("")
    : `<div class="muted">无候选。</div>`;

  $("#clarify-row").classList.toggle("hidden", data.state !== "needs_clarification");
  const sug = $("#action-suggestions");
  if ((data.next_action_suggestions || []).length) {
    sug.innerHTML = "建议动作: " + data.next_action_suggestions
      .map((s) => `<span class="chip">${esc(s)}</span>`).join("");
    sug.classList.remove("hidden");
  } else {
    sug.classList.add("hidden");
  }
  if (data.state === "resolved" && !opts.keepResult && searchState.selectedFileId) {
    const hit = searchState.candidates.find((c) => c.file_id === searchState.selectedFileId);
    toast(`已命中: ${hit ? hit.title : searchState.selectedFileId}`, "ok");
  }
}

async function doSearch() {
  const query = $("#sq").value.trim();
  if (!query) { toast("先输入查询内容", "err"); return; }
  const btn = $("#search-btn");
  busy(btn, true, "检索中…");
  try {
    searchState.selectedFileId = null;
    const data = await api("/v1/search-agent/search", { query });
    renderSearchResponse(data);
  } catch (e) {
    if (e.status === 503) toast("文件搜索服务不可用（后端未就绪）", "err");
    else toast("搜索失败: " + e.message, "err");
  } finally {
    busy(btn, false);
  }
}

async function doClarify(payload) {
  if (!searchState.sessionId) { toast("没有进行中的会话，请先搜索", "err"); return; }
  const btn = $("#clarify-btn");
  busy(btn, true, "收敛中…");
  try {
    const data = await api("/v1/search-agent/clarify", {
      session_id: searchState.sessionId, ...payload,
    });
    renderSearchResponse(data);
    $("#clarify-input").value = "";
  } catch (e) {
    if (e.status === 404) {
      toast("会话已过期，请重新搜索", "err");
      resetSession();
    } else toast("追问失败: " + e.message, "err");
  } finally {
    busy(btn, false);
  }
}

function resetSession() {
  searchState.sessionId = null;
  searchState.candidates = [];
  searchState.selectedFileId = null;
  $("#search-session").classList.add("hidden");
  $("#action-result").classList.add("hidden");
}

function showActionResult(r) {
  const host = $("#action-result-body");
  const parts = [];
  const statusCls = r.status === "ok" ? "badge-ok" : r.status === "failed" ? "badge-err" : "badge-muted";
  parts.push(`<div class="kv"><span class="badge ${statusCls}">${esc(r.action)} · ${esc(r.status)}</span>
    <span class="muted">${esc(r.file_title || r.file_id || "")}</span></div>`);
  parts.push(`<div>${esc(r.message || "")}</div>`);
  if (r.file_uri) {
    const isHttp = /^https?:\/\//.test(r.file_uri);
    // file:// URI 浏览器打不开（http 页面禁跳 file://），复制时剥掉前缀给纯路径：
    // macOS 粘到 Finder「前往文件夹」(⌘⇧G)、Linux 粘到文件管理器均可直达
    const copyValue = r.file_uri.startsWith("file://")
      ? r.file_uri.slice("file://".length)
      : r.file_uri;
    parts.push(`<div class="uri-line">${isHttp
      ? `<a href="${esc(r.file_uri)}" target="_blank" rel="noopener">${esc(r.file_uri)}</a>`
      : esc(r.file_uri)}
      <button class="btn btn-ghost btn-sm" data-copy="${esc(copyValue)}">复制路径</button></div>`);
  }
  if (r.windows_path) {
    // WSL 后端：给出可在 Windows 资源管理器直接使用的路径
    parts.push(`<div class="uri-line">🪟 Windows 路径: ${esc(r.windows_path)}
      <button class="btn btn-ghost btn-sm" data-copy="${esc(r.windows_path)}">复制</button></div>`);
  }
  if (r.annotations) parts.push(`<div>备注: ${esc(r.annotations)}</div>`);
  if (r.archived) parts.push(`<div><span class="badge badge-muted">已归档</span></div>`);
  if (r.share_payload) parts.push(`<pre>${esc(JSON.stringify(r.share_payload, null, 2))}</pre>`);
  if (r.compare_payload) parts.push(`<pre>${esc(JSON.stringify(r.compare_payload, null, 2))}</pre>`);
  if ((r.next_action_suggestions || []).length) {
    parts.push(`<div class="suggestions">建议: ${r.next_action_suggestions.map(s => `<span class="chip">${esc(s)}</span>`).join("")}</div>`);
  }
  host.innerHTML = parts.join("");
  $("#action-result").classList.remove("hidden");
}

async function doExecute(fileId, action, extra = {}) {
  try {
    const r = await api("/v1/search-agent/execute", {
      file_id: fileId, action, session_id: searchState.sessionId || undefined, ...extra,
    });
    showActionResult(r);
    toast(r.status === "ok" ? `${action} 成功` : `${action}: ${r.message || r.status}`,
      r.status === "ok" ? "ok" : "err");
  } catch (e) {
    toast(`${action} 失败: ` + e.message, "err");
  }
}

// 复制按钮（动作结果区；剪贴板 API 需要安全上下文，http://局域网IP 下会失败，给提示）
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("已复制", "ok");
  } catch {
    prompt("当前上下文无法用剪贴板 API，请手动复制：", text);
  }
}
$("#action-result-body").addEventListener("click", (ev) => {
  const copyBtn = ev.target.closest("[data-copy]");
  if (copyBtn) copyText(copyBtn.dataset.copy);
});

// 候选卡片事件委托
$("#candidates").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-act]");
  if (!btn) return;
  const card = btn.closest(".candidate");
  const fileId = card?.dataset.fileId;
  if (!fileId) return;
  switch (btn.dataset.act) {
    case "select": await doClarify({ selected_file_id: fileId }); break;
    case "open": doExecute(fileId, "open"); break;
    case "share": {
      const shareTo = prompt("分享给谁（昵称/账号）？");
      if (shareTo) doExecute(fileId, "share", { share_to: shareTo });
      break;
    }
    case "compare": {
      const sel = card.querySelector(`select[data-peer-for="${CSS.escape(fileId)}"]`);
      if (sel?.value) doExecute(fileId, "compare", { peer_file_id: sel.value });
      break;
    }
    case "annotate": {
      const note = prompt("写点备注：");
      if (note) doExecute(fileId, "annotate", { note });
      break;
    }
    case "archive":
      if (confirm("归档这个文件？")) doExecute(fileId, "archive");
      break;
  }
});

$("#search-btn").addEventListener("click", doSearch);
$("#sq").addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });
$("#clarify-btn").addEventListener("click", () => {
  const reply = $("#clarify-input").value.trim();
  if (reply) doClarify({ reply });
});
$("#clarify-input").addEventListener("keydown", (e) => { if (e.key === "Enter") doClarify({ reply: e.target.value.trim() }); });
$("#reset-session").addEventListener("click", resetSession);
$("#close-action-result").addEventListener("click", () => $("#action-result").classList.add("hidden"));

$("#rebuild-file-btn").addEventListener("click", async (ev) => {
  const btn = ev.currentTarget;
  busy(btn, true, "重建中…");
  try {
    const r = await api("/v1/search-agent/rebuild-index");
    $("#rebuild-file-result").textContent =
      `扫描 ${r.scanned} · 导入 ${r.imported} · 跳过 ${r.skipped} · 清理幽灵 ${r.removed} · 错误 ${r.errors}`;
    toast("文件索引重建完成", "ok");
  } catch (e) {
    toast("重建失败: " + e.message, "err");
  } finally {
    busy(btn, false);
  }
});

// ================================================================ 端云聊天
const chatLog = $("#chat-log");
let chatPending = false;

function addMsg(role, text, meta) {
  const empty = chatLog.querySelector(".chat-empty");
  if (empty) empty.remove();
  const el = document.createElement("div");
  el.className = "msg " + (role === "user" ? "msg-user" : "msg-assistant");
  el.textContent = text;
  if (meta) {
    const m = document.createElement("div");
    m.className = "msg-meta";
    m.innerHTML = meta;
    el.appendChild(m);
  }
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return el;
}

function chatMeta(d) {
  const srcCls = d.source === "edge" ? "badge-ok" : d.source === "cloud" ? "badge-violet" : "badge-muted";
  const chips = [
    `<span class="badge ${srcCls}">${esc(d.source || "none")}</span>`,
    d.reason ? `<span class="chip" title="路由原因">${esc(d.reason)}</span>` : "",
    d.used_model ? `<span class="chip">${esc(d.used_model)}</span>` : "",
    d.edge_confidence != null ? `<span class="chip">置信度 ${(d.edge_confidence * 100).toFixed(0)}%</span>` : "",
    d.escalated ? `<span class="badge badge-warn">⚡ 升级云端</span>` : "",
  ];
  return chips.join("");
}

async function sendChat() {
  if (chatPending) return;
  const input = $("#chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addMsg("user", message);
  chatPending = true;
  busy($("#chat-send"), true, "推理中…");
  const pending = addMsg("assistant", "端侧推理中，CPU 首 token 以秒计，长回复可能需要几分钟…");
  pending.classList.add("msg-pending");
  try {
    const d = await api("/v1/chat", { message, force_cloud: $("#force-cloud").checked });
    pending.classList.remove("msg-pending");
    pending.textContent = d.text || "(空回复)";
    const m = document.createElement("div");
    m.className = "msg-meta";
    m.innerHTML = chatMeta(d);
    pending.appendChild(m);
  } catch (e) {
    pending.classList.remove("msg-pending");
    pending.textContent = "请求失败: " + e.message;
  } finally {
    chatPending = false;
    busy($("#chat-send"), false);
    chatLog.scrollTop = chatLog.scrollHeight;
  }
}

$("#chat-send").addEventListener("click", sendChat);
$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

// ================================================================ 报销材料
function formValues(form) {
  const fd = new FormData(form);
  const out = {};
  for (const [k, v] of fd.entries()) {
    if (v instanceof File) continue;
    if (typeof v === "string" && v.trim() === "" && form.elements[k]?.type !== "checkbox") continue;
    out[k] = form.elements[k]?.type === "checkbox" ? form.elements[k].checked : v;
  }
  return out;
}

$("#collect-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const body = formValues(ev.target);
  try {
    const r = await api("/v1/expense/collect", body);
    const missing = (r.missing_required_types || []);
    $("#collect-result").innerHTML = `
      <div class="result-card">
        <div class="kv">
          <span class="badge badge-ok">已收进来</span>
          <span class="muted">${esc(r.material_id)} · ${esc(r.claim_id)}</span>
        </div>
        <div class="kv">
          <span class="chip chip-accent">金额 ${r.extracted_amount != null ? "¥" + r.extracted_amount : "未识别"}</span>
          <span class="chip chip-accent">日期 ${esc(r.extracted_date || "未识别")}</span>
          <span class="chip chip-accent">商户 ${esc(r.merchant || "未识别")}</span>
          <span class="chip">抽取置信度 ${(r.extraction_confidence * 100).toFixed(0)}%</span>
        </div>
        <div class="muted">本单已 ${r.claim_material_count} 件 · 总额 ${r.claim_total_amount != null ? "¥" + r.claim_total_amount : "—"}</div>
        ${missing.length ? `<div style="margin-top:6px">⚠️ 还缺: ${missing.map(t => `<span class="chip">${esc(t)}</span>`).join("")}</div>` : ""}
      </div>`;
    toast("材料已收进来", "ok");
  } catch (e) { toast("collect 失败: " + e.message, "err"); }
});

$("#esearch-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const v = formValues(ev.target);
  const body = { limit: 20 };
  if (v.keyword) body.keyword = v.keyword;
  if (v.claim_id) body.claim_id = v.claim_id;
  if (v.min_amount) body.min_amount = Number(v.min_amount);
  if (v.max_amount) body.max_amount = Number(v.max_amount);
  if (v.from_date) body.from_date = v.from_date;
  if (v.to_date) body.to_date = v.to_date;
  try {
    const r = await api("/v1/expense/search", body);
    const rows = (r.items || []).map((it) => `
      <tr>
        <td>${esc(it.title)}</td>
        <td><span class="chip">${esc(it.doc_type)}</span></td>
        <td>${esc(it.merchant || "—")}</td>
        <td class="num">${it.extracted_amount != null ? "¥" + it.extracted_amount : "—"}</td>
        <td>${esc(it.extracted_date || "—")}</td>
        <td class="num">${(it.score ?? 0).toFixed(2)}</td>
      </tr>`).join("");
    $("#esearch-result").innerHTML = `
      <div class="result-card">
        <div class="muted">共 ${r.total} 条${(r.missing_required_types || []).length
          ? ` · ⚠️ 缺: ${r.missing_required_types.map(t => `<span class="chip">${esc(t)}</span>`).join("")}` : ""}</div>
        ${rows ? `<table class="hits"><thead><tr>
          <th>标题</th><th>类型</th><th>商户</th><th>金额</th><th>日期</th><th>分值</th>
        </tr></thead><tbody>${rows}</tbody></table>` : '<div class="muted" style="margin-top:8px">没有匹配材料。</div>'}
      </div>`;
  } catch (e) { toast("search 失败: " + e.message, "err"); }
});

$("#export-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const v = formValues(ev.target);
  const body = { include_raw_text: !!v.include_raw_text };
  if (v.claim_id) body.claim_id = v.claim_id;
  try {
    const r = await api("/v1/expense/export", body);
    $("#export-result").innerHTML = `
      <div class="result-card">
        <div class="kv">
          <span class="badge badge-ok">已生成</span>
          <span class="muted">${esc(r.export_id)} · ${(r.materials || []).length} 件</span>
          <button class="btn btn-ghost btn-sm" id="copy-manifest">复制清单</button>
        </div>
        <pre class="manifest">${esc(r.manifest || "")}</pre>
      </div>`;
    $("#copy-manifest")?.addEventListener("click", () => copyText(r.manifest || ""));
  } catch (e) { toast("export 失败: " + e.message, "err"); }
});

$("#rebuild-expense-btn").addEventListener("click", async (ev) => {
  const btn = ev.currentTarget;
  busy(btn, true, "扫描中…");
  try {
    const r = await api("/v1/expense/rebuild-index");
    $("#rebuild-expense-result").textContent =
      `扫描 ${r.scanned} · 导入 ${r.imported} · 跳过 ${r.skipped} · 错误 ${r.errors}`;
  } catch (e) {
    toast("重建失败: " + e.message, "err");
  } finally {
    busy(btn, false);
  }
});
