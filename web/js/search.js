import { API_BASE, $, esc, toast, api, fmtTime, busy, limits } from "./core.js";

// ================================================================ 文件搜索
export const searchState = { sessionId: null, candidates: [], selectedFileId: null };

async function loadIndexStatus() {
  const host = $("#index-overview-body");
  try {
    const resp = await fetch(API_BASE + "/v1/search-agent/index-status", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const d = await resp.json();
    // 用后端下发的上限设置输入框 maxlength：上限只有 config 一个事实源，
    // 前端不再硬编码，也就不会再与后端漂移（此前前端 260 / 后端静默截断 120）。
    if (Number.isInteger(d.max_query_len) && d.max_query_len > 0) {
      limits.maxQueryLen = d.max_query_len;
      const sq = $("#sq");
      if (sq) sq.maxLength = d.max_query_len;
    }
    const sources = (d.sources || []).map((s) => `
      <div class="index-source" title="${esc(s.path)}">
        <span class="source-path">${esc(s.path)}</span>
        <span class="chip ${s.available ? "chip-accent" : ""}">${s.available ? `${s.indexed_files} 个文件` : "目录不可访问"}</span>
      </div>`).join("");
    host.innerHTML = `
      <div class="overview-stats">
        <div class="overview-stat"><strong>${d.indexed_files}</strong><span>已收录文件</span></div>
        <div class="overview-stat"><strong>${d.scan_recursive ? "递归" : "第一层"}</strong><span>扫描深度</span></div>
        <div class="overview-stat"><strong>${d.vector_ready ? "已就绪" : "不可用"}</strong><span>向量索引</span></div>
      </div>
      <p class="hint-line">${d.scan_recursive ? "包含子目录" : "仅索引以下目录的第一层文件，子目录尚未纳入"} · 自动检查间隔 ${d.scan_interval_seconds} 秒${d.last_indexed_at ? ` · 最近入库 ${esc(fmtTime(d.last_indexed_at))}` : ""}</p>
      <div class="index-sources">${sources || '<span class="muted">未配置扫描目录</span>'}</div>`;
  } catch (e) {
    host.textContent = "索引状态暂不可用：" + e.message;
  }
}
$("#refresh-index-status").addEventListener("click", loadIndexStatus);
loadIndexStatus();

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
      <span class="cand-rank">${idx + 1}</span>
      <span class="cand-title">${esc(c.title)}</span>
      <span class="cand-score" title="检索排序分，并非命中概率">匹配分 ${(c.score ?? 0).toFixed(3)}</span>
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

export function renderSearchResponse(data, opts = {}) {
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
    ? `<div class="muted">找到 ${searchState.candidates.length} 条候选；匹配分仅用于排序，请核对文件名和证据。</div>` + searchState.candidates.map(candidateCard).join("")
    : `<div class="empty-results">当前索引未找到候选。试试文件名、来源或时间线索；也可先检查上方索引范围。</div>`;

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

export async function doSearch() {
  const query = $("#sq").value.trim();
  if (!query) { toast("先输入查询内容", "err"); return; }
  const btn = $("#search-btn");
  busy(btn, true, "检索中…");
  try {
    searchState.selectedFileId = null;
    const data = await api("/v1/search-agent/search", { query, top_k: Number($("#search-limit").value) });
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
      session_id: searchState.sessionId, top_k: Number($("#search-limit").value), ...payload,
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
    // file:// URI 浏览器打不开（http 页面禁跳 file://），复制时优先用后端映射好的
    // windows_path（WSL / Windows 原生）；否则剥掉前缀给纯路径：macOS 粘到 Finder
    // 「前往文件夹」(⌘⇧G)、Linux 粘到文件管理器均可直达。注意 Windows URI 是
    // file:///C:/...，直接 slice("file://") 会多出一个前导斜杠，故优先 windows_path。
    const copyValue = r.windows_path
      || (r.file_uri.startsWith("file://")
        ? r.file_uri.slice("file://".length)
        : r.file_uri);
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
export async function copyText(text) {
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
    await loadIndexStatus();
  } catch (e) {
    toast("重建失败: " + e.message, "err");
  } finally {
    busy(btn, false);
  }
});

