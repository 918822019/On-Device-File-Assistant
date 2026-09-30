import { $, esc, toast, api, busy, limits } from "./core.js";
import { searchState, renderSearchResponse, doSearch } from "./search.js";

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
  const srcCls = d.source === "edge" ? "badge-ok" : ["cloud", "file_search"].includes(d.source) ? "badge-violet" : "badge-muted";
  const chips = [
    `<span class="badge ${srcCls}">${esc(d.source === "file_search" ? "本地文件检索" : (d.source || "none"))}</span>`,
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
  chatPending = true;
  busy($("#chat-send"), true, "判断中…");
  input.value = "";
  addMsg("user", message);
  const pending = addMsg("assistant", "正在判断是否需要查找本地文件…");
  pending.classList.add("msg-pending");
  try {
    const d = await api("/v1/chat", { message, force_cloud: $("#force-cloud").checked });
    pending.classList.remove("msg-pending");
    pending.textContent = d.text || "(空回复)";
    if (d.search) {
      const results = document.createElement("div");
      results.className = "chat-file-results";
      results.innerHTML = d.search.candidates.slice(0, 5).map((c) =>
        `<div class="chat-file-hit">📄 ${esc(c.title)} <span class="muted">${esc(c.evidence || "")}</span></div>`
      ).join("");
      const open = document.createElement("button");
      open.className = "btn btn-ghost btn-sm";
      open.textContent = d.search.candidates.length ? "查看候选 · 追问 · 打开原件" : "调整查询 · 检查索引范围";
      open.addEventListener("click", () => {
        searchState.selectedFileId = null;
        $("#sq").value = d.search.query;
        document.querySelector('.tab[data-tab="search"]').click();
        renderSearchResponse(d.search);
      });
      results.appendChild(open);
      pending.appendChild(results);
    }
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
$("#chat-search").addEventListener("click", () => {
  const query = $("#chat-input").value.trim();
  if (!query) { toast("先输入要搜索的文件线索", "err"); return; }
  if (query.length > limits.maxQueryLen) {
    toast(`文件搜索最多支持 ${limits.maxQueryLen} 个字符`, "err");
    return;
  }
  $("#sq").value = query;
  document.querySelector('.tab[data-tab="search"]').click();
  doSearch();
});
$("#chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

