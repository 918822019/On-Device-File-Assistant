/* 页面入口：功能模块各管自己的事件；导航在这里统一管理。 */
import { $ } from "./js/core.js";
import "./js/search.js";
import "./js/chat.js";
import "./js/expense.js";
import { loadMetrics } from "./js/metrics.js";

// ---------------------------------------------------------------- Tabs
$("#tabs").addEventListener("click", (ev) => {
  const btn = ev.target.closest(".tab");
  if (!btn) return;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === btn));
  document.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
  if (btn.dataset.tab === "metrics") loadMetrics();
});
