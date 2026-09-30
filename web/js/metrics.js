import { API_BASE, $, esc } from "./core.js";

// ================================================================ 复盘指标
function pct(v) {
  return v == null ? '<span class="muted">样本不足</span>' : (v * 100).toFixed(1) + "%";
}

function metricCard(label, value, sub) {
  return `<div class="metric-card">
    <div class="metric-label">${esc(label)}</div>
    <div class="metric-value">${value}</div>
    <div class="metric-sub">${sub}</div>
  </div>`;
}

export async function loadMetrics() {
  const body = $("#metrics-body");
  try {
    const resp = await fetch(API_BASE + "/v1/metrics");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const d = await resp.json();
    const e = d.expense || {}, f = d.file_search || {};
    body.innerHTML = `
      <h4 class="metrics-group">🧾 报销</h4>
      <div class="metrics-grid">
        ${metricCard("14 天回访率", pct(e.revisit_rate),
          `${e.claims_revisited ?? 0}/${e.claims_total ?? 0} 个报销单 · 窗口 ${e.revisit_window_days ?? 14} 天`)}
        ${metricCard("1 小时补齐率", pct(e.followup_completion_rate),
          `${e.followup_completed ?? 0}/${e.followup_needed_claims ?? 0} 个缺件单 · 窗口 ${e.followup_window_hours ?? 1}h`)}
        ${metricCard("字段纠正次数", e.corrections_total ?? 0,
          `涉及 ${e.corrected_materials ?? 0} 件材料 · 越少越好`)}
      </div>
      <h4 class="metrics-group">🔍 文件搜索</h4>
      <div class="metrics-grid">
        ${metricCard("追问收敛率", pct(f.clarify_resolve_rate),
          `${f.clarify_resolved ?? 0}/${f.needs_clarification ?? 0} 个追问会话最终 resolved`)}
        ${metricCard("收敛且执行动作", pct(f.clarify_exec_rate),
          `${f.clarify_resolved_executed ?? 0}/${f.needs_clarification ?? 0} · README 口径`)}
        ${metricCard("会话总量", f.sessions_total ?? 0,
          `直接命中 ${f.resolved_direct ?? 0} · 需追问 ${f.needs_clarification ?? 0} · 未找到 ${f.not_found ?? 0}`)}
      </div>`;
    $("#metrics-meta").textContent =
      `事件流共 ${d.events_total ?? 0} 条 · 生成于 ${d.generated_at || "—"}（比率分母为 0 时显示"样本不足"，不以 0% 冒充）`;
  } catch (e2) {
    body.innerHTML = `<div class="muted">指标加载失败: ${esc(e2.message)}（指标服务未装配时返回 503）</div>`;
  }
}

$("#metrics-refresh").addEventListener("click", loadMetrics);
