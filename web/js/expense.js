import { $, esc, toast, api, busy } from "./core.js";
import { copyText } from "./search.js";

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
        <div class="kv" style="margin-top:8px">
          <button type="button" class="btn btn-ghost btn-sm" id="toggle-correct">✏️ 纠正字段</button>
          <span class="muted">抽取有误时人工修正，计入复盘「纠正次数」</span>
        </div>
        <form id="correct-form" class="form-grid hidden">
          <label>金额<input name="extracted_amount" type="number" step="0.01" min="0" value="${r.extracted_amount ?? ""}"></label>
          <label>日期<input name="extracted_date" type="text" value="${esc(r.extracted_date || "")}" placeholder="YYYY-MM-DD"></label>
          <label>商户<input name="merchant" type="text" value="${esc(r.merchant || "")}"></label>
          <label>标题<input name="title" type="text" value="${esc(r.title || "")}"></label>
          <div class="span-2"><button type="submit" class="btn btn-primary btn-sm">提交纠正</button></div>
        </form>
      </div>`;
    $("#toggle-correct")?.addEventListener("click", () =>
      $("#correct-form").classList.toggle("hidden"));
    $("#correct-form")?.addEventListener("submit", async (ev2) => {
      ev2.preventDefault();
      const fd = new FormData(ev2.target);
      const body = { material_id: r.material_id };
      const amt = fd.get("extracted_amount");
      if (amt !== null && String(amt).trim() !== "") body.extracted_amount = Number(amt);
      for (const k of ["extracted_date", "merchant", "title"]) {
        const v = fd.get(k);
        if (v && String(v).trim() !== "") body[k] = String(v).trim();
      }
      try {
        const cr = await api("/v1/expense/correct", body);
        toast(cr.corrected_fields.length
          ? `已纠正: ${cr.corrected_fields.join(", ")}`
          : "传值与现值一致，无字段变化（不计数）", "ok");
        $("#correct-form").classList.add("hidden");
      } catch (e3) {
        toast("纠正失败: " + e3.message, "err");
      }
    });
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

