# API 参考与日志字典

> 服务默认监听 `0.0.0.0:9000`。所有接口返回 JSON；每次请求自动带 `trace_id`
> （响应头 `x-trace-id` 回传，也可由调用方通过请求头 `x-trace-id` 指定）。
> 错误响应统一为 `{"error": {code, message, status, trace_id, path, detail}}`。

## 端点总览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 健康检查，返回 `{"ok": true}` |
| POST | `/v1/chat` | 端云路由聊天 |
| POST | `/v1/embeddings` | 端侧文本向量化 |
| POST | `/v1/expense/collect` | 报销：收进来 |
| POST | `/v1/expense/search` | 报销：找回来 |
| POST | `/v1/expense/export` | 报销：拿出去 |
| POST | `/v1/expense/rebuild-index` | 报销：手动触发 watch 目录扫描 |
| POST | `/v1/expense/correct` | 报销：人工纠正抽取字段（计入复盘指标） |
| POST | `/v1/search-agent/search` | 文件搜索：首查 |
| POST | `/v1/search-agent/clarify` | 文件搜索：追问消歧 |
| POST | `/v1/search-agent/execute` | 文件搜索：命中后动作 |
| POST | `/v1/search-agent/rebuild-index` | 文件搜索：手动重建索引 |
| GET | `/v1/metrics` | 复盘指标（报销回访/补齐/纠正 + 文件搜索追问收敛漏斗） |

---

## 聊天与向量

### `POST /v1/chat`

```json
{
  "message": "请给我一个端侧部署 tiny llm 的方案",
  "force_cloud": false
}
```

返回字段：

- `source`: `edge` / `cloud` / `none`（云端与端侧都不可用时）
- `escalated`: 是否发生了边侧到云端的切换
- `reason`: 路由原因（如 `edge_ok`、`edge_low_confidence`、`policy_force_cloud_first`、
  `cloud_failed_edge_fallback` 等，全程透传便于排障）
- `used_model`: 实际调用模型 id
- `edge_confidence`: 端侧置信度（仅 edge 路径）
- `text`: 返回文本

### `POST /v1/embeddings`

```json
{
  "texts": ["你好", "请给我一个端侧部署方案"],
  "normalize": true
}
```

返回字段：

- `source`: 固定 `edge`
- `model`: 实际使用的 embedding 模型
- `embeddings`: 向量数组，按输入顺序返回

---

## 报销场景（收进来 / 找回来 / 拿出去）

### `POST /v1/expense/collect`

```json
{
  "claim_id": "reimbursement-2026-09-001",
  "source_app": "微信",
  "doc_type": "invoice",
  "title": "出差发票",
  "raw_text": "商户: 京东  金额: 128.00 元 日期:2026-09-01",
  "file_uri": "wechat://msg/xxx",
  "captured_at": "2026-09-01T09:00:00"
}
```

返回字段：

- `material_id`：材料唯一 ID
- `claim_id`：所属报销单（未传时用 `EXPENSE_DEFAULT_CLAIM_ID`）
- `extracted_amount` / `extracted_date` / `merchant`：正则自动抽取结果
- `needs_follow_up` / `missing_required_types`：是否提示缺失关键材料及缺什么
- `claim_material_count` / `claim_total_amount`：报销单当前件数与总额

### `POST /v1/expense/search`

```json
{
  "claim_id": "reimbursement-2026-09-001",
  "keyword": "京东 发票",
  "limit": 20
}
```

可选过滤：`doc_types`（数组）、`from_date` / `to_date`（ISO）、`min_amount` / `max_amount`。
不带 `keyword` 时按更新时间倒序返回（"上次找的材料先看到"）。

返回字段：

- `items`：匹配材料列表（含 `score`，关键词 0.7 + 语义 0.3 加权）
- `missing_required_types`：报销单缺失材料类型（仅传了 `claim_id` 时）
- `keyword` / `claim_id`：查询上下文回显

### `POST /v1/expense/export`

```json
{
  "claim_id": "reimbursement-2026-09-001"
}
```

也可用 `material_ids` 数组按件导出；`include_raw_text=true` 时清单携带原文。
`claim_id` 与 `material_ids` 至少填一个。

返回字段：

- `export_id`：本次导出 ID
- `manifest`：可直接复制的导出清单文本（提交财务前核对用）
- `materials`：导出的材料摘要列表

### `POST /v1/expense/rebuild-index`

无请求体。对 `EXPENSE_WATCH_DIR` 做一次扫描（一级子目录名作为 `claim_id`）。

```json
{
  "scanned": 10,
  "imported": 3,
  "skipped": 7,
  "errors": 0,
  "material_ids": ["…"]
}
```

> pdf/图片读不出文本属于预期跳过（计入 `skipped`，不计 `errors`）。

### `POST /v1/expense/correct`

```json
{
  "material_id": "bba3282cb754...",
  "extracted_amount": 130.5,
  "merchant": "京东世纪贸易"
}
```

只传要改的字段（`extracted_amount` / `extracted_date` / `merchant` / `title`），
None = 不改。返回 `corrected_fields`（实际改动列表；传值与现值一致时为空，
不计入复盘「纠正次数」）。`material_id` 不存在返回 404。

### `GET /v1/metrics`

无参数。返回报销（14 天回访率 / 1 小时补齐率 / 纠正次数）与文件搜索
（追问收敛率 / 收敛且执行动作率 / 会话漏斗）指标；比率分母为 0 时为
`null`（样本不足）。完整口径与已知偏差见 **[METRICS.md](METRICS.md)**。

---

## 个人文件搜索（search → clarify → execute）

### `POST /v1/search-agent/search`

```json
{
  "query": "上周群里发的聚餐照片",
  "top_k": 6,
  "force_disambiguation": false
}
```

返回字段：

- `state`: `resolved` / `needs_clarification` / `not_found`
- `session_id`: 追问会话 ID（即 search_id）
- `selected_file_id`: `resolved` 时默认取首个候选
- `candidates`: 每条候选附带 `score`、`evidence`（命中证据）、`visual_hints`、`matched_clues`
- `question`: 需要澄清时的追问话术
- `next_action_suggestions`: `打开/分享/备注/归档` 等建议动作

> 若配置了 `FILE_MEMORY_SOURCE_DIR`，请求会触发一次**节流异步**增量扫描
> （不阻塞本次查询；每个扫描窗口最多一次）。

### `POST /v1/search-agent/clarify`

```json
{
  "session_id": "上一轮返回的 session_id",
  "reply": "我想要群里那张7月的"
}
```

`reply`（补充线索）与 `selected_file_id`（直接确认某候选）二选一或都传；
`selected_file_id` 优先。线索支持三类：字面词、时间（"上周的"）、来源（"微信群里的"）。

返回结构同 `/search`，其中 `query` 字段回填本次会话的原始查询（便于客户端展示上下文）。
会话失效返回 404（重新发起搜索即可）。

### `POST /v1/search-agent/execute`

```json
{
  "file_id": "命中文件 id",
  "action": "open|share|compare|annotate|archive",
  "share_to": "老王",
  "peer_file_id": "需要对比时提供",
  "note": "这个是最终版"
}
```

- `open`：返回 `file_uri`，由端上打开
- `share`：需要 `share_to`，返回 `share_payload`
- `compare`：需要 `peer_file_id`，返回 `compare_payload`（双方时间、来源、预览、`recommended_keep` 推荐保留项）
- `annotate`：需要 `note`，持久化到 `data/file_state.json`（重启不丢）
- `archive`：归档标记，持久化到 `data/file_state.json`（重启不丢）

### `POST /v1/search-agent/rebuild-index`

无请求体。同步执行一次扫描 + 向量索引重建。

```json
{
  "scanned": 12,
  "imported": 8,
  "skipped": 4,
  "errors": 0,
  "removed": 1,
  "material_ids": ["fm_xxx"]
}
```

> `removed` 为本次清理的幽灵记录数（磁盘已删除但索引残留的文件）。
> `source_dir` 不可达时不会执行清理，防止误删全库。

---

## 日志字典（个人文件搜索）

每次请求自动带 `trace_id`（响应头 `x-trace-id` 返回）；同一搜索任务生成
`search_id` 串联 `/search → /clarify → /execute`。关键事件如下，均可直接 grep：

| # | event | 含义 |
|---|---|---|
| 1 | `service_init` | 服务启动，含 `faiss_enabled`、`faiss_ready` |
| 2 | `api.search.requested` | 进入 `/v1/search-agent/search` |
| 3 | `search.started` | 一次检索开始 |
| 4 | `search.candidate_pool` | 候选池来源（faiss 或全量回退） |
| 5 | `search.scored` | 打分完成后的候选规模与 top ids |
| 6 | `search.completed` | 首次检索返回 |
| 7 | `clarify.started` / `clarify.filtered` / `clarify.completed` | 追问各阶段 |
| 8 | `api.execute.requested` | 动作执行入口 |
| 9 | `action.*` | 文件动作（`action.open_ok`、`action.annotate_ok` 等） |
| 10 | `ingest.finished` | 一次扫描周期结果（scanned/imported/skipped/errors/removed） |
| 11 | `faiss.rebuild` | 向量索引重建耗时与结果 |
| 12 | `http.request` | 每次 HTTP 请求的统一链路指标 |

日志示例：

```json
{
  "event": "search.completed",
  "trace_id": "4e4f...",
  "search_id": "8a3f...",
  "state": "needs_clarification",
  "candidate_count": 3,
  "duration_ms": 24.81
}
```

```json
{
  "event": "api.execute.open_ok",
  "trace_id": "4e4f...",
  "file_id": "fm_1a2b...",
  "duration_ms": 1.9,
  "path": "/v1/search-agent/execute"
}
```
