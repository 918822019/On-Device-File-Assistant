# 端云结合 Agent（edge-first tiny LLM）

本仓库实现一个“端侧优先、云端可选”的轻量 Agent：

- 端侧：默认走 tiny LLM（量化推理）
- 云端：默认不启用；如需可选则接入 Gemini/Gemma 风格的 OpenAI 兼容接口（默认模型 id 为 Gemma4-E2B）
- 路由：命中大输入 / 指定关键词 / 强制参数时，可切到云端；也可以在配置里关闭 tiny-first。
- 端侧新增：embedding 接口采用 `google/embeddinggemma-300m`（默认从 ModelScope 加载）

---

## 1. 架构图（仓库级）

- [src/edge_cloud_agent/config.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/config.py)：配置读取与默认值（边侧、云侧、路由策略）
- [src/edge_cloud_agent/edge_runtime.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/edge_runtime.py)：端侧 tiny llm 加载与生成（含 int4 + gp32 回退链）
- [src/edge_cloud_agent/embedding_runtime.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/embedding_runtime.py)：端侧 embedding 模型加载与向量生成（`google/embeddinggemma-300m`）
- [src/edge_cloud_agent/cloud_client.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/cloud_client.py)：云端 API 客户端
- [src/edge_cloud_agent/routing.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/routing.py)：路由策略层
- [src/edge_cloud_agent/agent.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/agent.py)：路由与回退策略（Edge/Cloud 编排）
- [src/edge_cloud_agent/main.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/main.py)：FastAPI 入口，提供 `/health`、`/v1/chat`、`/v1/embeddings`
- [src/edge_cloud_agent/routers/chat.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/routers/chat.py)：聊天路由定义
- [src/edge_cloud_agent/routers/embeddings.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/routers/embeddings.py)：embedding 路由定义
- [src/edge_cloud_agent/routers/expense.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/routers/expense.py)：报销场景闭环路由（收进来/找回来/拿出去）
- [src/edge_cloud_agent/expense/storage.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/expense/storage.py)：报销材料本地持久化（本地 JSONL 存储）
- [src/edge_cloud_agent/expense/service.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/expense/service.py)：报销工作流服务（字段抽取、检索、导出）
- [src/edge_cloud_agent/expense/schemas.py](/Users/wzy/PycharmProjects/端云结合/src/edge_cloud_agent/expense/schemas.py)：报销场景请求与响应模型
- [requirements.txt](/Users/wzy/PycharmProjects/端云结合/requirements.txt)：依赖
- [.env.example](/Users/wzy/PycharmProjects/端云结合/.env.example)：环境变量模板

---

## 2. 运行时流程（核心）

1. 判断是否直接走云端
   - 触发条件：`force_cloud=true`、上下文过长（`ROUTE_MAX_INPUT_CHARS`）或关键词命中
2. 若未要求直达云端
   - `ROUTE_USE_TINYLLM=true`：先走 edge tiny llm；置信度不足/文本过短时可回退云端（若启用）
   - `ROUTE_USE_TINYLLM=false`：直接走云端（若启用）
3. 云端未启用时，统一退化到端侧处理并给出明确不可回退提示

---

## 3. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 4. 启动

```bash
cp .env.example .env
export PYTHONPATH=src
uvicorn edge_cloud_agent.main:app --host 0.0.0.0 --port 9000
```

或者直接运行脚本：`bash scripts/run.sh`

---

## 5. 关键环境变量（端侧 tiny llm 为默认）

### 通用与聊天

- `EDGE_MODEL_SOURCE`：端侧来源；`hf` 或 `modelscope`
- `EDGE_MODEL_ID`：端侧基础模型（默认：TinyLlama-1.1B-Chat）
- `EDGE_QUANTIZED_MODEL_ID`：量化模型 ID（可放 int4 ckpt）
- `EDGE_QUANTIZATION=int4-gp32`：量化首选策略
- `EDGE_QUANT_BITS`：GPTQ bits（默认 4）
- `EDGE_QUANT_GROUP_SIZE`：GPTQ group size（默认 32）
- `CLOUD_ENABLED=false`：当前默认关闭云端兜底
- `CLOUD_MODEL_ID`：云端模型 ID，默认 `google/gemma-4-e2b-it`
- `ROUTE_USE_TINYLLM=true`：`true` 代表端侧优先，`false` 代表云侧优先
- `ROUTE_MAX_INPUT_CHARS`：触发云端优先的上下文长度阈值
- `ROUTE_MIN_EDGE_CONFIDENCE`：端侧置信度低于该阈值回退云端

### embedding（端侧）

- `EDGE_EMBEDDING_SOURCE=modelscope`
- `EDGE_EMBEDDING_MODEL_ID=google/embeddinggemma-300m`
- `EDGE_EMBEDDING_LOCAL_DIR`：本地缓存目录（优先）
- `EDGE_EMBEDDING_MAX_LENGTH`：单条文本最大长度
- `EDGE_EMBEDDING_BATCH_SIZE`：单批向量化数量
- `EDGE_EMBEDDING_TRUST_REMOTE_CODE`
- `EDGE_EMBEDDING_TORCH_DTYPE`

### 报销场景（V1）

- `EXPENSE_STORE_PATH`：本地报销材料存储路径（默认 `data/expense_store.jsonl`）
- `EXPENSE_DEFAULT_CLAIM_ID`：未传 claim_id 时使用的默认报销单
- `EXPENSE_REQUIRED_DOC_TYPES`：V1 视角要求的关键材料类型，逗号分隔
- `EXPENSE_MAX_SEARCH_LIMIT`：`/v1/expense/search` 的安全上限
- `EXPENSE_ENABLE_EMBEDDING_SEARCH`：启用 embedding 搜索加权

### 个人文件搜索（V1）

- `FILE_MEMORY_STORE_PATH`：个人文件 JSONL 存储路径（默认 `data/personal_file_store.jsonl`）
- `FILE_MEMORY_SOURCE_DIR`：扫描根目录（示例：`/storage/emulated/0`）
- `FILE_MEMORY_SCAN_INTERVAL_SECONDS`：监听扫描间隔（秒）
- `FILE_MEMORY_SCAN_FILE_SUFFIXES`：扫描后缀白名单，逗号分隔
- `FILE_MEMORY_SCAN_RECURSIVE`：`true` 为递归扫描
- `FILE_MEMORY_TOP_K_DEFAULT`：默认返回候选数
- `FILE_MEMORY_MAX_QUERY_LEN`：查询文本截断长度
- `FILE_MEMORY_ENABLE_FAISS`：开启向量索引（默认 `true`），不可用时自动回退原有检索
- `FILE_MEMORY_FAISS_INDEX_PATH`：FAISS 落盘路径（默认 `data/personal_file_faiss.index`）
- `FILE_MEMORY_FAISS_CANDIDATE_MULTIPLIER`：FAISS 召回候选池放大倍数（默认 `4`，即先取 `top_k * 4`）
- `FILE_MEMORY_FAISS_TEXT_WEIGHT`：文本线索权重（默认 `0.65`）
- `FILE_MEMORY_FAISS_SEMANTIC_WEIGHT`：语义向量权重（默认 `0.30`）
- `FILE_MEMORY_FAISS_CLUE_WEIGHT`：来源/视觉/时间线索权重（默认 `0.25`）
- `FILE_MEMORY_FAISS_VERSION_BONUS`：版本相关候选加分因子（默认 `0.20`）
- `APP_LOG_LEVEL`：日志级别（默认 `INFO`）

日志约定：

- 每次请求都会自动带 `trace_id`，会在响应头 `x-trace-id` 返回。
- 同一个搜索任务会生成 `search_id`，串联 `/search -> /clarify -> /execute` 日志。
- 关键事件会记录 `event`, `trace_id`, `path`, `duration_ms` 等字段，便于按时间线排查问题。

### 5.1 个人文件搜索日志字典（V1）

以下是当前日志系统关键事件，便于快速排障。

1. `service_init`：服务启动时打印，包含 `faiss_enabled`、`faiss_ready`。
2. `api.search.requested`：进入 `/v1/search-agent/search`。
3. `search.started`：一次检索开始。
4. `search.candidate_pool`：命中候选池来源（faiss 或全量）。
5. `search.scored`：打分完成后的候选规模与 top ids。
6. `search.completed`：首次检索返回结果。
7. `clarify.started`：追问流程开始。
8. `clarify.completed`：追问返回结果。
9. `api.execute.requested`：动作执行入口。
10. `action.*`：文件动作日志（`action.open_ok`、`action.annotate_ok` 等）。
11. `ingest.finished`：一次扫描/重建周期的扫描与导入结果。
12. `http.request`：每次 HTTP 请求的统一链路指标。

日志示例（可用于 grep）：

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

---

## 6. 接口示例

### 6.1 `POST /v1/chat`

```json
{
  "message": "请给我一个端侧部署 tiny llm 的方案",
  "force_cloud": false
}
```

返回字段：

- `source`: `edge` / `cloud`
- `escalated`: 是否发生了边侧到云端的切换
- `reason`: 路由原因（如 `edge_ok`, `edge_low_confidence`, `policy_force_cloud_first` 等）
- `used_model`: 实际调用模型 id
- `edge_confidence`: 端侧置信度（仅 edge 路径）
- `text`: 返回文本

### 6.2 `POST /v1/embeddings`

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

### 6.3 `POST /v1/expense/collect`（报销场景：收进来）

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
- `claim_id`：所属报销单
- `extracted_amount`：抽取金额
- `needs_follow_up`：是否提示缺失关键材料
- `missing_required_types`：当前报销单还缺的资料类型

### 6.4 `POST /v1/expense/search`（报销场景：找回来）

```json
{
  "claim_id": "reimbursement-2026-09-001",
  "keyword": "京东 发票",
  "limit": 20
}
```

返回字段：

- `items`：匹配材料列表（含 `score`）
- `missing_required_types`：报销单缺失材料类型
- `keyword`/`claim_id`：查询上下文回显

### 6.5 `POST /v1/expense/export`（报销场景：拿出去）

```json
{
  "claim_id": "reimbursement-2026-09-001"
}
```

返回字段：

- `export_id`：本次导出 ID
- `manifest`：可直接复制的导出清单文本
- `materials`：导出的材料摘要列表

导出清单可用于提交给财务或在下一次报销复用时快速核对。

### 6.6 `POST /v1/search-agent/search`（个人文件搜索第一阶段）

```json
{
  "query": "上周群里发的聚餐照片",
  "top_k": 6,
  "force_disambiguation": false
}
```

返回字段：

- `state`: `resolved` / `needs_clarification` / `not_found`
- `session_id`: 追问会话 ID
- `selected_file_id`: 解析到确定命中时给出
- `candidates`: 每条候选附带 `evidence`、`visual_hints`、`matched_clues`
- `next_action_suggestions`: `打开/分享/备注/归档` 等建议动作

### 6.7 `POST /v1/search-agent/clarify`（追问消歧）

```json
{
  "session_id": "上一轮返回的 session_id",
  "reply": "我想要群里那张7月的"
}
```

### 6.8 `POST /v1/search-agent/execute`（命中后动作）

```json
{
  "file_id": "命中文件 id",
  "action": "open|share|compare|annotate|archive",
  "share_to": "老王",
  "peer_file_id": "需要对比时提供",
  "note": "这个是最终版"
}
```

`share` 会返回 `share_payload`，`compare` 会返回 `compare_payload`（含时间、来源、推荐保留项），`annotate`/`archive` 会保存在执行上下文中。

### 6.9 `POST /v1/search-agent/rebuild-index`（个人文件索引重建）

返回字段同上，包含本轮扫描的 `scanned/imported/skipped/errors` 以及 `material_ids`。

```json
{
  "scanned": 12,
  "imported": 8,
  "skipped": 4,
  "errors": 0,
  "material_ids": ["fm_xxx"]
}
```

---

## 7. 场景化产品目标（V1）

1. 先只打通「报销」这个场景，做成三动作闭环：
   - 收进来：用户把材料分享给助手，立即可检索
   - 找回来：关键词/时间段/金额快速检索
   - 拿出去：导出可提交的材料清单
2. 复盘指标（只看“用户再次回来”）：
   - 同一 `claim_id` 14 天内再次使用 search/export 的占比
   - 收集后 1 小时内补齐材料的成功率
   - 用户对自动抽取字段的纠正次数（越少越好）



---

## 8. 运行脚本与常用命令

- 本地快速启动（推荐）

```bash
make run
```

- 安装依赖

```bash
make install
```

- 查看文档与排障

- [docs/QUICKSTART.md](/Users/wzy/PycharmProjects/端云结合/docs/QUICKSTART.md)
- [docs/ARCHITECTURE.md](/Users/wzy/PycharmProjects/端云结合/docs/ARCHITECTURE.md)

---

## 9. 推荐配置（先跑通端侧）

```bash
EDGE_QUANTIZATION=int4-gp32
CLOUD_ENABLED=false
ROUTE_USE_TINYLLM=true
EDGE_EMBEDDING_SOURCE=modelscope
EDGE_EMBEDDING_MODEL_ID=google/embeddinggemma-300m
```

如需云端可选 Gemma4 兜底：

```bash
CLOUD_ENABLED=true
CLOUD_MODEL_ID=google/gemma-4-e2b-it
```

---

## 7. Android APK (端侧首版)

本仓库新增了 Android MVP 客户端，目录：`android/`。

- 手机端目标：
  - 与现有后端 `/v1/search-agent/search`、`/v1/search-agent/clarify`、`/v1/search-agent/execute`、`/v1/search-agent/rebuild-index` 打通。
  - 支持“搜索 -> 追问 -> 执行动作（打开/分享/归档/备注/对比）”。
  - 预留前台服务 `EdgeRuntimeService`，用于后续接入端侧推理、监听与增量索引。

- 启动要点：
  1. 先启动 Python API（默认监听 `0.0.0.0:9000`）。
  2. 用 Android Studio 打开 `android/` 目录。
  3. 若运行在真机或模拟器，请把 `android/app/build.gradle.kts` 中 `BuildConfig.API_BASE_URL` 调整为可达地址。

- 默认模拟器地址：`http://10.0.2.2:9000`。

后续我会继续把“端侧服务能力”从服务桩升级到真正的本地模型推理分发（例如文件监听、embedding、tinyllm 小模型路由），并保持同一套 UI 动作闭环不变。

端侧日志补充：
- 页面会展示 API 请求日志（search / clarify / execute / rebuild）；
- 服务端扫描日志会展示在“端侧运行日志”列表，覆盖 `watcher_registered / scan_scheduled / scan_done / scan_failed`。
- 日志文件落盘到 `android/app/filesDir/runtime_events.log`。
