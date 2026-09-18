# 端云结合 Agent（edge-first Gemma4-E2B）

本仓库实现一个“端侧优先、云端可选”的轻量 Agent：

- 端侧 LLM：`google/gemma-4-E2B-it`（Gemma4 E2B，any-to-any，权重约 9.54 GiB / bfloat16）。
  E2B 是 Google 面向端侧的型号，本机以非量化方式在 CPU 上推理。
- 端侧 embedding：`google/embeddinggemma-300m`（768 维，bfloat16）
- 云端：默认不启用（`CLOUD_ENABLED=false`），云端模型选型未定。启用时走 OpenAI 兼容的
  `/chat/completions` 接口，`CLOUD_MODEL_ID` 仅为透传给该接口的字符串。
- 路由：命中大输入 / 指定关键词 / 强制参数时可切云端；也可在配置里关闭 edge-first。

> **设备约束**：transformers 5.x 的 SDPA 在 Apple Silicon 的 MPS 上会产出 NaN 且结果
> 非确定性（同一输入多次运行余弦值漂移），故本机 `EDGE_DEVICE` 与
> `EDGE_EMBEDDING_DEVICE` 均固定为 `cpu`。详见 [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)。

> **历史说明**：早期版本端侧为 TinyLlama-1.1B + int4/gp32 量化，`gemma-4-e2b-it` 当时被
> 配置在云端位置。该量化链路依赖 `optimum` 且在本机无 CUDA 无法工作，现已弃用；
> 相关代码（`_QUANT_DIRECT_MODES` / `_GPTQ_MODES` 回退链）仍保留，可通过
> `EDGE_QUANTIZATION` 重新启用。

---

## 1. 架构图（仓库级）

- [src/edge_cloud_agent/config.py](src/edge_cloud_agent/config.py)：配置读取与默认值（边侧、云侧、路由策略）
- [src/edge_cloud_agent/edge_runtime.py](src/edge_cloud_agent/edge_runtime.py)：端侧 LLM 加载与生成（当前走非量化路径；GPTQ/BNB 回退链保留但需 `optimum` + CUDA）
- [src/edge_cloud_agent/embedding_runtime.py](src/edge_cloud_agent/embedding_runtime.py)：端侧 embedding 模型加载与向量生成（`google/embeddinggemma-300m`）
- [src/edge_cloud_agent/cloud_client.py](src/edge_cloud_agent/cloud_client.py)：云端 API 客户端
- [src/edge_cloud_agent/routing.py](src/edge_cloud_agent/routing.py)：路由策略层
- [src/edge_cloud_agent/agent.py](src/edge_cloud_agent/agent.py)：路由与回退策略（Edge/Cloud 编排）
- [src/edge_cloud_agent/main.py](src/edge_cloud_agent/main.py)：FastAPI 入口，提供 `/health`、`/v1/chat`、`/v1/embeddings`、`/v1/expense/*`、`/v1/search-agent/*`
- [src/edge_cloud_agent/routers/chat.py](src/edge_cloud_agent/routers/chat.py)：聊天路由定义
- [src/edge_cloud_agent/routers/embeddings.py](src/edge_cloud_agent/routers/embeddings.py)：embedding 路由定义
- [src/edge_cloud_agent/routers/expense.py](src/edge_cloud_agent/routers/expense.py)：报销场景闭环路由（收进来/找回来/拿出去）
- [src/edge_cloud_agent/expense/storage.py](src/edge_cloud_agent/expense/storage.py)：报销材料本地持久化（本地 JSONL 存储）
- [src/edge_cloud_agent/expense/service.py](src/edge_cloud_agent/expense/service.py)：报销工作流服务（字段抽取、检索、导出）
- [src/edge_cloud_agent/expense/schemas.py](src/edge_cloud_agent/expense/schemas.py)：报销场景请求与响应模型
- [src/edge_cloud_agent/routers/personal_search.py](src/edge_cloud_agent/routers/personal_search.py)：个人文件搜索路由（search/clarify/execute/rebuild-index）
- [src/edge_cloud_agent/personal_search/service.py](src/edge_cloud_agent/personal_search/service.py)：检索打分、澄清会话、动作执行核心服务
- [src/edge_cloud_agent/personal_search/ingest.py](src/edge_cloud_agent/personal_search/ingest.py)：目录扫描、增量导入与幽灵文件清理
- [src/edge_cloud_agent/personal_search/vector_index.py](src/edge_cloud_agent/personal_search/vector_index.py)：FAISS 向量索引封装（不可用时自动回退）
- [src/edge_cloud_agent/personal_search/storage.py](src/edge_cloud_agent/personal_search/storage.py)：个人文件 JSONL 存储（批量落盘 + 原子写）
- [src/edge_cloud_agent/personal_search/schemas.py](src/edge_cloud_agent/personal_search/schemas.py)：个人文件搜索请求与响应模型
- [src/edge_cloud_agent/text_utils.py](src/edge_cloud_agent/text_utils.py)：共享分词器（两条业务线统一口径）
- [tests/](tests/)：业务层单测（35 例，`pytest tests/`）
- [requirements.txt](requirements.txt)：依赖
- [.env.example](.env.example)：环境变量模板

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
cp .env.example .env    # 首次
make run
```

或者直接执行：

```bash
export PYTHONPATH=src
uvicorn edge_cloud_agent.main:app --host 0.0.0.0 --port 9000
```

或者：`bash scripts/run.sh`

三种方式的 `.env` 都会生效——包 `__init__.py` 会 `load_dotenv(override=False)`。
且**显式设置的环境变量优先于 `.env`**，因此可以临时覆盖而不必改文件：

```bash
CLOUD_ENABLED=true CLOUD_API_BASE=https://your-endpoint/v1 make run
```

---

## 5. 关键环境变量（端侧 Gemma4-E2B 为默认）

### 通用与聊天

- `EDGE_MODEL_SOURCE`：端侧来源；`hf` 或 `modelscope`
- `EDGE_MODEL_ID`：端侧基础模型（当前：`google/gemma-4-E2B-it`）
- `EDGE_LOCAL_DIR`：本地权重目录；设置后优先于 `EDGE_MODEL_ID` 直接加载
- `EDGE_DEVICE`：推理设备。`auto` 交给 accelerate 决定（Apple Silicon 上会落 MPS）；
  **本机须设为 `cpu`**，原因见文首设备约束
- `EDGE_DTYPE`：权重精度。`auto` = 沿用模型 `config.json` 声明值（E2B 为 bfloat16）；
  也可显式 `bfloat16` / `float16` / `float32`
- `EDGE_QUANTIZATION`：量化策略。`none`（当前默认）= 不量化，按原生精度直接加载；
  `int4-gp32` 等取值会进入 GPTQ/BNB 回退链，需额外安装 `optimum`，且本机无 CUDA 不可用
- `EDGE_QUANTIZED_MODEL_ID`：量化 ckpt 的模型 ID，仅在启用量化时使用
- `EDGE_QUANT_BITS` / `EDGE_QUANT_GROUP_SIZE`：GPTQ 参数（默认 4 / 32）
- `CLOUD_ENABLED=false`：当前默认关闭云端兜底
- `CLOUD_MODEL_ID`：云端模型 ID，透传给 OpenAI 兼容接口；云端选型未定
- `CLOUD_API_BASE`：云端接口地址。**注意默认值 `http://127.0.0.1:8000/v1` 指向本机 8000 端口**，
  启用前务必改成真实的云端地址
- `ROUTE_USE_TINYLLM=true`：`true` 代表端侧优先，`false` 代表云侧优先
  （变量名沿用早期 TinyLlama 时代，现端侧为 E2B，名称已名不副实；因涉及代码改动暂未重命名）
- `ROUTE_MAX_INPUT_CHARS`：触发云端优先的上下文长度阈值
- `ROUTE_MIN_EDGE_CONFIDENCE`：端侧置信度低于该阈值回退云端

### embedding（端侧）

- `EDGE_EMBEDDING_SOURCE`：`modelscope` 走 `snapshot_download`；其他值（如 `local`）跳过下载直接加载
- `EDGE_EMBEDDING_MODEL_ID=google/embeddinggemma-300m`
- `EDGE_EMBEDDING_LOCAL_DIR`：本地权重目录（优先）
- `EDGE_EMBEDDING_DEVICE`：同 `EDGE_DEVICE`，**本机须设为 `cpu`**
- `EDGE_EMBEDDING_MAX_LENGTH`：单条文本最大长度
- `EDGE_EMBEDDING_BATCH_SIZE`：单批向量化数量
- `EDGE_EMBEDDING_TRUST_REMOTE_CODE`
- `EDGE_EMBEDDING_TORCH_DTYPE`：`bfloat16`（本机验证为确定且无 NaN）/ `float16` / `float32` / `auto`

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

V1 打通两个端侧场景，业务层设计与现状详见 [docs/BUSINESS_LAYER.md](docs/BUSINESS_LAYER.md)。

1. 「报销」三动作闭环：
   - 收进来：用户把材料分享给助手，立即可检索
   - 找回来：关键词/时间段/金额快速检索
   - 拿出去：导出可提交的材料清单
2. 「个人文件搜索」消歧闭环（search → clarify → execute）：
   - 模糊口述（“上周群里发的聚餐照片”）→ 追问收敛 → 命中后动作（打开/分享/对比/备注/归档）
   - 已接通 Android MVP 客户端（见下文 §10）
3. 复盘指标（只看“用户再次回来”）：
   - 报销：同一 `claim_id` 14 天内再次使用 search/export 的占比；收集后 1 小时内补齐材料的成功率；用户对自动抽取字段的纠正次数（越少越好）
   - 文件搜索：追问后最终 resolved 并执行动作的会话占比（指标口径待定，暂无采集代码）



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

- [docs/QUICKSTART.md](docs/QUICKSTART.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/BUSINESS_LAYER.md](docs/BUSINESS_LAYER.md)：业务层梳理（报销/个人文件搜索）与遗留清单

---

## 9. 推荐配置（本机 macOS / Apple Silicon 已验证）

```bash
# 端侧 LLM：Gemma4-E2B，非量化，CPU
EDGE_MODEL_SOURCE=hf
EDGE_MODEL_ID=google/gemma-4-E2B-it
EDGE_LOCAL_DIR=<repo>/models/google/gemma-4-E2B-it
EDGE_QUANTIZATION=none
EDGE_DEVICE=cpu
EDGE_DTYPE=auto

# 端侧 embedding：embeddinggemma-300m，CPU + bfloat16
EDGE_EMBEDDING_SOURCE=local
EDGE_EMBEDDING_MODEL_ID=google/embeddinggemma-300m
EDGE_EMBEDDING_LOCAL_DIR=<repo>/models/google/embeddinggemma-300m
EDGE_EMBEDDING_DEVICE=cpu
EDGE_EMBEDDING_TORCH_DTYPE=bfloat16

# 云端未定，保持关闭
CLOUD_ENABLED=false
ROUTE_USE_TINYLLM=true
```

权重需预先下载到 `models/`（已 gitignore）：E2B 约 9.54 GiB，embedding 模型约 1.1 GiB。

云端选型确定后再启用：

```bash
CLOUD_ENABLED=true
CLOUD_API_BASE=<真实的 OpenAI 兼容地址>/v1   # 默认值指向本机 8000，必须先改
CLOUD_API_KEY=<key>
CLOUD_MODEL_ID=<云端模型 id>
```

---

## 10. Android APK (端侧首版)

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
