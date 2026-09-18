# 快速上手（Quickstart）

## 0. 目标

本项目默认行为：端侧优先（`google/gemma-4-E2B-it`），云端可选但选型未定。默认 `CLOUD_ENABLED=false`，因此只跑端侧。

## 1. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 需要 transformers **5.x**（4.57 及以下不认识 `gemma4` 架构）。当前 requirements.txt 钉 5.17.0。

## 1.1 下载权重到 models/（已 gitignore）

```bash
# E2B 约 9.54 GiB，embedding 模型约 1.1 GiB
hf download google/gemma-4-E2B-it --local-dir models/google/gemma-4-E2B-it
```

`google/embeddinggemma-300m` 若走 ModelScope：`snapshot_download(..., cache_dir="models")`
会落到 `models/google/embeddinggemma-300m`。

## 2. 配置环境变量

```bash
cp .env.example .env
```

核心几项：

- `CLOUD_ENABLED=false`：云端未定，先只跑端侧
- `ROUTE_USE_TINYLLM=true`：默认 edge-first
- `EDGE_QUANTIZATION=none`：不量化，按模型原生精度加载（量化链路需 `optimum` + CUDA，本机不可用）
- `EDGE_DEVICE=cpu` / `EDGE_EMBEDDING_DEVICE=cpu`：**Apple Silicon 必须设 cpu**，
  transformers 5.x 的 SDPA 在 MPS 上会产出 NaN 且结果非确定性
- `EDGE_LOCAL_DIR` / `EDGE_EMBEDDING_LOCAL_DIR`：指向上一步下载的 `models/` 目录
- `EDGE_EMBEDDING_TORCH_DTYPE=bfloat16`：本机验证为确定且无 NaN

## 3. 启动服务

```bash
make run
```

或直接执行：

```bash
bash scripts/run.sh
```

## 4. 健康检查

```bash
curl http://127.0.0.1:9000/health
```

返回 `{"ok": true}` 即可。

## 5. 调用聊天接口

```bash
curl -X POST http://127.0.0.1:9000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请给我一个端侧部署 tiny llm 的方案","force_cloud":false}'
```

返回会包含：

- `source`
- `escalated`
- `reason`
- `used_model`
- `edge_confidence`
- `text`

## 6. 调用 embedding 接口（本地）

```bash
curl -X POST http://127.0.0.1:9000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"texts":["你好","请给我一个端侧部署方案"],"normalize":true}'
```

返回字段说明：

- `source`：固定 `edge`
- `model`：当前使用的 embedding 模型（默认 `google/embeddinggemma-300m`）
- `embeddings`：文本向量列表，按输入顺序返回

## 7. 开启云端兜底（可选，选型未定）

```bash
export CLOUD_ENABLED=true
export CLOUD_API_BASE=<真实的 OpenAI 兼容地址>/v1
export CLOUD_API_KEY=<key>
export CLOUD_MODEL_ID=<云端模型 id>
```

> **注意**：`CLOUD_API_BASE` 默认为空，启用云端时必须显式配置（为空时云端调用会给出
> 明确的配置错误提示）。云端调用已有异常保护：失败时依次回退「已有端侧结果 →
> 端侧推理 → 明确降级提示（used_source=none）」，不会返回 HTTP 500。

云端可达时，满足以下条件会走云端：

- 请求长度过长（`ROUTE_MAX_INPUT_CHARS`）
- 命中关键词（`ROUTE_FORCE` 相关关键词集合）
- 用户 `force_cloud=true`
- 端侧置信度低于阈值 `ROUTE_MIN_EDGE_CONFIDENCE`

## 8. 报销场景（V1）三动作闭环

### 8.1 收进来（share-in）

```bash
curl -X POST http://127.0.0.1:9000/v1/expense/collect \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "reimburse-2026-09-14",
    "source_app": "wechat",
    "doc_type": "invoice",
    "title": "地铁发票",
    "raw_text": "商户：地铁集团   金额: 12.00 元   日期:2026-09-14"
  }'
```

### 8.2 找回来（look-up）

```bash
curl -X POST http://127.0.0.1:9000/v1/expense/search \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "reimburse-2026-09-14",
    "keyword": "地铁",
    "limit": 10
  }'
```

### 8.3 拿出去（export）

```bash
curl -X POST http://127.0.0.1:9000/v1/expense/export \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "reimburse-2026-09-14"
  }'
```

返回的 `manifest` 可直接用于财务提交前核对。

## 9. 常见问题（排查顺序）

1. 服务启动报 import 错误
   - 先确认 `export PYTHONPATH=src`
2. 端侧加载失败
   - 检查 `EDGE_QUANTIZED_MODEL_ID` 与 `EDGE_QUANTIZATION`
3. embedding 初始化失败
   - 检查 `EDGE_EMBEDDING_*` 是否正确
4. 需要验证云端
   - 请求体中加入 `"force_cloud": true`
