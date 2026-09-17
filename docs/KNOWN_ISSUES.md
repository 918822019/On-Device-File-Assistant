# 已知问题（Known Issues）

环境配置过程中排查出的问题清单。**已修复**项保留记录以免回归，**未修复**项需要在后续处理。

验证环境：macOS / Apple Silicon（MPS 可用）/ 48 GiB RAM / Python 3.11.15 / transformers 5.17.0。

---

# 一、已修复

## R1. 端侧 LLM 完全无法加载（缺 `optimum`）

**原现象**：启动时 `Edge 初始化失败: No package metadata was found for optimum`，
`/v1/chat` 恒定返回固定文案，无任何推理发生。

**原根因**：默认 `EDGE_QUANTIZATION=int4-gp32` 命中 `_QUANT_DIRECT_MODES`，
`_resolve_model_id()` 解析到 GPTQ 仓库 `TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ`。
该仓库 `config.json` 自带 `quantization_config`，transformers 加载时走
`AutoHfQuantizer.merge_quantization_configs`，**强制要求 `optimum`**——而 requirements.txt 未声明。

四级回退链因此全灭：

| 步骤 | 分支 | 结果 |
|---|---|---|
| 1 | `_QUANT_DIRECT_MODES` 直接加载 GPTQ ckpt | 缺 optimum → 被 `except` 吞掉 |
| 2 | `GPTQConfig` 量化加载 | 查 optimum 版本失败 → 被吞掉 |
| 3 | bitsandbytes int4 | macOS 上 bitsandbytes 0.42.0 编译时无 GPU 支持 → 被吞掉 |
| 4 | fp16 最终兜底 | **仍加载同一个 GPTQ 仓库** → 再撞 optimum，且无 try/except → 抛出 |

**修复方式**：端侧改用 `google/gemma-4-E2B-it` + `EDGE_QUANTIZATION=none`，
量化链路整体不再触发。`/v1/chat` 现返回 `reason: edge_ok` 与真实生成结果。

**遗留**：回退链本身的缺陷未动（见 O5），仅在重新启用量化时才会暴露。

---

## R2. `_dtype()` 兜底分支返回 float16，导致 float32 不可达

**原代码**（`embedding_runtime.py`）：

```python
def _dtype(self):
    if (...).lower() in {"bf16", "bfloat16"}: return torch.bfloat16
    if (...).lower() in {"fp16", "float16"}:  return torch.float16
    return torch.float16          # ← bug：应为 float32
```

设 `EDGE_EMBEDDING_TORCH_DTYPE=float32` 时 `cfg.torch_dtype` 确实是 `"float32"`，
但 `_dtype()` 落入兜底仍返回 `torch.float16`。实测：

| 配置值 | 修复前实际 dtype | 修复后实际 dtype |
|---|---|---|
| `bfloat16` | bfloat16 | bfloat16 |
| `float16` | float16 | float16 |
| `float32` | **float16**（被吞） | **float32** ✅ |

**修复方式**：补上 `float32` 分支，兜底改为返回 `"auto"`（交由 transformers 读取模型 config）。
`edge_runtime.py` 原先把 `torch_dtype=torch.float16` **硬编码**在两处，同样改为可配置的 `_dtype()`。

---

## R3. MPS + float16 下 embeddinggemma-300m 输出全 NaN

`device_map="auto"` 在 Apple Silicon 上落到 `mps:0`，该模型在 MPS + fp16 下 forward 产出 NaN
（768 维全部污染），mean-pooling 后 `cos()` 为 `nan`。

**修复方式**：新增 `EDGE_DEVICE` / `EDGE_EMBEDDING_DEVICE` 配置项，本机设为 `cpu`。
CPU + bfloat16 实测 5/5 次逐位相同（`cos=0.647385 / 0.496369`）。

**遗留**：MPS 路径本身的问题未解决，且在 transformers 5.x 下更严重，见 O1。

---

## R4. `torch_dtype=` 在 transformers 5.x 已废弃

每次加载都打印 `` `torch_dtype` is deprecated! Use `dtype` instead! ``，
涉及 `embedding_runtime.py` 1 处、`edge_runtime.py` 4 处。

**修复方式**：全部改为 `dtype=`。实测 5.17.0 下 `dtype="auto"` 会读取模型 config.json 的
`dtype` 字段（E2B → bfloat16；embeddinggemma-300m → float32）。

---

## R5. 模型缓存落在 `/tmp`，重启后 1.1GB 权重复下载

`_download_if_modelscope()` 的兜底 cache_dir 是 `/tmp/modelscope_embedding_cache`，
macOS 重启即清空。

**修复方式**：权重迁至项目内 `models/`（已 gitignore），`*_LOCAL_DIR` 指向该处，
`*_SOURCE` 设为 `local` 以跳过下载分支。启动时 `/health` 就绪时间从 >30s（下载中）降至 **5–6s**。

---

## R6. `EDGE_QUANTIZATION=none` 仍会尝试 bitsandbytes int4

**原代码**：`_load_model_with_quantization()` 的前两个分支有 `mode` 守卫，
但 bitsandbytes 块**无条件执行**。因此 `mode="none"` 会跳过前两个分支后直接落入 int4 量化尝试——
在无 GPU 的机器上对 9.54GB 模型做一次注定失败的加载。

**修复方式**：函数开头增加未量化早返回分支：

```python
if mode in {"", "none", "no", "off", "false", "0"}:
    return AutoModelForCausalLM.from_pretrained(model_source, dtype=dtype,
                                                device_map=device_map, trust_remote_code=True)
```

实测日志确认生效：`量化已关闭（EDGE_QUANTIZATION='none'），按 auto 直接加载`，
不再出现 bitsandbytes 相关 warning。

---

## R7. `.env` 从未被 Python 代码加载（只有 `scripts/run.sh` 会 source）

**原现象**：`python-dotenv` 在 requirements.txt 里声明并已安装，但**全 src 无任何 import**。
`.env` 仅由 `scripts/run.sh` 的 `source .env` 注入。而 README 第 4 节教的主启动方式是直接跑：

```bash
cp .env.example .env
export PYTHONPATH=src
uvicorn edge_cloud_agent.main:app --host 0.0.0.0 --port 9000
```

照此启动时 `.env` **完全被忽略**。实测：`EdgeConfig().model_id` 得到代码硬编码默认值
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`，而非 `.env` 里配置的 `google/gemma-4-E2B-it`——
即按 README 操作会直接掉进 R1 那条坏掉的量化路径。

**修复方式**：在 `edge_cloud_agent/__init__.py` 调用 `load_dotenv(repo_root/".env", override=False)`。
放在包 `__init__.py` 是必须的——因为 config.py 的默认值在 import 时绑定（见 O6），
dotenv 必须早于任何子模块 import 注入。现在 uvicorn / make run / pytest / 直接 import
四种入口都能读到 `.env`。

**验证**：不经 run.sh 直接 import，`EdgeConfig.model_id` 正确得到 E2B；
显式设置的环境变量仍优先于 `.env`（`override=False` 语义）。

---

## R8. `scripts/run.sh` 无条件 source `.env`，覆盖调用方显式环境变量

**原现象**：`set -a; source .env; set +a` 会用 `.env` 里的值**覆盖**调用方传入的环境变量，
优先级与惯例相反。导致 `CLOUD_ENABLED=true bash scripts/run.sh` 这类临时覆盖完全失效——
排查 O3 时正是被这一点误导，一度以为云端已被调用。

**修复方式**：改为逐行读取 `.env`，仅对「尚未在环境中设置」的键赋值，跳过空行/注释/非法键名，
与 Python 侧 `load_dotenv(override=False)` 保持同一语义。

**验证**：`CLOUD_ENABLED=true CLOUD_API_BASE=... bash scripts/run.sh` 现在能把覆盖值送进服务进程。

---

## R9. `_edge_only()` 丢弃 reason 参数，硬编码返回 `"edge_ok"`

**原现象**：路由判定应走云端但 `CLOUD_ENABLED=false` 时，`ask()` 调
`_edge_only(messages, reason=decision.reason)`，而该函数成功分支把 reason 硬编码成 `"edge_ok"`，
真实的路由判定（`policy_force_cloud_first` / `policy_tiny_disabled`）被丢弃。
响应因此无法反映「本来想去云端」这一事实——这正是 R7/R8 排查过程中误判的直接原因。

**修复方式**：透传为 `f"{reason}_edge_only"`，异常分支透传为 `f"{reason}_edge_error"`，
并在改由端侧应答时打一条 warning 说明云端未启用。

---

## R10. 云端调用缺少异常保护，兜底路径自己会变成 HTTP 500

**原现象**（即原 O3）：两个问题叠加。

1. `CLOUD_API_BASE` 默认值 `http://127.0.0.1:8000/v1` 指向**本机 8000 端口**，
   该端口在本机被 `arxiv_fetcher.web:app` 占用。实测
   `POST http://127.0.0.1:8000/v1/chat/completions` → **404**，
   其真实路由是 `/api/search`、`/api/paper/{id}` 等，无任何 OpenAI 兼容端点
2. `agent.py` 的 `_ask_cloud` 与 `_call_cloud` 外层**没有 try/except**，
   `cloud_client.py` 的 `resp.raise_for_status()` 抛出的 `HTTPError` 一路冒泡到
   `main.py` 的全局兜底 handler

结果：启用 `CLOUD_ENABLED=true` 而地址不对时，`/v1/chat` 从「200 + 降级文案」
**退化成 HTTP 500 `internal_error`**，比不开云端更糟。这与 R1 是同一种模式：
**回退路径自己没有被保护**，于是回退机制在真正需要它时失效。

**修复方式**：

- `cloud_client.py`：`api_base` 为空时抛出带明确指引的 `RuntimeError`（不再拼出畸形 URL）；
  并 strip 尾部 `/` 避免出现 `//chat/completions`
- `agent.py`：`_call_cloud` 捕获所有异常返回 `None` 并记 warning；`_ask_cloud` 在云端失败时
  依次回退——已有端侧结果则保留 → 否则再试端侧推理 → 都不行才返回
  `used_source="none"`、`model=""` 的诚实降级提示（新增 `_cloud_unavailable_msg`）
- `config.py`：`CLOUD_API_BASE` 默认值改为空字符串，强制显式配置

**验证**（云端指向 arxiv-fetcher，即必然 404）：

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `force_cloud=true` + 端侧可用 | HTTP 500 | HTTP 200，`cloud_failed_edge_fallback`，返回端侧真实回答 |
| 云端失败 + 端侧已答但低置信 | HTTP 500 | HTTP 200，`edge_low_confidence_cloud_failed_kept_edge`，保留端侧结果 |
| 云端失败 + 端侧也不可用 | HTTP 500 | HTTP 200，`used_source="none"`、`model=""`、明确提示 |
| 云端关闭（基线） | `edge_ok` | `edge_ok`，行为不变 |

服务端 0 个 5xx、0 个未处理异常；云端失败降为一条 warning：
`云端调用失败 api_base='http://127.0.0.1:8000/v1' model='': 404 Client Error`。

---

## R11. `generate()` 无条件 `output_scores=True`，为每步留存全词表 logits

**原现象**：`EdgeRuntime.generate()` 固定传 `output_scores=True` +
`return_dict_in_generate=True`，于是 transformers 为**每一个**生成步保留一份
(batch, vocab) 张量。本模型 `vocab_size=262144`，`EDGE_MAX_NEW_TOKENS=220` 时
每请求驻留约 **220 MiB**，且随 `max_new_tokens` **线性增长**（1024 步约 1 GiB）。

而唯一的消费者 `_estimate_confidence()` 只读前 6 步：

```python
window = min(6, len(scores))
for step_scores in scores[:window]:
```

即保留了 220 步、只用 6 步。

**修复方式**：改用 `_ConfidenceRecorder`（一个 logits processor），在生成过程中就地
记录前 6 步的 max-softmax，超出窗口只做一次长度判断，不留存任何张量。
`generate()` 不再传 `output_scores` / `return_dict_in_generate`；
`_estimate_confidence()` 已成为死代码，一并删除。窗口大小提为类常量
`_CONFIDENCE_WINDOW = 6`，与原实现一致。

**实测收益——需要如实说明**：

| 指标 | 旧实现 | 新实现 |
|---|---|---|
| 生成速度（预热后交替对照 4 轮） | 12.88s / 220 tok = 17.08 tok/s | 12.88s / 220 tok = 17.08 tok/s |
| 每请求 scores 驻留 | 220 MiB | **0** |
| confidence 数值 | — | 4 个 prompt **逐位相同**（差 `0.00e+00`） |
| 生成文本 | — | 逐字相同 |

**速度没有改善（比值 1.000）**。保留这个改动的理由是内存：端侧目标是 Android
而非 macOS，每请求 220 MiB 的额外驻留在手机上远比为 48 GiB 的开发机重要，
且该开销会随 `max_new_tokens` 线性放大。

---

### ⚠️ 排查教训：首次 `generate()` 的预热成本会伪造出巨大的性能差异

本次修复的**初始动机是错的**。我先后得出「快 3.1 倍」「快 1.30 倍」两个结论，
两者都是测量假象：

| 测量方式 | 得出的结论 | 错在哪 |
|---|---|---|
| 进程内连续两次 generate，`output_scores=True` 排第一 | 快 3.1 倍 | 第一次调用吃满预热成本 |
| 同上，把新实现排第一 | 快 1.30 倍 | 顺序反过来，结论就反过来了 |
| **先预热 2 次丢弃，再交替测量 4 轮** | **0% 差异** | — |

预热成本量级：**进程内第一次 `generate()` 比后续慢约 25%，且 RSS 增长约 3.6 GiB**
（kernel 懒初始化、allocator arena 扩张、9.5 GiB 权重首次触页）。
我曾把这 3.6 GiB 错误归因给 `output_scores`。

因此，本项目任何推理性能对比都必须：

1. **先预热**（跑 1–2 次短生成并丢弃结果）
2. **交替测量**（A/B/A/B 而非 AAA/BBB），并去掉各自第一轮
3. **多轮取均值**，单轮差值在 ±5% 内视为无差异

另：`resource.getrusage().ru_maxrss` 在 macOS 上单位是**字节**，不是 KB；
按 KB 换算会把结果放大 1024 倍（本次排查中一度把 1.1 GiB 读成 1124 GiB）。

---

# 二、未修复

## O1. transformers 5.x 的 SDPA 在 MPS 上产出 NaN 且非确定性 ⚠️ 最重要

同一模型、同一批文本、bf16，多次重复运行的完整矩阵：

| transformers | device | attn | NaN | 确定性 |
|---|---|---|---|---|
| 4.57.6 | MPS | 默认 | 0 | ✅ 16/16 **逐位相同** |
| 4.57.6 | CPU | 默认 | 0 | ✅ 3/3 相同 |
| **5.17.0** | MPS | 默认 sdpa | **3072** | ❌ 8/8 全 NaN |
| 5.17.0 | MPS | 显式 sdpa | 3072 | ❌ 6/8 NaN，通过的 2 次值还互不相同 |
| 5.17.0 | MPS | eager | 0 | ⚠️ **8 次出现 6 种不同结果**，cos 在 0.462~0.779 间跳变 |
| **5.17.0** | CPU | 默认 | 0 | ✅ 6/6 相同，且与 4.57.6 CPU **逐位一致** |

关键结论：

1. **升级本身不改变数值行为**——5.17.0+CPU 与 4.57.6+CPU 逐位一致，问题精确出在 5.x 的 SDPA×MPS 路径
2. `attn_implementation="eager"` 能消除 NaN，但**引入非确定性**：同一输入的余弦值会漂移，
   甚至语义排序反转（「天气」偶尔高于「收据」）
3. 非确定性比 NaN 更危险：NaN 会报错，而漂移的向量会**静默污染 faiss 索引**，
   导致同一查询在不同时刻召回不同结果
4. `PYTORCH_ENABLE_MPS_FALLBACK=1` 无效（3/3 仍 NaN）

**当前规避**：`EDGE_DEVICE=cpu` / `EDGE_EMBEDDING_DEVICE=cpu`。
**代价**：放弃 MPS 加速。**风险**：任何人把这两个值改回 `auto` 就会静默踩坑。

建议后续：加启动自检，当 device 解析为 mps 且 transformers >= 5.0 时打 ERROR 级日志或直接拒绝启动。

---

## O2. `/v1/chat` 降级响应误报 `source` 与 `used_model`

端侧不可用时 `agent.py::_edge_unavailable_msg()` 返回：

```json
{
  "source": "edge",                     // 声称端侧回答，实际无任何推理
  "used_model": "TinyLlama/...",        // 取自 cfg.model_id，该模型从未加载成功
  "reason": "edge_unavailable_cloud_disabled",
  "text": "端侧模型未就绪且未开启云端兜底，当前仅支持本地 tiny llm 推理。"
}
```

- **HTTP 200**，耗时约 2ms（无推理的证据）
- 任意输入返回**逐字节相同**的响应
- 只有 `reason` 与固定 `text` 说了真话

只看状态码或 `source` 字段的调用方会误判为成功。依赖缺失会在启动日志里喊出来，
**误报不会**——这是本清单里最容易咬人的一项。

建议：降级时 `source` 应为 `"none"`/`"unavailable"`，`used_model` 置空，
或直接把该情形改为 HTTP 503（与 `routers/embeddings.py` 对未就绪的处理保持一致——
那里就是抛 503）。

---

## O4. `local_cache_dir` 双重语义冲突

`EDGE_LOCAL_DIR` / `EDGE_EMBEDDING_LOCAL_DIR` 被两处代码以不同含义使用：

- `_resolve_model_id()`：当作**可直接 `from_pretrained` 的模型目录**
- `_download_if_modelscope()`：当作 **modelscope 的缓存根目录** `cache_dir`

设置该值后，`_load_model()` 会把本地路径当模型 ID 传给 `snapshot_download()`，
下载必然失败 → 被 `except` 捕获 → warning「ModelScope 下载失败，回退直接加载模型」→
再用本地路径 `from_pretrained` 才成功。功能可用，但每次启动多一条误导性 warning。

**规避**：填 `*_LOCAL_DIR` 时把对应 `*_SOURCE` 设为非 `modelscope` 的值（如 `local`）。
本机 `.env` 已如此配置。

---

## O5. 量化回退链第 4 级复用 GPTQ 仓库而非 `cfg.model_id`

`_load_model_with_quantization()` 的最后一级兜底本意是「放弃量化、退回普通模型」，
但它使用的仍是 `model_source`（= `resolved_model`，已指向 GPTQ 仓库）而非 `self.cfg.model_id`，
因此永远无法真正降级到可加载的模型。且该分支**没有 try/except**，异常直接抛出。

当前因 `EDGE_QUANTIZATION=none` 不会走到这里。若重新启用量化，该缺陷会立刻复现 R1。

---

## O6. 配置默认值在 import 时绑定，运行期改环境变量无效

`config.py` 的所有 Config 均为 `@dataclass(frozen=True)`，默认值直接写在类体里：

```python
source: str = os.getenv("EDGE_EMBEDDING_SOURCE", "modelscope")
```

`os.getenv()` 在**模块 import 时求值一次**，此后修改 `os.environ` 不会反映到新实例上。

影响：无法在同一进程内切换配置做多组对比测试；任何「先 import 再设环境变量」的调用方式
都会静默拿到旧值。

排查本项目配置问题时，务必在启动进程**之前**（shell 层或 `scripts/run.sh` 的 `.env` 加载）设好环境变量。

**与 R7 的关系**：正因为默认值在 import 时绑定，`load_dotenv()` 必须放在包的
`__init__.py` 里——晚于任何子模块 import 都会失效。这也是 R7 修复方案的约束条件。

**排查陷阱**：在同一进程内改 `os.environ` 后重新构造 `EdgeConfig()` 不会生效。
做多组配置对比时必须**每组一个独立子进程**，并在 Python 启动前设好环境变量。
（本次排查 O3 时就因此得到过一批无效结果。）

---

## O7. `modules.json` 引用的 `4_Normalize` 目录不存在

ModelScope 下载的 `google/embeddinggemma-300m` 快照中，`modules.json` 声明了 5 个模块
（Transformer / Pooling / 2_Dense / 3_Dense / Normalize），但目录里只有
`1_Pooling`、`2_Dense`、`3_Dense`，缺 `4_Normalize`。

**当前无影响**：本项目用 `AutoModel` 取 `last_hidden_state` 后自行做 mean-pooling + L2 归一化，
不经过 sentence-transformers，因此不读 `modules.json`。

但需注意：**`2_Dense` / `3_Dense` 投影层当前也未被使用**，接口返回的是 768 维 hidden state
的均值池化结果，而非该模型设计的投影向量。若日后改用 `SentenceTransformer` 加载，
会因缺 `4_Normalize` 报错，同时向量语义也会与现在不同（进而使已建的 faiss 索引失效）。

---

## O8. `used_model` 返回本地路径而非模型 ID

设置 `EDGE_LOCAL_DIR` 后，`_resolve_model_id()` 返回本地目录路径，
该值一路传到响应的 `used_model` 字段：

```
"used_model": "/Users/wzy/.../models/google/gemma-4-E2B-it"
```

泄露本机绝对路径，且调用方无法据此识别实际模型。建议响应中改填 `cfg.model_id`。

---

## O9. E2B 的 128K 上下文能力被配置严重低估

`gemma-4-E2B-it` 的 `text_config.max_position_embeddings = 131072`（128K），
而 `.env` 仍是 TinyLlama 时代留下的：

- `EDGE_MAX_INPUT_TOKENS=1400`
- `EDGE_MAX_NEW_TOKENS=220`
- `ROUTE_MAX_INPUT_CHARS=1800`（超过就试图转云端，而云端未启用）

即长度略超 1800 字符的请求会被路由判定为「应走云端」，但云端关闭，
最终仍退回端侧并给出降级提示。

**⚠️ 修正：这些阈值不应调大。** 本机 CPU 实测 prefill 约 **30 tok/s**（64/128/256/512
tokens 分别为 32.8/33.3/33.9/40.1 ms per token，近似线性，28 层 sliding_attention 抑制了
O(n²)）。按此推算，`EDGE_MAX_INPUT_TOKENS=1400` 光是 prefill 就需 **约 47 秒**才吐出第一个字；
若按模型规格调到 8192，prefill 需数分钟，请求根本不可用。

也就是说 1400/220/1800 这组数字**并非「TinyLlama 时代的遗留错误」，而大致就是
CPU 推理可承受的上限**。真正的约束是运行硬件而非模型能力——E2B 的 128K 上下文要在
端侧 NPU/GPU（Android，见 `android/app/src/main/cpp/`）上才有意义。

另：decode 实测约 17 tok/s，且 4–5 线程即饱和（1/2/4/5/8/12/15 线程分别为
7.1/10.5/14.6/15.8/16.5/16.4/18.1 tok/s，隐含带宽 72→185 GB/s）。这说明 decode 是
**内存带宽瓶颈**：每生成一个 token 都要把 9.51 GiB 权重完整读一遍，加线程无效，
只有量化（减少每 token 字节数）能提速。

因此本项在 macOS 开发机上**不建议调整**；迁移到 Android 端侧后应按实际 NPU 吞吐重新标定。

---

## O10. 服务启动时同步加载模型，阻塞 `/health`

启动流程同步加载端侧 LLM（9.54 GiB）与 embedding（1.1 GiB）。权重已在本地时
`/health` 约 5–6s 就绪；但首次部署或缓存被清空时会触发下载，`/health` 在下载完成前不可达
（实测 >30s），健康检查/编排系统可能误判为启动失败。

建议改为后台异步预热 + `/health` 立即返回就绪状态（并在就绪前对相关接口返回 503）。
