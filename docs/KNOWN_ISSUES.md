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
`.env` 仅由 `scripts/run.sh` 的 `source .env` 注入。而 README（当时版本）教的主启动方式是直接跑：

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

## R12. 拆分重构漏改导入，`_now_iso` 已移走但 ingest 仍在引用 → 应用完全起不来

**原现象**：一次分层重构把 `_now_iso` 从 `personal_search/service.py` 移到
`common/time_utils.py`（改名 `now_iso`），但 `personal_search/ingest.py:23` 的
`from .service import PersonalFileSearchService, _now_iso` 没跟着改。后果是
`import edge_cloud_agent.main` 直接 `ImportError`，**10 个模块连锁失败、8 个测试
文件无法 collection**，应用完全无法启动。这个状态在工作区里存活了整段时间，
因为唯一能发现它的手段（跑一次 app 或 pytest）在重构后没有执行过。

**为什么没被及时发现**：项目当时**没有安装任何 linter**（`make lint` 是空桩）。
但要注意——装上 ruff 也**发现不了**这个错误：ruff/pyflakes 只做单文件分析，
**不跨模块解析导入**，`from .service import _now_iso` 在语法上完全合法。
能发现它的只有类型检查器（mypy/pyright）或「把每个模块都 import 一遍」。
考虑到引入类型检查器会给这个依赖 torch/transformers/faiss 的代码库带来海量
基线噪音，选择了后者。

**修复方式**：改用 `common.time_utils.now_iso` 的公开名（不保留 `_now_iso` 别名
——私有下划线会掩盖「这是跨模块共享工具」这一事实），并把 8 个测试文件的导入
指向迁移后的新家（`relevance.*` / `sessions.*` / `text_utils.tokenize`）。

**防回归**：新增 `tests/test_import_smoke.py`，用 `pkgutil.walk_packages`
逐模块 import（parametrize 形式，失败时直接显示是哪个模块），并断言
`main.app` 能装配出 `/health` 与各条 `/v1` 路由。已证伪验证：把坏导入放回去，
测试精确指名 `ingest.py:29`。

---

## R13. `config.py` 默认值与 `.env.example` 相反，新克隆会走进已弃用路径

**原现象**：`.env` 被 gitignore，所以**新克隆环境跑的就是 `config.py` 的默认值**，
而有 6 项默认值与 `.env.example`（本机验证过的正确值）相反：

| 配置项 | 原默认值 | `.env.example` | 后果 |
|---|---|---|---|
| `EDGE_MODEL_ID` | TinyLlama-1.1B | google/gemma-4-E2B-it | 加载错误的模型 |
| `EDGE_QUANTIZED_MODEL_ID` | TheBloke/TinyLlama-GPTQ | 空 | **打开量化会静默加载完全无关的权重** |
| `EDGE_QUANTIZATION` | `int4-gp32` | `none` | 走进已弃用且依赖 CUDA 的 GPTQ→BNB 回退链 |
| `EDGE_DEVICE` | `auto` | `cpu` | Apple Silicon 上落 MPS → O1 的 NaN |
| `EDGE_EMBEDDING_TORCH_DTYPE` | `float16` | `bfloat16` | MPS+float16 输出**全 NaN 向量** |
| `EDGE_EMBEDDING_DEVICE` | `auto` | `cpu` | 向量非确定性 → 直接污染语义检索 |

最隐蔽的是第 2 项：`int4-gp32` + TinyLlama GPTQ 仓库的组合意味着新环境不只是
「慢」或「有 NaN」，而是**加载了一个和文档宣称完全不同的模型**。

代码里的注释当时已经在自相矛盾——`device` 的注释写着「本机请设为 cpu，
详见 KNOWN_ISSUES」，而默认值偏偏是 `auto`。

**修复方式**：6 项全部对齐到 `.env.example`，并把注释从「请自行覆盖」改成
「解释默认值为何如此」。`CLOUD_MODEL_ID` 也对齐为空（尊重「云端选型未定、
强制显式配置」的立场），同时补上 `cloud_client` 里**缺失的对称校验**——此前
`api_base` 有清晰的中文 RuntimeError，而 `model` 完全不校验，空值会变成
`"model": ""` 发给远端换回一个不透明的 4xx，再被 orchestrator 吞成通用降级提示。

**对本机无影响**：`.env` 存在且这 6 项的值与新默认值逐字相同，故改动只影响新克隆。

**防回归**：新增 `tests/test_config_env_parity.py`，用 **AST 静态解析**
`os.getenv(KEY, default)` 的字面默认值与 `.env.example` 逐项比对
（刻意不 import config——包 `__init__` 会 `load_dotenv()`，import 拿到的是被本机
`.env` 覆盖后的值，恰好掩盖了要测的东西）。数值比较容忍 `0.3` vs `0.30` 这类
纯格式差异；`ALLOWED_DIVERGENCE` 白名单当前为空，且有一条测试会在白名单条目
恢复一致时提醒删除豁免。附带校验 `.env.example` 里每个键都真的被代码读取。

---

## R14. `/health` 恒绿 + 全链路静默降级 → 后端整体坏掉也无从察觉

**原现象**：`bootstrap.py` 里**每一个**服务装配失败都是 `except Exception → None +
logger.warning`（metrics、embedding runtime、expense、personal file，连 warm scan
失败也只 warning），而 `/health` 无条件 `return {"ok": True}`。两者叠加的结果是：
后端可以整体坏掉而外部毫无察觉。

最典型的场景是 embedding runtime 加载失败——搜索会**静默退化为纯关键词匹配**，
仍然返回 200 和看起来正常的结果，只是少了语义召回，没有任何信号提示这一点。

**修复方式**：`/health` 补充 `degraded` 布尔与 `services` 明细（逐项报告
edge_llm / embedding_runtime / personal_file_service / expense_service / metrics
是否装配成功），但**保持 HTTP 200 且 `ok=true` 不变**。

状态码不能改的原因：`deploy.sh` 与 `service.sh` 用 `curl -fsS ... | grep -q '"ok"'`
做启动门禁，`-f` 会把 503 当成失败并死等到 `HEALTH_TIMEOUT`（900s）超时；
`web/js/core.js` 用 `resp.ok && data.ok` 驱动状态点。

`cloud_enabled` **不计入** `degraded`——它默认 false，是配置选择而非故障，
计入会让每个默认安装恒定报告降级，告警随即失去意义。索引就绪度也不在此处，
仍由 `/v1/search-agent/index-status` 负责。

前端加 `.dot-warn` 黄点（复用已有的 `--warn` 变量），文案列出缺失的服务名，
hover 显示完整清单。

---

## R15. 两个推理运行时零线程安全，并发 `/v1/chat` 会同时前向同一个模型

**原现象**：`EdgeRuntime.generate()` 与 `EdgeEmbeddingRuntime.embed()` 都没有任何锁
（对比：storage / refresh / sessions / ingest 全都用了 `Lock`/`RLock`）。而本项目
**22 个端点全部是同步 `def`、零个 `async def`**，FastAPI 会把它们丢进同一个 anyio
线程池（默认 40 个 token），因此两个并发 `/v1/chat` 会同时对同一个 HF 模型对象调
`generate`。HF 模型自身没有内部同步；CPU 上也没有并行跑两个 5.1B 前向的余量，
只会互相踩内存并把两边延迟一起拖长。

embedding 侧更严重：`bootstrap` 只构造**一个** `EdgeEmbeddingRuntime` 实例并同时
注入 `expense_service` 与 `personal_file_service`，后台 ingest/watch 线程会和 HTTP
请求线程并发调用 `embed`。

**修复方式**：两个运行时各加一把 `threading.RLock`（用 RLock 而非 Lock：当前没有
重入路径，但万一将来 `_generate_locked` 里调用了同样持锁的方法，RLock 不会自死锁），
实际前向逻辑拆到 `_generate_locked` / `_embed_locked`。就绪校验与空输入早返回
**留在锁外**——模型没加载好时应立刻报错，而不是排在一次长生成后面等锁。

**连带修复**：`/health` 改为 `async def`。加锁后并发 chat 会排队持锁，极端情况下
把 40 槽的线程池占满；若 `/health` 也是同步的，它会被一起饿死 → `deploy.sh` 健康
门禁失败 → systemd 把一个「健康但繁忙」的服务重启掉。该函数只读 `app.state`、
没有任何阻塞 IO，跑在事件循环上即可永不被业务负载阻塞。

**防回归**：新增 `tests/test_inference_concurrency.py`。运行时在 `__init__` 里就加载
真实模型（9.5 GiB + 300 MB），无法在测试中构造，故用 `object.__new__` 绕过
`__init__` 注入桩模型，断言 8 线程并发下的**前向并发峰值 == 1**。

每个正向断言都配一条**反证**：绕过锁直接调 `*_locked` 时必须**能**并发，否则
`peak == 1` 可能只是桩模型本身不并发、断言就是空的。（第一版测试正是栽在这里：
helper 自带了一把 RLock，导致「把生产代码的锁换成空操作」也无法让它失败。）
另有一条源码级断言覆盖「`__init__` 确实创建了 `_inference_lock`」——因为 helper
绕过 `__init__`，这一行被删时其余测试全绿，而生产环境会在首次推理时 AttributeError。

---

## R16. FAISS 的 index 与 ids.json 错位时，`search()` 静默返回**错误的文件**

**原现象**：`build()` 先写 index 文件、再写 `.ids.json`，两次独立写入之间崩溃会留下
错位组合（典型：index 已更新为 N 个向量，ids.json 仍是上一轮的 M 条）。而
`_load_index()` 只判断「index 存在且 ids 非空」就算就绪，`is_ready()` 里**完全没有
长度校验**。

后果比抛异常更糟：`search()` 是**按位置**把命中下标映射到 file_id 的
（第 i 个向量 → `_ids[i]`），错位状态下 `idx >= len(self._ids)` 的守卫只跳过后半段，
**前 M 个位置会用旧 ids 映射到新向量上**，返回的是别的文件，且不报错。

另外 `_persist_ids` 直接写目标文件（非原子），而 `PersonalFileStore` 与
`FileStateStore` 都正确使用了 tmp + `os.replace`——三个存储里只有它漏了。

**修复方式**：
1. `_persist_ids` 改为 tmp + `os.replace` 原子写，与另两个存储口径一致；
2. `is_ready()` 把 `len(self._ids) == self._index.ntotal` 作为**硬条件**；
3. `_load_index()` 检测到错位时记一条 `faiss.ids_mismatch` warning（含
   `index_vectors` / `ids_count` / `index_path`，便于排查），然后判为未就绪。

判为未就绪即可自愈：store 是唯一事实源，FAISS 只是可丢弃的缓存，下一轮扫描会全量
重建。故无需引入跨两个文件的事务。

**防回归**：新增 `tests/test_vector_index_consistency.py`（用 venv 里真实的
faiss 1.10.0），构造 ids 偏短、偏长、半截 JSON 三种错位状态，断言均判为未就绪且
`search()` 返回空而非错误结果；并验证错位可通过 rebuild 自愈、位置映射本身正确。
已证伪：撤掉长度校验后 4 条用例失败。

---

## R17. ingest 稳态 IO = O(语料总字节)/轮，且 `read_bytes()` 会把整个文件读进内存

**原现象**：两个问题叠加。

其一，判重 hash 实现为 `md5(path.read_bytes()[:1024*1024])`——`read_bytes()` 会把
**整个文件**读进内存**再**切片。而扫描后缀白名单里含 `.mp4` / `.mov`，一个几 GB 的
视频就是一次几 GB 的内存分配。两条业务线各有一份完全相同的实现
（`_short_sha1` / `_short_hash`，且名字里的 sha1 是假的，实际是 md5）。

其二，这个 hash 是**变更检测键**，`run_once` 对每个发现的文件**每轮都先算 hash
再判断有没有变**。于是稳态下每 120s 就要读一遍全量语料的前 1MB。另外
`_sweep_deleted_files` 对每个已入库条目再做一次 `path.exists()`——在 WSL 的 9P
跨系统调用上，这两项都是主要成本。

**修复方式**：
1. 提取到 `common/file_io.py::content_hash()`，**流式**读取（256 KiB 分块，只读
   满 `max_bytes` 就停），消除整文件进内存；两条业务线共用一份，顺带消除重复实现。
   输出与旧实现**逐字节等价**（同样的字节序列喂给同一个 md5），因此存量索引里的
   `file_hash` 不会全部失配、进而触发整库重导 + 用 300M 模型重新 embedding。
   异常口径也保持一致（任何失败返回 `""` 而不抛出，让调用方按「取不到 hash」处理，
   而不是把该文件计成扫描错误）。
2. 新增 **size + mtime 快路径**：`PersonalFileItem` 增加 `file_mtime_ns` 字段，
   一次 `stat()` 同时取 size 与 mtime，两者都没变就**连 hash 都不算**。
   稳态下每个文件从「读 1MB」降为「一次 stat」。
3. `_build_item` 接受调用方传入已算好的 hash/size/mtime——此前变更文件会被
   stat 两次、hash 两次（`_build_item` 内部又算了一遍）。
4. `_sweep_deleted_files` 接受 `known_existing`（本轮 `_discover_files` 已枚举到的
   路径集合），命中即跳过 stat。未命中时**仍回落到 `path.exists()`**：
   `_discover_files` 只返回白名单后缀且不在排除目录里的文件，一个仍然存在、只是被
   移出扫描范围的文件不该因此被误删，故语义与逐条 stat 完全一致。
5. `ingest.finished` 日志新增 `rehashed` / `mtime_backfilled` / `state_pruned` /
   `force_rehash`。`rehashed` 是本轮真正读了文件头部算 hash 的数量，稳态下应接近 0；
   若长期等于 `scanned`，说明快路径没生效。

**已知取舍（刻意保留，并被测试钉住）**：size 与 mtime 都不变的变更，快路径发现不了。
这正是云同步/备份恢复的行为，而本项目恰好以 OneDrive/iCloud/微信目录为主要目标。
为此 `run_once` 增加 `force_rehash` 参数：周期扫描传 False，UI 的「重建索引」按钮
传 True 做全量权威 hash 校验，作为这种场景的显式兜底出口。

**向后兼容**：旧记录没有 `file_mtime_ns`（`from_dict` 回落 None），会被当作「未知」
而重算一次 hash，然后**就地回填元数据**（`dataclasses.replace`，不重读内容、
不重算 embedding），下一轮即回到快路径。若不回填，这些文件会永久停留在慢路径。
回填也计入 flush 条件，否则重启后又是一轮全量重算。

**防回归**：新增 `tests/test_ingest_fastpath.py`（12 条）+ `test_file_io.py` 里的
hash 等价性用例（覆盖空文件 / 恰好 1 MiB / 超过 1 MiB / 含 NUL 二进制）。
三条已证伪：去掉快路径 → 4 条失败；去掉 `path.exists()` 回落 → 捕获误删场景；
去掉 mtime 回填 → 旧记录永久慢路径。

---

## R18. 前端查询上限 260 vs 后端静默截断 120，尾部线索被无声丢弃

**原现象**：`web/index.html` 的搜索框 `maxlength="260"`、`chat.js` 也硬编码 260，
而后端 `FILE_MEMORY_MAX_QUERY_LEN` 默认 120，`service.start_search` 直接
`query[:120]` **静默截断**。121~260 字符时，用户输入的尾部线索（往往正是「上周的」
「微信群里的」这类关键限定）被丢掉，不报错、不提示，只表现为「搜不准」。

**修复方式**：不采用「把两个魔数手动对齐」的做法——那只是把下一次漂移推迟到
有人改 config 的那天。改为让上限只有**一个事实源**：
`IndexStatusResponse` 新增 `max_query_len`（取自 cfg），`search.js` 在已有的
`loadIndexStatus()` 里据此设置输入框 `maxLength` 并写入 `core.js` 导出的共享
`limits` 对象，`chat.js` 改用 `limits.maxQueryLen`。`index.html` 的静态
`maxlength` 保留为 120（与后端默认值一致），只作为首屏与后端不可达时的兜底。

同时把截断从静默改为可观测：真正发生截断时记一条 `search.query_truncated`
warning（含 `original_query_len` / `limit` / `dropped_chars`）。**只记长度不记原文**
——本项目刻意不记录查询文本（既有日志只有 `query_digest` 与 `query_len`），
不在此引入内容泄露。

`limits` 放在 `core.js` 而非各模块自持，依赖 ES module 的 live binding；
所有模块必须**裸引** `./core.js`，若某个模块改成 `./core.js?v=x`，浏览器模块注册表
会把它当成另一个模块，`limits` 就会出现两份实例（search.js 写的 chat.js 看不到），
且 cache-busting 也不会因此生效（服务端已对 `/web/` 发 `no-cache`）。
这条约束由 `test_web_contract.py` 钉住。

**防回归**：新增 `tests/test_web_contract.py`，把链路
`index.html 静态值 == core.js limits 初值 == .env.example == config 默认值`
整体锁住（后两者的相等由 `test_config_env_parity.py` 保证）。已证伪：把
`maxlength` 改回 260 即失败。

---

## R19. Android 主启动路径两处崩溃：缺前台服务权限 + minSdk 28 撞 API 29 符号

**原现象（两处都在 `onCreate` 路径上）**：

1. manifest 只声明了 `FOREGROUND_SERVICE`，但 service 声明了
   `android:foregroundServiceType="dataSync"` 且 `targetSdk 34`。Android 14 起
   这两者必须配对，否则 `startForeground()` 抛 `SecurityException`。
2. `minSdk = 28`，但代码用到 `MediaStore.Downloads.EXTERNAL_CONTENT_URI`、
   `MediaStore.VOLUME_EXTERNAL_PRIMARY`（`EdgeRuntimeService.kt:156-157`）与
   `MediaStore.Files.FileColumns.RELATIVE_PATH`（`LocalFileIndexStore.kt:153,173`），
   三者均为 **API 29** 引入。在 Android 9 上会抛 `NoClassDefFoundError` /
   `NoSuchFieldError`，而 `registerWatchObservers()` 在 `onCreate` 里**没有
   `runCatching` 包裹**，`RELATIVE_PATH` 又出现在每次索引扫描的投影列里——
   属于必崩路径而非边缘情况。文件里其它三处 `SDK_INT` 守卫
   （TIRAMISU/M/O）都在，偏偏这几个 API 29 符号没有。

**修复方式**：
1. 补 `FOREGROUND_SERVICE_DATA_SYNC`。该权限 API 34 引入，但低版本系统会忽略
   未知的 `uses-permission`，故无需加版本条件。
2. `minSdk` 提到 **29**。DFlash 图需要的 Vulkan 1.1 入口自 API 28 起即具备，
   故这一改动不影响原生推理；相比之下加 `SDK_INT` 守卫要求后续每个贡献者都记得，
   而提高地板能永久消灭这一整类问题。顺带清掉两处因此（其实早在 minSdk 28 时就已）
   变成死分支的守卫：`pendingIntentImmutableFlag()` 的 `SDK_INT >= M` 恒真、
   `ensureForegroundChannel()` 的 `SDK_INT < O` 恒假。

已用 `./gradlew :app:assembleDebug` 实际构建验证通过（含 manifest 合并）。

---

## R20. 备注/归档标记永不剪枝，同路径新文件会**继承**旧文件的备注

**原现象**：幽灵清理只删 `PersonalFileStore` 里的条目，不会动 `FileStateStore`，
所以被删文件的备注与归档标记永久残留（无界增长）。

但这不只是内存问题：`file_id = "fm_" + md5(path.as_posix())[:14]`，**同一路径永远
得到同一个 file_id**。文件被删除后若该路径上出现一个内容不同的新文件，旧备注会
凭空贴到新文件上——用户看到的是一条与当前内容毫无关系的历史标记。

**修复方式**：`FileStateStore.prune(valid_file_ids)` 剪掉不在索引中的条目，
经 `service.prune_file_state()` 暴露；`run_once` 在 `removed > 0` 时调用，
`force_rehash`（手工重建）时也跑一次，让升级前积累的存量孤儿有确定的清理时机。
无变化时不落盘，避免每轮扫描都重写一次 JSON。剪枝失败被兜住并记 warning，
不影响扫描结果（下轮还会再试）。

**防回归**：`test_ingest_fastpath.py` 中 5 条用例，含直接复现上述继承场景的
`test_new_file_at_same_path_does_not_inherit_old_annotation`。已证伪：断开接线后
4 条失败。

---

## R21. 澄清 evidence 无界增长，且它直接展示在候选卡片上

**原现象**：`rerank_within` 每轮做 `evidence=f"{candidate.evidence}；{evidence}"`，
而**澄清轮数没有上限**（`turn` 只用于日志）。`evidence` 不只是内部字段——
`web/js/search.js` 把它渲染成候选卡片上的「证据: …」，所以一次长对话会让这个
字符串和卡片一起无限变长。

**修复方式**：抽出 `_merge_evidence()`，只保留最近 `EVIDENCE_MAX_SEGMENTS`（4）段，
被省略时以「…」开头提示证据不完整。保留最新而非最旧，因为新线索更能解释当前排序；
完整的命中类型仍在 `matched_clues` 里（它是去重集合，不随轮数线性增长）。

**防回归**：`test_rerank_evidence_does_not_grow_across_turns` 连跑 12 轮 rerank，
断言 evidence 长度收敛而非线性增长。已证伪：改回无条件拼接即失败。

---

## R22. `make test` / `make install` 跑的是 PATH 上的 conda 解释器，不是项目 venv

**原现象**：Makefile 用裸命令（`pytest tests/`、`pip install -r requirements.txt`），
而本机 PATH 上的 `pytest`/`python3`/`pip` 全部解析到 `~/miniconda3/bin`。后果：

- `make test` 实际跑的是 **miniconda 的 Python 3.14 + fastapi 0.139**，而
  `requirements.txt` 钉的是 fastapi 0.116 → **4 个测试失败**，且失败原因与真实
  运行环境无关（fastapi 0.139 起 `include_router` 不再把路由摊平进 `app.routes`，
  而是包成惰性 `_IncludedRouter`，导致遍历 `app.routes` 找不到任何 `/v1` 路由）；
- `make install` 会把依赖装进 conda base，污染环境。

**修复方式**：Makefile 顶部解析 `$(PY)`，按 `.venv` → `$VIRTUAL_ENV`、
`bin/python` 或 `Scripts/python.exe` 的顺序探测（与 `scripts/run.py` 同款逻辑），
找不到才回落 `python3`。`install` / `test` / `lint` 一律走 `$(PY) -m ...`。
`install` 同时装 `requirements-dev.txt`（否则 `make test` / `make lint` 在全新环境
里根本跑不起来）。新增 `make check` = `lint` + `test`。

顺带修掉那条被 fastapi 版本差异暴露的脆弱测试：`test_app_object_is_constructible`
改用 `app.openapi()["paths"]` 而非遍历 `app.routes`——前者是公开契约，在 0.116 与
0.139 下都返回同样的 14 条路径（正好对上 API.md 声称的 14 个端点）。

**另**：`.venv/bin` 下所有 console script 的 shebang 都指向已不存在的
`.venv-new/bin/python`（venv 被移动或重命名过），因此 `pip`、`accelerate`、
`fastapi` 等脚本全部不可用；`python -m pip` 可正常工作。`$(PY) -m` 的写法顺带
绕开了这个问题。ruff 例外——它是原生二进制，不带 shebang。

---

## R23. 项目没有任何 linter，`make lint` 是空桩

**原现象**：`requirements-dev.txt` 只有 `pytest>=8.0`（且是全项目唯一一行没有精确
钉版的依赖），`make lint` 的内容是 `echo "Add formatter/test checks here when needed."`。
R12 那个坏导入能长期存活，直接原因就是这个。

**修复方式**：装 `ruff==0.16.9`（精确钉版，与运行时依赖同风格），新增 `ruff.toml`，
`make lint` 接上 `ruff check`。

配置的关键在于**只开能指出真实缺陷的规则族**：ruff 0.16 的默认规则集极宽
（A/BLE/C/D/DTZ/S/TRY/PLR/… 几乎全开），直接跑会在本项目产出上千条与刻意设计
冲突的告警，lint 门禁随即沦为噪音而被绕过。实测：不配置时 1572 条，其中
**1500 条是 RUF001/002/003 把中文全角标点（：，（）；）当成「歧义字符」**——
这是中文代码库的正确写法，必须关闭。配置后 72 条真实发现，全部处理完毕。

`ruff.toml` 里**每条 ignore 都写明了原因**，避免后人当成疏漏「顺手修掉」：
`BLE001`/`S110`/`S112`（全链路静默降级是刻意设计，代价已由 R14 的 `/health`
明细补偿）、`DTZ005`/`DTZ006`（时间戳统一 naive UTC，见 `common/time_utils.py`）、
`SIM103`（项目一致用 guard clause 风格，折叠成 `return not (a and b)` 反而更难读）、
`SIM105`、`SIM108`。per-file 豁免同样带理由：`vector_index.py` 的 `F821`/`UP037`
（numpy 是可选依赖，故意在 `__init__` 的 try 块里局部导入，`np` 只出现在字符串注解中，
且文件顶部有 `from __future__ import annotations`，注解根本不会被求值）、
`scripts/sources.py` 的 `SIM112`（**假阳性**：Windows 环境变量字面量就是
`%OneDrive%` 这个混合大小写，改成大写会让 OneDrive 探测直接失效）、
`tests/*` 的 `C408`（fixture 构造器一律 `base = dict(field=value, ...)` 再
`.update()` 后 `**base` 展开，免去键的引号且与数据类构造器签名逐字对应）。

**刻意不接 `ruff format`**：它会重排 27 个文件，纯外观改动应单独成一个提交，
不要和功能性修改混在一起。

**能力边界（写进 ruff.toml 顶部与 requirements-dev.txt）**：ruff/pyflakes 只做
单文件分析，**不跨模块解析导入**，抓不到 R12 那类错误。不要用本配置替代
`tests/test_import_smoke.py`。

顺带修掉的真实发现：`expense/service.py` 里 `class _ExtractedFields` 被插在
**import 块中间**（5 个 E402 的根因，重构时替换动态匿名类留下的）；
4 处 `subprocess.run` 补 `check=False` 把 best-effort 意图写明；
`coverage.py` 的 `zip()` 补 `strict=True`（roots 与 sources 由同一列表推导而来、
长度必须恒等，显式声明可防止将来重构把两者拆开后 zip 静默截断）；
3 处可变类属性补 `ClassVar`；`scripts/run.py`、`sources.py` 有 shebang 却是 644，
补可执行位；`_infer_doc_type` 死代码与孤儿 `suffix` 变量（同一处重构残留）；
`test_personal_search.py` 里的 `_utcnow_naive` 本地副本（「now_iso 四份拷贝」之一，
四份已全部收敛到 `common/time_utils.py`）；4 处 docstring 续行被重构压到顶格
（用 AST 扫描全仓找出，含模块级函数——第一版扫描漏了 `col_offset == 0` 的情况）。


---

## R24. 5 份手搓的 `tmp + os.replace`，漏一处就是静默数据损坏

**原现象**：原子落盘在 5 个地方各写了一遍 —— `expense/storage.py`、
`personal_search/storage.py`（同一文件里两处：JSONL 快照与备注/归档 JSON）、
`personal_search/vector_index.py`、`common/source_discovery.py`。

代价不是行数，而是**新增站点时会漏**。这不是假设：FAISS 的 ids 边车此前就是
直接写目标文件，直到 R16 排查错位问题时才补上原子写。读取端虽然有容错
（坏行跳过、JSON 解析失败回落空状态），但那会把一次崩溃静默降级成「索引丢了
一半」，比直接报错难查得多。

**修复方式**：`common/file_io.py` 新增 `_atomic_tmp()` contextmanager 与
`atomic_write_text()` / `atomic_write_lines()`，5 个站点全部改为调用它。三个要点：

- `.tmp` 必须与目标**同目录**：`os.replace` 是 rename(2) 语义，跨文件系统会抛
  EXDEV，而系统临时目录与 `data/` 往往不在同一个卷上（macOS 上 `/tmp` 还是符号链接）。
- 异常分支要 `unlink` 残留的 `.tmp`：调用方下次成功写入虽然会覆盖同名文件，
  但在那之前它是一个看起来像正常产物的残留，且占着磁盘配额。
- `atomic_write_lines()` 接受**生成器**、边迭代边写。JSONL 快照每行都带
  embedding（当前 1.2 MB / 59 条），先 `"".join(...)` 拼成一整个字符串等于把整份
  索引在内存里再复制一份 —— 这与 R17 减少内存分配的方向相反。

**一处刻意的行为变更**：`newline="\n"` 成为默认，数据文件在所有平台上都用 LF。
此前 JSONL/JSON 走 `open("w")` 默认换行翻译，在 Windows 上会写成 CRLF，而
`.env` 早就显式写了 `newline="\n"`。所有读取端都是换行无关的（逐行 `strip()`
后 `json.loads`、或 `read_text` 后解析），因此不改变任何解析结果；macOS/Linux
上字节完全不变。

**防回归**：`tests/test_common_primitives.py`，含「写入中途失败时目标文件保持
旧内容」「失败后不留 `.tmp`」「`.tmp` 必须与目标同目录」。已证伪：把 `os.replace`
换成 `shutil.copyfile` → 同目录断言失败；去掉 `except` 分支的 `unlink` → 残留
断言失败。

**生产验证**：对真实的 `data/personal_file_store.jsonl`（1199311 字节 / 59 条）
调用 `flush()`，落盘后**逐字节不变**。

顺带：`.gitignore` 补 `/tmp/` 与 `/prefixcache/`（仓库根两个空的运行时目录，
仓内无任何代码引用，应是在仓库根跑 tiny-llm 二进制时落的；Git 不跟踪空目录，
所以它们平时不出现在 `git status` 里，一旦有产物写入就会立刻变成待提交内容）。

---

## R25. 两份 `cosine_similarity`，而阈值是按同一个口径调出来的

**原现象**：`personal_search/relevance.py:113` 与 `expense/service.py:352`
是两份逻辑逐行相同的实现 —— 连「输入长度不一致时按最小维度对齐」这个并不显然
的取舍都一样，只差 `<= 0.0` 与 `<= 0` 的写法。

风险不在于行数，而在于**只会改一份**。两条业务线的相似度分数各自喂给按同一
口径调出来的阈值：personal 侧的 0.7/0.3 新旧分融合权重（R21）、expense 侧的
命中判定。一旦有人只在一边调整了归一化或零向量兜底，两条线的分数会静默分叉，
而阈值不会报错 —— 只会表现为「报销搜索突然变得很松/很紧」。

**修复方式**：抽到 `common/vectors.py`，两处调用点改引用。刻意保持纯 Python
循环而不换 numpy：embedding 是 768 维、单次查询只算几十条候选，numpy 的 import
与数组构造开销远大于计算本身；更重要的是 numpy 在本项目是**可选依赖**
（见 `vector_index` 的局部导入约定），`common/` 不能依赖它。

**防回归**：8 例参数化（同向/正交/反向/非单位向量/空向量/零向量/长度不一致）
+ 尺度不变性，另有 `test_both_business_lines_share_one_cosine_implementation`
断言 `relevance.cosine_similarity is common.vectors.cosine_similarity`，并扫源码
确认两个业务模块里不再出现本地定义。已证伪：在 `relevance.py` 里重新定义一份
→ 该测试失败。

---

## R26. 两个 store 是逐字符几乎相同的两份实现，而修复只落在一边

**原现象**：`PersonalFileStore` 与 `ExpenseStore` 有 **10 个同名同形方法**
（`__init__` / `_load` / `_persist` / `add_or_update` / `flush` / `get` /
`get_by_file_uri` / `list_all` / `get_many` / `delete_by_file_uri`），
`_load` 与 `_persist` 几乎逐字符一致。两者真正的差异只有三点：行类型、主键
字段名（`file_id` vs `material_id`）、以及 expense 多两个业务查询。

最实际的后果不是行数，而是**修一边忘另一边**。R20 那一轮给 personal 侧补的
「同一 id 换了 URI 时清理旧二级索引键」，expense 侧就没有：改过来源文件的报销
材料会在 `_by_file_uri` 里留下悬空键 —— 按旧 URI 查得到，`delete_by_file_uri`
删的却是新 URI，旧键永远留着。

**修复方式**：`common/jsonl_store.py` 的 `JsonlSnapshotStore[RowT]`。子类只声明
三个类属性（`row_type` / `id_attr` / `index_attr`）加业务专属查询
（`list_by_claim`、`all_claim_ids`）；公开的 `get_by_file_uri` /
`delete_by_file_uri` 保留为一行委托，调用方（ingest、service、路由）零改动。

顺带统一掉的三处不一致，都是**更严的那一边胜出**：

1. expense 的 `_load` 不校验空主键，空 id 的行会全部挤在 `""` 键上互相覆盖；
   现在两条线都丢弃这类行。
2. expense 的 `add_or_update` 不清理悬空索引键（即上述原现象）；现在清理。
3. `PersonalFileStore.get_many` 的循环变量叫 `material` —— 从 expense 抄过来
   的残留，正好是「两份拷贝」的物证。

`_persist` 里先 `list(self._items.values())` 拷贝一份再序列化：
`atomic_write_lines` 是边迭代边写的，直接传 `values()` 视图的话，任何并发写入
都会让迭代抛 `RuntimeError: dictionary changed size during iteration`。所有
写方法确实都持锁，但**读方法刻意不持锁**（快照语义下读到的是稍旧但自洽的视图，
加锁会让检索热路径与后台扫描的落盘互相阻塞），所以这个隐患不能靠「调用方守
规矩」来关。拷贝一次的代价相对逐行 `json.dumps` 可以忽略。

**行数说明（不美化）**：两个 store 合计 405 行 → 251 行，加上新增的 170 行基类，
净增 16 行。收益不是更短，而是**持久化语义从此只有一处定义**。

**防回归**：32 例，含 6 线程并发写 300 行后落盘完整、坏行跳过不丢其余记录、
空主键丢弃、悬空键清理、「旧键若已被别的 id 占用则不得误删」、
`persist=False` + `flush()` 只落盘一次、20 次连续写入不留 `.tmp`。
已证伪 7 项全部被抓到：去掉悬空键清理、去掉空主键守卫、坏行改为抛出、
`_persist` 丢掉 `ensure_ascii=False`、失败不清理 `.tmp`、原子写退化为
`copyfile`、业务线重新 fork 一份 cosine。

其中一项**第一次没抓到**，值得记下来：`ensure_ascii` 的断言原本写在直接测
`atomic_write_lines` 的用例里，根本没经过 `_persist`。补了
`test_store_persists_utf8_not_ascii_escapes` 钉住 store 自己的序列化口径 ——
这个退化是静默的（`json.loads` 两种都吃），只会让中文标题从 3 字节 UTF-8
膨胀成 6 字节 `\uXXXX`，1.2 MB 的快照平白涨一截，且再也没法用编辑器直接看。

**生产验证**：真实 59 条记录 `from_dict`/`to_dict` 往返 0 失配，59 条
`file_uri` 二级索引全部可查，`flush()` 后逐字节不变。

---

## R27. `runtime/` 兼容垫片是一份只为让自己继续存在而存在的代码

**原现象**：`src/edge_cloud_agent/runtime/` 是 6 个文件、共 30 行的纯 re-export。
`src/` 与 `scripts/` 对它**零引用**，唯一使用者是 `tests/test_architecture_layers.py`
里验证垫片自身可用的 3 条断言 —— 即一份只被「证明它还活着」的测试养着的代码。

**判断依据**：这次分层重构尚未提交，`edge_cloud_agent.runtime.*` 从来没有作为
已发布路径存在过，因此没有真实的兼容对象。保留它的成本是结构性的：
`docs/PROJECT_STRUCTURE.md` 要为它多解释一行，读代码的人要花时间确认
「engines 和 runtime 哪个才是真的」。

**修复方式**：整目录删除。模块数 56 → 52，`test_import_smoke.py` 的数量哨兵
同步上调（`>= 50` → `>= 52`，否则删完刚好卡在阈值上、余量为零）。

**防回归**：`test_legacy_runtime_shim_is_gone` 同时断言目录不存在**且**
`__import__("edge_cloud_agent.runtime")` 抛 ImportError。前者挡文件被恢复，
后者挡有人通过 `sys.modules` 或 `.pth` 把这个路径重新塞回来。

---

## R28. 层边界此前只有文档、没有任何机器校验

**原现象**：`docs/PROJECT_STRUCTURE.md` 写着依赖方向
`routers → agents / personal_search / expense → llm / engines → common`，
但没有任何东西保证它成立。`test_import_smoke.py` 只能发现「模块坏了」，
发现不了「依赖错了方向」—— 层边界破了不会有任何报错，只会在几个月后表现为
「改 `common/` 要跑全量测试」「某个业务模块删不掉」这类说不清来源的成本。

**修复方式**：`test_architecture_layers.py` 改为用 AST 静态扫全仓 import
（相对导入按 `level` 还原成绝对路径），把**当前实际存在**的 15 条跨包依赖边
固化成 `ALLOWED_EDGES` 白名单：

```text
routers          → agents, analytics, common, engines, expense, personal_search
agents           → llm, personal_search
personal_search  → analytics, common, engines
expense          → common, engines
analytics        → common
llm              → engines
engines          → （无）
common           → （无）
```

选白名单而不是「禁止某几个方向」：新增任何一条边都会红，逼出一次显式决定，
而不是让新依赖悄悄混进来。用 AST 而非运行时 import 追踪，是因为静态扫描连
**未使用**的 import 也算，`# noqa: F401` 绕不过去。

另有 `test_documented_edges_still_exist` 做反向校验：白名单里已经消失的边必须
删掉，否则它会慢慢变成一份过期的架构图 —— 而一份看起来权威其实过期的架构图，
比没有架构图更误导人。

**顺带发现**：当前实际依赖图里有一条文档没写的边 —— `personal_search → analytics`
（`search_flow.py:7` 的 `from ..analytics import record_safe`，首轮检索埋点）。
它不违反方向（业务层之间、且 analytics 只依赖 common），已补进白名单与
`docs/PROJECT_STRUCTURE.md` 的依赖方向说明。

**已证伪 4 项**：让 `common/` 反向依赖 `routers`、白名单里留一条过期边、
恢复 `runtime/` 垫片、让 `personal_search` 直接依赖 `routers` —— 全部被抓到。


---

## R29. expense 的摄取循环是 personal 侧的一份**旧快照**，四项修复只落在一边

**原现象**：`expense/ingest.py` 与 `personal_search/ingest.py` 有 7 处同形构造
（`IngestResult`、`_as_file_uri`、`_discover_files`、`_read_text_*`、
`_sweep_deleted_*`、`run_once` 骨架、`start_watch_loop`）。与 R26 的 store 一样，
问题不是行数，而是 **personal 侧后来修的东西 expense 侧一个都没有**：

| 能力 | personal 侧 | expense 侧（改前） |
|---|---|---|
| 目录遍历 | `os.walk` + 原地剪枝，`followlinks=False` | `rglob("*")`，**无法跳过整棵子树** |
| 变更检测 | size + mtime_ns 快路径，稳态 0 字节内容读取 | 每个文件每轮读头部 1 MB 算 md5 |
| 幽灵清理 | 复用本轮枚举结果，命中即跳过 `stat` | 每条已入库记录一次 `stat` |
| 并发保护 | `_SCAN_LOCK` 串行化 | **无锁** |
| 单文件失败 | 计入 `errors` + 记日志，继续 | 只有 `collect()` 被 try 包住，`stat`/`hash` 阶段的异常掀掉整轮 |

`rglob("*")` 那条尤其讽刺：personal 侧的注释里正是为此改掉它的 ——
「WSL 下跨 9P 扫 `/mnt/c` 时 `node_modules`/`AppData` 级目录会让扫描成本爆炸」。
而 expense 的 watch 循环默认**每 120 秒跑一次**，也就是说这份 O(watch 目录总字节)
的 IO 是持续发生的，不是一次性的。

无锁那条是真并发缺陷：`run_once` 有两个入口（watch 线程与 `/v1/expense/rebuild-index`），
「读 existing → 判定 → 写」这个复合操作不是原子的，store 自己的锁保护不到它。

**修复方式**：抽出两个业务无关的 `common/` 模块，两侧共用同一份实现：

- `common/fs_scan.py`：`iter_files()`（遍历 + 剪枝 + 后缀过滤 + 云占位符门控 + 排序）、
  `fingerprint()` / `is_unchanged()`（size + mtime_ns 变更检测）、
  `reachable_roots()` / `owning_root()` / `is_ghost()`（按根可达的幽灵判定）
- `common/watch_loop.py`：`IngestResult` + `run_watch_loop()`

expense 侧因此一次性拿到全部五项能力，并新增 `ExpenseMaterial.file_mtime_ns`
（`from_dict` 用 `.get()`，存量 JSONL 向后兼容；缺失时按「未知」处理，重算一次
hash 后就地回填）。`/v1/expense/rebuild-index` 同步改为传 `force_rehash=True`，
与 personal 侧的「重建索引」语义对齐。

顺带删掉两处**纯转发**的 `_as_file_uri()` 包装（`return as_file_uri(path)`）——
它们存在的唯一作用是让「URI 生成有几份实现」这个问题需要靠一条测试来盯。

**防回归**：`tests/test_fs_scan.py`（30 例）+ `tests/test_expense_ingest.py`
新增 13 例。共做了 13 项变异：**11 项当场被抓到** —— 去掉 expense 快路径、
`force_rehash` 不再绕过、去掉 mtime 回填、幽灵清理不再复用扫描结果、per-file try
失效、去掉扫描锁、`is_ghost` 去掉 `path.exists()` 回落、`iter_files` 改为遍历后
过滤、不再排序、后缀比较不再小写归一、expense 不写 `file_mtime_ns`；
**1 项是测试漏洞**（下述第 1 条，已补）；**1 项是语义等价的变异**（下述第 3 条，
不是漏洞）。R30 另有 5 项。

其中**两条第一次没抓到**，都是测试本身的问题，记下来：

1. `from_dict` 丢掉 `file_mtime_ns` 时，原有的内存态用例全绿 —— 同一进程内
   第二轮扫描用的是完整的内存对象，只有**重启后的第一轮**会退回慢路径。
   补了 `test_file_mtime_ns_survives_a_jsonl_roundtrip`：从同一路径重新构造
   store 与 service，断言重启后第一轮仍然 0 次 hash。补完再变异即被抓到。
2. 我写的 `test_sweep_still_removes_a_file_that_left_the_scan_scope` 前提是错的：
   把 `m.txt` 改名成 `m.bin` 之后，记录里的旧 URI 指向的路径**确实不存在了**，
   清理它是正确行为。真正要防的是「文件仍在磁盘、只是本轮没被枚举到」
   （后缀被移出白名单、或移进排除目录）。已改写为改 `watch_file_suffixes`，
   并补一条改名的对照组 `test_sweep_removes_record_whose_path_really_is_gone`。

还有一条**证伪脚本自身**的坑值得单独记：第一版把测试名写成裸函数名而不是完整
node id，pytest 收集不到任何用例（`no tests ran in 0.00s`），却因退出码非 0 被
判成「抓到了」—— 14 条**全部是假阳性**，看起来完美。改成完整 node id、并在每次
变异**之前先跑一遍基线确认该用例真的存在且是绿的**之后，才暴露出上面两个真漏洞。

另有一条变异是**语义等价的**，不是测试漏洞：`is_unchanged` 里
`recorded_mtime_ns is None` 的显式守卫被后面的相等比较覆盖（`None == <int>`
恒为 False）。穷举 256 组 `None`/`int` 输入，删掉守卫后行为差异 0 处。保留它是
为了把「未知 ≠ 没变」这条不变量写在代码里，已在 docstring 中注明它当前是冗余的。

---

## R30. 两个 watch 循环都是 `except Exception: pass`，守护线程的持续失败完全静默

**原现象**：两侧 `start_watch_loop` 是同一段十行代码：

```python
while not stop_event.is_set():
    try:
        run_once(service, cfg)
    except Exception:
        pass          # ← 连一行日志都没有
    stop_event.wait(delay)
```

watch 循环是索引保持新鲜的**唯一自动机制**。它一旦持续失败（源目录权限变了、
store 落盘失败、embedding 运行时挂了），外部看到的现象只是「新文件搜不到」——
与 R14「`/health` 恒绿」是同一类可观测性幻觉，而且更隐蔽：连降级路径都没走到，
`/health` 的 `services` 明细里也不会体现，因为服务本身是就绪的。

**修复方式**：`common/watch_loop.run_watch_loop()` 统一循环骨架：

- `tick` 抛异常**永不打断循环**（否则一次偶发错误就让索引永久停更），但每次
  失败都记一条 warning，含异常类型名、`consecutive_failures`、`total_failures`
- 连续失败计数在成功一轮后归零 —— 「偶发一次」与「一直在坏」必须能区分开
- 起止各记一条 info（`loop_started` 带 `interval_seconds`，`loop_stopped` 带
  `ticks` 与 `total_failures`）。此前守护线程是否活着，在日志里完全看不出来
- 只记异常**类型名**不记 message：消息里常带完整文件路径，与项目「日志不落
  用户内容」的口径冲突（同 `ingest.item_failed`）
- `interval <= 0` 夹到 1 秒下限并记 warning。`Event.wait(0)` 立即返回，配上一个
  永不 set 的 `stop_event` 就是 100% CPU 的忙等循环，而 `*_INTERVAL_SECONDS`
  是环境变量，写错成 0 不会有任何报错。正的小数（测试与快速轮询）不受影响

各业务模块的 `start_watch_loop` 保留原签名（`bootstrap.py` 零改动），只负责把
自己的 `run_once` 与命名传进去。**串行锁没有上移**：它要保护的是 store 与向量
索引的写入，而 `run_once` 有多个入口（watch 线程、`/search` 节流刷新、
`/rebuild-index`），只有包在 `run_once` 里才覆盖得到全部。

**顺带修掉一处结构问题**：`personal_search/ingest.py` 里有个私有 `_is_windows()`
（`return os.name == "nt"`），是 `common/path_utils.is_windows_native()` 的
**第三份**平台判定拷贝，而且更弱 —— 后者支持 `FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP`
覆盖，那是非 Windows 机器上测试 Windows 分支的唯一入口。现统一走 `path_utils`，
并由 `test_platform_predicate_is_shared_with_path_utils` 用 AST 钉住
（不能用字符串搜索：`fs_scan` 的 docstring 里为说明这段历史提到了 `os.name`，
第一版就是这么假阳的）。

**防回归**：`tests/test_watch_loop.py`（18 例）。已证伪 5 项：回到
`except: pass`、不夹取非正 interval、异常打断循环、记 message 而非类型名、
连续计数不归零。

其中「记 message 而非类型名」第一次没抓到：我断言的是 `record.getMessage()`，
而那是固定消息串 `watch-loop-tick-failed`，extra 载荷不在里面。但 extra **会**被
上一轮加的 `_ExtraFormatter` 渲染进真实日志输出 —— 也就是说这条泄漏在生产日志里
是真的，只是我的测试看不见。改为直接断言 `record.exception`。

---

## R31. `IngestResult.material_ids` 在 personal 侧装的是 file_id，而它是 wire 契约

**原现象**：两份 `IngestResult` 副本字段完全相同，都叫 `material_ids` ——
personal 侧那份是照抄 expense 留下的名字，装的其实是 `file_id`
（`material_ids.append(item.file_id)`）。这个名字还一路漏进了 HTTP 响应：
`RebuildIndexResponse.material_ids`。

**为什么不能直接改名**：Android 客户端用
`@SerializedName("material_ids")` 钉住了这个 wire 名
（`android/app/src/main/kotlin/com/example/filememoryagent/model/ApiModels.kt:123`）。
改字段名不会报错，只会让客户端**静默拿到空列表** —— Gson 对未知字段是忽略而非报错。

**修复方式**：内部名改为业务中性的 `imported_ids`，wire 名保持 `material_ids`，
映射发生在 routers 层（两个端点各一行 + 注释说明为什么不能改）。
`IngestResult` 收敛为 `common/watch_loop.py` 里的单一定义，两侧共用。

**未做的事**：没有动 wire 字段名。要改就得同时发一版 Android 并处理旧客户端，
收益（API 命名准确）不足以抵这个协调成本。`docs/API.md` 与两处 schema 的注释
已写明这个历史包袱。


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
"used_model": "<repo>/models/google/gemma-4-E2B-it"
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

## O11. 文件索引与部署机器/形态绑定，不可迁移

`file_id = "fm_" + md5(path.as_posix())[:14]`（`personal_search/service.next_file_id`），
`file_uri` 同样由路径派生。同一份文件在不同部署形态下路径字面不同：

| 部署形态 | 同一文件的路径 | file_id / file_uri |
|---|---|---|
| WSL | `/mnt/c/Users/x/a.png` | 与 Windows 原生**不同** |
| Windows 原生 | `C:/Users/x/a.png` | 与 WSL **不同** |
| macOS / Linux | `/Users/x/a.png` | 机器间路径一致时可复用 |

后果：把 `data/personal_file_store.jsonl` + FAISS 索引从一种形态搬到另一种
（如 WSL → Windows 原生）会全量失配重导，且备注/归档（`file_state.json`
按 file_id 关联）全部失联；旧记录会被幽灵清理按根清掉。

**处置**：明确为设计边界——索引与部署机器绑定，换形态请重建索引（备注/归档
如需保留可手工迁移 file_state.json 的键）。**不改** file_id 派生方式：改动会
让现有全部部署的 file_id 变化，代价远大于收益。可选 P2：提供
`scripts/migrate_file_ids.py`（旧 id → 新 id，同时迁移 file_state.json）。

关联：Windows 原生首次从历史 `file://C:/...`（非法形式）索引升级时，
`file:///C:/...` 新 URI 首查不命中旧记录，会按「新增」重读内容 + 重算一次
embedding（file_id 不变，备注不丢，属一次性成本）；POSIX 平台 URI 字面
不变，零成本。

## O12. Windows 长路径（MAX_PATH 260）超限文件计入扫描 errors

微信（`WeChat Files/wxid_.../Msg/Attach/...`）与 OneDrive 的深层嵌套容易超过
260 字符限制，`stat`/`open` 抛 `OSError` → 该文件计入 ingest 的 `errors`
（per-file try/except 兜住，不中断扫描），内容不可检索。

处置：建议开启系统级长路径支持（注册表
`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1`，需重启）。
**不引入** `\\?\` 前缀方案：会改变 file_path/file_uri/file_id 字面值，
破坏索引稳定性（见 O11）。

---

## O13. 指标事件流无界增长，但**不能**简单按时间/条数压缩

`MetricsEventStore` 是追加式 JSONL + 同步维护的内存列表，`_load()` 在启动时读入
全部历史，`append()` 永不淘汰，`list_all()` 每次复制整个列表。没有任何轮转或压缩。

**为什么暂不修**：`docs/METRICS.md` 定义的指标里，`revisit_rate`（14d 窗口）与
`followup_completion_rate`（1h 窗口）是有窗口的，但 `corrections_total`、
`clarify_resolve_rate`、`clarify_exec_rate` 是**全时段累计**。因此任何「丢弃 N 天前
事件」的压缩策略都会**静默改变这三个已文档化指标的口径**——报表数字会变，
而使用者不会收到任何提示。这比无界增长本身更糟。

同时只限制内存而不限制文件也不解决问题：文件仍会无限增长，启动时的全量读取
（以及读取耗时）照旧。要真正约束磁盘就得引入轮转（保留 `.1` 世代），而那同样会
让累计指标失去早期数据。

**当前实际风险很低**：`data/metrics_events.jsonl` 目前是 6.3 KB 量级；即使增长
1000 倍到 6 MB，启动加载也在亚秒级，内存占用可忽略。

**若将来确需约束，可选方案（按侵入性递增）**：
1. 把三个累计指标改为**显式窗口口径**（如「近 90 天」），同步更新 METRICS.md，
   然后才能安全地按窗口压缩事件流；
2. 引入可配置的 `METRICS_MAX_EVENTS`（默认 0 = 不限制，保持现有语义），
   超限时丢弃最旧事件并记 warning 说明累计指标现在只覆盖较短窗口；
3. 事件流轮转 + 归档文件离线聚合（把累计计数沉淀为快照，而非依赖全量事件）。

无论选哪种，都必须**先改文档口径、再改代码**，并在 `/v1/metrics` 响应里暴露
实际覆盖的时间范围，让口径变化对使用者可见。
