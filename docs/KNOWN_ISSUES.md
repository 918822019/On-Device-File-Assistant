# 已知问题（Known Issues）

本文件记录 2026-09-16 环境配置过程中排查出、但**本次未修复**的问题。
每项均已在 macOS 15 / Apple Silicon / Python 3.11.15 上实测复现。

---

## 1. Edge LLM 完全无法加载：缺少 `optimum`（未修复，已知悉）

**现象**：服务启动时 `WARNING orchestrator ... Edge 初始化失败: No package metadata was found for optimum`，
`/v1/chat` 的端侧路径不可用。服务本身仍能启动，`/health`、`/v1/embeddings`、expense、personal_search 等
不依赖 LLM 的接口正常。

**根因**：默认 `EDGE_QUANTIZATION=int4-gp32` 命中 `EdgeRuntime._QUANT_DIRECT_MODES`，
于是 `_resolve_model_id()` 解析到 GPTQ 仓库 `TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ`。
该仓库 `config.json` 自带 `quantization_config`，transformers 加载时会走
`AutoHfQuantizer.merge_quantization_configs`，**强制要求 `optimum`**——而 `requirements.txt` 未声明该依赖。

`edge_runtime.py::_load_model_with_quantization()` 的四级回退链因此全灭：

| 步骤 | 分支 | 结果 |
|---|---|---|
| 1 | `_QUANT_DIRECT_MODES` 直接加载 GPTQ ckpt | 缺 optimum → 被 `except` 吞掉，仅 warning |
| 2 | `GPTQConfig` 量化加载 | `GPTQConfig.__post_init__` 查 optimum 版本 → 同样失败，被吞掉 |
| 3 | bitsandbytes int4 兜底 | macOS 上 bitsandbytes 0.42.0 编译时无 GPU 支持 → 失败，被吞掉 |
| 4 | fp16 最终兜底（`edge_runtime.py:144`） | **仍然加载同一个 GPTQ 仓库** → 再撞 optimum，且此分支无 try/except → 异常向上抛出，Edge 初始化失败 |

**关键代码缺陷**：第 4 级兜底本意是「放弃量化、退回普通模型」，但它复用了
`resolved_model`（已指向 GPTQ 仓库）而非 `cfg.model_id`，因此永远无法真正降级到可加载的模型。

**已验证的绕过方式**（本次未采用）：设 `EDGE_QUANTIZATION=none` 或置空，
`_is_quant_mode()` 返回 False，`_resolve_model_id()` 改返回普通仓库
`TinyLlama/TinyLlama-1.1B-Chat-v1.0`，不再需要 `optimum`。
代价：需下载约 2.2GB fp16 权重，内存占用高于 int4。

**注意**：即使装上 `optimum`，本机为 macOS arm64 无 CUDA，GPTQ kernel（gptqmodel/auto-gptq）
大概率仍不可用。量化推理的真实目标是 Android 端（见 `android/app/src/main/cpp/`），
本机开发更适合走非量化路径。

---

## 2. `EmbeddingRuntime._dtype()` 兜底分支返回 float16，float32 不可达

```python
def _dtype(self):
    if (self.cfg.torch_dtype or "").lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if (self.cfg.torch_dtype or "").lower() in {"fp16", "float16"}:
        return torch.float16
    return torch.float16          # ← 应为 torch.float32
```

设 `EDGE_EMBEDDING_TORCH_DTYPE=float32` 时，`cfg.torch_dtype` 确实是 `"float32"`，
但 `_dtype()` 落入兜底分支仍返回 `torch.float16`。实测三种取值的真实结果：

| `EDGE_EMBEDDING_TORCH_DTYPE` | 实际 model dtype | MPS 上结果 |
|---|---|---|
| `bfloat16` / `bf16` | `torch.bfloat16` | ✅ 正常 |
| `float16` / `fp16` | `torch.float16` | ❌ 全 NaN |
| `float32` | `torch.float16`（兜底吞掉） | ❌ 全 NaN |

即当前只有 bf16 一条路能在 Apple Silicon 上得到可用向量。

---

## 3. MPS + float16 下 embeddinggemma-300m 输出 NaN

`device_map="auto"` 在 Apple Silicon 上把模型放到 `mps:0`，而该模型在 MPS + fp16 下
forward 产出 NaN（768 维全部污染），mean-pooling 后 `cos()` 为 `nan`。

实测对照（同一模型、同一批文本）：

| device | dtype | NaN/Inf 个数 |
|---|---|---|
| mps:0 | float16 | 2304（全污染） |
| mps:0 | bfloat16 | 0 ✅ |
| mps:0 | float32 | 0 ✅ |
| cpu | float32 | 0 ✅ |
| cpu | bfloat16 | 0 ✅ |

结论：**只有 fp16 + MPS 这一组合有问题**，device_map="auto" 本身无需改动。
本机已通过 `.env` 设 `EDGE_EMBEDDING_TORCH_DTYPE=bfloat16` 规避（见第 2 项：这是唯一可配置的可用值）。

---

## 4. `local_cache_dir` 双重语义冲突

`EDGE_EMBEDDING_LOCAL_DIR` / `EDGE_LOCAL_DIR` 被两处代码以不同含义使用：

- `_resolve_model_id()`：当作**可直接 `from_pretrained` 的模型目录**
- `_download_if_modelscope()`：当作 **modelscope 的缓存根目录** `cache_dir`

设置该值后，`_load_model()` 会把本地路径当模型 ID 传给 `snapshot_download()`，
下载必然失败 → 被 `except` 捕获 → warning「ModelScope 下载失败，回退直接加载模型」→
再用本地路径 `from_pretrained` 才成功。功能上可用，但每次启动都多一条误导性 warning。

**规避**：填 `*_LOCAL_DIR` 时，把对应 `*_SOURCE` 设为非 `modelscope` 的值（如 `local`），
`_download_if_modelscope()` 会直接原样返回、跳过下载分支。本机 `.env` 已如此配置。

---

## 5. 配置默认值在 import 时绑定，运行期改环境变量无效

`config.py` 的所有 Config 均为 `@dataclass(frozen=True)`，默认值直接写在类体里：

```python
source: str = os.getenv("EDGE_EMBEDDING_SOURCE", "modelscope")
```

`os.getenv()` 在**类定义时（即模块 import 时）求值一次**，此后修改 `os.environ` 不会反映到
`EmbeddingConfig()` 的新实例上。

影响：
- 无法在同一进程内切换配置做多组对比测试
- 任何依赖「先 import 再设环境变量」的调用方式都会静默拿到旧值

排查本项目配置问题时，务必在启动进程**之前**（shell 层或 `scripts/run.sh` 的 `source .env`）设好环境变量。

---

## 6. `torch_dtype=` 在 transformers 4.57 已废弃

升级后每次加载模型都会打印 `` `torch_dtype` is deprecated! Use `dtype` instead! ``。
涉及 `embedding_runtime.py`（`AutoModel.from_pretrained`）与 `edge_runtime.py`
（`AutoModelForCausalLM.from_pretrained`，共 4 处）。

4.57.6 中该 kwarg 仍然生效，但升级到 transformers 5.x 时会失效，届时需改为 `dtype=`。

---

## 7. `modules.json` 引用的 `4_Normalize` 目录不存在

ModelScope 下载的 `google/embeddinggemma-300m` 快照中，`modules.json` 声明了 5 个模块
（Transformer / Pooling / 2_Dense / 3_Dense / Normalize），但目录里只有
`1_Pooling`、`2_Dense`、`3_Dense`，缺 `4_Normalize`。

**当前无影响**：本项目用 `AutoModel` 直接取 `last_hidden_state` 后自行做 mean-pooling + L2 归一化，
并不经过 sentence-transformers，因此不读 `modules.json`。
但若日后改用 `SentenceTransformer` 加载，会因此报错；且 `2_Dense`/`3_Dense` 投影层
当前也未被使用，输出的是 768 维 hidden state 而非模型设计的投影向量。

---

## 8. 服务启动时同步下载大模型，阻塞 `/health`

启动流程会同步加载端侧模型，其中 embedding 模型约 1.13GB。首次启动时
`/health` 在下载完成前不可达（实测超过 30s 无响应），健康检查/编排系统可能误判为启动失败。

模型已缓存到本地后不再是问题（本机加载仅 1.7s），但首次部署或缓存被清空后会复现。
建议后续改为后台异步预热 + `/health` 立即返回就绪状态。
