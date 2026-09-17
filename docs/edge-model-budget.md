# 端侧模型体积与内存预算（gemma-4-E2B-it）

面向 Android 端侧选型的实测数据与估算。数据来源见每节标注。

> **测量环境**：macOS / Apple M5 Pro（5 性能核 + 10 能效核）/ 48 GiB RAM /
> Python 3.11.15 / transformers 5.17.0 / torch 2.6.0。
> **手机端数字为工程估算，非实测**，依据见 §4。

---

## 1. 权重构成（实测）

从 `model.safetensors` 头部直接读取，未加载张量。总 5.123B 参数 / 9.54 GiB（bf16）。

| 组件 | 参数量 | bf16 | int4 (Q4_K_M) | 占比 | 本项目是否使用 |
|---|---|---|---|---|---|
| `language_model` | 4.647B | 8.66 GiB | 2.61 GB | 90.7% | ✅ |
| `audio_tower` | 0.305B | 0.57 GiB | — | 6.0% | ❌ 无 Processor |
| `vision_tower` | 0.167B | 0.31 GiB | — | 3.3% | ❌ 无 Processor |
| `embed_vision` / `embed_audio` | 0.003B | ~0 | — | <0.1% | ❌ |

**`language_model` 内部的最大单项**：

| 张量组 | 参数量 | bf16 | int4 | 占全模型 |
|---|---|---|---|---|
| **`embed_tokens_per_layer.weight`（PLE）** | **2.349B** | **4.38 GiB** | **1.32 GB** | **45.8%** |
| `layers.N.mlp.{gate,up,down}_proj` × 3 | 1.557B | 2.91 GiB | 0.72 GB | 30.4% |
| `embed_tokens.weight`（与 lm_head tied） | 0.403B | 0.75 GiB | 0.19 GB | 7.9% |
| `layers.N.self_attn.{q,o}_proj` | 0.264B | 0.50 GiB | 0.12 GB | 5.2% |
| `layers.N.self_attn.{k,v}_proj` | 0.034B | 0.06 GiB | 0.02 GB | 0.6% |
| PLE 相关投影 | 0.042B | 0.09 GiB | 0.02 GB | 0.8% |

关键张量形状：

```
model.language_model.embed_tokens.weight            [262144, 1536]   BF16
model.language_model.embed_tokens_per_layer.weight  [262144,  8960]  BF16   <- 8960 = 35 层 × 256
model.language_model.per_layer_model_projection     [  8960,  1536]  BF16
```

---

## 2. 「effective 2B」的实现机制（实测自 config.json）

`enable_moe_block = False`，**不是 MoE**，没有专家稀疏性可省。压缩来自三处：

| 机制 | config 字段 | 效果 |
|---|---|---|
| **PLE**（Per-Layer Embeddings） | `hidden_size_per_layer_input = 256`<br>`vocab_size_per_layer_input = 262144` | 每层用 256 维的低秩输入嵌入，而非 1536 维全量 |
| **KV 共享** | `num_kv_shared_layers = 20` | 35 层中约 20 层复用其他层的 KV，独立 KV 层仅约 43% |
| **混合注意力** | `layer_types`：28 × `sliding_attention`（`sliding_window = 512`）<br>+ 7 × `full_attention`（`global_head_dim = 512`） | 80% 的层 KV 封顶在 512 token，与上下文长度无关 |

其他相关字段：`num_key_value_heads = 1`（**单 KV 头**）、`head_dim = 256`、
`global_head_dim = 512`、`hidden_size = 1536`、`intermediate_size = 6144`、
`use_double_wide_mlp = True`、`tie_word_embeddings = True`、
`max_position_embeddings = 131072`。

> 注：`text_config` 是**逐层异构**的。直接读 `config.text_config.head_dim` 会抛
> `AmbiguousGlobalPerLayerAttributeError`，需经 `per_layer_config[i]` 访问，
> 或在 config 上设 `allow_global_per_layer_attribute_access = True`。

---

## 3. PLE 的访问模式：适合留在磁盘流式读取

**这是决定端侧内存的关键一项。**

PLE 的用法是**按 token id 取行**（gather），不是按层取列切片：

```
第 L 层、token id = T 的贡献 = embed_tokens_per_layer[T, L*256 : (L+1)*256]
```

在当前 `[262144, 8960]` 布局下，一个 token 的全部 PLE 贡献恰好是**一整行连续的
8960 × 2 = 17.9 KB**。因此：

- ✅ 每 token 只需一次 17.9 KB 的连续闪存读取，对 mmap / 流式加载是最理想的模式
- ✅ **不需要重排张量**。若改成 `[35, 262144, 256]`，每层内部连续，但每 token 的读取
  反而变成 35 次跨步访问——对实际访问模式更差
- ⚠️ prefill 阶段读取量 = `n_tokens × 17.9 KB`，8K 上下文约 143 MB（一次性）

**结论：引擎选型时第一个要验证的能力就是「PLE 能否 mmap/流式加载而不整体驻留」。**
这一项直接决定常驻内存是 ~1.5 GB 还是 ~2.9 GB（见 §5）。

---

## 4. decode 速度估算模型（已在本机验证）

decode 阶段每生成一个 token 都要把常驻权重完整读一遍，因此是**内存带宽瓶颈**：

```
tok/s ≈ 有效内存带宽 ÷ 每 token 需读字节数
```

**本机 bf16 实测验证**（预热后交替对照 4 轮）：

| 量 | 值 |
|---|---|
| 每 token 需读权重 | 9.51 GiB = 10.21 GB |
| 实测 decode | 17.08 tok/s |
| 反推隐含带宽 | **174.4 GB/s** |
| 线程饱和实测上限 | 185 GB/s |
| 吻合度 | **94%** → 模型成立 |

线程缩放数据（同样证明是带宽瓶颈而非算力瓶颈）：

| threads | 1 | 2 | 4 | 5 | 8 | 12 | 15 |
|---|---|---|---|---|---|---|---|
| tok/s | 7.1 | 10.5 | 14.6 | 15.8 | 16.5 | 16.4 | 18.1 |
| 隐含带宽 GB/s | 72 | 108 | 149 | 162 | 168 | 167 | 185 |

**4–5 线程即饱和**（torch 默认取 5 = 性能核数，已接近最优），拉到 15 线程只多 15%。

prefill 实测（本机 CPU）：约 **30 tok/s**，64/128/256/512 tokens 分别为
32.8 / 33.3 / 33.9 / 40.1 ms per token，近似线性（28 层 sliding attention 抑制了 O(n²)）。

---

## 5. 手机端估算

每 token 需读 = 常驻权重 1.29 GB（Q4_K_M）+ PLE 行 gather 17.9 KB + KV cache。

| 平台 | 峰值带宽 | 理论 tok/s | **实测量级（×0.65）** |
|---|---|---|---|
| 旗舰 LPDDR5X（8 Gen3 / 天玑9300） | 77 GB/s | 60 | **~39** |
| 中高端 LPDDR5X | 60 GB/s | 46 | **~30** |
| 中端 LPDDR5 | 45 GB/s | 35 | **~23** |
| 上一代 LPDDR5 | 34 GB/s | 26 | **~17** |
| 低端 LPDDR4X | 25 GB/s | 19 | **~13** |

KV cache（int8，已计入 `num_kv_shared_layers = 20`）：

| 上下文 | bf16 全独立 | bf16 共享后 | **int8 共享后** |
|---|---|---|---|
| 2K | 42 MiB | 18 MiB | **9 MiB** |
| 8K | 126 MiB | 54 MiB | **27 MiB** |
| 32K | 462 MiB | 198 MiB | **99 MiB** |
| 128K | 1806 MiB | 774 MiB | **387 MiB** |

**上下文不是内存瓶颈**——8K 只占 27 MiB。这是单 KV 头 + 80% 层 sliding window + KV 共享三者叠加的结果。

### 常驻 RAM 汇总（Q4_K_M，ctx = 8K）

| 方案 | 常驻权重 | KV | 运行时开销 | **RAM 合计** | 磁盘 |
|---|---|---|---|---|---|
| **PLE 流式加载** | 1.29 GB | 0.03 GB | 0.15–0.30 GB | **≈1.5–1.6 GB** | 2.61 GB |
| PLE 也常驻 | 2.61 GB | 0.03 GB | 0.15–0.30 GB | **≈2.8–2.9 GB** | 2.61 GB |

int4 格式选择的影响（纯文本 4.647B 参数）：

| 格式 | bpw | 磁盘 |
|---|---|---|
| 理论下界 | 4.00 | 2.32 GB |
| GPTQ / AWQ | 4.25 | 2.47 GB |
| GGUF Q4_K_M | 4.50 | 2.61 GB |

差距仅 12%，格式选择不是主要矛盾。

---

## 6. 估算的不确定性（必须连同数字一起看）

1. **×0.65 的降额系数是假设值**，非实测。真实 CPU 推理通常落在峰值带宽的 50–80%
2. **带宽模型假设 int4 反量化零开销**。ARM CPU 若缺少 int4 点积指令，反量化的计算成本
   可能反超带宽成为瓶颈，实际速度会明显低于上表——**这是最大的风险项**
3. **若引擎走 NPU/GPU 而非 CPU，上表全部作废**。NPU 通常有原生 int4/int8 通路可能更快，
   但也可能因算子覆盖不全而回退 CPU
4. 常驻 RAM 中「运行时开销 0.15–0.30 GB」为经验估值，未实测
5. 未计入 Android 的 per-process 内存限制与 LMK（low memory killer）行为；
   1.5 GB 常驻的 native 分配是否会被系统回收，需在目标机型上验证

**所有手机端数字都需在目标机型上实测确认后才可用于决策。**

---

## 7. 与其他候选的对比

本机已有的可直接加载的小模型（transformers 5.17.0 实测通过）：

| 模型 | 位置 | bf16 体积 | 本机 CPU 生成速度 | int4 估算 |
|---|---|---|---|---|
| `google/gemma-4-E2B-it` | 项目内 `models/` | 9.54 GiB | 17.1 tok/s | ~2.6 GB |
| `Qwen2.5-0.5B-Instruct` | `~/PycharmProjects/train_backhand/pretrained/` | 0.92 GiB | 约 4 倍于 E2B | ~0.3 GB |
| `Qwen3.5-0.8B` | `~/.cache/modelscope/models/` | 1.75 GiB | 慢（缺 `causal_conv1d` / `flash-linear-attention`，退回参考实现） | ~0.5 GB |

**选型时应一并评估的问题**：E2B 是为通用多模态助手设计的，而本项目场景是
报销票据 + 个人文件检索。即使 E2B 能压到 ~1.5 GB 常驻，其体积仍是
Qwen2.5-0.5B 的约 5 倍。能力差距是否值这个体积，需要按实际任务评测，
不宜只看模型规格。

---

## 8. 待验证清单（引擎选型时逐项确认）

- [ ] **PLE 能否 mmap / 流式加载而不整体驻留**（决定 1.5 GB vs 2.9 GB）
- [ ] 是否支持 `gemma4` / `Gemma4ForConditionalGeneration` 架构
- [ ] 是否支持 `layer_types` 混合注意力（28 sliding window=512 + 7 full，`global_head_dim` 与 `head_dim` 不同）
- [ ] 是否支持 `num_kv_shared_layers` 的 KV 共享
- [ ] 是否支持 `tie_word_embeddings`
- [ ] 是否支持 262144 词表的 tokenizer（`tokenizer.json` 32 MB）
- [ ] int4 反量化在目标 ARM CPU 上是否有 SIMD/点积加速
- [ ] 能否只加载 `language_model` 而跳过 vision/audio 塔
- [ ] 是否有 NPU/GPU 后端，以及 gemma4 算子的覆盖程度
- [ ] 目标机型上的实际常驻内存与 LMK 存活情况
