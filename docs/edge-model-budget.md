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

> **精算修正**：上表按「独立层比例 (35−20)/35 = 0.43」粗估。在
> [gemma4-reference-spec.md](gemma4-reference-spec.md) §3.6 查明 KV 共享的确切结构后
> （自有 KV 仅层 0–14，共享方只有 layer 13/14，且 layer 13 虽是 sliding 层却需一份额外的
> **全长**副本），精算值为 2K=10 MiB / 8K=**31 MiB** / 32K=115 MiB（int8）。
> 与粗估差约 15%，**结论不变**：上下文远不是瓶颈，权重才是。

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

## 8. 自研引擎（tiny-llm）已验证项

这张清单原本是给「选第三方引擎」用的。现在走的是自研路线
（`third_party/tiny-llm`，分支 `feat/gemma4-ple`），逐项状态如下。
**判据写在每一项后面** —— 「已验证」必须说明是在什么条件下验证的，
否则和没验证一样（本文档 §6 的教训：估算值必须与实测值分开标注）。

| 项 | 状态 | 判据 / 缺口 |
|---|---|---|
| `gemma4` 架构支持 | ✅ | `ModelType::kGemma4`；假模型导出→加载→前向→greedy decode 全通 |
| `layer_types` 混合注意力 | ✅ | 逐层 `head_dim_of(i)`；两套 RoPE（sliding=default/1e4、full=**proportional**/1e6+0.25）。C++ ↔ HF 逐位置对齐，29 位置最差 rel 1.8e-5、argmax 逐步一致 |
| sliding window | ✅ | 窗口起点处误差**无跳变**（pos 7=9.8e-7 → 8=3.1e-6 → 9=2.3e-6）。⚠️ 只在 window=8 的假模型上验过；真模型 window=512 未测 |
| `num_kv_shared_layers` KV 共享 | ✅ | 零拷贝（共享层与提供方同一物理槽位）；提供方 = 每种 layer_type 在边界前的最后一层。**同上，只在 6 层假模型上验过** |
| `tie_word_embeddings` | ✅ | `lm_head_ = embed_`；并修掉了假模型生成器给 tied 两键填独立随机数导致「参考实现自己是坏的」的坑（见设计文档 §7.1） |
| **PLE 流式加载不整体驻留** | ✅ 机制 | `--ple-ssd`：整表留盘、每 token pread 一行。常驻 vs 留盘 **logits 逐字节 IDENTICAL**，且留盘路径也与 HF 对齐。⚠️ **省内存的绝对量只在 0.19 MB 的假表上验过**；真表 4.7 GB 的收益是推算值 |
| 只加载 `language_model`，跳过 vision/audio | ✅ | 导出器 `SKIP_PREFIXES` 在**读取阶段**就跳过，多模态塔连内存都不进 |
| int4 反量化的 ARM SIMD 加速 | ⬜ 未做 | gemma4 的 i4 导出还没写（任务 #23）。Qwen 侧已有 `matvec_i4_sdot*` NEON 变体可复用，`kMaxInDim=16384` > 加倍宽 intermediate 12288 ✅ |
| Android arm64 交叉编译 | ✅ | `scripts/build_android.sh` 产出 ARM aarch64 ELF（interpreter `/system/bin/linker64`），NEON 变体与新 kernel 均已编入 |
| **目标机型实测** | ❌ 阻塞 | `adb devices` 为空，且无 AVD / 系统镜像，emulator 起不来。需要接一台真机（任务 #24） |
| 262144 词表 tokenizer | ❌ 未接 | C++ 侧只吃 token id 序列，detokenize 仍在 Python（这是一个有意的既定决策，不是缺口） |
| NPU/GPU 后端 | ❌ 未做 | 当前 gemma4 只有 CPU 路径。Vulkan 后端存在但未接 gemma4 算子 |

### 8.1 真模型尚未导出（当前最大缺口）

上面所有 ✅ 都是在**缩小版假模型**上得到的：6 层 / hidden 64 /
head_dim 32+64 / vocab 512 / ple_dim 16。真模型是 35 层 / hidden 1536 /
head_dim 256+512 / vocab 262144 / ple_dim 256。

导出器目前把全部张量以 fp32 驻留，真模型峰值约 **20.5 GB**（仅 PLE 表
就 9.4 GB），而本机空闲约 18.7 GB —— 强行跑会换页，且波及同机其他服务，
故**没有**导出。需要先把 `write_tqwen` 改成两遍流式（第一遍只读
safetensors 元数据算 offset，第二遍逐张量载入→转换→写出→释放），
峰值可降到单个张量。这是任务 #21，也是 #22（真模型导出 + 常驻内存实测）
和 #23（i4）的共同前置项。

**已经用真 config 算过（不是跑过）的部分**：kernel 的硬编码上限全部通过
（导出期新增校验，见 `validate_runtime_limits`）——

| 常量 | 上限 | 真模型实际值 |
|---|---|---|
| `kMaxHalf`（rope_ref / rope_neon） | 256 | sliding head_dim 256 → half **128** ✅ |
| `kMaxAngles`（proportional_rope_ref） | 512 | 0.25×512//2 = **64** ✅ |
| `kMaxTensorName` | 64 | 最长 **48**（`model.layers.34.post_per_layer_input_norm.weight`）✅ |
| `kMaxInDim`（matvec_i4_sdot*） | 16384 | 加倍宽 intermediate **12288** ✅ |

即：真模型的**形状**不会撞上任何硬编码上限；未验证的是真形状下的
**数值**与**内存/速度**。
