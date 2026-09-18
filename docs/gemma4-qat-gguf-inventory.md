# google/gemma-4-E2B-it-qat-q4_0-gguf 的完整清单与转换对照

来源：ModelScope `google/gemma-4-E2B-it-qat-q4_0-gguf`，文件
`gemma-4-E2B_q4_0-it.gguf`（3194.3 MiB）。
**本文档的所有数字都是从 GGUF 头部实测解析出来的**，不是从 README 抄的
（该仓库 README 是 0 字节）。解析方式：GGUF 的元数据与张量表全在文件头部，
用 HTTP Range 请求取前 16 MB 即可解析出全部 541 个张量的名字/维度/dtype/偏移，
**不需要下载 3.2 GB**。

这是 #28 的依据：PTQ 4-bit 的实测天花板是 64.42%（f16 基线 97.57%，
见 `edge-model-budget.md` §5.5.2），必须换算法档次；QAT 的质量上限高于任何
PTQ，且是官方产物、无需校准语料与 Hessian。

---

## 1. 文件与架构

```
GGUF version 3     张量数 541     元数据键 49
general.architecture = gemma4
general.size_label   = 4.6B        general.license = apache-2.0
general.file_type    = 2           general.quantization_version = 2
general.base_model.0.repo_url = https://huggingface.co/google/gemma-4-E2B-it
```

## 2. 张量清单（按类归并，×N 表示层数）

⚠️ **GGUF 的 dims 是 [in, out]，与 HF 的 [out, in] 相反。**
例如 `blk.0.attn_q.weight` 在 GGUF 里是 `[1536, 2048]`，
而 HF 的 `self_attn.q_proj.weight` 是 `[2048, 1536]`。转换时必须转置。

| GGUF dtype | ×N | dims (GGUF 顺序) | 名字 |
|---|---|---|---|
| F32 (0) | 1 | [1536] | `output_norm.weight` |
| F16 (1) | 1 | [1536, 8960] | `per_layer_model_proj.weight` |
| F32 (0) | 1 | [256] | `per_layer_proj_norm.weight` |
| **Q6_K (14)** | 1 | [8960, 262144] | **`per_layer_token_embd.weight`** ← PLE 表 |
| F32 (0) | 1 | [256] | `rope_freqs.weight` |
| **Q6_K (14)** | 1 | [1536, 262144] | **`token_embd.weight`** ← embed（tied lm_head） |
| Q4_0 (2) | 35 | [1536, 2048] | `blk.N.attn_q.weight` |
| Q4_0 (2) | 15 | [1536, 256] | `blk.N.attn_k.weight` ← 只有非共享层有 |
| Q4_0 (2) | 15 | [1536, 256] | `blk.N.attn_v.weight` ← 只有非共享层有 |
| Q4_0 (2) | 35 | [2048, 1536] | `blk.N.attn_output.weight` |
| F32 (0) | 35 | [256] | `blk.N.attn_q_norm.weight` |
| F32 (0) | 15 | [256] | `blk.N.attn_k_norm.weight` ← 只有非共享层有 |
| F32 (0) | 35 | [1536] | `blk.N.attn_norm.weight` |
| F32 (0) | 35 | [1536] | `blk.N.post_attention_norm.weight` |
| F32 (0) | 35 | [1536] | `blk.N.ffn_norm.weight` |
| F32 (0) | 35 | [1536] | `blk.N.post_ffw_norm.weight` |
| F32 (0) | 35 | [1536] | `blk.N.post_norm.weight` |
| Q4_0 (2) | 35 | [1536, 6144 或 12288] | `blk.N.ffn_gate.weight` |
| Q4_0 (2) | 35 | [1536, 6144 或 12288] | `blk.N.ffn_up.weight` |
| Q4_0 (2) | 35 | [6144 或 12288, 1536] | `blk.N.ffn_down.weight` |
| Q4_0 (2) | 35 | [1536, 256] | `blk.N.inp_gate.weight` ← PLE 子块 gate |
| Q4_0 (2) | 35 | [256, 1536] | `blk.N.proj.weight` ← PLE 子块 projection |
| F32 (0) | 35 | [1] | `blk.N.layer_output_scale.weight` ← `layer_scalar` |

**KV 共享层（15..34，共 20 层）没有 `attn_k` / `attn_v` / `attn_k_norm`**，
所以这三类各只有 15 份 —— 与我们 `is_dead_kv_weight` 过滤死权重后的结果
（540 张量 + 1 个 V4 扩展 = 541）**完全一致**。这是两条独立路径对同一
架构事实的交叉验证。

### 量化方案（Google 官方的选择）

| 类别 | dtype | bpw | 说明 |
|---|---|---|---|
| matmul 权重（含 PLE 子块的两个投影） | **Q4_0** | 4.5 | 32 值/块，1 个 fp16 scale，**对称**（zero 恒为 8） |
| `token_embd` 与 `per_layer_token_embd`（PLE 表） | **Q6_K** | 6.5625 | 两张查表都保到 6-bit |
| `per_layer_model_proj` | F16 | 16 | |
| 所有 norm、`layer_scalar`、`rope_freqs` | F32 | 32 | |

体积核算（与实测 3194.3 MiB 吻合）：
PLE 2.349B × 6.5625/8 = 1.93 GB，embed 0.4027B × 6.5625/8 = 0.33 GB，
matmul ~1.85B × 4.5/8 = 1.04 GB，合计 ≈ 3.30 GB = 3.07 GiB。

**与我们独立得出的方案对比**：我们选「embed f16 + PLE 表 i4 + matmul i4」，
官方选「embed Q6_K + PLE 表 Q6_K + matmul Q4_0」。
embed 保高精度两边一致；但**官方给 PLE 表 6-bit，而我们的消融显示 PLE 表
量化只值 0.56 个一致率点**（保 f16 的代价是磁盘 +3.2 GiB）。
两者不矛盾：官方按体积权衡选了 6-bit 折中，我们按敏感度实测发现它不敏感。
转换时应保留官方的 Q6_K（磁盘 1.93 GB，介于我们 f16 的 4.48 GB 与
i4 的 1.26 GB 之间），因为 QAT 的 scale 是训练出来的，不该再由我们二次量化。

⚠️ 注意官方把 **PLE 子块的两个投影（`inp_gate` / `proj`）也量化成了 Q4_0**，
而 cyankiwi 的 compressed-tensors 版把它们全部 ignore 掉。两个社区/官方配方
在这一点上**相反**。我们的实测是「保 f16 值 +3.37 个一致率点」，
所以转换后可以 A/B：直接用官方 Q4_0 vs 把它们转成 f16。

## 3. 名字映射（GGUF → .tqwen / HF）

| GGUF | .tqwen |
|---|---|
| `token_embd.weight` | `model.embed_tokens.weight` |
| `per_layer_token_embd.weight` | `model.embed_tokens_per_layer.weight` |
| `per_layer_model_proj.weight` | `model.per_layer_model_projection.weight` |
| `per_layer_proj_norm.weight` | `model.per_layer_projection_norm.weight` |
| `output_norm.weight` | `model.norm.weight` |
| `rope_freqs.weight` | **无对应** —— 我们在 forward 里现算 inv_freq，不落盘 |
| `blk.N.attn_norm.weight` | `model.layers.N.input_layernorm.weight` |
| `blk.N.post_attention_norm.weight` | `model.layers.N.post_attention_layernorm.weight` |
| `blk.N.ffn_norm.weight` | `model.layers.N.pre_feedforward_layernorm.weight` |
| `blk.N.post_ffw_norm.weight` | `model.layers.N.post_feedforward_layernorm.weight` |
| `blk.N.post_norm.weight` | `model.layers.N.post_per_layer_input_norm.weight` |
| `blk.N.attn_{q,k,v}.weight` | `model.layers.N.self_attn.{q,k,v}_proj.weight` |
| `blk.N.attn_output.weight` | `model.layers.N.self_attn.o_proj.weight` |
| `blk.N.attn_{q,k}_norm.weight` | `model.layers.N.self_attn.{q,k}_norm.weight` |
| `blk.N.ffn_{gate,up,down}.weight` | `model.layers.N.mlp.{gate,up,down}_proj.weight` |
| `blk.N.inp_gate.weight` | `model.layers.N.per_layer_input_gate.weight` |
| `blk.N.proj.weight` | `model.layers.N.per_layer_projection.weight` |
| `blk.N.layer_output_scale.weight` | `model.layers.N.layer_scalar` |

⚠️ `blk.N.post_norm.weight` 映射到 **`post_per_layer_input_norm`**，
不是「最终的 norm」。最终 norm 是全局的 `output_norm.weight`。
五个 norm 的名字在两套体系里错位得很厉害，逐个核对过才敢写这张表。

⚠️ `rope_freqs.weight` 是 [256] F32 —— 预计算的 inv_freq 表。
我们的 forward 用 `proportional_rope_ref` 现算，不消费它。
但它的存在是一个交叉验证点：full 层 `rope.dimension_count = 512`、
sliding 层 `dimension_count_swa = 256`，而 256 = 512/2，
与「inv_freq 长度 = head_dim/2」一致。

## 4. 元数据 → 我们的 config / V4Ext

| GGUF 键 | 值 | 对应 |
|---|---|---|
| `gemma4.block_count` | 35 | `n_layers` |
| `gemma4.embedding_length` | 1536 | `hidden_size` |
| `gemma4.embedding_length_per_layer_input` | 256 | `ple_dim_per_layer` |
| `gemma4.attention.head_count` | 8 | `n_heads` |
| `gemma4.attention.head_count_kv` | 1 | `n_kv_heads` |
| `gemma4.attention.key_length` | 512 | `head_dim_full` |
| `gemma4.attention.key_length_swa` | 256 | `head_dim`（sliding） |
| `gemma4.attention.value_length` / `_swa` | 512 / 256 | 与 key_length 一致（`attention_k_eq_v=False` 但维度相同） |
| `gemma4.attention.shared_kv_layers` | 20 | `n_kv_shared_layers` |
| `gemma4.attention.sliding_window` | 512 | `sliding_window` |
| `gemma4.attention.sliding_window_pattern` | 35 个布尔 | **等价于 `layer_types`**：True=sliding、False=full。实测 False 出现在下标 4,9,14,19,24,29,34 —— 与 config.json 的 `layer_types` **完全一致** |
| `gemma4.feed_forward_length` | 35 个整数 | **逐层 intermediate_size**：前 15 个 6144、后 20 个 12288。与 `use_double_wide_mlp` + KV 共享同判据一致 |
| `gemma4.final_logit_softcapping` | 30.0 | `final_logit_softcapping` |
| `gemma4.rope.freq_base` | 1000000.0 | `rope_theta_full` |
| `gemma4.rope.freq_base_swa` | 10000.0 | `rope_theta`（sliding） |
| `gemma4.rope.dimension_count` / `_swa` | 512 / 256 | 旋转维度（full 层 `partial_rotary_factor=0.25` ⇒ 512×0.25=128 对 ⇒ 256 维？需与 config.json 复核） |
| `gemma4.attention.layer_norm_rms_epsilon` | 1e-6 | `rms_norm_eps` |
| `gemma4.context_length` | 131072 | `max_position_embeddings` |
| `tokenizer.ggml.eos_token_id` | 1 | ⚠️ **与 config.json 的 `[1, 106]` 不同**：GGUF 只有单值 1。我们的 exporter 取列表末位 106（`<turn|>` 回合结束符），对话生成必须停在这里 —— 用 GGUF 的 1 会停不下来 |

**`rope.dimension_count` 这一项需要复核**：config.json 的
`rope_parameters.full_attention.partial_rotary_factor = 0.25`，
head_dim=512 ⇒ 旋转 128 维 ⇒ `rope_angles = 64`；
而 GGUF 的 `dimension_count = 512`。两者语义不同
（一个是 head_dim，一个可能是「旋转嵌入的总维度」= 2×rope_angles×2？），
转换器不要直接把它当 rotary_dim 用，**一律以 config.json 的
partial_rotary_factor 为准**，并用 `rope_freqs.weight` 的实际长度（256）交叉核对。

## 5. 转换器需要实现的两种反量化

### Q4_0（matmul 权重）
```
每块 32 个值 = 2 字节 fp16 scale(d) + 16 字节 nibble
dequant: w[i] = d * (q[i] - 8)          q ∈ [0,15]，**对称**（zero 恒 8）
nibble 序：低 nibble 在前（qs[i] = q[2i] | (q[2i+1] << 4)）
```
⇒ **与我们的 `Dtype::kI4` 布局直接对应**：group_size=32、zero=8.0、
低 nibble 在前（`dequant_i4_row` 的 `(i%2==0) ? p&0xF : p>>4` 完全一致）。
转换只是把每块的 2 字节 scale 扩成我们的 4 字节头 `[scale_fp16 | zero_fp16=8.0]`。

⇒ **顺带解锁 `sdot5_mt`**：该 kernel 要求对称量化（zero 恒 8），
Q4_0 天然满足，而它比 `sdot4_mt` 少算 zero 修正项与激活前缀和。

### Q6_K（两张查表）
```
超块 256 值 = 16 个子块 × 16 值
每超块：ql[128B] + qh[64B]（6-bit 量化值拆成低4位与高2位）
        + scales[16B]（16 个子块的 6-bit scale，打包）+ fp16 d
```
这是 k-quant 里较复杂的一种，需要严格按 llama.cpp 的
`block_q6_K` 定义实现，**不能凭记忆写**（6-bit scale 的打包方式
极易错，而错了是静默乱码）。

⇒ 转换目标是 **f16**（不是我们的 i4）：embed 本来就走 f16 路径，
PLE 表本来留盘、f16 行读已实现（`ple_dtype_ == kF16` 分支）。
这样只需实现 Q6_K → f16 一个方向，不需要 Q6_K → i4 的再量化
（再量化会毁掉 QAT 训练出来的 scale，是净损失）。

## 5.1 已验证的事实（下载后实测，不是推测）

下载完整文件（3,349,516,256 B）并装了 `gguf` 0.19.0 之后实测确认：

**① `gguf.quants.dequantize` 提供 Q4_0 与 Q6_K 的 numpy 参考实现**，
故 §5 里「Q6_K 不能凭记忆写」的风险**已消除**——直接用它，不必自己实现
`block_q6_K`。块大小实测 `GGML_QUANT_SIZES`：Q4_0 = (32, 18)、
**Q6_K = (256, 210)**，210 = ql[128] + qh[64] + scales[16] + d[2]，
与 §5 推的布局一致。
（注意 `dequantize` 收 `np.ndarray` 而不是 `bytes`，要先 `np.frombuffer(..., np.uint8)`。）

**② 张量表结束于 15,815,534，数据区起点 15,815,552**（64 B 对齐）。
即 §1 的「Range 取前 16 MB 就够」是**刚好够**——词表 262144 个 token 使
kv 区长达 15.8 MB。取 4 MB 会在解析 `tokenizer.ggml.tokens` 时越界报错。

**③ 布局：数据不需要转置，只需把 dims 标签反序。**
GGUF 的 `ne[0]` 是最快轴，而 `ne = [in, out]`，所以内存里已经是
`[out, in]` 行主序 —— 恰好就是我们的 .tqwen 布局。实测（`blk.0.attn_k.weight`
Q4_0，dims=[1536,256]，对照原始 HF 的 `k_proj.weight` [256,1536]）：

    A 不转置 reshape[dims[1], dims[0]]  相对 L2 =  34.14%
    B 转置                              相对 L2 = 120.32%

B 的 120% 已接近「两个无关同范数向量」的 √2 ≈ 141%，即完全不相关；
A 的 34% 明显有结构 ⇒ **A 正确**。
⚠️ 这是个高价值陷阱：「dims 是反的」很容易被误解成「数据要转置」，
一旦转置就静默毁掉全部权重（形状仍然合法，不报错）。

**④ QAT 把权重训漂了 —— 这改变了验收标准。**
用**未量化的 F32 张量**做控制组（不含任何量化误差），对照原始 HF checkpoint：

| GGUF 张量 | 对应 HF | 相对 L2 差异 |
|---|---|---|
| `blk.0.attn_k_norm.weight` | `layers.0.self_attn.k_norm.weight` | **0.000%（逐位相同）** |
| `blk.0.attn_norm.weight` | `layers.0.input_layernorm.weight` | 0.525% |
| `blk.7.ffn_norm.weight` | `layers.7.pre_feedforward_layernorm.weight` | 0.533% |
| `output_norm.weight` | `model.norm.weight` | 0.792% |
| `blk.0.layer_output_scale.weight` | `layers.0.layer_scalar` | **17.123%**（0.02087 vs 0.01782） |

`attn_k_norm` 逐位相同**证明我的 GGUF 解析与布局是对的**（否则不可能精确匹配）；
而其余张量的差异就是 **QAT 训练造成的权重漂移**，`layer_scalar` 漂了 17%。
上面 A 方案那 34% 的相对差 = QAT 漂移 + Q4_0 量化误差，两者叠加，
**不是转换错误**。

⇒ **`tools/eval_i4_quality.py` 的判据（argmax 一致率 vs 原始 HF bf16）
对 QAT 模型无意义。** 那个判据只对 PTQ 成立：PTQ 的目标是逼近原模型，
所以「与原模型的一致率」是对的量；而 QAT 是**另一个模型**，
它不追求逼近原模型，只追求自己在 4-bit 下好用。
拿 QAT 去比原模型，量到的是 QAT 训练的漂移量，与转换正确性无关。

**QAT 的验收标准应改为两条：**
1. **转换保真度**：转换后的 .tqwen 反量化值必须与 GGUF 的
   `quants.dequantize` 结果逐位一致（Q4_0 是无损转码，见 §5；
   Q6_K→f16 是精度**升格**，f16 的 11 位尾数多于 Q6_K 的 ~6 位，
   故也应无损）。这条能验证转换器，且与原模型无关。
2. **任务质量**：直接用 QAT 模型生成、看输出是否连贯正确；
   或与 llama.cpp 跑同一个 GGUF 的输出对照（同权重、不同引擎，
   差异应只来自算子实现）。
   可选的定量口径：把 QAT 的 F32 张量（norm/layer_scalar）与
   Q4_0/Q6_K 反量化结果一起灌回 HF 模型结构，得到「QAT 的 fp32 等价模型」，
   再以它为参考算 argmax 一致率 —— 这才是与 PTQ 可比的口径。

## 6. 转换器的工作量与顺序

1. **GGUF 读取器**：头部（magic/version/张量数/kv 数）→ kv 对（含 ARR/STR
   嵌套）→ 张量表（name/ndim/dims/dtype/offset）→ 数据区（按 offset，
   需加上张量表结束后的对齐偏移）。**支持 Range 请求只取头部**用于清单核对。
2. **Q4_0 → 我们的 i4**：扩头即可，最简单，先做，可单独验证。
3. **Q6_K → f16**：按 `block_q6_K` 实现，是主要工作量与主要风险。
   验证方式：先与 llama.cpp 对同一张量反量化结果逐位对照
   （不要只靠"看起来对"）。
4. **名字映射 + 转置**（dims 反序）。
5. **V4Ext 合成**：从 GGUF 元数据生成，`ple_row_bytes` 按 PLE 的
   **目标 dtype**（f16 ⇒ 8960×2 = 17920）算，不能用文件主 dtype。
6. **EOS 用 106 而不是 GGUF 的 1**（见 §4 的警告）。
7. 用 `tools/eval_i4_quality.py` 验收：argmax 一致率目标 ≥95%
   （f16 基线 97.57%）。**不要用 PPL**（softcapping 下无效，见 §5.6）。

## 7. 未选的替代方案

`unsloth/gemma-4-E2B-it-qat-GGUF` 有 `UD-Q4_K_XL`（2499 MiB，
unsloth 的 imatrix 动态量化，质量通常高于 q4_0）与 `UD-Q2_K_XL`（2085 MiB）。
但 **Q4_K 是超块格式**（256 值/块、8 个子块、6-bit scale/min 分离打包），
读取器工作量比 Q4_0 大一个数量级，且 XL 变体是逐层混合位宽，
需要先解析每个张量的实际 dtype。留作 Q6_K 路径走通之后的可选升级。

`unsloth/gemma-4-E2B-it-qat-mobile-GGUF` 只含 `UD-Q2_K_XL`（2085 MiB），
是 2-bit，体积最小但质量风险最高，且同样是 Q2_K 超块格式。
它还带 MTP（multi-token prediction）权重 —— 那是一种投机解码 drafter，
与本项目「gemma4 不做投机解码」的既定决策不符，不需要。
