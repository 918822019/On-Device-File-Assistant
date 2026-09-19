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

⚠️ **GGUF 的 dims 标签是 [in, out]，与 HF 的 [out, in] 相反 —— 但数据
不需要转置，只需把 dims 标签反序。** GGUF 的 `ne[0]` 是最快轴，
`ne=[in,out]` ⇒ 内存里**已经是 [out,in] 行主序**，恰好就是我们的 .tqwen 布局。
例如 `blk.0.attn_q.weight` 的 dims 是 `[1536, 2048]`，而它的内存布局与
HF `self_attn.q_proj.weight` `[2048, 1536]` 逐字节相同。
实测对照见 §5.1 ③（转置方案相对 L2 = 120% ≈ 两个无关向量的 √2）。
⚠️ 「dims 是反的」极易被误读成「数据要转置」，一旦转置会静默毁掉全部权重。

| GGUF dtype | ×N | dims (GGUF 顺序) | 名字 |
|---|---|---|---|
| F32 (0) | 1 | [1536] | `output_norm.weight` |
| F16 (1) | 1 | [1536, 8960] | `per_layer_model_proj.weight` |
| F32 (0) | 1 | [256] | `per_layer_proj_norm.weight` |
| **Q6_K (14)** | 1 | [8960, 262144] | **`per_layer_token_embd.weight`** ← PLE 表 |
| F32 (0) | 1 | [256] | `rope_freqs.weight` |
| **Q6_K (14)** | 1 | [1536, 262144] | **`token_embd.weight`** ← embed（tied lm_head） |
| Q4_0 (2) | **28 + 7** | [1536, 2048] / **[1536, 4096]** | `blk.N.attn_q.weight` ← sliding 28 层 / **full 7 层** |
| Q4_0 (2) | **12 + 3** | [1536, 256] / **[1536, 512]** | `blk.N.attn_k.weight` ← 只有非共享层（0..14）有 |
| Q4_0 (2) | **12 + 3** | [1536, 256] / **[1536, 512]** | `blk.N.attn_v.weight` ← 同上 |
| Q4_0 (2) | **28 + 7** | [2048, 1536] / **[4096, 1536]** | `blk.N.attn_output.weight` |
| F32 (0) | **28 + 7** | [256] / **[512]** | `blk.N.attn_q_norm.weight` |
| F32 (0) | **12 + 3** | [256] / **[512]** | `blk.N.attn_k_norm.weight` ← 只有非共享层有 |
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

⚠️ **本表最初把逐层变化压平了**（写成 `attn_q` ×35 全 `[1536,2048]`、
`attn_k_norm` ×15 全 `[256]`），照它写转换器会有 **10 个张量形状错**。
上表是重新解析全部 541 个张量后的实测分布。

**head_dim 是逐层变的**：full 层（head_dim 512）= **{4, 9, 14, 19, 24, 29, 34}
共 7 层**，其余 28 层是 sliding（head_dim 256）。这 7 个下标与
`gemma4.attention.sliding_window_pattern` 里 False 的位置**完全一致**（§4）。
ffn 则是前 15 层窄（6144）、后 20 层宽（12288）。

**KV 共享层（15..34，共 20 层）没有 `attn_k` / `attn_v` / `attn_k_norm`**，
所以这三类各只有 15 份 —— 与我们 `is_dead_kv_weight` 过滤死权重后的结果
（540 张量 + 1 个 V4 扩展 = 541）**完全一致**。这是两条独立路径对同一
架构事实的交叉验证。

✅ **转换器已把这条交叉验证做成了强制门禁**：把 GGUF 张量表经名字映射后，
与 `read_plan_from_dir`（HF config.json + safetensors 元数据）产出的计划
逐张量比对，集合差与 shape 差都必须为 0，否则拒绝转换。
实测：**540 个张量，集合差 0、shape 差 0**，唯一有意跳过的是
`rope_freqs.weight`（forward 现算 inv_freq，不落盘）。

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
⚠️ nibble 配对**与我们不同**，见下方「必须重排」
```

⚠️⚠️ **nibble 必须重排。本节最初写的是「低 nibble 在前
（qs[i] = q[2i] | (q[2i+1] << 4)），与我们的 kI4 布局直接对应，
转换只是把 2 字节 scale 扩成 4 字节头」—— 那是错的，已按实测改正。**

llama.cpp 的 `dequantize_row_q4_0`：

```c
for (j = 0; j < QK4_0/2; ++j) {
    y[j]           = d * ((qs[j] & 0xF) - 8);
    y[j + QK4_0/2] = d * ((qs[j] >>  4) - 8);   // ← 相距 16，不是相距 1
}
```

即 Q4_0 **一个字节里的两个 nibble 相距 16**（低 = 位置 j，高 = 位置 j+16）；
而我们 `kernels/ref_ops.h` 的 `dequant_i4_row` 是
`(i%2==0) ? packed[i/2]&0xF : packed[i/2]>>4`，即**相距 1**
（低 = 位置 2i，高 = 位置 2i+1）。两者不同，正确的重排是（`i < 8`）：

```
packed[i]     = (qs[2i] & 0xF) | ((qs[2i+1] & 0xF) << 4)
packed[8 + i] = (qs[2i] >>  4) | ((qs[2i+1] >>  4) << 4)
```

scale 两字节**原样保留**（那是 QAT 训出来的，动它就白做 QAT 了）；
zero 两字节写 fp16(8.0) = `00 48`（⚠️ 不是 `00 41`；我第一版手算错了，
被自测里的断言当场抓住 —— 所以那个断言是「反解回 8.0」而不是「等于某常量」）。

实测（`tools/selftest_gguf_transcode.py`，含变异测试）：

| 做法 | 与 `gguf.quants.dequantize` 逐位一致 |
|---|---|
| **重排后** | 合成 **200/200** + 8 个真实张量全过 |
| 直接拷贝 qs（本节原来的说法） | **0/200**，平均相对误差 **133.9%** |

133.9% 已接近「两个无关同范数向量」的 √2 ≈ 141% —— 照错的做等于把
**全部 matmul 权重换成垃圾**，而形状合法、不报错、输出是「通顺的乱码」。

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
（注意 `dequantize` 收 `np.ndarray` 而不是 `bytes`，要先 `np.frombuffer(..., np.uint8)`。
且 `dequantize` 返回的是**扁平** f32，要自己按 `reshape(rows, ne[0])` 重塑。）

**①bis ⚠️ Q6_K → f16 不是无损的（§5 与下面 ① 原来都说「精度升格故无损」，
那个推理是错的）。** 错在只数了 Q6_K 值域的 6 位，没注意它的值是
`d(fp16) × 6-bit 子块 scale × (q-32)` —— **两个量化量的乘积**最多需要
11+6+6 ≈ 23 位尾数，f16 的 11 位装不下。实测：

| | `token_embd` | `per_layer_token_embd` |
|---|---|---|
| 非零元素被 f16 改动的比例 | 64.7 ~ 72.5% | 75.7 ~ 76.8% |
| 相对误差 中位 / 均值 | 1.48e-4 / 1.57e-4 | 1.57e-4 / 1.66e-4 |
| max | 4.866e-4 | 4.880e-4 |
| 超过 1 ULP(2^-10) 的元素 | **0** | **0** |

即误差**严格限于 f16 的 1 ULP（2^-11 = 4.883e-4）**。
判定「可接受」的依据是与已测过的敏感度对比：PLE 表走 i4（权重误差 ~9.6%）
实测只值 0.56 个 argmax 一致率点，而这个扰动比它小约 **600 倍**。
要逐位无损只有两条路：存 f32（PLE 8.75 GiB 磁盘，常驻内存不变），
或在 C++ 侧实现原生 Q6_K dtype（新 kernel + 绑定 + 流式路径）。

⇒ **验收标准 ① 对 f16 张量必须比 `float16(dequantize)` 而不是 `dequantize`**
（转换器里就是这么写的）；对 i4 与 f32 张量则是严格逐位一致。

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
4. **名字映射 + dims 标签反序（数据不转置，见 §5.1 ③）**。
   ✅ 已完成，并做成强制门禁：与 HF 侧 `read_plan_from_dir` 的计划
   逐张量比对，实测 540 个张量集合差 0、shape 差 0。
5. **V4Ext 合成**：从 GGUF 元数据生成，`ple_row_bytes` 按 PLE 的
   **目标 dtype**（f16 ⇒ 8960×2 = 17920）算，不能用文件主 dtype。
6. **EOS 用 106 而不是 GGUF 的 1**（见 §4 的警告）。
7. 验收。**不要用 PPL**（softcapping 下无效，见 §5.6）；
   也**不要**用「vs 原始 HF 的 argmax 一致率」（QAT 权重已训漂，见 §5.1 ④）。
   正确口径见 §5.1 末尾的两条，实测结果见 §8。

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

---

## 8. 转换结果与实测（#28 已完成）

工具：`tools/gguf_reader.py`（GGUF v3 读取器）、
`tools/gguf_to_tiny_gemma4.py`（转换器）、
`tools/materialize_qat_hf.py`（造 QAT 参考 checkpoint）、
`tools/selftest_gguf_transcode.py`（含变异测试的自测）。

```
python tools/gguf_to_tiny_gemma4.py \
    --gguf   models/ms_cache/google/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf \
    --hf-dir models/google/gemma-4-E2B-it \
    --out    models/tiny/gemma4-qat-q4_0.tqwen --verify
```

| | 值 |
|---|---|
| 输出文件 | `models/tiny/gemma4-qat-q4_0.tqwen` |
| 文件总字节 | **6,696,074,240**（6.24 GiB） |
| 常驻权重 | **1905.87 MB** |
| PLE 留盘 | 4480.00 MB（f16，`--ple-ssd`，**不占常驻内存**） |
| 峰值 RSS | **2,026,143,744 B = 1.887 GiB** |
| group_size | **32**（Q4_0 定死，不是可调参数） |
| zero | 恒 **8.0** ⇒ 对称 ⇒ `sdot5_mt` 可用 |
| EOS | 106（取自 HF config 的 `[1,106]` 末位，**不是** GGUF 的 1） |
| 转换耗时 | **9~11 s** |

常驻 1906 MB 比旧 RTN i4 的 1776 MB 多 **7.3%**，两处可解释：
① group_size 64→32 使 i4 从 4.5 bpw 变 5.0 bpw（+11%，约 +108 MB）；
② `per_layer_model_projection` 官方保 F16 而我们原来量化成 i4（+20 MB）。

### 8.1 验收 ①：转换保真度 —— 通过

`--verify` 把写好的 .tqwen 读回来解码，与 `gguf.quants.dequantize` 对照：
**540 个张量全部一致**（i4/f32 严格逐位；f16 比 `float16(dequantize)`，
见 §5.1 ①bis 的 1-ULP 论证）。

i4 的解码用 `verify_i4_roundtrip.dequant_i4_py` —— 那是照抄 C++
`dequant_i4_row` 逐行翻译的**独立**实现，刻意不复用导出器代码，
否则导出器与验证器一起错就对照不出来。

### 8.2 验收 ②：任务质量 —— 通过（门槛 95%）

参考 = `materialize_qat_hf.py` 造的 **HF bf16 QAT checkpoint**
（权重 = GGUF 反量化值，config/tokenizer 复制原始 checkpoint）。
造好后 `eval_i4_quality.py` 零改动可用，只换 `--hf-dir`。

⚠️ 该参考加载时会报 audio/vision 塔 MISSING 并随机初始化 —— 只含语言塔是
有意的，随机初始化不影响纯文本前向，只多占约 0.9 GB。

**名字映射的正确性另有独立验证**：QAT 没漂移的张量，materialize 后应与
原始 checkpoint 吻合，且漂移量要复现 §5.1 ④ 那张表。实测 5 个张量
**逐个精确复现**（0.000 / 0.525 / 0.533 / 0.792 / 17.123%）。
这一步是必要的：5 个 norm 的名字错位得很厉害而形状都是 [1536]，
接错了不会报形状错，只会静默劣化。

语料 = f16 模型 greedy 生成的 534 token（与 §5.4/§5.5.2 的 PTQ 评测同一份：
本次对照运行**精确复现**了 f16 = 97.57%、rtn-i4 = 60.11% 两个旧数字，
证明语料逐 token 相同、口径可比）。

| 配置 | 参考 | argmax 一致率 | margin 中位 | decode |
|---|---|---|---|---|
| f16（原始权重） | HF-**原始** bf16 | **97.57%** ← 精度地板 | 2.67 | 30.70 ms/tok（32.6 tok/s） |
| rtn-i4（RTN gs=64, sdot4_mt） | HF-原始 bf16 | 60.11% | 2.39 | 20.36 ms/tok（49.1） |
| **qat `neon_mt`（W4A32）** | HF-**QAT** bf16 | **97.75%** ✅ | 1.70 | 61.5 ms/tok（16.3） |
| qat `sdot5_mt`（W4A8 对称） | HF-QAT bf16 | 88.95% | 1.61 | **21.43 ms/tok（46.7）** |
| qat `sdot4_mt`/`sdot3_mt`/`sdot2_mt`/`sdot_mt` | HF-QAT bf16 | 88.20% | 1.66 | 23.78 ms/tok（42.0） |
| f16（原始权重） | HF-**QAT** bf16 | 65.92% | 2.67 | — ← **QAT 漂移量本身** |

（速度均为**顺序单独运行**、固定 128 个 decode token、EOS 关闭时测得。
⚠️ 并发跑多个进程会把数字污染到 2~3 倍差 —— 我第一次三个内核并跑，
decode 从 21 ms/tok 变成 38 ms/tok。）

**结论：QAT 把 argmax 一致率从 PTQ 的 60.11%（最好 64.42%）拉到 97.75%，
已在 f16 精度地板 97.57% 之上 ⇒ 4-bit 权重本身几乎不再损失质量。**

最后一行 65.92% 是这次最有价值的对照：它是**转换零误差**的情况
（HF bf16 直接前向，没有我们的引擎参与），却只有 65.92%。
⇒ 「vs 原始模型的一致率」这个判据对 QAT **物理上不可能达到 95%**，
它量的是 QAT 的 4-bit 权重与原模型之差，而不是转换质量。§5.1 ④ 的判断被实测坐实。

### 8.3 W4A8 的激活量化成了新瓶颈

RTN 时代 W4A8（60.11%）与 W4A32（61.99%）只差 **1.9 点**；
QAT 下 W4A8（88.20~88.95%）与 W4A32（97.75%）差 **8.8~9.6 点**。
原因：QAT 已把**权重**量化误差压到近零，于是 W4A8 剩下的主要误差源
变成了**激活的 int8 量化**，之前它被巨大的权重误差掩盖了。

四个 W4A8 变体（sdot/sdot2/sdot3/sdot4 的 _mt）给出**完全相同**的
88.20% 与 margin 1.66 —— 这是预期的一致性检查（它们是同一数学的逐级优化，
数值等价）。`sdot5_mt` 的 88.95% 差 4 个位置，是去掉 zero 修正项后
f32 累加顺序不同导致的近平局翻转，不是 bug。

**但 88.95% 在文本上并不可见地劣化。** 同一个 prompt：

> `neon_mt`（97.75%）：内存带宽是指\*\*内存单位时间内可以传输的数据量\*\*，**它**决定了数据在处理器和内存之间传输的速度。
> `sdot5_mt`（88.95%）：\*\*内存带宽是指内存单位时间内可以传输的数据量，决定了数据在处理器和内存之间传输的速度。\*\*

两者语义完全相同，差别只在 **markdown 加粗的范围**和一个「它」字
（`1018` = `**`、`238010` = `它`）。
⇒ 那 11 个点的不一致**集中在低风险的格式/近平局位置**，不在内容上。
argmax 一致率会系统性过度惩罚这类位置 —— 用它当唯一判据时，
88.95% 听起来像「不可用」，实际不是。

### 8.4 部署取舍（未决，见 §9）

| 方案 | 一致率 | decode | 常驻 | RSS | 评价 |
|---|---|---|---|---|---|
| f16 | 97.57% | 32.6 tok/s | 4348 MB | 4.54 GiB | 6 GB 机放不下 |
| QAT + `neon_mt`（W4A32） | **97.75%** | 16.3 tok/s | 1906 MB | 1.89 GiB | 质量最好，但 W4A32 只有 31 GB/s 有效带宽（f16 是 84%×185 GB/s），说明 i4 的 `neon_mt` 是**计算受限**、内核未优化 |
| QAT + `sdot5_mt`（W4A8） | 88.95% | **46.7 tok/s** | 1906 MB | 1.89 GiB | 最快，质量在文本上看不出差别 |

i4 的价值是**内存**（1.89 vs 4.54 GiB）而不是速度 —— 这点没变。

## 9. 遗留项

> #30 完成后，第 1 条的优先级下降：`sdot6_mt` 已经给出 96.07%@49.0 tok/s，
> W4A32 路径（97.75%@16.3 tok/s）只剩「需要精确参考」这一个用途。

1. **W4A32 的 i4 内核未优化**：`neon_mt` 只有 16.3 tok/s、有效带宽 31 GB/s，
   而 f16 的 `neon_mt_kv_nt` 能到 84%×185 GB/s。i4 读得更少却更慢，
   说明瓶颈在反量化的 ALU 而非带宽。优化它就能同时拿到 97.75% 与高速。
2. ~~**W4A8 的激活精度**~~ **已完成（#30，见 §8.5）**：per-group 激活缩放
   把 88.95% 拉到 **96.07%**，速度不变（49.0 tok/s），新内核 `sdot6_mt`。
   注：int16 激活**没做也不必做** —— SDOT 是 int8×int8，提到 int16 就用不了
   SDOT，会失去 §8.3 分析出的那个 2× MAC 吞吐优势。
3. **`unsloth` 的 `UD-Q4_K_XL`**（2499 MiB，imatrix 动态量化，质量通常高于
   q4_0）仍未试；Q4_K 是超块格式，读取器工作量比 Q4_0 大一个数量级（§7）。
4. **Android 真机**（#24）仍阻塞在无设备。

### 8.5 分组激活 scale 把 W4A8 从 88.95% 拉到 96.07%（#30 已完成，速度不变）

§8.3 把 W4A8 的 8.8 点差距归因于「激活的 int8 量化成了主误差源」。
本次把这个归因**从推断变成实测**，并据此修掉了它。

**先做预检再写内核**（`tools/probe_activation_int8.py`）：在真实前向上 hook
每个 Linear 的输入，比较「整条向量一个 int8 scale」（= sdot5 的做法）与
「每 32 个元素一个 scale」的点积相对误差。判据用**点积层面的能量比**
`sqrt(sum((xq-x)^2)/sum(x^2))`，不用逐元素误差 —— 大量小元素被压成 0 对点积
几乎无影响，而一个大元素错了影响很大，逐元素口径会误导。

| Linear | in_dim | 全局误差 | 分组误差 | 改善 | outlier 比 |
|---|---|---|---|---|---|
| `mlp.down_proj` | 12288 | 6.359% | 1.013% | **6.27×** | 11.8 |
| `mlp.down_proj` | 6144 | 5.740% | 0.987% | 5.82× | 10.8 |
| `per_layer_input_gate` | 1536 | 5.335% | 0.927% | 5.76× | 12.4 |
| `self_attn.o_proj`(full 层) | 4096 | 4.818% | 1.035% | 4.65× | 12.3 |
| `per_layer_model_projection` | 1536 | 0.723% | 0.482% | 1.50× | 1.6 |
| **按 in_dim×层数加权** | | **4.627%** | **0.929%** | **4.98×** | |

outlier 比 = 全局 max / 各组 max 的中位数。10~12 意味着全局 scale 被少数通道
撑大、其余通道被压成 0/±1 —— 正是猜测的机制，且重灾区就是 in_dim 最大的
`down_proj`。（`per_layer_model_projection` 只有 1.50×，说明并非所有 Linear
都有强 outlier，但计算量大的那几个都有。）

**实现**：新增 `kernels/matvec/matvec_i4_sdot6.cpp`，注册 `sdot6` / `sdot6_mt`。
**不动 sdot5**，否则失去 A/B 对照。

改动比预想的还小：sdot5 的外层循环**本来就按权重组走**，每组算一次
`A = w_scale * scale_x`、组内 SDOT 累加到 int32、组结束才乘 A 进 f32 累加器。
所以只需把 `scale_x` 从标量换成按组索引的数组（`float scale_x` →
`const float *scale_x`，`* scale_x` → `* scale_x[g]`），SDOT 结构与 int32
累加完全不动 —— **组内仍是精确整数运算，跨组才进浮点，精度模型不变**。
激活量化改成逐组求 amax，每次 matvec 只调一次、被 out_dim 行摊薄，
且每组 32 个 float = 128 B 常驻 L1，与原来「整条两遍」同阶。

**实测**（534-token 语料、参考 = QAT bf16 checkpoint、顺序单跑、固定 128 decode token）：

| 内核 | argmax 一致率 | margin 中位 | decode |
|---|---|---|---|
| `sdot5_mt`（全局激活 scale） | 88.95% | 1.61 | 20.25 ms/tok（49.4 tok/s） |
| **`sdot6_mt`（分组激活 scale）** | **96.07%** ✅ | 1.70 | 20.41 ms/tok（49.0 tok/s） |
| `neon_mt`（W4A32 精确） | 97.75% | 1.70 | 61.5 ms/tok（16.3 tok/s） |

**+7.12 点，越过 95% 门槛，速度代价在噪声内（49.4 → 49.0 tok/s）。**
剩下的 1.68 点是 int8 激活量化的残余误差。

另一个印证：`sdot6_mt` 的 greedy 生成序列与 `neon_mt`（W4A32 精确参考）
**md5 完全相同**（`3002012b7d97`），而 `sdot5_mt` 是 `4a1542fe8cf2`。
单线程 `sdot6` 与 `sdot6_mt` 也相同。

⇒ **拿到了 W4A32 的输出、W4A8 的速度。** §8.4 的取舍表因此作废：
不再需要在 97.75%@16.3 与 88.95%@46.7 之间选，`sdot6_mt` 是 96.07%@49.0。
i4 现在在质量与速度上**同时**优于 f16（97.57%@32.6），且常驻内存只有
1906 MB（f16 是 4348 MB）。

**测试**（`tests/test_matvec_i4.cpp`，236 → 243，0 failed）：
除了「与自己的分组朴素参考一致」（g64 / g32 / 残组 / 单线程 / 多线程），
另加两条**证明改动真的生效**的：
- `sdot6_differs_from_sdot5_on_outliers`：构造一个 100× 的 outlier 通道，
  以精确 f32 为裁判，断言 sdot6 与 sdot5 结果不同**且** sdot6 的 rel_l2
  小于 sdot5 的 1/3。防的是「分组 scale 写了但没生效」—— 那种情况下
  sdot6 仍会与自己的参考一致、通过全部自洽测试，只有跨内核对比才暴露。
- `sdot6_bitexact_mt_vs_st`：线程池只切行、不改算术，必须逐位一致。

朴素参考 `matvec_i4_w4a8_grouped_naive` 是**新写**的，刻意不复用
`matvec_i4_w4a8_naive`（那份用全局 scale），否则「参考与被测一起错」。
