# gemma-4-E2B-it 文本前向参考规范（阶段 0 交付物）

供 tiny-llm 实现 gemma4 时逐算子对照。**本文档所有公式均直接摘自
`transformers/models/gemma4/modeling_gemma4.py`（transformers 5.17.0）并逐项与
`models/google/gemma-4-E2B-it/` 的 config.json / safetensors 头部交叉验证**，
行号在每处标注。未经推导猜测。

> ⚠️ 本文档存在的理由：gemma4 有 **7 处会导致「能跑、不崩溃、但输出垃圾」的静默数值陷阱**
> （见 §7）。这类错误无法靠冒烟测试发现，只能靠逐算子对齐。

---

## 1. 全局常量（已核实）

```
hidden_size                  = 1536
num_hidden_layers            = 35
vocab_size                   = 262144
intermediate_size            = 6144      ← 仅层 0..14；层 15..34 为 12288
num_attention_heads          = 8
num_key_value_heads          = 1         （GQA 8:1）
head_dim                     = 256       ← 仅 sliding 层；full 层为 512（global_head_dim）
sliding_window               = 512
num_kv_shared_layers         = 20
rms_norm_eps                 = 1e-6
hidden_activation            = gelu_pytorch_tanh    ← 不是 SiLU
attention_bias               = False
tie_word_embeddings          = True
final_logit_softcapping      = 30.0
attention_k_eq_v             = False     → use_alternative_attention 全为 False，v_proj 各层独立
use_bidirectional_attention  = None      → is_causal = (None != "all") = True
max_position_embeddings      = 131072
hidden_size_per_layer_input  = 256       （PLE 维度）
vocab_size_per_layer_input   = 262144
enable_moe_block             = False     → 无 MoE 分支
```

派生常量：

```
主 embed 缩放            = hidden_size**0.5            = 39.191836...
PLE embed_scale          = hidden_size_per_layer_input**0.5 = 16.0
per_layer_model_proj_scale = hidden_size**-0.5         = 0.0255158...
per_layer_input_scale    = 2.0**-0.5                   = 0.7071068...
attention scaling        = 1.0                          ← 硬编码，见 §7 陷阱 1
first_kv_shared_layer_idx = 35 - 20 = 15
```

RoPE **两套配置**（`config.text_config.rope_parameters`，按 layer_type 取）：

| layer_type | rope_theta | rope_type | partial_rotary_factor | 实际旋转维度 |
|---|---|---|---|---|
| `sliding_attention` | **10000.0** | `default` | 1.0（未设） | head_dim 256 全量 |
| `full_attention` | **1000000.0** | `proportional` | **0.25** | 512 × 0.25 = **128** |

`layer_types` 实测分布：28 × sliding + 7 × full，full 层号 = **4, 9, 14, 19, 24, 29, 34**（每 5 层一个）。

---

## 2. 逐层异构表（实测自 safetensors，非推导）

| 层号区间 | layer_type | intermediate_size | head_dim | q_proj | k/v_proj | o_proj | 自有 KV |
|---|---|---|---|---|---|---|---|
| 0–14 | 混合（full 在 4,9,14） | **6144** | 256 / 512 | [2048,1536] / [4096,1536] | [256,1536] / [512,1536] | [1536,2048] / [1536,4096] | ✅ |
| 15–34 | 混合（full 在 19,24,29,34） | **12288** | 256 / 512 | 同上 | 同上 | 同上 | ❌ 共享 |

实测分布：`intermediate_size` = {6144: 15 层, 12288: 20 层}；`head_dim` = {256: 28 层, 512: 7 层}。

**两条规则共用同一个判据**（`modeling_gemma4.py:1064-1067` 与 `:1183-1184`）：

```python
first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers   # = 15
is_kv_shared_layer        = layer_idx >= first_kv_shared_layer_idx
use_double_wide_mlp       = config.use_double_wide_mlp and is_kv_shared_layer
intermediate_size         = config.intermediate_size * (2 if use_double_wide_mlp else 1)
```

即**层 15–34 同时具备「KV 共享」和「双倍宽 MLP」两个属性**，层 0–14 两者都无。

每层张量清单（`model.language_model.layers.N.*`）：

```
input_layernorm.weight              [1536]
post_attention_layernorm.weight     [1536]
pre_feedforward_layernorm.weight    [1536]
post_feedforward_layernorm.weight   [1536]
post_per_layer_input_norm.weight    [1536]     ← PLE 子块，Qwen 无
layer_scalar                        [1] bf16   ← 逐层不同，Qwen 无
mlp.gate_proj.weight                [I, 1536]  I = 6144 或 12288
mlp.up_proj.weight                  [I, 1536]
mlp.down_proj.weight                [1536, I]
per_layer_input_gate.weight         [256, 1536]  ← PLE
per_layer_projection.weight         [1536, 256]  ← PLE
self_attn.q_proj.weight             [8*hd, 1536]
self_attn.k_proj.weight             [1*hd, 1536]   ← 仅层 0..14 存在
self_attn.v_proj.weight             [1*hd, 1536]   ← 仅层 0..14 存在
self_attn.o_proj.weight             [1536, 8*hd]
self_attn.q_norm.weight             [hd]
self_attn.k_norm.weight             [hd]           ← 仅层 0..14 存在
self_attn.v_norm.weight             [hd]           ← 仅层 0..14 存在，且 with_scale=False
```

> **层 15–34 没有 k_proj / v_proj / k_norm / v_norm**（`modeling_gemma4.py:1196`
> `if not self.is_kv_shared_layer:` 才创建）。导出时不要为这些层生成这些张量，
> 绑定逻辑也必须容忍它们缺失。

> **`v_norm` 是 `Gemma4RMSNorm(head_dim, eps, with_scale=False)`** —— 无缩放参数的
> RMSNorm，作用于 value。多数模型不 norm V，容易漏。

---

## 3. 前向全流程

### 3.1 输入嵌入（`modeling_gemma4.py:1575-1577`, `:1444-1456`, `:1646`）

```
embed_tokens = Gemma4TextScaledWordEmbedding(vocab, 1536, embed_scale=1536**0.5)
inputs_embeds = embed_tokens(input_ids) * 39.191836...
```

**主嵌入自带 ×sqrt(hidden_size) 缩放**，且 tied 到 lm_head。

### 3.2 PLE 上下文分量（`project_per_layer_inputs`，`:1755-1788`）

```
P = per_layer_model_projection(inputs_embeds)          # Linear 1536 → 8960
P = P * 0.0255158...                                    # = hidden_size**-0.5
P = reshape(P, [B, S, 35, 256])
P = per_layer_projection_norm(P)                        # RMSNorm(256, eps=1e-6)，对最后一维
```

注意 `per_layer_model_projection` 权重形状是 **[8960, 1536]**，`per_layer_projection_norm`
是 **RMSNorm(256)**（作用在 reshape 之后的最后一维，即每层的 256 维切片上）。

### 3.3 PLE token 分量（`get_per_layer_inputs`，`:1711-1753`）

```
T = embed_tokens_per_layer(input_ids) * 16.0            # embed_scale = 256**0.5
T = reshape(T, [B, S, 35, 256])
```

`embed_tokens_per_layer` 权重形状 **[262144, 8960]**，8960 = 35 × 256。
**这就是需要流式加载的 2.349B 参数大表**（fp16 4.7 GB / int4 ~1.32 GB）。

### 3.4 两个分量合并（`:1788`）

```
per_layer_inputs = (P + T) * 0.7071068...               # = 2**-0.5
```

若为多模态输入（`input_ids is None`）则只有 P，不乘 2**-0.5。**纯文本路径永远走合并分支。**

合并后的张量形状 `[B, S, 35, 256]`，在第 i 层取 `per_layer_inputs[:, :, i, :]`（`:1690`）。

> **对 PLE 流式的含义**：查表发生在**进入层循环之前、对全部 token 一次性完成**，
> 不是逐层查。因此实现时应：
> - prefill N 个 token → 读 N 行 × 17.9 KB（fp16），N=1400 时约 25 MB
> - decode → 每 token 读 1 行 17.9 KB
> - 立即与 P 合并成紧凑的 `[S, 35, 256]` 中间量后释放行缓冲
> - 中间量大小：S=8192 时 8192×35×256×4B = **293 MB fp32**（可分块处理）
>
> 4.7 GB 的表**永远不需要整体驻留**。

### 3.5 Decoder Layer（`Gemma4TextDecoderLayer.forward`，`:1384-1441`）

```python
residual = h
h = input_layernorm(h)                                   # RMSNorm(1536)
h = self_attn(h, ...)                                    # 见 §3.6
h = post_attention_layernorm(h)
h = residual + h

residual = h
h = pre_feedforward_layernorm(h)
h = mlp(h)                                               # 见 §3.7
h = post_feedforward_layernorm(h)
h = residual + h

# ---- PLE 子块（第三个残差块，在 attention 和 MLP 之后）----
residual = h
h = per_layer_input_gate(h)                              # Linear 1536 → 256
h = act_fn(h)                                            # gelu_pytorch_tanh
h = h * per_layer_input                                  # 逐元素门控，[.,256] × [.,256]
h = per_layer_projection(h)                              # Linear 256 → 1536
h = post_per_layer_input_norm(h)                         # RMSNorm(1536)
h = residual + h

h = h * layer_scalar                                     # ← 每层最后乘标量
return h
```

**共 3 个残差子块**（attention / MLP / PLE），Qwen 只有 2 个。
`enable_moe_block=False`，跳过 `:1414-1427` 的 MoE 分支。

`layer_scalar` 实测各层不同（bf16）：layer0=0.01782、layer1=0.22266、layer2=0.79297、
layer3=0.28711、layer4=0.49805、layer5=0.63672、layer14=0.02856…。
初始化为 `ones(1)`（`:1367`、`:1510`），checkpoint 里是训练后的值，**必须逐层读取，不可假设为 1**。

### 3.6 Attention（`Gemma4TextAttention.forward`，`:1214-1276`）

```python
q = q_proj(h).view(B, S, 8, hd)
q = q_norm(q)                       # RMSNorm(hd)，有 scale
q = apply_rotary_pos_emb(q, cos, sin)     # 按 layer_type 取对应 RoPE 配置
q = q.transpose(1, 2)

if is_kv_shared_layer:              # 层 15..34
    k, v = shared_kv_states[layer_type]        # 按 layer_type 取，不是按层号
else:                               # 层 0..14
    k = k_proj(h).view(B, S, 1, hd)
    v = v_proj(h).view(B, S, 1, hd)            # attention_k_eq_v=False，故 v_proj 存在
    k = k_norm(k)                              # RMSNorm(hd)，有 scale
    k = apply_rotary_pos_emb(k, cos, sin)
    k = k.transpose(1, 2)
    v = v_norm(v).transpose(1, 2)              # RMSNorm(hd)，with_scale=False

if past_key_values is not None and not is_kv_shared_layer:
    k, v = past_key_values.update(k, v, layer_idx)
if store_full_length_kv:
    shared_kv_states[layer_type] = (k, v)      # 只有层 13、14 会写

attn = attention(q, k, v, mask, scaling=1.0, sliding_window=self.sliding_window)
#   eager: repeat_kv(k, 8) → matmul(q, k^T) * 1.0 → mask → softmax → matmul(·, v)
h = attn.reshape(B, S, 8*hd)
h = o_proj(h)
```

**KV 共享的精确结构**（照抄 `:1183-1188` 判据计算，已核实）：

```
store_full_length_kv = (not is_kv_shared_layer) and
                       layer_idx == 最后一次出现该 layer_type 的下标（在层 0..14 内）
=> 提供方只有两层：layer 13（sliding）、layer 14（full）
```

- `shared_kv_states` 是**按 `layer_type` 索引的 2 槽字典**，不是按层号的映射表
- 层 15–34 中 16 个 sliding 层读 layer 13 的 KV，4 个 full 层（19,24,29,34）读 layer 14 的 KV
- 源码注释强调：即使有 Cache 也必须走 `shared_kv_states`，因为 sliding 层的 Cache
  过窗后不再保留全长状态，而共享方需要**全长**
- **推论：layer 13 本身是 sliding 层（自有 Cache 窗口封顶 512），但它提供给共享的副本必须全长**
  → 需要一份额外的全长 KV 存储

### 3.7 MLP（`Gemma4TextMLP`，`:1061-1078`）

```python
h = down_proj( act_fn(gate_proj(x)) * up_proj(x) )      # act_fn = gelu_pytorch_tanh
```

即 **GEGLU 结构，不是 SwiGLU**。`intermediate_size` 按 §2 逐层不同。

### 3.8 输出（`:1864-1867`）

```
h = norm(h)                                    # 最终 RMSNorm(1536)
logits = lm_head(h)                            # = embed_tokens.weight（tied），[262144, 1536]
logits = tanh(logits / 30.0) * 30.0            # final_logit_softcapping
```

文本 attention **没有** logit softcapping（`:340`、`:835` 的 tanh 分别属于
`Gemma4AudioAttention` 和 `Gemma4VisionRotaryEmbedding`，与文本路径无关）。

---

## 4. KV cache 内存精算（据 §3.6 结构）

自有 KV 的层：0–14 共 15 层 = 12 个 sliding（head_dim 256，窗口封顶 512）+ 3 个 full（4,9,14，head_dim 512，全长）。
外加 layer 13 的全长共享副本（sliding 型，head_dim 256）。layer 14 本身即全长，无需额外副本。

| 上下文 | bf16 | int8 |
|---|---|---|
| 2K | 20.0 MiB（6.0 固定 + 12.0 + 2.0） | 10.0 MiB |
| 8K | 62.0 MiB（6.0 固定 + 48.0 + 8.0） | 31.0 MiB |
| 32K | 230.0 MiB（6.0 固定 + 192.0 + 32.0） | 115.0 MiB |

> 这比 `docs/edge-model-budget.md` §5 里按「独立层比例 0.43」粗估的 27 MiB@8K(int8)
> 略大（精算 31 MiB），因为粗估没有计入 layer 13 的全长副本。结论不变：**上下文不是内存瓶颈**。

---

## 5. 与 tiny-llm 现状的差距（已核实）

### 5.1 已具备，可直接用

| 能力 | 位置 |
|---|---|
| **partial rotary** | `runtime/backend.h:214-221` 已有 `rotary_dim = head_dim * partial_rotary_factor` 的接口；`runtime/backend_cpu.h:87-90` 有 CPU 实现 → full 层的 0.25 因子白捡 |
| i4 / GPTQ matvec kernel 全家族 | `kernels/matvec/matvec_i4_*`，按「打包布局 + (out_dim,in_dim,group_size)」驱动，**与架构无关** |
| RMSNorm | `kernels/rmsnorm/` |
| SSD 单行 pread 模式 | `runtime/qwen_forward_token.cpp:118-148` + `ExpertStore::read_bytes` |
| Python 侧 tokenizer | `tools/tokenize_prompt.py`（262144 词表无痛） |

### 5.2 缺失，必须新增

| 缺口 | 说明 | 严重度 |
|---|---|---|
| **gelu_pytorch_tanh kernel** | `kernels/` 下只有 `silu/`（`silu_ref.cpp`、`swiglu_neon.cpp`、`swiglu_ref.cpp`），**无 gelu**。且 `qwen_forward_token.cpp:468` 的 FFN 硬编码 SwiGLU。需要新写 gelu_tanh + GEGLU 融合算子（ref + neon） | 高（MLP 与 PLE 门控都要用） |
| **final_logit_softcapping** | 全仓 grep `softcap`/`capping` 零命中。需在 logits 输出前加 `tanh(x/30)*30` | 中（漏了会改变采样分布） |
| **两套 rope_theta** | `ModelConfig.rope_theta`（`model_loader.h:54`）与 `TinyHeader.rope_theta`（`tiny_format.h:236`）都是**单一 float**。gemma4 需要 sliding=1e4 / full=1e6 两个 | 中（V4Ext 加一个字段） |
| **sliding window attention** | `attention_decode_ref(...)` 对 `t∈[0,seq_len)` 全量 online softmax，调用点永远传 `pos+1`；全仓无 sliding/window 概念 | 高 |
| **逐层不同 head_dim** | `head_dim` 是文件头单一字段 → `ModelConfig.head_dim` 单值 | 高 |
| **逐层不同 intermediate_size** | 同上，6144 / 12288 两档 | 中 |
| **KV 共享** | `KvCache` 是单块同构 arena `[2][n_layers][n_kv_heads][max_seq_len][head_dim]`，层号直接寻址，无 logical→physical 间接 | 高 |
| **v_norm（with_scale=False）** | 无缩放参数的 RMSNorm 变体，需确认现有 rmsnorm kernel 能否传空 scale | 低 |
| **layer_scalar** | 逐层标量乘法，Qwen 无此概念 | 低 |
| **4 个 layernorm/层** | Qwen 每层 2 个，gemma4 有 5 个（含 post_per_layer_input_norm） | 低（`LayerWeights` 扩字段） |
| **PLE 大表绑定与流式** | 见 §6 | 中（有先例） |

---

## 6. PLE 流式加载的落点（已核实的三处改动）

1. **`runtime/model_loader.cpp:60-62` `is_offloadable()`** —— 目前只认
   `.mlp.experts.` 子串和 `== "model.embed_tokens.weight"`。需加入 PLE 表名，
   建议改为读 V4Ext 的「大表留盘」标志位而非继续堆名字匹配。
2. **tied 守卫（两处）** —— `model_loader.cpp:258-260` 与 `:513-515` 都硬编码
   `"model.embed_tokens.weight"`。gemma4 的 HF 名是 `model.language_model.embed_tokens.weight`。
   **若 exporter 映射成别的名字，守卫静默失效 → tied embed 被卸载 → lm_head 指针悬空。**
   建议：泛化守卫 + 加载期断言 `tied_embeddings != 0 && embed 被卸载 → fail-fast`。
3. **`qwen_model.cpp:508-528` 偏移绑定** —— 照 `embed_file_offset_` 的模式加
   `ple_file_offset_` / `ple_row_bytes_`（**行字节数在导出时算好写入 V4Ext**，
   运行期不再计算，理由见 §7 陷阱 5）。

---

## 7. 七处静默数值陷阱（漏掉任何一个都「能跑但输出垃圾」）

| # | 陷阱 | 后果 | 依据 |
|---|---|---|---|
| 1 | **attention `scaling = 1.0`**，不是 `head_dim**-0.5` | 注意力分布差 sqrt(256)=16 倍或 sqrt(512)=22.6 倍 | `:1178` 硬编码 `self.scaling = 1.0`；`:825-826` 只在 scaling 为 None 时才用 `head_dim**-0.5` |
| 2 | **主 embed 要 ×sqrt(1536)=39.19** | 全部激活差 39 倍 | `:1575-1577` `embed_scale=config.hidden_size**0.5` |
| 3 | **PLE 查表要 ×16.0**，合并后要 ×2^-0.5，投影后要 ×1536^-0.5 | PLE 子块量级全错 | `:1595`、`:1597`、`:1603`、`:1788` |
| 4 | **激活是 gelu_pytorch_tanh 不是 SiLU** | MLP 与 PLE 门控全错 | `:1376`、`:1073` `ACT2FN[config.hidden_activation]`，config 值 `gelu_pytorch_tanh` |
| 5 | **PLE 行字节数随 dtype 变**（fp16=17920B，i4 是每组 68B in-band 的分组布局） | 读错偏移 → 输出垃圾但不崩溃。**源码注释记录过同类事故：硬编码 `sizeof(float)` 使 fp16 embed 的 CosSim 掉到 0.87、argmax 全错** | `qwen_forward_token.cpp:126-131` 注释；`tiny_format.h:59-77` |
| 6 | **`layer_scalar` 逐层不同**，初始化为 1 但 checkpoint 里是 0.0178~0.79 | 若假设为 1，每层输出量级错 | `:1367`、`:1440`、`:1510`；实测值见 §3.5 |
| 7 | **`v_norm` 是 with_scale=False 的 RMSNorm**，且层 15–34 根本没有 k/v 权重 | 漏 norm → V 量级错；为共享层生成不存在的张量 → 绑定失败或读到垃圾 | `:1197`、`:1196` |

外加一处行为差异（非数值陷阱但会影响正确性）：**`final_logit_softcapping=30.0`**
必须在 lm_head 之后、采样之前应用（`:1864-1867`）。

---

## 8. 阶段 0 的验收标准

在写任何真权重相关代码之前，先完成：

1. `tools/make_fake_gemma4_model.py` —— 生成一个**缩小版**随机 gemma4
   （建议 `num_hidden_layers=6`（保证同时含 sliding 与 full，且跨越 KV 共享边界）、
   `hidden_size=64`、`hidden_size_per_layer_input=16`、`vocab_size=512`），
   **同时**产出 `.tqwen` 与 HF `state_dict`。
   缩小时必须保留：两套 head_dim、两套 rope_theta、partial_rotary_factor=0.25、
   `num_kv_shared_layers` 使共享边界落在层中间、`layer_scalar` 取非 1 的随机值、
   `use_double_wide_mlp` 造成的两种 intermediate_size。
2. `tools/align_fake_gemma4_model.py` —— 抄 `tools/align_fake_qwen35_model.py` 骨架
   （`run_cpp`:65 + `main`:117），C++ `--dump-logits` vs HF eager 前向，逐位置比对。
3. **验收门槛**：
   - 逐位置 logits 最大绝对误差 < **1e-5**（与现有 qwen35 对齐脚本同标准）
   - argmax 全序列一致
   - **额外**：PLE 子块的中间量（`per_layer_inputs[:, :, i, :]`）也要单独比对一次，
     因为 §7 陷阱 3/5 的错误会被后续残差稀释，只在最终 logits 上比对可能看不出来
   - **额外**：断言 `layer_scalar` 全部生效（把某层 scalar 改成 2.0，logits 必须变化）

只有阶段 0 通过，才有资格进入阶段 1（V4Ext + exporter）。
