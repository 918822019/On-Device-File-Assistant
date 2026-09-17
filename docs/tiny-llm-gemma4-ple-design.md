# tiny-llm 支持 gemma4 + PLE 流式加载：改动面设计

目标：让 `third_party/tiny-llm`（tinyqwen 引擎）能加载并推理 `google/gemma-4-E2B-it`，
且 2.349B 参数的 PLE 大表**留盘流式读取**、不常驻内存。

> 本文档中标注「已核实」的结论均由直接读源码 / 读 safetensors 头部得到；
> 标注「调查」的来自对 tiny-llm 的架构勘察，未逐行复核。
> 模型侧数据来自 `models/google/gemma-4-E2B-it/`（commit 时的实测）。

---

## 1. 结论摘要

**规模：中型特性，不是重写。** C++ 侧约 10–14 个文件、2500–4000 行（含测试）；
Python 工具链 3–4 个新脚本、1200–2000 行。

**最大的一块可复用**：PLE 流式加载**已有完全同构的先例**——`embed_tokens` 的 SSD
单行卸载。预计 100–200 行即可。

**最高风险不在 PLE，在 KvCache**：需要同时支持「逐层不同 head_dim」和「跨层 KV 共享」，
而 KvCache 被 decode、两条 prefill、投机解码的 checkpoint/restore/truncate 和多个测试共用。

---

## 2. gemma4 结构事实（已核实，读自 safetensors 头部与 config.json）

### 2.1 整体

```
model_type = gemma4        architectures = ['Gemma4ForConditionalGeneration']
hidden_size = 1536         num_hidden_layers = 35        vocab_size = 262144
intermediate_size = 6144   use_double_wide_mlp = True    tie_word_embeddings = True
sliding_window = 512       max_position_embeddings = 131072
enable_moe_block = False   ← 不是 MoE，没有专家稀疏性可省
```

### 2.2 逐层异构的注意力维度（关键难点）

`layer_types` = 28 × `sliding_attention` + 7 × `full_attention`，
full 层号为 **4, 9, 14, 19, 24, 29, 34**（每 5 层一个）。

| 张量 | sliding 层 | full 层 |
|---|---|---|
| `self_attn.q_proj.weight` | [2048, 1536] = 8×**256** | [4096, 1536] = 8×**512** |
| `self_attn.k_proj.weight` | [256, 1536] = 1×256 | [512, 1536] = 1×512 |
| `self_attn.v_proj.weight` | [256, 1536] | [512, 1536] |
| `self_attn.o_proj.weight` | [1536, 2048] | [1536, 4096] |
| `self_attn.q_norm.weight` | [256] | [512] |
| `self_attn.k_norm.weight` | [256] | [512] |

即 **GQA 8:1，head_dim 逐层类型不同（256 / 512）**。
对应 config 的 `head_dim=256` 与 `global_head_dim=512`、`num_key_value_heads=1`。

### 2.3 每层张量清单（language_model）

```
input_layernorm.weight              [1536]
post_attention_layernorm.weight     [1536]
pre_feedforward_layernorm.weight    [1536]     ← Gemma2/3 式 pre+post 双 norm
post_feedforward_layernorm.weight   [1536]     ← Qwen 只有 input/post_attention 两个
post_per_layer_input_norm.weight    [1536]     ← PLE 相关
layer_scalar                        [1] bf16   ← Qwen 无此物，见 §2.5
mlp.{gate,up}_proj.weight           [6144, 1536]
mlp.down_proj.weight                [1536, 6144]
per_layer_input_gate.weight         [256, 1536]  ← PLE
per_layer_projection.weight         [1536, 256]  ← PLE
self_attn.{q,k,v,o}_proj / {q,k}_norm          （维度见 §2.2）
```

### 2.4 全局张量（language_model）

| 张量 | 形状 | 参数 | fp16 | int4 | 能否留盘 |
|---|---|---|---|---|---|
| `embed_tokens.weight` | [262144, **1536**] | 0.403B | 805 MB | ~226 MB | **不能**（tied，见 §4.2） |
| `embed_tokens_per_layer.weight` | [262144, **8960**] | **2.349B** | **4.7 GB** | ~1.32 GB | **能且应该** |
| `per_layer_model_projection.weight` | [8960, 1536] | 0.014B | 26 MB | — | 常驻 |
| `per_layer_projection_norm.weight` | [256] | — | — | — | 常驻 |
| `norm.weight` | [1536] | — | — | — | 常驻 |

8960 = 35 层 × 256。

### 2.5 两个必须从 HF 参考实现推导、不可猜的点

1. **`layer_scalar`**：逐层一个 bf16 标量，且**各层取值不同**（实测 layer0=0.0178、
   layer1=0.2227、layer2=0.7930、layer3=0.2871、layer4=0.4980、layer14=0.0286…）。
   Qwen 全系没有这个张量，其作用（残差缩放？输出门控？）必须读
   `transformers` 的 `Gemma4TextModel` 实现确认。
2. **PLE 的组合公式**：gemma4 用的是
   `per_layer_input_gate` [256,1536] + `per_layer_projection` [1536,256] +
   `post_per_layer_input_norm` [1536] + 全局 `per_layer_model_projection` [8960,1536] +
   `per_layer_projection_norm` [256]。
   **checkpoint 里没有任何 `altup_*` / `laurel_*` 张量**，即 gemma4 的 PLE 与
   gemma3n 的 AltUp/Laurel 不是同一套机制，不能照搬 llama.cpp 的 altup 路径。

> 附带：`text_config` 是逐层异构的，直接读 `config.text_config.head_dim` 会抛
> `AmbiguousGlobalPerLayerAttributeError`，需经 `per_layer_config[i]` 访问。

---

## 3. tiny-llm 侧现状（调查 + 关键点已核实）

### 3.1 四层抽象，但 forward 是每架构手写

| 层 | 位置 | gemma4 需要做什么 |
|---|---|---|
| arch 枚举 | `runtime/tiny_format.h:185-194`<br>`ModelType{kQwen2=0,kQwen35=1,kQwen35MoE=2,kQwen3MoE=3}` | 加 `kGemma4=4` |
| 磁盘头 | `TinyHeader`(192B) + `V2Ext` + `V3Ext`；`kFormatVersion=3` | **需要 V4Ext**（见 §4.1） |
| 内存配置 | `runtime/model_loader.h:44-177` `ModelConfig` | 加字段 + 校验（`model_loader.cpp:561-587`） |
| 张量绑定 | `runtime/qwen_model.cpp` `QwenModel::create()` ~:440-760<br>容器 `LayerWeights`（`qwen_model.h:373-440`） | 加 gemma4 绑定分支 |
| forward | `qwen_forward_token.cpp`(742行)、`qwen_forward_prefill.cpp`(832行)、`qwen_forward_prefill_qwen35.cpp`(915行) | 新写 gemma4 路径或加分支 |

**没有通用图执行器**，这是工作量的主要来源。

### 3.2 `.tqwen` 格式对 PLE 无需改动（已核实字段宽度）

`TensorEntry`（`tiny_format.h:302-309`，`static_assert(sizeof==120)`）：
`name[64] / dtype(u32) / ndim(1..4) / shape u64[4] / offset u64 / nbytes u64`。

PLE 表 262144×8960：ndim=2 ✅、shape u64 ✅、fp16 下 nbytes=4.7e9 < u64 ✅、
数据区 64B 对齐天然满足 ✅。**格式本身够用**，要改的是加载策略（§4.2）。

### 3.3 已有的 SSD 卸载基建（PLE 可直接复用）

`runtime/expert_store.{h,cpp}`：

- `bool open(path, err)` — `O_RDONLY`
- `bool read_bytes(uint64_t offset, size_t nbytes, void *out)` — **任意偏移裸 pread**
- `const ExpertWeights get(layer, expert)` — LRU 命中零拷贝，miss 淘汰 + pread
- `set_cache_slots(n)` / `set_cache_budget(bytes, err)` / `drop_page_cache()`
- **刻意用 pread 而非 mmap**：mmap 的 readahead 会把未激活的专家拉进 page cache
  （`expert_store.h:11-13`）
- 文档结论（`docs/moe_offload.md`）：slots=4 最优；slots=4096 反而更慢并把机器推进 swap

**已核实的同构先例**——`embed_tokens` 单行卸载（`qwen_forward_token.cpp:118-148`）：

```cpp
const void *embed_src = embed_;
if (embed_file_offset_ != 0) {
    // 行字节数随 dtype 变：fp32 = hidden*4，fp16 = hidden*2。
    // 硬编码 sizeof(float) 会让 fp16 embed 读到错误偏移的数据
    // （实测 CosSim 掉到 0.87、argmax 全错）。
    const size_t row_bytes = hidden * (embed_dtype_ == Dtype::kF16 ? 2 : 4);
    if (!expert_store_ || !expert_store_->read_bytes(
            embed_file_offset_ + token_id * row_bytes, row_bytes, embed_row_.data())) {
        std::fprintf(stderr, "tinyqwen: embed 卸载 pread 失败 (token=%d)\n", token_id);
        std::abort();
    }
    embed_src = embed_row_.data();
}
```

偏移绑定在 `qwen_model.cpp:508-528`（`embed_file_offset_`，`ev->data == nullptr` 时走卸载分支）。

**PLE 就是这个模式的放大版**：行字节数 8960×2 = 17.9 KB（fp16），
每 token 读一次整行即可供 35 层使用（见 §5 风险 R3）。

### 3.4 当前缺失的能力（已核实）

| 能力 | 现状 |
|---|---|
| sliding window attention | **完全没有**。`attention_decode_ref(q,k,v,seq_len,max_seq_len,n_heads,n_kv_heads,head_dim,scale,out)` 对 `t∈[0,seq_len)` 全量 online softmax，调用点永远传 `pos+1`。全仓 grep `sliding\|window` 只命中 Python 侧模型配置 |
| 逐层不同 head_dim | **不支持**。head_dim 是文件头单一字段 → `ModelConfig.head_dim` 单值 |
| 跨层 KV 共享 | **不支持**。`KvCache` 是单块同构 arena `[2][n_layers][n_kv_heads][max_seq_len][head_dim]`，层号直接寻址；唯一的压缩映射是 GDN 混合架构里 full 层→紧凑下标（`model_loader.h:153-159`），没有 layer→shared-slot 间接表 |
| C++ tokenizer | **刻意不做**（`main.cpp:26-27`）。Python 侧 `tools/tokenize_prompt.py` 预分词 → `--tokens-json` 消费；输出只有 token id，detokenize 也在 Python 侧 |
| JNI 层 | **完全没有**。Android 部署是 `adb push` 单个 CLI + `adb shell` 运行（`scripts/run_android.sh`，设备目录 `/data/local/tmp/tinyqwen`） |

好消息：attention 是 dispatch 注册表算子（`kernels/dispatch.h:326,342-346`），
**加带 window-start 参数的新变体不动老实现**；i4 kernel 家族
（`matvec_i4_{ref,neon,neon_mt,sdot,sdot2..5}`）完全按
「打包布局 + (out_dim, in_dim, group_size)」驱动，**与架构无关，零新 kernel**。

---

## 4. 三个必须先定的设计决策

### 4.1 V4Ext 头扩展

`kFormatVersion=3`，而 **V3Ext 的 32 字节 reserved 已被 MoE 用掉**。
`tiny_format.h:22-25` 有硬纪律：**改格式必须同时改 exporter**。

V4Ext 至少需要容纳：

```
sliding_window            u32     512
global_head_dim           u32     512      （head_dim=256 走原字段）
n_full_attn_layers        u32     7
full_attn_layer_bitmap    u64     位图标记哪 7 层是 full（比存层号列表省空间）
kv_shared_layers          u32     20
ple_hidden_per_layer      u32     256
ple_total_dim             u32     8960
ple_file_offset           u64     PLE 表在文件中的偏移
ple_row_bytes             u32     按 dtype 算好的行字节数（避免运行期再算错，见 R2）
```

`layer_scalar` 是逐层张量，走正常 TensorEntry 即可，不占头部。

### 4.2 tied embedding 不可卸载 —— 守卫目前硬编码 Qwen 名字（已核实，高风险）

`model_loader.cpp:47-63` 的注释明确写着：

```
//  **tied 模型不能卸载 embed**：tied 时 lm_head = embed，而 lm_head 每
//  token 要全读 1187 MB，卸载会让 lm_head 指针悬空。由调用方守卫。
bool is_offloadable(const std::string &name) {
    return name.find(".mlp.experts.") != std::string::npos ||
           name == "model.embed_tokens.weight";
}
```

守卫有**两处**，且都硬编码了 Qwen 的名字：

- `model_loader.cpp:258-260`
  ```cpp
  const bool tied_embed = (name == "model.embed_tokens.weight" && h.tied_embeddings != 0);
  if (is_offloadable(name) && !tied_embed) continue;
  ```
- `model_loader.cpp:513-515` 稀疏加载判据里同样的条件

`TinyHeader.tied_embeddings` 字段已存在（`model_loader.h:55`，读取于 `model_loader.cpp:552`），
**这部分不用改格式**。

**风险**：gemma4 的 HF 名是 `model.language_model.embed_tokens.weight`。
如果 exporter 的 `plan_tensors` 把它映射成别的名字，这两处守卫会**静默失效** →
tied 的 embed 被卸载 → **lm_head 指针悬空**。这正是注释警告的失败模式，
而且不会立刻崩溃，只会产出垃圾 logits。

**决策项**：exporter 应把 gemma4 的 embed 也命名成 `model.embed_tokens.weight`
（复用现有守卫），还是泛化守卫为「读 header 的 tied 标志 + 一个 offload 白名单」？
建议后者，并**加一个加载期断言**：`tied_embeddings != 0 && embed 被卸载 → 直接 fail-fast`。

同时 `is_offloadable()` 需要新增 PLE 的名字/规则。建议引入通用的
「大表留盘」标志位放 V4Ext，而不是继续堆名字子串匹配。

### 4.3 lm_head 常驻成本（对调查报告头号风险的更正）

架构勘察把「最大风险」记为：

> 262144×8960 的 lm_head：若 tied 且表被卸载，`project_lm_head_raw`（`qwen_model.h:192`）
> 要求整表在 RAM——4.7GB fp16 装不下 …… 这是 PLE 需求里唯一没有现成先例的部分

**这条把两张表搞混了**（已核实）：

- `262144×8960`（2.349B，fp16 4.7 GB）是 **`embed_tokens_per_layer`（PLE）**，
  在**输入侧**、按 token id 取行、**可流式**
- **lm_head 是 tied 的 `embed_tokens`，`[262144, 1536]` = 0.403B**，
  fp16 **805 MB** / int4 **~226 MB**

所以 lm_head 必须常驻（因为 tied），但 805 MB fp16 / 226 MB int4 **完全可接受**，
不构成阻塞。`project_lm_head_raw` 要求整表在 RAM 这一点是对的，
只是那张表是 0.403B 而非 2.349B。

**结论：头号风险降级。** 风险重心移到 KvCache（§5 R1）。

---

## 5. 风险清单（按严重度排序）

### R1 — KvCache 改造的连带面（最高）

需要同时支持：① 逐层不同 head_dim（256/512）；② `num_kv_shared_layers=20` 的
layer→物理槽映射。

而 `KvCache` 被以下全部共用：`qwen_forward_token.cpp`、两条 prefill 路径、
**投机解码的 checkpoint/restore/truncate**（`qwen_model.h:261-278`）、多个测试。
属于「改一处、验十处」。

缓解：先把 sliding/full 两套 arena 分开（28 层 × head_dim 256 + 7 层 × head_dim 512），
避免引入通用的逐层 head_dim 数组；KV 共享用一个 `logical→physical` 间接表，
默认恒等映射，只有 gemma4 才启用。**必须为投机解码路径单独加回归测试。**

### R2 — PLE 行字节数算错会静默降级，不崩溃

`qwen_forward_token.cpp` 的注释记录过一次真实事故：硬编码 `sizeof(float)`
导致 fp16 embed 读到错误偏移，**CosSim 掉到 0.87、argmax 全错**，但程序不报错。

PLE 更危险：它有 fp16 / bf16 / i4 三种可能布局，i4 还是分组量化
（每组 68B in-band：scale fp16 + zero fp16 + 64B packed uint4，`tiny_format.h:59-77`），
行字节数不是简单的 `dim × sizeof`。

缓解：把 `ple_row_bytes` **在导出时算好写进 V4Ext**，运行期不再计算；
并在对齐脚本里对 PLE 路径单独做数值断言（不只是 argmax 一致，还要 CosSim > 0.999）。

### R3 — PLE 每 token 读一次整行 vs 每层读一次切片

`embed_tokens_per_layer` 布局是 `[262144, 8960]`，一个 token 的**全部 35 层**
PLE 贡献恰好是**一整行连续的 17.9 KB**。

- ✅ 正确做法：每 token 一次 `read_bytes` 读整行到 scratch（8960 × fp32 = 35 KB），
  35 层各取自己的 256 维切片
- ❌ 错误做法：每层单独 pread 512 B × 35 次 → 35 倍 syscall，且都是同一行的不同偏移

**不需要重排张量**。若改成 `[35, 262144, 256]`，每层内部连续但每 token 变成 35 次跨步读，
对实际访问模式**更差**。（这一点与我先前的口头判断相反，以本条为准。）

待实测：4.7 GB 表上随机取行，Android UFS 的延迟与 page cache 局部性。
可能需要行级小 LRU（`ExpertStore` 的 Slot/LRU 可泛化，但当前 `make_key` 是
`layer<<20|expert`，`expert_store.h:182-184`）或 `POSIX_FADV_WILLNEED` 预取。

### R4 — `layer_scalar` 与 PLE 组合公式不可猜

见 §2.5。必须读 transformers 的 `Gemma4TextModel` / `Gemma4TextLayer` 参考实现，
逐算子对齐。**建议第一步就做 `make_fake_gemma4_model.py` + `align_fake_gemma4_model.py`**，
用随机小模型把公式钉死，再碰真权重。

### R5 — 后端接口波及

若给 attention 加 window 参数，`IBackend`（`runtime/backend.h`）的 CPU / CUDA / Vulkan
三个实现都要跟着改。缓解：先用「CPU-only + ref fallback」圈住范围，
Vulkan/CUDA 留 stub。

### R6 — 无 JNI 层

如果最终目标是 App 内集成而非 `adb shell` 跑分，JNI 是完全空白的额外工程。
当前 `端云结合/android/app/src/main/cpp/jni_bridge.cpp` 只有一个返回字符串的桩。

---

## 6. 建议分期

| 阶段 | 内容 | 产出 | 依赖 |
|---|---|---|---|
| **0** | 读 transformers `Gemma4TextModel` 参考实现，钉死 `layer_scalar` 与 PLE 组合公式；写 `make_fake_gemma4_model.py` + `align_fake_gemma4_model.py` | 随机小模型上 C++ 与 HF logits 逐位置对齐（TOL=1e-5） | 无 |
| **1** | V4Ext 头 + `ModelConfig` 字段 + `kGemma4` 枚举 + exporter `export_gemma4_to_tiny.py` | 能把真权重导出成 `.tqwen` 并通过 192B 头校验 | 阶段 0 |
| **2** | 张量绑定分支 + gemma4 forward（**先全 full-attention、不共享 KV、PLE 常驻**） | 真模型 greedy decode 出正确文本 | 阶段 1 |
| **3** | sliding window attention 变体 + 逐层 head_dim + KvCache 两套 arena | 混合注意力正确，投机解码回归通过 | 阶段 2 |
| **4** | KV 共享（`num_kv_shared_layers=20`）+ logical→physical 间接表 | 内存下降，数值不变 | 阶段 3 |
| **5** | **PLE 流式**：`is_offloadable` 泛化 + tied 守卫 fail-fast + `read_bytes` 整行读 | 常驻 RAM 从 ~2.6 GB 降到 ~1.3 GB | 阶段 2（可与 3/4 并行） |
| **6** | i4 量化导出 + Android 交叉编译 + 目标机型实测 | 端侧可用 | 阶段 5 |

**为什么阶段 0 必须最先做**：`layer_scalar` 和 PLE 公式猜错的后果是
「能跑、不崩溃、输出垃圾」，与 R2 同类。先在随机小模型上对齐，
是唯一能把这类错误变成显式失败的办法。

**为什么 PLE 流式排在阶段 5 而不是更早**：它依赖阶段 2 的正确 forward，
且本身改动量小（有先例）。先让模型算对，再优化内存。

---

## 7. 待确认问题

1. **以哪个 clone 为准？** `~/code/tiny-llm`（分支 `codex/speculative-decoding`，153 commits，
   HEAD `6d8427b`，含 101 个未跟踪的实验产物）vs `~/Desktop/code/tiny-llm/tiny-llm`
   （分支 `main`，139 commits，HEAD `e382c1b`）。已确认前者包含后者的 HEAD。
   当前 submodule 指向前者的 `6d8427b`。
2. **是否需要投机解码兼容 gemma4？** 现有 EAGLE3/dflash 基建是围绕 Qwen3-0.6B 的，
   gemma4 的 KV 共享 + 逐层 head_dim 会让 checkpoint/restore/truncate 复杂化。
   建议阶段 3 先只保证「不破坏 Qwen 的投机解码」，gemma4 的投机解码另立议题。
3. **是否需要 App 内集成（JNI）？** 若需要，是独立于本设计的额外工程。
4. **vision/audio 塔是否彻底不导出？** 本项目无 Processor、纯文本路径，
   建议 exporter 直接跳过，省 0.476B 参数（int4 约 0.22 GB）。
   但要注意 `embed_vision` / `embed_audio` 的 `embedding_projection` 是否被文本路径引用
   （已核实：它们是 `[1536,768]` / `[1536,1536]` 的独立投影，文本路径不需要）。
5. **detokenize 是否需要进引擎？** 当前只在 Python 侧解码。App 内集成时必须解决。

---

## 8. 复用清单（可直接拿来用的）

| 资产 | 位置 | 用途 |
|---|---|---|
| `ExpertStore::read_bytes` | `runtime/expert_store.{h,cpp}` | PLE 整行 pread |
| embed 单行卸载模式 | `qwen_forward_token.cpp:118-148`、`qwen_forward_prefill.cpp:180-190,561-570` | PLE 流式的直接模板 |
| `embed_file_offset_` 绑定 | `qwen_model.cpp:508-528` | PLE offset 绑定 |
| i4 kernel 全家族 | `kernels/matvec/matvec_i4_*` | gemma4 所有线性层，零新 kernel |
| `dequant_i4_row` | `qwen_model.h:83`、`tools/quantize_embed_i4.py` | PLE 行按 i4 反量化 |
| dispatch 注册表 | `kernels/dispatch.h:326,342-354` | 加 attention window 变体 |
| 对齐脚本骨架 | `tools/align_fake_qwen35_model.py`（`run_cpp`:65 + `main`:117） | 抄成 gemma4 版 |
| `write_tqwen` | `tools/export_qwen_to_tiny.py:290` | 通用写文件，可直接复用 |
| Android 交叉编译 | `scripts/build_android.sh`（arm64-v8a、android-28、NDK 四级探测） | 产物 `build-android/runtime/tinyqwen` |
| 冒烟流程 | `scripts/add_model.sh`（导出→192B 头校验→分词→生成→解码→token 多样性断言） | 加 gemma4 case |

## 9. 需要新写的

| 文件 | 内容 |
|---|---|
| `runtime/tiny_format.h` | `kGemma4=4`、V4Ext 结构、`kFormatVersion=4` |
| `runtime/model_loader.{h,cpp}` | `ModelConfig` 新字段 + 校验 + `is_offloadable` 泛化 + tied fail-fast 断言 |
| `runtime/qwen_model.{h,cpp}` | gemma4 绑定分支、`LayerWeights` 扩展（4 个 norm、PLE 三件套、layer_scalar） |
| `runtime/gemma_forward_*.cpp` | decode + prefill（或在现有 forward 加分支） |
| `runtime/kv_cache.{h,cpp}` | 两套 arena + logical→physical 映射 |
| `kernels/attention/attention_decode_window_*.cpp` | 带 window-start 的变体 |
| `tools/export_gemma4_to_tiny.py` | 或扩展 `detect_model_type`(:537，目前非 qwen 直接 `sys.exit`) + `plan_tensors`(:682) |
| `tools/make_fake_gemma4_model.py` | 随机小模型，双格式导出 |
| `tools/align_fake_gemma4_model.py` | logits 逐位置对齐 |
| `model_gemma4_*.yaml` | 注册表（`tools/model_registry.py` 读取） |
| `scripts/add_model.sh` | 加 gemma4 case（目前硬编码调 `export_qwen_to_tiny*.py`） |
