# gemma4 decode 路径 CPU 算子逐位算术清单(Vulkan fp32 bitwise 复现用)

> 调研对象:`third_party/tiny-llm`,入口 `runtime/qwen_forward_gemma4.cpp`。
> 验证方法:源码 + **build-gemma4 实际产物反汇编**(objdump,Apple clang 21.0.0,arm64-apple-darwin25.3.0,Release `-O3 -DNDEBUG`,无 `-ffast-math`、无显式 `-ffp-contract`)。
> 配置基线:`--ops-impl neon --matvec-impl sdot6_mt`(gemma4 = i4 权重 + f16 tied embed/lm_head)。

## 0. 全局编译事实(影响所有算子)

1. **编译器 contraction 是开启的**(clang 默认 `-ffp-contract=fast/on`)。反汇编证实:凡是源码里写成标量表达式 `a*b+c` 的地方(ref 系 kernel、forward 里的 PLE 合并),二进制里都是 `fmadd/fmla`(单次舍入);而 **NEON intrinsic 路径(vmulq/vfmaq/vsubq)不会被再融合**,保持指令级语义。
   ⇒ 「逐位复现」的目标必须是**编译产物的行为**,不是源码文本的行为。两者在 ref 系 kernel 上不一致。
2. 超越函数全部来自 **Apple libm**:`powf`、`___sincosf_stret`(sin/cos 成对融合调用,不是独立 sinf/cosf!)、`expf`、`tanhf`。`sqrt`/除法用 IEEE 正确舍入指令 `fsqrt`/`fdiv`(与 GPU 一致)。**没有任何 rsqrt 近似/牛顿迭代**。
   ⚠️ **2026-09-20 更新(tiny-llm 4227fad)**:gemma4 golden 路径的 `expf`/`tanhf` 已替换为 vendored 的 `tq_expf/tq_expm1f/tq_tanhf`(kernels/math/portable_math.*,与 GLSL 转录同源,见 §10 路线 b 落地记录);本条的 Apple libm 描述适用于替换前的产物,`powf/sincosf`(rope 表,host 数据)仍走平台 libm。contraction 依赖点(gemma4 golden 路径)已全部钉死为显式 `std::fmaf`:attention l 更新、geglu/gelu inner、PLE 合并、proportional_rope——§3/§4/§5 的「编译实际」形态现在是**源码显式保证**,不再依赖编译器行为(R7 在这些点消解)。
3. `vaddvq_f32` 的 lowering 已从反汇编确认为 `faddp.4s v,v,v` + `faddp.2s` = **`((l0+l1)+(l2+l3))`**(成对树形,不是顺序)。
4. gemma4 全部维度:hidden=1536、head_dim=256(sliding)/512(full)、ple_dim=256、inter=6144/12288、lm_head in_dim=1536。**全是 16 的倍数** ⇒ rmsnorm_neon / attention dot_neon / rope_neon 主循环、matvec_f16 的 32 元素主循环都**恰好整除,所有标量尾段在 gemma4 上永不执行**(尾段里存在与主循环不同语义的 fmadd,可整体忽略)。
5. 若 golden 换平台(Android NDK / Linux)重采:libm 与 contraction 默认值都会变,本清单需按新产物重新反汇编核对。

---

## 1. rmsnorm

### ref:`kernels/rmsnorm/rmsnorm_ref.cpp` — `rmsnorm_ref`(L47)
- (b) 运算序列:
  ```
  sumsq(double) = 0
  for i in 0..n-1:  xi = (double)x[i];  sumsq += xi*xi      // L52-56,double 顺序累加
  mean_sq = (float)(sumsq / n)                              // L60,double 除法→float 舍入
  rms     = sqrtf(mean_sq + eps)                            // L65,fsqrt 正确舍入
  scale   = 1.0f / rms                                      // L69,fdiv 正确舍入
  y[i]    = (x[i] * scale) * weight[i]                      // L73-76,两次独立 fmul,无 FMA
  ```
- (c) 归约:**double 顺序累加**。编译产物把平方用 SIMD 算(`fcvtl`+`fmul.2d`,float→double 平方**在 double 里精确无舍入**),但 `fadd d1` 链严格按元素顺序 → 与源码顺序累加逐位一致;标量尾段 `fmadd d` 因乘积精确也与源码等价。
- (d) FMA:尾段 `fmadd`(数值上无害,见上)。
- (e) 超越函数:`sqrtf`(正确舍入,平台无关)。
- (g) GLSL:需要 **fp64 顺序累加**(或 double-float 模拟)。但 `--ops-impl neon` 下**不会被选中**,仅为 ref golden 场景。

### neon:`kernels/rmsnorm/rmsnorm_neon.cpp` — `rmsnorm_neon`(L47,注册名 "neon",L111)
- (b) 运算序列(全 fp32):
  ```
  s0..s3 = vec4(0)
  主循环(16 元素/轮,L54-63): s_c = vfmaq_f32(s_c, v, v)   // FMA:acc += x²(乘积不单独舍入)
  ssum = (s0+s1) + (s2+s3)                                  // L66,lane-wise;编译为 (s1+s0),(s3+s2) 再相加——加法可交换,逐位等价
  [4 元素循环 L68-70 / 标量尾 L74-76:gemma4 不执行]
  sumsq = vaddvq_f32(ssum) = ((l0+l1)+(l2+l3))              // L73
  mean_sq = sumsq / (float)n                                // L81,fp32 fdiv(注意:与 ref 的 double 除法不同)
  rms = sqrtf(mean_sq + eps); scale = 1.0f / rms            // L83-86
  pass2(4 元素/轮,L94-99): y = vmulq(vmulq(x, scale), weight)   // 两次乘法,顺序同 ref
  ```
- (c) 归约:4×vec4 累加器,`s_c` 的 lane j 累加 `i ≡ 4c+j (mod 16)` 的元素;合并 `((s0+s1)+(s2+s3))` lane-wise,再横向 `((l0+l1)+(l2+l3))`。
- (d) FMA:**显式** `vfmaq_f32`(pass1);pass2 纯乘法。
- (e) `sqrtf`、`fdiv`(均正确舍入)。
- (f) **ref 与 neon 不逐位一致**:double 顺序累加 vs fp32 FMA 4 路树形累加 + `mean` 的除法精度不同(~1e-7 相对差,源码头注 L20-23 亦承认)。
- (g) **GLSL:容易**。fp32 全程;复现 4×vec4 accumulator 的跨步分配、merge 顺序、`((l0+l1)+(l2+l3))` 横向加即可;`precise` 限定 + 显式 `fma()`;sqrt/div 用 IEEE 语义的 `sqrt()`/除法(**不要用 inverseSqrt**)。无尾段。

---

## 2. rope(default,sliding 层;forward L376 经 backend_->rope)

### ref:`kernels/rope/rope_ref.cpp` — `rope_ref`(L57)
- (b) 表(每次调用重算,**无缓存**,L74-85):
  ```
  for i in 0..half-1:
    exponent = -(float)(2i) / (float)head_dim          // fdiv,正确舍入
    inv_freq = powf(theta, exponent)                   // Apple libm powf(float 重载!)
    angle    = (float)pos * inv_freq                   // fmul
    (sn[i], cs[i]) = sincosf(angle)                    // 编译为 ___sincosf_stret 成对调用
  ```
- apply(L92-105,rotate_half 配对 (i, i+half)):
  - 源码:`r0 = x0*c - x1*s; r1 = x1*c + x0*s`
  - **编译实际(反汇编 0x308-0x348,8 对/轮向量循环)**:
    ```
    r0 = fma(x0, c, -fl(x1*s))     // 先 fmul(x1,-s) 舍入,再 fmla 精确加 x0*c
    r1 = fma(x1, c,  fl(x0*s))     // 先 fmul(x0,s) 舍入,再 fmla 精确加 x1*c
    ```
    即 contraction 选择了「x·c 融合、x·s 先舍入」的形态。
- (f)/(g):`--ops-impl neon` 下不选 ref;若要对齐 ref golden,需按上面的**融合形态**写 GLSL(fma + 预舍入乘),而非源码文本形态。

### neon:`kernels/rope/rope_neon.cpp` — `rope_neon`(L50,注册名 "neon",L121)
- (b) 表与 ref **逐字相同**(同 powf + sincosf,L60-69)⇒ 表值逐位一致。
- apply 主循环(4 对/轮,L78-89,gemma4 half=128/256 全走此路):
  ```
  r0 = vsubq_f32(vmulq_f32(x0,c), vmulq_f32(x1,s))   // fl(fl(x0*c) - fl(x1*s)),非融合!
  r1 = vaddq_f32(vmulq_f32(x1,c), vmulq_f32(x0,s))   // fl(fl(x1*c) + fl(x0*s))
  ```
  intrinsic 直译为 `fmul.4s×4 + fsub.4s + fadd.4s`(反汇编确认,无 fmla)。
- 尾段(half%4≠0 时,L92-100)是标量源码表达式 → 编译被 contraction 成 fmla/fmadd;**gemma4 不执行**。
- (f) **ref(编译态)与 neon 不逐位一致**:FMA 融合 vs 乘加分离(文件头注 L18-20 自己就点明了这一风险)。
- (g) **GLSL:容易**(旋转本体)+ **host 预计算表**。`precise` 防驱动重结合;cos/sin 表建议在 CPU 用与 rope_ref 完全相同的 `powf/sincosf` 代码逐 pos 生成后上传(表只依赖 (theta, pos),每 token 2 组:theta=1e4),彻底绕开在 shader 里移植 powf/sincosf。

---

## 3. proportional_rope(full 层;forward L384 dispatch 直连)

### `kernels/rope/proportional_rope_ref.cpp` — `proportional_rope_ref`(L56)
- **唯一实现**:`proportional_rope_registry` 从未注册任何变体(全仓 grep 无 `TINYQWEN_PROPORTIONAL_ROPE_VARIANT` 使用),dispatch 通用入口(`kernels/dispatch.cpp:886`)查表落空后**直调 ref**。与 `--ops-impl` 无关。
- (b) 表(L88-95,仅 `rope_angles` 项,补零维度恒等通过、被化简跳过——文件头注 L23-29 论证了恒等性):
  ```
  exponent = -(float)(2i) / (float)head_dim    // ⚠️ 分母是 head_dim(512),不是 rotary_dim
  inv_freq = powf(theta=1e6, exponent)
  angle    = (float)pos * inv_freq
  (sn,cs)  = sincosf(angle)
  ```
  `rope_angles = int(double(prf) * hd) / 2`(forward L382-383,double 乘法后取整)。
- apply(L104-111):只旋转 `i < rope_angles` 的 (i, i+half) 对;`[rope_angles, half)` 原样保留。
  **编译实际**:与 rope_ref 完全同构的向量循环 + FMA 融合:
  ```
  r0 = fma(x0, c, -fl(x1*s));  r1 = fma(x1, c, fl(x0*s))
  ```
  rope_angles=64(真模型 full 层)%8==0 ⇒ 全走融合向量循环。
- (d) FMA:**编译器 contraction 注入**(源码没写)。
- (f) ⚠️ **关键不对称**:`--ops-impl neon` 下,sliding 层(rope_neon)是**非融合**乘加,full 层(proportional_rope_ref 编译态)是**融合** FMA——GPU 侧两种层必须用**不同的旋转算术**!
- (g) **GLSL:容易**——显式 `fma(x0, c, -(x1*s))` / `fma(x1, c, x0*s)`(内层 `x1*s` 先单独舍入)+ host 预计算表(theta=1e6)。

---

## 4. attention_decode(forward L415,dispatch 直连;sliding 窗口用基址平移 L405-416,纯寻址无算术)

### ref:`kernels/attention/attention_decode_ref.cpp` — `attention_decode_ref`(L57)
- (b) 每 q-head 独立,online softmax 沿 t **顺序**扫描:
  ```
  m=-inf, l=0, oh[hd]=0
  for t in 0..seq-1:
    dot(double) = Σ_i (double)q[i] * k_t[i]        // L100-105,double 顺序累加(乘积精确)
    s = (float)dot * scale                          // L107;gemma4 scale=attention_scaling(硬编码 1.0,×1.0 逐位不变但保留该 fmul)
    m_new = max(m, s)                               // fcsel
    rescale = expf(m - m_new);  p = expf(s - m_new) // L114-116,Apple libm expf,每 t 两次
    oh[i] = fl( fl(oh[i]*rescale) + fl(p*vt[i]) )   // L120-127;编译为 fmul.4s,fmul.4s,fadd.4s(向量化但未融合)
    l = fl( fl(l*rescale) + p )                     // L130-131;编译为 fmul+fadd,未融合(反汇编 0x400-0x404)
  inv_l = 1.0f / l;  oh[i] *= inv_l                 // L136-140,fdiv + fmul(编译为 fmul.4s)
  ```
- (e) `expf` ×2/位置(参数恒 ≤0)。
- (g) double 顺序点积 → 需 fp64 或模拟;`--ops-impl neon` 下不选,仅 ref golden 场景。

### neon:`kernels/attention/attention_decode_neon.cpp` — `attention_decode_neon`(L92,注册名 "neon",L277)
- (b) 结构同 ref,三处不同:
  1. **dot_neon**(L53-81):4×vec4 fp32 FMA 累加器,16 元素/轮 `a_c = vfmaq_f32(a_c, q块, k块)`;merge `(a0+a1)+(a2+a3)` lane-wise;`vaddvq = ((l0+l1)+(l2+l3))`。hd=256/512 整除 16 ⇒ 无尾段。`s = dot * scale`(fmul)。
  2. **oh 更新**(L140-147):`o = vmulq(oh, rescale); o = vfmaq(o, p, vt)` ⇒ `oh' = fl( fl(oh*rescale) + exact(p*vt) )` —— 第二项 **FMA 融合**(与 ref 的 mul+add 不同)。
  3. **l 更新**(L155):`l = l*rescale + p` 标量表达式,**编译为 `fmadd`(融合)**(反汇编 0x1ac);ref 是 fmul+fadd 两次舍入。
- 归一化(L162-170):`inv_l = 1/l`(fdiv),`oh *= inv_l`(fmul.4s)。
- (e) `expf` ×2/位置,同 ref(但输入 s 已因 dot 不同而不同)。
- (f) **ref 与 neon 不逐位一致**(dot 累加域/结构、oh 融合、l 融合三处)。
- (g) **GLSL:最难的一个**。
  - dot:容易(复现 4×vec4 FMA 跨步 + merge + `((l0+l1)+(l2+l3))`)。
  - online softmax 对 t **本质顺序**(每步 rescale 依赖上一步的 m/l/oh)⇒ GPU 必须每 head 一个线程(或 workgroup)沿 t 串行循环复现,不能改成并行两遍 softmax(那样舍入路径完全不同)。
  - `expf`:~~必须移植 Apple libm expf~~ **已落地 §10 路线 b**:CPU 与 shader 同源 `tq_expf`(4227fad),golden 在替换后采集。shader 见 runtime/vulkan/gemma4_attention.comp(每 q-head 一 invocation,沿 t 串行)。
  - f16kv 变体(L182-273)gemma4 不用(forward L167-172 显式 abort `--kv-f16`)。

---

## 5. geglu / gelu_tanh(forward L454 / L482,dispatch 直连)

### `kernels/gelu/geglu_ref.cpp` — `geglu_ref`(L60)、`kernels/gelu/gelu_ref.cpp` — `gelu_tanh_ref`(L52)
- **无 neon 变体**(gelu_registry/geglu_registry 无任何注册),dispatch(dispatch.cpp:925/936)恒落 ref ⇒ 这两个 ref **就是实际运行实现**。
- (b) 源码:
  ```
  v3   = (v*v)*v                                  // 左结合,两次 fmul
  inner = v + C*v3                                // C = 0.044715f
  t    = tanhf(K * inner)                         // K = 0.7978845608028654f(float 常量 = fl(√(2/π)))
  y    = (0.5f*v) * (1.0f + t)                    // 左结合
  geglu: gate[i] = y * up[i]                      // 额外一次 fmul
  ```
- **编译实际**:`inner` 被 contraction 融合为 **`fma(C, v3, v)`**(标量尾 `fmadd s1,s1,s8,s0`;向量循环 `fmul.4s×3 + fmla.4s`)。其余保持分离:`fmul`(K·inner) → `bl tanhf`(向量循环里也是逐元素标量调用)→ `fadd`(1+t)→ `fmul`((0.5v)·(1+t),0.5v 提前算)。
- (d) FMA:编译器注入于 inner;源码警告(L14-20 / gelu_ref 头注):**运算顺序不可重排**,重排引入 ~6e-3 偏差。
- (e) `tanhf`(Apple libm)。
- (f) ref=唯一实现,无 ref/neon 分歧问题。
- (g) **GLSL:需移植 tanhf 多项式**;其余容易——`precise` + `fma(C, (v*v)*v, v)`,常量按 float 位模式写死(0.044715f、0.7978845608f 的精确 fp32 位模式),乘法顺序照抄。

---

## 6. matvec_f16(lm_head;forward L527-531)

### 实际选择路径(--ops-impl neon --matvec-impl sdot6_mt,gemma4 = i4 文件 + f16 tied embed/lm_head)
- `runtime/edge_setup.cpp:12` `select_matvec_registries`:模型 dtype=kI4 → **is_i4 分支**(L41-77):
  - i4 注册表 ← `sdot6_mt`(q/k/v/o/gate/up/down/ple 投影全走 i4,不在本清单范围,见 Phase 1);
  - f32 注册表:`sdot6_mt` 不存在 → 回退 `neon_mt_kv_nt`;
  - **f16 注册表:强制 `set_matvec_f16_impl_by_name("neon_mt_kv_nt")`(L69-71,注释明说防「lm_head 落 ref 的 12 倍静默减速」)**。
- f16 注册表只有两个名字:`"ref"`(matvec_f16_ref.cpp:54)、`"neon_mt_kv_nt"`(matvec_f16_neon_mt_kv_nt.cpp:658;`"neon_mt"` 在 f16 表**不存在**,edge_setup 的二级回退在 aarch64 上不会触发)。
- dispatch 通用入口 `kernels/dispatch.cpp:233` 用 `g_f16_current` ⇒ **lm_head 实际执行 `matvec_f16_neon_mt_kv_nt`**。

### ref:`kernels/matvec/matvec_f16_ref.cpp` — `matvec_f16_ref`(L36)
```
每行 o: acc(double)=0
  for i: acc += double(half_to_float(w[i])) * x[i]     // L45,f16→f32→double 均精确,乘积在 double 精确
  y[o] = (float)acc                                     // L48
```
编译产物:SIMD 位运算做 f16→f32(与 half_to_float 语义一致)、`fcvtl`→`fmul.2d`(精确)→**顺序** `fadd d` 链(已核对反汇编加法顺序=元素顺序)。归约:**double 顺序**。GLSL 复现需 fp64 —— 但本配置下不选它。

### neon:`kernels/matvec/matvec_f16_neon_mt_kv_nt.cpp` — `dot_row_f16`(L69)/ `dot_row_f16_nt`(L155,LDNP 版,算术相同)/ 入口 `matvec_f16_neon_mt_kv_nt`(L568)
- (b) 每行(in_dim=1536=48×32 ⇒ **只走 32 元素主循环,无任何尾段**):
  ```
  acc0..acc3 = vec4(0)
  每轮 i += 32(L85-98):
    acc0 = fma(f32(w[i+0 :4]), x[i+0 :4], acc0);  acc0 = fma(f32(w[i+4 :8]), x[i+4 :8], acc0)
    acc1 = fma(f32(w[i+8 :12]),x[i+8 :12],acc1);  acc1 = fma(f32(w[i+12:16]),x[i+12:16],acc1)
    acc2 = …(i+16..23,两次)   acc3 = …(i+24..31,两次)
    // f32(w) = vcvt_f32_f16,精确;每链每轮按 low4→high4 顺序做两次 FMA(顺序敏感!)
  sum01 = acc0+acc1; sum23 = acc2+acc3; t = sum01+sum23      // L116-118,lane-wise
  total = vaddvq_f32(t) = ((l0+l1)+(l2+l3))                  // L119
  ```
  即 `acc_c` 的 lane j 累加 `i ≡ 8c+4·{0,1}+j (mod 32)`,链内按块顺序 FMA。
- (c) 归约:4 链 × vec4,链内**顺序 FMA**(每轮 2 次/链),链间 `(0+1)+(2+3)`,横向成对。
- (d) FMA:显式 `vfmaq_f32`,fp32 累加。
- 16B 对齐分支(L211 `dot_row`):LDNP vs 普通加载,**算术逐位相同**;lm_head 行步长 3072B,对齐成立走 _nt。
- 多线程 RowPool(L535 起,阈值 L543=262144 元素;lm_head 402M 元素 ⇒ 多线程):**按行切分、行间独立 ⇒ 每行结果与线程数无关,确定性逐位可复现**。
- (f) **ref 与 neon 不逐位一致**(double 顺序 vs fp32 4 链 FMA 树;头注 L27-28 门禁容差 5e-3 也印证)。
- (g) **GLSL:容易~中等**。纯 fp32:每 workgroup/线程复现 4 条 vec4 FMA 链 + 相同跨步 + 相同 merge;f16→f32 用 `unpackFloat2x16`(精确);必须显式 `fma()` 且 `precise`。

---

## 7. argmax(`kernels/argmax/`)

- `argmax_ref.cpp:37` `argmax_ref`:单遍,严格 `>` 更新 ⇒ **平票取最小下标**(与 numpy/torch 一致,头注 L21-24)。
- `argmax_neon.cpp:47` `argmax_neon`(注册名 "neon",L99,`--ops-impl neon` 时选中):两遍法。pass1 `vmaxq`+`vmaxvq`(编译为 `fmaxv.4s`;max 是精确运算,归约顺序无关);pass2 顺序扫描找**第一个** `==m`(lane0 优先)⇒ 语义与 ref 完全一致。
- (f) **ref 与 neon 逐位/语义一致**(唯一无分歧的算子)。
- (g) **GLSL:容易**。有序归约保 (max, 最小下标) 即可。
- ⚠️ 注意 forward 的 top-k 路径(L553 `top_k_logits_gemma4`,用 `std::partial_sort`,不稳定排序):平票时 indices 顺序是实现定义的。逐位对拍请只走 `argmax` 路径(L556)。

---

## 8. dequant_i4_row(PLE i4 行反量化)

- `kernels/ref_ops.h:221-244` — inline `dequant_i4_row`(被 forward L273/L298 调用;布局同 tiny_format.h i4 packing)。
- (b) 每元素:
  ```
  组布局(小端): [scale_fp16(2B) | zero_fp16(2B) | packed_uint4(gs/2 B)],低 nibble 在前
  scale = half_to_float(scale_h); zero = half_to_float(zero_h)      // ref_ops.h:51,纯位操作,精确(含非规格化/inf/NaN)
  v = nibble(0..15)
  out = (float(v) - zero) * scale                                    // L239:scvtf 精确 → fsub → fmul
  ```
- (c) 无归约。(d) 无 FMA((a-b)*c 无可融合形态)。(e) 无超越函数。
- (g) **GLSL:容易**——整数位操作解 nibble + fp16→fp32(`unpackFloat2x16` 或手写位运算,均精确)+ `precise` 的 sub/mul。

---

## 9. dispatch 层(--ops-impl neon 时各算子落到谁)

- `kernels/dispatch.cpp:833` `set_ops_impl_by_name("neon")`:任一注册表含 "neon" 即接受(main.cpp:788 调用)。
- 通用入口按 `ops_impl_name()` 查表,查不到**直调 *_ref**:
  | 入口(dispatch.cpp) | registry 里的变体 | neon 时实际执行 |
  |---|---|---|
  | rmsnorm(L863) | ref 兜底 + "neon"(rmsnorm_neon.cpp:111) | **rmsnorm_neon** |
  | rope(L874) | "neon"(rope_neon.cpp:121) | **rope_neon** |
  | proportional_rope(L886) | **空**(无任何注册) | **proportional_rope_ref(编译含 FMA 融合)** |
  | attention_decode(L899) | "neon"(attention_decode_neon.cpp:277) | **attention_decode_neon** |
  | gelu_tanh(L925) | **空** | **gelu_tanh_ref(编译含 FMA inner)** |
  | geglu(L936) | **空** | **geglu_ref(同上)** |
  | argmax(L947) | "neon"(argmax_neon.cpp:99) | **argmax_neon** |
- backend 转发:`runtime/backend_cpu.cpp:109-131`(rmsnorm/rope/attention 均转 dispatch 通用入口);forward 里 attention/geglu/gelu/proportional_rope/argmax/matvec_f16 是 dispatch 自由函数直连。
- matvec f16 注册表:入口 dispatch.cpp:233;选择逻辑见 §6(edge_setup.cpp:69)。

### 9b. forward 内嵌的散装算术(不在 kernels,但 GPU 同样要逐位复现)
| 位置(qwen_forward_gemma4.cpp) | 运算 | 编译实际 |
|---|---|---|
| L200 embed scale | `hidden[j] *= es` | fmul,平凡 |
| L241 PLE P 分量 | `ple_ctx[j] *= ps` | fmul |
| **L310 PLE 合并** | `(ctx + tok*ts) * merge` | **`fmla.4s`:`fma(tok, ts, ctx)` 再 ×merge(FMA 融合!)** |
| L361-376 QK/V-norm | per-head rmsnorm(同 §1;V-norm weight=全 1,×1.0 精确) | 同 rmsnorm_neon |
| L430/L446/L470/L492 残差 | `hidden[j] += normed[j]` | fadd |
| L508 layer_scalar | `hidden[j] *= ls` | fmul |
| **L547 logit softcapping** | `tanhf(logit * (1/cap)) * cap` | fmul → `bl tanhf`(Apple libm)→ fmul;`inv = 1.0f/cap` host 算 |

---

## 10. 汇总表:算子 × bitwise 复现难度(GLSL precise)

| 算子(neon 配置下实际执行的实现) | 难度 | 说明 |
|---|---|---|
| rmsnorm(rmsnorm_neon) | **容易(需复现归约结构)** | 4×vec4 FMA + `((s0+s1)+(s2+s3))` + `((l0+l1)+(l2+l3))`;sqrt/div IEEE;无尾段 |
| rope(rope_neon,sliding 层) | **容易 + host 表** | 非融合 mul/mul/sub;cos/sin 表由 host 用同版 powf/sincosf 预计算上传 |
| proportional_rope(ref,full 层) | **容易 + host 表(但形态不同!)** | 必须按编译态融合形:`fma(x0,c,-fl(x1*s))` / `fma(x1,c,fl(x0*s))` |
| attention_decode(neon) | **shader 已写,待真机对拍** | dot 4 链 FMA + online softmax 沿 t 串行(每 head 一 invocation)已按本节结构复刻进 gemma4_attention.comp;expf = 同源 tq_expf(4227fad);l 更新已钉 fmaf。剩余风险 = Adreno 的 precise 除法/截断合规性(math_probe + attention parity 验证) |
| geglu / gelu_tanh(ref) | **需移植多项式** | `tanhf` 逐位;inner 必须 `fma(C,(v*v)*v,v)`;乘法顺序不可重排(源码自证 6e-3 风险) |
| softcapping(forward L547) | **需移植多项式** | 同 tanhf |
| matvec_f16(neon_mt_kv_nt,lm_head) | **容易~中等(需复现归约结构)** | 4 链×每轮 2 FMA、跨步 mod 32、`(0+1)+(2+3)`+成对横加;fp16→fp32 精确;MT 行独立不影响逐位 |
| argmax(neon) | **容易** | 平票取最小下标;max 精确 |
| dequant_i4_row | **容易** | 位操作 + (v-zero)*scale |
| PLE 合并(L310) | **容易** | 注意是 `fma(tok,ts,ctx)*merge`,非源码文本的两次舍入 |
| (对照)rmsnorm_ref / attention_ref / matvec_f16_ref | **需 fp64 或降级** | double 顺序累加;GLSL 无 fp64 时需 double-float 模拟(代价高)或放弃对 ref golden 的逐位、改对 neon golden |

### 落地建议
1. **golden 一律用 `--ops-impl neon --matvec-impl sdot6_mt` 的实机产物采集**(本文所有「编译实际」都以此为准);ref 路径含 double 累加,不建议作为 GPU 逐位目标。
2. expf/tanhf 两条路线:(a) 从 opensource.apple.com 的 libm 抠 expf/tanhf 多项式移植进 shader;(b) **推荐**:在 tiny-llm 里把 `std::exp/std::tanhf` 替换为内嵌的可移植实现,重采 golden——shader 与 CPU 共享同一份多项式源码,一劳永逸且跨平台(Android NDK 的 libm 与 Apple 不同,路线 (a) 换平台即失效)。
   **路线 (b) 已落地(4227fad,golden 采集之前 = 零重采成本)**,并修正两个原文判断:
   - bionic `expf` = ARM optimized-routines(bionic Android.bp:33 链 `libarm-optimized-routines-math`)——但 OR expf 内部走 **double** 路径(z/r/y 全是 double),移动 GPU 无 `shaderFloat64`,**无法照抄进 GLSL**;
   - bionic `tanhf` = FreeBSD `s_tanhf.c`(OR 没有标量 tanhf),内部调 `expm1f`。
   实际 vendor:`tq_expf` ← FreeBSD-11 `e_expf.c`(经典纯 fp32 Cephes 系),`tq_expm1f` ← bionic `s_expm1f.c`,`tq_tanhf` ← bionic `s_tanhf.c`(内部 expm1f→tq_expm1f)。CPU TU 以 `-ffp-contract=off` 编译,GLSL 全 `precise`,常量双侧位模式,逐语句对应——CPU↔GPU 逐位一致由构造保证,与平台 libm 解耦。真机 math_probe 对拍(test_vulkan_i4.cpp)验证 Adreno 侧前提(precise 除法 correctly-rounded、int() 截断向零)。host 实测:vs Apple libm ≤1ulp(expf/expm1f,96-98% 逐位同)、≤2ulp(tanhf);p2_mid 16 ids 替换前后逐位不变,p1_short 贪心翻转但连贯。
3. rope 表 host 预计算:每 token 上传 `[half]×2` float(theta=1e4 与 1e6 各一张),用 CPU 同代码路径生成 ⇒ powf/sincosf 完全不进 shader。
4. GLSL 侧纪律:全程 `precise` 限定;`fma()` 显式写;禁止依赖 `inverseSqrt`/驱动 fast-math;归约结构按本清单的 lane 跨步与合并顺序 1:1 摆放。
5. 平台迁移红线:换编译器/OS(如 Android NDK)后,ref 系 kernel 的 contraction 形态与 libm 都会变,须重新反汇编核对本文 §0/§2/§3/§5 的融合结论。
