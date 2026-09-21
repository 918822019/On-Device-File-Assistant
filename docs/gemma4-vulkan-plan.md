# gemma4-E2B i4 × Android Vulkan GPU 推理 —— 立项计划与状态

> 状态快照:2026-09-20。工作全部在 `third_party/tiny-llm`(分支 feat/gemma4-ple),
> 每阶段一组 commit + 数据回填其 `docs/optimization_log.md`。
> 逐位复现的算术依据:[gemma4-cpu-op-bitwise-inventory.md](gemma4-cpu-op-bitwise-inventory.md)。

## 已确认的范围决策

| 项 | 决策 |
|---|---|
| 载体 | CLI(adb push tinyqwen),不做 App/JNI 集成(另立项) |
| 真机 | OnePlus PLK110 / Android 16 / Adreno 840 / 16GB 统一内存 |
| 验收 | **先正确后性能**:每阶段 GPU generated_ids / slot dump 与 CPU golden 逐位一致;性能只记录,最终给完整对比表 |
| 模型 | 主 `gemma4-qat-q4_0-plei4.tqwen`(QAT 对称 i4 + PLE i4),副 `gemma4-e2b-i4.tqwen`(RTN 非对称,GPU v1 不支持,仅 CPU golden) |
| 技术路线 | **整段 forward 执行器**(仿 DFlashVulkanEngine,每 token 一次 submit);不走逐算子 backend_vulkan 扩展(实测负收益 2.2×,gemma4 35 层会放大到 400+ 次同步/token) |
| CPU/GPU 分工 | GPU:全部 i4 matvec + 逐层算子;CPU:prefill(数值锚点)、PLE 查表(--ple-ssd,每 token pread ~5.6KB + 上传 35.8KB)、softcapping/argmax(v1 回读 1MB logits) |
| golden 口径 | **真机 CPU i4(QAT+sdot6+neon ops)**,不是 f16(f16 与 i4 本有 2.25% 分叉,混用不可归因);ref 系算子是 double 累加,与 neon 逐位不同,一律采 neon 路径 |

## 关键设计决策

- **i4 GPU 布局**:不改文件格式、不加导出变体;init 时 host 重排(in-band 68B/组 → packed nibble 连续块 + fp16 位模式 scales 数组),`gemma4_vk_repack_i4` 为引擎/harness 共用实现
- **i4 kernel 算术**:逐位复刻 sdot6 的 W4A8(激活分组 amax→int8 对称量化;组内 int32 点积精确、顺序无关、subgroup 并行;跨组 fp32 **g 升序顺序累加**,每 lane 冗余同序;GLSL `precise` 防 contraction;手写 round-half-away-from-zero)
- **PLE**:保持 --ple-ssd 留盘 + 每 token 上传 [35,256] fp32(~10µs);PLE 表一字节不上 GPU。内存预算峰值 ~4.0–4.4GB / 16GB ✅
- **KV**:GPU 双池 + 跨层共享按 kv_cache 的 slot_map 拓扑;sliding 池环形(window 512,回绕语义与 CPU 一致)、full 池线性;head_dim 256/512 用 specialization 区分
- **prefill v1 留 CPU**(锚点 + initialize_from_cpu 导入 KV);GPU batched prefill 列计划外
- **rope**:cos/sin 表 host 每 token 用与 CPU 内核同一份 powf/sincosf 生成后上传(GPU 零超越函数);sliding=分离形、full=编译态融合形(清单 §2/§3,⚠️ NDK 产物须重新反汇编核对)
- **expf/tanhf**(attention/geglu 需要):Android bionic 与 ARM optimized-routines 同源,移植其多项式进 shader;不可逐位时触发降级检查点(提请拍板,不静默降级)

## 阶段与状态

| Phase | 内容 | 门禁 | 状态 |
|---|---|---|---|
| 0 | CPU i4 真机基线 + golden 采集(`scripts/collect_gemma4_golden.sh`:2 模型 × 3 prompt × {16,64} tok × 2 运行,含逐层 g4_dump、热门禁、meminfo) | golden 两次运行逐位可复现 | 脚本就绪(2b79b83),宿主机配方已验证;**待真机** |
| 1 | i4 W4A8 Vulkan kernel(quantize + matvec)+ host 封装 + 逐位对拍 harness(`tests/test_vulkan_i4.cpp`,9 档 gemma4 真实形状 + 残组;止损点 B 计时探针) | GPU vs CPU sdot6 fp32 逐位一致 | 真机首跑 6 失败已全部归因(3 harness bug:probe 相对偏移/arena 扩容清零/测试循环反向——均已修;2 驱动偏差 H1/H2——exact_arith 已落地;1 FTZ——tq_ftz 双侧对齐);止损点 B 数据已采(lm_head GPU 外推 22.7ms vs CPU 164.6ms ≈ **7.2×**;down_proj GPU 1.80ms vs sdot6 0.90ms/mt 3.57ms 且 mt p95 过热);**待重推真机复跑** |
| 2 | Gemma4VulkanEngine 骨架 v0(CPU 编排逐语句复制 forward + GPU i4 matvec)+ `--gemma4-vk` CLI 接线 + rmsnorm/rope bitwise shader(已写未接线) | v0 端到端 generated_ids 逐位 = CPU golden | **真机 PASS**(2026-09-21:v0 p1 ids 与 golden 逐位一致) |
| 3 | 专有算子 GPU 化(rmsnorm/rope 接线 → attention/geglu/elementwise shader)→ 35 层全链一次录制一次 submit | 全部 slot 10+i + 99 逐位一致;transcendental 不可逐位 → 降级检查点 | **代码全部落地**(exact_arith 后 shader 全绿 + 新增 elementwise/kv_write shader + kernels 引擎模式 API(engine_setup/rec_*/单 command buffer 录制+保守屏障)+ 引擎 v1(TINYQWEN_G4V=v1,35 层全 GPU 一次 submit,rope 表 host 逐字复制生成,theta+pos 键控缓存,延迟 KV 导入修 prefill 时序)+ engine_ops 四算子真机对拍用例);**真机 PASS**(2026-09-21:v1 e2e p1+p2 ids 与 golden 逐位一致,13cd595;含 H3 sqrt 判定+sqrt_exact、rmsnorm WorkGroupID 修复、KV 共享槽位化) |
| 4 | PLE 每 token 上传接通 + lm_head(f16,GPU 或留 CPU)+ softcap/argmax CPU 回读 | 端到端逐位 = Phase 0 golden | **已并入 v1 落地**:PLE 块全 GPU(ple_gate matvec+gelu mode1+slice 乘+ple_proj matvec),ple_ctx 每 token 上传;lm_head **刻意留 CPU**——止损点 B 数据:fma_exact 化后 GPU 外推 116.9ms > CPU nt_kv_mt 27.1ms,且省 805MB GPU 副本(f16 shader+API 保留,性能阶段再启用);softcap/argmax = CPU 回读(v0 同款代码);**真机 PASS**(随 v1 e2e 验证,2026-09-21) |
| 5 | main.cpp 正式集成(`--backend vulkan` 对 gemma4-i4 分支到执行器、内存预检计入 GPU 副本、verify_android.sh BACKEND 参数化) | host CTest + 真机 tests 全绿 + CPU 回归 | 最小接线已提前落地(--gemma4-vk);正式 gate 改造未开始 |
| 6 | bench_gemma4_vulkan.sh(CPU vs GPU 配对、热门禁、RSS/温度)+ optimization_log/android.md/本文档回填 | bench 内嵌逐位比对 | 未开始 |

## 风险与止损点

| # | 风险 | 探测时点 | 处置 |
|---|---|---|---|
| R1 | 带宽无优势(统一内存共享 LPDDR5X,decode 带宽 bound) | Phase 1 计时探针 | **止损点 B**:lm_head/down_proj 单 kernel GPU > CPU sdot6 且全层外推无优势 → 终止,数据回填 |
| R2 | bitwise 不可达(expf/tanhf/归约穷尽移植仍分叉) | Phase 3 | **已判定并消解**(真机实测):Adreno 840 对 ExtInst Fma(H1,拆成 mul+add)与 FDiv(H2,rcp+mul)**不合规**,其余原语(mul/add/sub/floor/ceil/int/unpackHalf2x16/subgroupAdd)合规;FTZ 于 subnormal。对策 = `exact_arith.comp.inc`(fma_exact/div_exact,只用合规原语构造 correctly-rounded FMA/除法),全部 shader 已替换,离线 489 万+ 样本 vs Fraction 黄金基准 0 失配;FTZ 双侧显式 tq_ftz 对齐。bitwise 由构造恢复,**止损点 A 不再需要**;待真机重跑 parity 最终确认(见清单 §10.1) |
| R3 | 单 buffer/分配上限 | Phase 2 | 按层分片 VkBuffer |
| R4 | GPU 持续降频 | Phase 6 | 报告冷热两态 |
| R5 | KV 共享拓扑复杂度超估 | Phase 2 | v1 收缩:仅 QAT 对称 + 固定 ctx + --ple-ssd(已在 create() gate 落地) |
| R6 | 激活量化舍入不一致 | Phase 1 | harness 先单独对拍 quantize(int8 逐位)再对拍整 matvec(已落地) |
| R7 | NDK contraction 与 Apple clang 不同 | Phase 0/3 | **golden 路径已消解**(4227fad):attention l 更新/geglu·gelu inner/PLE 合并/proportional_rope 的 contraction 依赖全部钉死为显式 `std::fmaf`(= host 反汇编验证的形态),编译器行为不再影响逐位语义;剩余范围仅非 golden 的 ref 变体 |

**总体止损**:R1/R2 触发且不接受降级 → 结论「GPU 路线不成立」,CPU i4(--ple-ssd + sdot6_mt)维持部署推荐;已完成 kernel/harness 作 experimental 保留,负结果回填 optimization_log。

**预期管理**:Adreno 840 无独立显存,decode 每 token 读 ~1.9GB 权重打同一条 LPDDR5X;GPU 收益来源是带宽利用率与并行 + 释放 CPU,合理目标 **1.0–1.6× vs CPU i4**,不是数倍加速。

## 文件地图(tiny-llm 侧)

- `runtime/vulkan/gemma4_{quant_x_i4,matvec_i4,rmsnorm,rope}.comp` — 已交付的 4 个 shader
- `kernels/math/portable_math.{h,cpp}` + `runtime/vulkan/portable_math.comp.inc` — tq_expf/tq_expm1f/tq_tanhf,CPU/GLSL 同源逐语句对应(-ffp-contract=off ↔ precise)
- `runtime/vulkan/gemma4_{math_probe,geglu,attention}.comp` — Phase 3 前置 shader(对拍探针 / geglu·gelu_tanh 双模式 / 每 head 串行 online softmax)
- `runtime/vulkan/gemma4_matvec_f16.comp` — lm_head f16 matvec(Phase 4 前置,4 链×mod-32 归约复刻;in_dim%32 gate)
- `runtime/gemma4_vk_kernels.{h,cpp}` + stub — i4 内核 host 封装(arena 寻址,repack 共用)
- `runtime/gemma4_vulkan.{h,cpp}` + stub — 引擎(v0 混合形态)
- `tests/test_vulkan_i4.cpp` — 逐位对拍 + 计时探针(仅 TINYQWEN_HAS_VULKAN 注册;含 math/geglu/attention 三个真机 parity 用例)
- `tests/test_portable_math.cpp` — vendored math 精度门禁(特殊值精确 + 扫描 ≤2ulp + 单调性 + vs 平台 libm 诊断)
- `scripts/collect_gemma4_golden.sh` — Phase 0 采集
- `runtime/main.cpp` — `--gemma4-vk` 最小接线
