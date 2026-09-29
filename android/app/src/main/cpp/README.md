# Native edge runtime

`libedge_runtime.so` is built by the Android Gradle/CMake build from the
`third_party/tiny-llm` runtime and kernel OBJECT libraries. It uses the shared
`select_matvec_registries()` setup, so INT4 Gemma's FP16 tied lm_head does not
fall back to a scalar kernel. The app targets API 28+ and also links the FP16 Vulkan DFlash graph.

JNI owns one session: CPU greedy with one `.tqwen` model, or Vulkan DFlash
with an FP16 target plus an FP16 draft (`width` 2..8, no wider than the
checkpoint). The DFlash graph uses CPU prefill and shared-GPU draft/verify
with logical KV rollback. It checks memory for two CPU copies plus GPU copies
before loading and returns draft/verify timings after generation. All calls
are serialized. The tokenizer is not part of the C++ runtime: callers must
pass token IDs produced by the exact target model's tokenizer.

The model is imported into `Context.noBackupFilesDir/models/model.tqwen` through
the Android document picker; no weights are bundled into the APK.
