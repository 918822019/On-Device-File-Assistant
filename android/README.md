# Android APK 骨架（文件记忆找回 Agent）

本目录提供一个可运行的 Android 客户端 MVP，目标：

- 连接现有 Python 服务 `/v1/search-agent/search`。
- 支持 `search -> clarify -> execute` 的第一阶段闭环。
- 包含本地文件监听 Service（`EdgeRuntimeService`），以及已链接 tiny-llm 的 JNI CPU / Vulkan DFlash 推理入口（token ID 模式）。
- 内置日志视图，便于串联 `trace_id` 与搜索动作。

## 目录

- `app/src/main/kotlin/com/example/filememoryagent/MainActivity.kt`：Compose UI 主界面。
- `app/src/main/kotlin/com/example/filememoryagent/network/PersonalSearchApi.kt`：Retrofit API 定义。
- `app/src/main/kotlin/com/example/filememoryagent/repository/SearchRepository.kt`：网络层与统一日志。
- `app/src/main/kotlin/com/example/filememoryagent/runtime/EdgeRuntimeService.kt`：前台服务模板。
- `app/src/main/kotlin/com/example/filememoryagent/ui/MainViewModel.kt`：交互状态机。
- `app/src/main/kotlin/com/example/filememoryagent/model/ApiModels.kt`：接口模型。
- `app/src/main/kotlin/com/example/filememoryagent/runtime/index/LocalFileIndexStore.kt`：本地文件索引存储/扫描。
- `app/src/main/res/xml/file_paths.xml`：FileProvider 路径。

## 本地运行

1. 启动你的 Python 后端（默认 `http://10.0.2.2:9000` 对应 Android 模拟器访问主机服务）。
2. 在 `android/app/build.gradle.kts` 中调整 `API_BASE_URL`（如内网 IP 或手机 IP）。
3. 用 Android Studio 打开 `android/` 目录并运行。 

默认会在界面提供：
- 输入模糊检索文本，点“搜索”；
- 进入追问时输入补充线索，点“继续确认”；
- 对候选执行 `打开/分享/归档/备注/对比` 动作。

已补充的端侧能力（不依赖模型）：
- 服务启动自动建索引 + 周期扫描（5 分钟）。
- 文件变更监听（MediaStore）触发去抖动增量扫描。
- 索引持久化到 `app/noBackupFilesDir/metadata/local_file_index.json`（不备份、卸载即删；旧 `filesDir` 位置自动迁移）。
- 文件打开动作支持 content/http/file 路径，无法直接打开时自动复制 URI 到剪贴板。
- 运行时存储权限申请与不足引导。

## 与后续端侧服务对齐点

- 文件监听由 `EdgeRuntimeService` 负责；端侧模型会话目前由 App 中的 `NativeRuntime` 单例持有。
- 后续仍需将分词/解码、自然语言聊天与服务层路由接入 JNI 模型会话；EmbeddingGemma 仍由 Python 服务提供。

## 日志系统

- 页面底部“日志”：
  - 会显示 API 调用链路与前端动作日志。
  - 可快速确认 `search / clarify / execute / rebuild` 是否成功提交和返回。
- “端侧运行日志”：
  - 记录服务创建、扫描触发、扫描结果、权限告警和异常。
  - 可快速确认 `MediaStore` 是否在持续触发重建。
- 持久化日志文件：
  - 文件名：`runtime_events.log`，位于 `app/noBackupFilesDir/logs/`（超 256KB 自动轮转，保留最近 300 行；旧 `filesDir` 位置自动迁移；分享时经 `cacheDir` 中转以适配 FileProvider）。
  - 发生在应用内部存储，适合抓日志排障。

日志事件示例：
```text
[14:23:10] [edge_runtime] scan_done | reason=service_start scanned=20 upserted=20 unchanged=18 removed=2 dur_ms=120
[14:23:13] [edge_runtime] scan_failed | reason=manual msg=...
```


## 本机 tiny-llm 推理（token ID 模式）

Android Vulkan 构建要求 minSdk 28、SDK CMake 3.22.1、NDK 27.2.12479018（含 glslc）：

```bash
cd android && ./gradlew :app:assembleDebug
```

APK 包含 arm64-v8a 的 `libedge_runtime.so`，通过 CMake 直接链接子模块 runtime +
kernel 注册器；无需将模型打入 APK。打开「端侧模型推理」→ 选取 `.tqwen` 文件
（例如 `models/tiny/gemma4-e2b-i4.tqwen`）→ App 流式复制到
`noBackupFilesDir/models/model.tqwen` 并加载。导入前需先把模型文件放到手机文件
选择器可访问的位置；不会从 `/data/local/tmp` 隐式读取或自动下载。

CPU greedy 使用 `QwenModel::forward_token`。Vulkan DFlash 可再导入同一目标模型训练出的
FP16 草稿 `.tqwen`（保存为 `noBackupFilesDir/models/draft.tqwen`），设置块宽 2..8，
由 JNI 完成 CPU prefill、同设备 proposal/target 验证及 KV 回退。内存不足或模型/后端
不匹配会拒绝加载；生成后显示 draft/verify 等阶段耗时。手机端分词器尚未接入，因此
输入框**只接受与目标模型同一 tokenizer 的 token IDs**，输出也是 IDs。
可先在开发机为相同 HF 模型生成 prompt IDs：

```bash
cd third_party/tiny-llm
.venv/bin/python tools/tokenize_prompt.py --model ../../models/google/gemma-4-E2B-it \
  --prompt "你好" --chat --out /tmp/gemma4_prompt.json
```

把 JSON 中的 `tokens` 列表复制到 App 输入框（逗号或空格分隔），点击「生成 32 token」。
文件搜索页仍连接 Python 服务，不使用此实验推理会话。
