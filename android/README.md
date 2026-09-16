# Android APK 骨架（文件记忆找回 Agent）

本目录提供一个可运行的 Android 客户端 MVP，目标：

- 连接现有 Python 服务 `/v1/search-agent/search`。
- 支持 `search -> clarify -> execute` 的第一阶段闭环。
- 包含前台 Service（`EdgeRuntimeService`），用于后续接入端侧模型推理。
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
- 索引持久化到 `app/filesDir/local_file_index.json`。
- 文件打开动作支持 content/http/file 路径，无法直接打开时自动复制 URI 到剪贴板。
- 运行时存储权限申请与不足引导。

## 与后续端侧服务对齐点

- `EdgeRuntimeService` 现在是前台服务桩，后续可在这里替换为
  - 文件监听（media store / content observer）
  - 本地模型推理请求（tiny llm 与 embedding）
  - 与云端路由策略的降级链路

## 日志系统

- 页面底部“日志”：
  - 会显示 API 调用链路与前端动作日志。
  - 可快速确认 `search / clarify / execute / rebuild` 是否成功提交和返回。
- “端侧运行日志”：
  - 记录服务创建、扫描触发、扫描结果、权限告警和异常。
  - 可快速确认 `MediaStore` 是否在持续触发重建。
- 持久化日志文件：
  - 文件名：`runtime_events.log`，位于 `app/filesDir`。
  - 发生在应用内部存储，适合抓日志排障。

日志事件示例：
```text
[14:23:10] [edge_runtime] scan_done | reason=service_start scanned=20 upserted=20 unchanged=18 removed=2 dur_ms=120
[14:23:13] [edge_runtime] scan_failed | reason=manual msg=...
```
