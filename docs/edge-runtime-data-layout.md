# 端侧运行时与数据管理约定

端侧不运行 Python。Android 层负责生命周期、权限、文件发现和用户动作，
C++ 层负责 TinyLlm、EmbeddingGemma 和 FAISS。

## 数据目录

所有端侧生成物都放在 Context.noBackupFilesDir：

    noBackupFilesDir/
    |- models/gemma-e2b-int4/
    |- models/embeddinggemma-300m/
    |- vectors/personal-files/
    |- metadata/files.db
    |- state/personal-files.manifest.json
    `- logs/runtime_events.log

禁止把模型、FAISS、数据库和运行日志写入 Download、Documents、DCIM 或
公共存储根目录。卸载应用时，应用专属模型、索引、元数据和日志会随应用
数据一起删除，并且不会被 Android 自动备份后在重装时恢复。

## 生命周期

EdgeRuntimeService.onCreate 负责初始化 EdgeDataPaths、检查 native runtime、
按需加载模型和 FAISS、注册 ContentObserver。

EdgeRuntimeService.onDestroy 负责取消任务、注销观察者，并释放 C++ 模型、
线程池和 FAISS。停止服务不等于删除数据；卸载应用才删除应用专属数据。

## 索引关系

FAISS 不是主数据库。主数据至少保存 file_id、content_uri、content_hash、
size_bytes、date_modified、source_hint、mime_type、index_status、
embedding_model 和 embedding_dimension。

FAISS 只保存向量和内部位置映射。manifest 保存模型版本、向量维度、索引
版本和构建时间。FAISS 损坏或模型升级时，必须能从元数据重新构建。

## C++ 接入原则

- Kotlin 不依赖 TinyLlm、EmbeddingGemma 或 FAISS 的内部类型。
- JNI 只暴露 load、generate、embed、search、release 等稳定能力。
- native 层必须有明确的 release，停止服务时调用。
- 模型文件不存在时不能自动下载到公共目录，只能返回可观测错误。
- 模型加载、推理、向量化、索引读写都写入统一运行日志。
