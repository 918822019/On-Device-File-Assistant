package com.example.filememoryagent.runtime.data

import android.content.Context
import java.io.File

/**
 * 端侧数据目录的唯一入口。
 *
 * 所有模型、向量索引、元数据和运行日志都放在 noBackupFilesDir。
 */
class EdgeDataPaths(context: Context) {
    private val root = context.noBackupFilesDir
    val models: File = File(root, "models")
    val vectors: File = File(root, "vectors")
    val metadata: File = File(root, "metadata")
    val state: File = File(root, "state")
    val logs: File = File(root, "logs")

    fun ensureDirectories() {
        listOf(models, vectors, metadata, state, logs).forEach { directory ->
            check(directory.exists() || directory.mkdirs()) {
                "无法创建端侧数据目录: " + directory.absolutePath
            }
        }
    }

    fun modelDirectory(modelId: String): File = File(models, modelId)
    fun vectorDirectory(indexId: String): File = File(vectors, indexId)
    fun manifestFile(indexId: String): File = File(state, indexId + ".manifest.json")
}
