package com.example.filememoryagent.runtime.index

import android.content.ContentResolver
import android.content.ContentUris
import android.content.Context
import android.database.Cursor
import android.provider.MediaStore
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.example.filememoryagent.runtime.logging.RuntimeEventLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.security.MessageDigest
import java.util.Locale

private const val TAG = "LocalFileIndexStore"
private const val MAX_INDEX_SIZE = 3000
private const val INDEX_FILE_NAME = "local_file_index.json"

/**
 * 本地文件索引条目（仅用于端侧监听/调度，不参与模型推理）。
 */
data class LocalIndexedFile(
    val fileId: String,
    val contentUri: String,
    val displayName: String,
    val mimeType: String,
    val sourceHint: String,
    val relativePath: String,
    val dateModified: Long,
    val sizeBytes: Long,
)

/**
 * 一次扫描结果汇总，用于服务日志与后续排障。
 */
data class LocalScanResult(
    val scanned: Int,
    val upserted: Int,
    val removed: Int,
    val unchanged: Int,
    val durationMs: Long,
    val reason: String,
)

/**
 * 端侧本地索引存储：
 * - 监听 MediaStore 变更事件。
 * - 定时/事件触发扫描。
 * - 在应用私有目录持久化索引 JSON（可用于服务重启后恢复）。
 */
class LocalFileIndexStore(context: Context) {

    private val appContext = context.applicationContext
    // 按 edge-runtime-data-layout.md 约定放 noBackupFilesDir/metadata/（不参与
    // 系统备份、卸载即删）。旧版本落在 filesDir，init 时自动迁移一次。
    // 目标形态是 metadata/files.db；当前先保持 JSON，格式升级另行排期。
    private val storeDir = File(context.noBackupFilesDir, "metadata")
    private val storeFile = File(storeDir, INDEX_FILE_NAME)
    private val legacyStoreFile = File(context.filesDir, INDEX_FILE_NAME)
    private val gson = Gson()

    private val _itemsById = LinkedHashMap<String, LocalIndexedFile>()

    fun snapshot(): List<LocalIndexedFile> {
        synchronized(_itemsById) {
            return _itemsById.values.toList()
        }
    }

    fun size(): Int {
        synchronized(_itemsById) {
            return _itemsById.size
        }
    }

    private fun loadFromDiskUnsafe(): MutableMap<String, LocalIndexedFile> {
        if (!storeFile.exists()) {
            return linkedMapOf()
        }
        return runCatching {
            val raw = storeFile.readText()
            val type = object : TypeToken<List<LocalIndexedFile>>() {}.type
            val saved: List<LocalIndexedFile> = gson.fromJson(raw, type)
            linkedMapOf<String, LocalIndexedFile>().apply {
                saved.forEach { item ->
                    put(item.fileId, item)
                }
            }
        }.getOrElse {
            RuntimeEventLog.log(appContext, "file_index", "load_from_disk_failed", storeFile.absolutePath)
            linkedMapOf()
        }
    }

    private fun persistUnsafe(items: Map<String, LocalIndexedFile>) {
        runCatching {
            storeDir.mkdirs()
            val text = gson.toJson(items.values.toList())
            storeFile.writeText(text)
            RuntimeEventLog.log(appContext, "file_index", "persist_ok", "count=${items.size} path=${storeFile.absolutePath}")
        }.onFailure {
            RuntimeEventLog.log(appContext, "file_index", "persist_failed", it.message.orEmpty())
            Log.w(TAG, "local-index-persist-failed", it)
        }
    }

    private fun inferSourceHint(path: String, mimeType: String): String {
        // path 与 mimeType 分别 lowercase，不再重复拼接 mimeType
        val normalizedPath = path.lowercase(Locale.getDefault())
        val normalizedMime = mimeType.lowercase(Locale.getDefault())
        val combined = normalizedPath + normalizedMime
        return when {
            combined.contains("wechat") || combined.contains("微信") || combined.contains("weixin") -> "wechat"
            combined.contains("mail") || combined.contains("邮箱") || combined.contains("gmail") -> "email"
            // dcim 是 Android 相机标准目录
            combined.contains("camera") || combined.contains("dcim") || combined.contains("photo") -> "camera"
            combined.contains("screenshot") || combined.contains("截图") -> "screenshot"
            combined.contains("download") || combined.contains("downloads") -> "downloads"
            combined.contains("documents") || combined.contains("文档") || combined.contains("docs") || combined.contains("doc") -> "document"
            combined.contains("pdf") || combined.contains("ppt") || combined.contains("excel") -> "office"
            // startsWith 只对 mimeType 本身有意义，拼接后的 combined 永远不会以 "image/" 开头
            normalizedMime.startsWith("image/") -> "gallery"
            normalizedMime.startsWith("video/") -> "video"
            else -> "local"
        }
    }

    private fun computeFileId(uri: String): String {
        val digest = MessageDigest.getInstance("MD5")
            .digest(uri.toByteArray())
            .joinToString(separator = "") { b -> String.format("%02x", b) }
        return digest.take(16)
    }

    private fun readString(cursor: Cursor, index: Int): String {
        return if (index >= 0) cursor.getString(index) ?: "" else ""
    }

    private fun readLong(cursor: Cursor, index: Int): Long {
        return if (index >= 0) cursor.getLong(index) else 0L
    }

    private fun collectFromFilesUri(resolver: ContentResolver): List<LocalIndexedFile> {
        val projection = arrayOf(
            MediaStore.Files.FileColumns._ID,
            MediaStore.Files.FileColumns.DISPLAY_NAME,
            MediaStore.Files.FileColumns.MIME_TYPE,
            MediaStore.Files.FileColumns.SIZE,
            MediaStore.Files.FileColumns.DATE_MODIFIED,
            MediaStore.Files.FileColumns.RELATIVE_PATH,
            MediaStore.Files.FileColumns.DATA,
        )
        val collection = MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY)
        val sortOrder = "${MediaStore.Files.FileColumns.DATE_MODIFIED} DESC"

        val cursor = resolver.query(
            collection,
            projection,
            null,
            null,
            "$sortOrder LIMIT $MAX_INDEX_SIZE",
        ) ?: return emptyList()

        cursor.use {
            val idxId = it.getColumnIndex(MediaStore.Files.FileColumns._ID)
            val idxName = it.getColumnIndex(MediaStore.Files.FileColumns.DISPLAY_NAME)
            val idxType = it.getColumnIndex(MediaStore.Files.FileColumns.MIME_TYPE)
            val idxSize = it.getColumnIndex(MediaStore.Files.FileColumns.SIZE)
            val idxDate = it.getColumnIndex(MediaStore.Files.FileColumns.DATE_MODIFIED)
            val idxRel = it.getColumnIndex(MediaStore.Files.FileColumns.RELATIVE_PATH)
            val idxData = it.getColumnIndex(MediaStore.Files.FileColumns.DATA)

            val out = ArrayList<LocalIndexedFile>()
            while (it.moveToNext()) {
                val id = readLong(it, idxId)
                if (id <= 0L) {
                    continue
                }

                val name = readString(it, idxName)
                val mimeType = readString(it, idxType)
                if (name.isBlank() || mimeType.isBlank()) {
                    continue
                }

                val contentUri = ContentUris.withAppendedId(collection, id).toString()
                val pathHint = buildString {
                    append(readString(it, idxRel))
                    if (length == 0) {
                        append(readString(it, idxData))
                    }
                }

                val sourceHint = inferSourceHint(pathHint, mimeType)
                out.add(
                    LocalIndexedFile(
                        fileId = computeFileId(contentUri),
                        contentUri = contentUri,
                        displayName = name,
                        mimeType = mimeType,
                        sourceHint = sourceHint,
                        relativePath = pathHint,
                        dateModified = readLong(it, idxDate),
                        sizeBytes = readLong(it, idxSize),
                    ),
                )
            }
            return out
        }
    }

    /**
     * 启动时恢复历史索引（若存在），然后按最新扫描结果增量更新。
     */
    suspend fun rebuildFromMediaStore(resolver: ContentResolver, reason: String): LocalScanResult =
        withContext(Dispatchers.IO) {
            val startedAt = System.currentTimeMillis()

            val previous = synchronized(_itemsById) {
                if (_itemsById.isEmpty()) {
                    loadFromDiskUnsafe()
                } else {
                    LinkedHashMap(_itemsById)
                }
            }

            synchronized(_itemsById) {
                _itemsById.clear()
                previous.forEach { (key, value) ->
                    _itemsById[key] = value
                }
            }

            val collected = collectFromFilesUri(resolver)
            // 命中 LIMIT 上限说明可能有文件被静默截断，日志显式标注
            val truncated = collected.size >= MAX_INDEX_SIZE

            val collectedIds = collected.map { it.fileId }.toSet()
            val previousIds = previous.keys.toSet()

            var newCount = 0
            var updatedCount = 0
            var unchangedCount = 0

            synchronized(_itemsById) {
                collected.forEach { item ->
                    val existing = previous[item.fileId]
                    if (existing == null) {
                        newCount++
                    } else if (existing.dateModified != item.dateModified || existing.sizeBytes != item.sizeBytes) {
                        updatedCount++
                    } else {
                        unchangedCount++
                    }
                    _itemsById[item.fileId] = item
                }
                _itemsById.keys.retainAll(collectedIds)
            }

            val current = synchronized(_itemsById) { LinkedHashMap(_itemsById) }
            persistUnsafe(current)

            // removed = 旧索引中存在但本次扫描未命中的文件（集合差）
            val removedCount = (previousIds - collectedIds).size
            val duration = System.currentTimeMillis() - startedAt
            val result = LocalScanResult(
                scanned = collected.size,
                upserted = newCount + updatedCount,
                removed = removedCount,
                unchanged = unchangedCount,
                durationMs = duration,
                reason = reason,
            )
            Log.i(
                TAG,
                "local-index-rebuild reason=$reason scanned=${result.scanned}" +
                    " upserted=${result.upserted} unchanged=${result.unchanged}" +
                    " removed=${result.removed} truncated=$truncated cost_ms=${result.durationMs}",
            )
            RuntimeEventLog.log(
                appContext,
                "file_index",
                "rebuild",
                "reason=$reason scanned=${result.scanned} upserted=${result.upserted} unchanged=${result.unchanged} removed=${result.removed} truncated=$truncated dur_ms=${result.durationMs}",
            )
            result
        }

    init {
        // 一次性迁移旧位置（filesDir）的索引文件；日志等非资产，失败直接丢弃。
        runCatching {
            storeDir.mkdirs()
            if (legacyStoreFile.exists() && !storeFile.exists()) {
                legacyStoreFile.renameTo(storeFile)
            }
        }.onFailure { Log.w(TAG, "index-store-migrate-failed", it) }
        // 预加载历史索引，保障服务启动后立刻有可读快照。
        val initial = runCatching { loadFromDiskUnsafe() }.getOrElse { linkedMapOf() }
        RuntimeEventLog.log(appContext, "file_index", "store_init", "count=${initial.size} path=${storeFile.absolutePath}")
        synchronized(_itemsById) {
            _itemsById.putAll(initial)
        }
        Log.i(
            TAG,
            "local-index-store-init count=${_itemsById.size} path=${storeFile.absolutePath}",
        )
    }
}
