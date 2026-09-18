package com.example.filememoryagent.runtime.logging

import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.PrintWriter
import java.io.StringWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private const val TAG = "RuntimeEventLog"
private const val MAX_LINES = 300
private const val LOG_FILE_NAME = "runtime_events.log"

// 日志文件轮转阈值：超过后只保留最后 MAX_LINES 行（与内存口径一致）。
// 此前文件逐行追加永不裁剪，长期运行无界膨胀。
private const val MAX_FILE_BYTES = 256L * 1024

/**
 * 端侧运行时统一日志桶。
 *
 * 用途：
 * - 服务端扫描、重建、权限告警等关键事件的本地可见化；
 * - 与界面展示共享最新日志；
 * - 同时持久化到应用私有文件，便于排障。
 *
 * 落盘位置遵循 docs/edge-runtime-data-layout.md 约定：
 * `noBackupFilesDir/logs/runtime_events.log` —— 不参与系统备份、
 * 不随重装恢复、卸载即删（旧版本写在 filesDir，初始化时自动迁移）。
 */
object RuntimeEventLog {

    private var appContext: Context? = null
    private val formatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val _lines = MutableStateFlow<List<String>>(emptyList())
    val lines: StateFlow<List<String>> = _lines

    /** 日志文件的唯一路径入口，避免多处硬编码。 */
    fun file(context: Context): File = File(File(context.noBackupFilesDir, "logs"), LOG_FILE_NAME)

    fun initialize(context: Context) {
        appContext = context.applicationContext
        runCatching {
            val logFile = file(context)
            logFile.parentFile?.mkdirs()
            migrateLegacyFile(context, logFile)
            if (logFile.exists()) {
                _lines.value = logFile.useLines { it.toList().takeLast(MAX_LINES) }
            }
            log(context, "runtime_log", "initialized", "loaded=" + _lines.value.size)
        }.onFailure { Log.w(TAG, "runtime-log-load-failed", it) }
    }

    /** 旧版本落盘在 filesDir；存在旧文件则迁移一次（日志非资产，迁移失败直接丢弃）。 */
    private fun migrateLegacyFile(context: Context, target: File) {
        val legacy = File(context.filesDir, LOG_FILE_NAME)
        if (!legacy.exists()) {
            return
        }
        runCatching {
            if (target.exists()) {
                legacy.delete()
            } else {
                legacy.renameTo(target)
            }
        }.onFailure { Log.w(TAG, "runtime-log-migrate-failed", it) }
    }

    fun logException(context: Context?, source: String, event: String, error: Throwable, detail: String = "") {
        val stack = StringWriter().also { error.printStackTrace(PrintWriter(it)) }.toString()
        log(context, source, event, detail + " error=" + error.message.orEmpty() + " stack=" + stack)
    }

    fun clear(context: Context?) {
        _lines.value = emptyList()
        context?.let { runCatching { file(it).delete() } }
        log(context, "runtime_log", "cleared")
    }

    fun text(): String = _lines.value.joinToString("\n")

    fun log(context: Context?, source: String, event: String, detail: String = "") {
        val ts = System.currentTimeMillis()
        val msg = detail.takeIf { it.isNotBlank() }?.let { " | $it" } ?: ""
        val line = "[${formatter.format(Date(ts))}] [$source] $event$msg"
        _lines.value = (_lines.value + line).takeLast(MAX_LINES)
        Log.i(TAG, line)

        persistIfNeed(context, line)
    }

    private fun persistIfNeed(context: Context?, line: String) {
        val targetContext = context ?: appContext
        if (targetContext == null) {
            return
        }
        runCatching {
            val logFile = file(targetContext)
            logFile.parentFile?.mkdirs()
            trimIfNeeded(logFile)
            logFile.appendText("$line\n")
        }.onFailure {
            Log.w(TAG, "runtime-log-persist-failed", it)
        }
    }

    /** 文件超过阈值时裁剪到最后 MAX_LINES 行；未超阈值时只有一次 length() stat 的开销。 */
    private fun trimIfNeeded(logFile: File) {
        if (!logFile.exists() || logFile.length() <= MAX_FILE_BYTES) {
            return
        }
        runCatching {
            val tail = logFile.readLines().takeLast(MAX_LINES)
            logFile.writeText(tail.joinToString(separator = "\n", postfix = "\n"))
        }.onFailure { Log.w(TAG, "runtime-log-trim-failed", it) }
    }
}
