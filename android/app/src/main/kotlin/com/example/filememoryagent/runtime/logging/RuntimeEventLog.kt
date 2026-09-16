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

/**
 * 端侧运行时统一日志桶。
 *
 * 用途：
 * - 服务端扫描、重建、权限告警等关键事件的本地可见化；
 * - 与界面展示共享最新日志；
 * - 同时持久化到 app 内部文件，便于排障。
 */
object RuntimeEventLog {

    private var appContext: Context? = null
    private val formatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    private val _lines = MutableStateFlow<List<String>>(emptyList())
    val lines: StateFlow<List<String>> = _lines

    fun initialize(context: Context) {
        appContext = context.applicationContext
        runCatching {
            val logFile = File(context.filesDir, LOG_FILE_NAME)
            if (logFile.exists()) {
                _lines.value = logFile.useLines { it.toList().takeLast(MAX_LINES) }
            }
            log(context, "runtime_log", "initialized", "loaded=" + _lines.value.size)
        }.onFailure { Log.w(TAG, "runtime-log-load-failed", it) }
    }

    fun logException(context: Context?, source: String, event: String, error: Throwable, detail: String = "") {
        val stack = StringWriter().also { error.printStackTrace(PrintWriter(it)) }.toString()
        log(context, source, event, detail + " error=" + error.message.orEmpty() + " stack=" + stack)
    }

    fun clear(context: Context?) {
        _lines.value = emptyList()
        context?.let { runCatching { File(it.filesDir, LOG_FILE_NAME).delete() } }
        log(context, "runtime_log", "cleared")
    }

    fun text(): String = _lines.value.joinToString("\n")

    fun file(context: Context): File = File(context.filesDir, LOG_FILE_NAME)

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
            val logFile = File(targetContext.filesDir, LOG_FILE_NAME)
            logFile.appendText("$line\n")
        }.onFailure {
            Log.w(TAG, "runtime-log-persist-failed", it)
        }
    }
}
