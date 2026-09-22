package com.example.filememoryagent.runtime

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.database.ContentObserver
import android.net.Uri
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import com.example.filememoryagent.runtime.logging.RuntimeEventLog
import com.example.filememoryagent.runtime.index.LocalFileIndexStore
import com.example.filememoryagent.runtime.index.LocalIndexedFile
import com.example.filememoryagent.runtime.index.LocalScanResult
import com.example.filememoryagent.runtime.data.EdgeDataPaths
import com.example.filememoryagent.runtime.native.NativeRuntime
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.cancel

private const val NOTIFICATION_CHANNEL = "edge_runtime"
private const val NOTIFICATION_ID = 1001
private const val WATCH_INTERVAL_MS = 5 * 60 * 1000L
private const val DEBOUNCE_MS = 1200L

/**
 * 端侧常驻服务：
 * 1) 注册 MediaStore 观察者，监听文件变更；
 * 2) 周期/事件触发本地索引扫描；
 * 3) 暴露给 UI 与后续模块的本地索引快照。
 */
class EdgeRuntimeService : LifecycleService() {

    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val handler = Handler(Looper.getMainLooper())
    private val indexStore = LocalFileIndexStore(this)
    private val dataPaths = EdgeDataPaths(this)

    private var periodicJob: Job? = null
    private var debounceJob: Job? = null
    private val observers = mutableListOf<ContentObserver>()
    private var lastScanSummary: LocalScanResult? = null

    // onCreate 完成（目录就绪 + 观察者注册尝试完毕）后置位；此前 isReady() 恒 true，是桩。
    @Volatile
    private var runtimeReady = false

    inner class LocalBinder : Binder() {
        fun service(): EdgeRuntimeService = this@EdgeRuntimeService

        fun isReady(): Boolean = runtimeReady

        fun localIndexSize(): Int = indexStore.size()

        fun localIndexSnapshot(): List<LocalIndexedFile> = indexStore.snapshot()

        fun lastScanMs(): Long = lastScanSummary?.durationMs ?: -1L

        fun lastScanReason(): String = lastScanSummary?.reason ?: "未扫描"
    }

    override fun onCreate() {
        super.onCreate()
        ensureForegroundChannel()
        startForeground(NOTIFICATION_ID, buildNotification())

        runCatching { dataPaths.ensureDirectories() }
            .onSuccess { RuntimeEventLog.log(this, "edge_runtime", "data_directories_ready", "root=" + dataPaths.models.parent) }
            .onFailure { RuntimeEventLog.logException(this, "edge_runtime", "data_directories_failed", it) }
        RuntimeEventLog.log(this, "edge_runtime", "native_runtime_status", NativeRuntime.status())
        RuntimeEventLog.log(this, "edge_runtime", "service_created", "notification_started")
        registerWatchObservers()
        startPeriodicScan()
        scheduleScan("service_start")
        runtimeReady = true
        Log.i(TAG, "edge-runtime-created")
        RuntimeEventLog.log(this, "edge_runtime", "service_initialized", "watchers_registered=${observers.size}")
    }

    override fun onBind(intent: Intent): IBinder {
        Log.i(TAG, "edge-runtime-bind")
        RuntimeEventLog.log(this, "edge_runtime", "service_bound")
        return binder
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent != null) {
            when (intent.action) {
                ACTION_TRIGGER_REBUILD -> {
                    RuntimeEventLog.log(this, "edge_runtime", "start_command", "manual_rebuild")
                    scheduleScan("manual")
                }
                ACTION_FORCE_REBUILD -> {
                    RuntimeEventLog.log(this, "edge_runtime", "start_command", "manual_force_rebuild")
                    scheduleScan("manual_force")
                }
                ACTION_STOP -> {
                    getSharedPreferences("edge_runtime_preferences", MODE_PRIVATE).edit()
                        .putBoolean("runtime_enabled", false)
                        .apply()
                    RuntimeEventLog.log(this, "edge_runtime", "stop_requested", "source=notification_or_ui")
                    stopRuntime()
                    return START_NOT_STICKY
                }
            }
        }
        return Service.START_STICKY
    }

    override fun onDestroy() {
        val observerCount = observers.size
        observers.forEach { contentResolver.unregisterContentObserver(it) }
        observers.clear()
        periodicJob?.cancel()
        debounceJob?.cancel()
        serviceScope.cancel()
        Log.i(TAG, "edge-runtime-destroyed")
        RuntimeEventLog.log(this, "edge_runtime", "service_destroyed", "observers=" + observerCount + " jobs_cancelled=true")
        super.onDestroy()
    }

    private fun stopRuntime() {
        RuntimeEventLog.log(this, "edge_runtime", "stopping", "observers=" + observers.size)
        stopSelf()
    }

    private fun registerWatchObservers() {
        if (!hasStoragePermissions()) {
            Log.w(TAG, "content-watchers-skipped-no-permission")
            RuntimeEventLog.log(this, "edge_runtime", "watcher_register_skip", "missing_storage_permission")
            return
        }
        val uris = listOf(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            MediaStore.Audio.Media.EXTERNAL_CONTENT_URI,
            MediaStore.Downloads.EXTERNAL_CONTENT_URI,
            MediaStore.Files.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY),
        )

        uris.forEach { uri ->
            val observer = object : ContentObserver(handler) {
                override fun onChange(selfChange: Boolean, changedUri: Uri?) {
                    Log.i(TAG, "media-observer-change uri=$uri changed=$changedUri")
                    scheduleScan("content_change:${uri.lastPathSegment}")
                }
            }
            contentResolver.registerContentObserver(uri, true, observer)
            observers.add(observer)
        }
        Log.i(TAG, "content-observers-registered count=${observers.size}")
        RuntimeEventLog.log(this, "edge_runtime", "watcher_registered", "count=${observers.size}")
    }

    private fun startPeriodicScan() {
        periodicJob = serviceScope.launch {
            while (isActive) {
                if (!hasStoragePermissions()) {
                    Log.w(TAG, "periodic-scan-skipped-no-permission")
                    RuntimeEventLog.log(this@EdgeRuntimeService, "edge_runtime", "periodic_skip", "no_permission")
                } else {
                    scheduleScan("periodic")
                }
                delay(WATCH_INTERVAL_MS)
            }
        }
    }

    private fun scheduleScan(reason: String) {
        if (!hasStoragePermissions()) {
            RuntimeEventLog.log(this, "edge_runtime", "scan_skip", "reason=$reason no_permission")
            return
        }
        RuntimeEventLog.log(this, "edge_runtime", "scan_scheduled", "reason=$reason")
        debounceJob?.cancel()
        debounceJob = serviceScope.launch {
            delay(DEBOUNCE_MS)
            runScan(reason)
        }
    }

    private suspend fun runScan(reason: String) {
        runCatching {
            val summary = indexStore.rebuildFromMediaStore(contentResolver, reason)
            lastScanSummary = summary
            Log.i(
                TAG,
                "scan-done reason=${summary.reason} scanned=${summary.scanned} upserted=${summary.upserted}" +
                    " unchanged=${summary.unchanged} removed=${summary.removed} dur=${summary.durationMs}ms",
            )
            RuntimeEventLog.log(
                this@EdgeRuntimeService,
                "edge_runtime",
                "scan_done",
                "reason=${summary.reason} scanned=${summary.scanned} upserted=${summary.upserted} unchanged=${summary.unchanged} removed=${summary.removed} dur_ms=${summary.durationMs}",
            )
        }.onFailure {
            Log.e(TAG, "scan-failed reason=$reason", it)
            RuntimeEventLog.log(
                this@EdgeRuntimeService,
                "edge_runtime",
                "scan_failed",
                "reason=$reason msg=${it.message.orEmpty()}",
            )
        }
    }

    private fun hasStoragePermissions(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            hasPermission(Manifest.permission.READ_MEDIA_IMAGES) &&
                hasPermission(Manifest.permission.READ_MEDIA_VIDEO) &&
                hasPermission(Manifest.permission.READ_MEDIA_AUDIO)
        } else {
            hasPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            permission,
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun buildNotification(): Notification {
        val stopIntent = Intent(this, EdgeRuntimeService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this,
            1002,
            stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or pendingIntentImmutableFlag(),
        )
        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL)
            .setContentTitle("个人文件记忆服务")
            .setContentText("文件监听与本地索引已就绪")
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setOngoing(true)
            .addAction(android.R.drawable.ic_delete, "停止监听", stopPendingIntent)
            .build()
    }

    private fun pendingIntentImmutableFlag(): Int {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            PendingIntent.FLAG_IMMUTABLE
        } else {
            0
        }
    }

    private fun ensureForegroundChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) {
            return
        }
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL,
            "Edge Runtime Channel",
            NotificationManager.IMPORTANCE_LOW,
        )
        channel.description = "个人文件监听服务"
        manager.createNotificationChannel(channel)
    }

    companion object {
        private const val TAG = "EdgeRuntimeService"
        const val ACTION_TRIGGER_REBUILD = "action_trigger_rebuild"
        const val ACTION_FORCE_REBUILD = "action_force_rebuild"
        const val ACTION_STOP = "action_stop"
    }
}
