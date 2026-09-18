package com.example.filememoryagent

import android.content.ActivityNotFoundException
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.Manifest
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.core.content.ContextCompat.startForegroundService
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.filememoryagent.model.FileSearchCandidate
import com.example.filememoryagent.runtime.EdgeRuntimeService
import com.example.filememoryagent.runtime.logging.RuntimeEventLog
import com.example.filememoryagent.ui.ActionResult
import com.example.filememoryagent.ui.MainViewModel
import java.io.File
import androidx.core.content.FileProvider

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { grantMap ->
        val allGranted = grantMap.values.all { it }
        if (allGranted) {
            RuntimeEventLog.log(this, "permission", "storage_granted")
            startEdgeRuntimeService()
            Toast.makeText(this, "已获得存储权限，开始监听文件变更", Toast.LENGTH_SHORT).show()
        } else {
            RuntimeEventLog.log(this, "permission", "storage_denied")
            Toast.makeText(this, "未授予全部权限，端侧监听受限", Toast.LENGTH_LONG).show()
            openPermissionSetting()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        RuntimeEventLog.initialize(this)
        if (hasStoragePermissions() && isEdgeRuntimeEnabled()) {
            startEdgeRuntimeService()
        } else {
            permissionLauncher.launch(requiredPermissions())
        }

        setContent {
            MaterialTheme {
                val vm: MainViewModel = viewModel()
                val state by vm.state.collectAsState()
                val action by vm.actionEvents.collectAsState()
                val edgeRuntimeLogs by RuntimeEventLog.lines.collectAsState()
                val context = LocalContext.current
                val snackbarHostState = remember { SnackbarHostState() }

                Scaffold(
                    snackbarHost = { SnackbarHost(hostState = snackbarHostState) },
                    topBar = {
                        CenterAlignedTopAppBar(
                            title = { Text("文件记忆找回 Agent") },
                        )
                    },
                ) { padding ->
                    ScreenContent(
                        state = state,
                        onQueryChanged = vm::setQuery,
                        onSearch = vm::search,
                        onClarifyChanged = vm::setClarifyInput,
                        onClarify = vm::clarify,
                        edgeRuntimeLogs = edgeRuntimeLogs,
                        onExecute = { fileId, action, shareTo, peer, note ->
                            vm.execute(fileId, action, shareTo, peer, note)
                        },
                        onRebuild = vm::rebuildIndex,
                        onShareToChanged = vm::setShareTo,
                        onNoteChanged = vm::setNote,
                        onCopyLogs = { copyLogs(this, RuntimeEventLog.text()) },
                        onShareLogs = { shareLogs(this, RuntimeEventLog.file(this)) },
                        onClearLogs = { RuntimeEventLog.clear(this) },
                        onStopService = { stopEdgeRuntimeService() },
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(padding),
                    )
                }

                LaunchedEffect(action) {
                    when (val event = action) {
                        is ActionResult.Open -> {
                            openOrCopy(context, event.uri, event.title)
                            vm.markActionEventHandled()
                            snackbarHostState.showSnackbar("已定位到原件: ${event.title}")
                        }
                        is ActionResult.Share -> {
                            shareFromPayload(context, event.payload, event.fallbackText)
                            vm.markActionEventHandled()
                        }
                        is ActionResult.Message -> {
                            snackbarHostState.showSnackbar(event.message)
                            vm.markActionEventHandled()
                        }
                        is ActionResult.Error -> {
                            snackbarHostState.showSnackbar(event.message)
                            vm.markActionEventHandled()
                        }
                        null -> { /* no-op */ }
                    }
                }
            }
        }
    }

    private fun requiredPermissions(): Array<String> {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            arrayOf(
                Manifest.permission.READ_MEDIA_IMAGES,
                Manifest.permission.READ_MEDIA_VIDEO,
                Manifest.permission.READ_MEDIA_AUDIO,
            )
        } else {
            arrayOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
            )
        }
    }

    private fun hasStoragePermissions(): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_MEDIA_IMAGES,
            ) == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_MEDIA_VIDEO,
                ) == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_MEDIA_AUDIO,
                ) == PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.READ_EXTERNAL_STORAGE,
            ) == PackageManager.PERMISSION_GRANTED
        }
    }

    private fun startEdgeRuntimeService() {
        getSharedPreferences(RUNTIME_PREFS, MODE_PRIVATE).edit()
            .putBoolean(RUNTIME_ENABLED, true)
            .apply()
        RuntimeEventLog.log(this, "ui", "start_edge_service")
        val intent = Intent(this, EdgeRuntimeService::class.java)
        startForegroundService(
            this,
            intent,
        )
    }

    private fun stopEdgeRuntimeService() {
        getSharedPreferences(RUNTIME_PREFS, MODE_PRIVATE).edit()
            .putBoolean(RUNTIME_ENABLED, false)
            .apply()
        RuntimeEventLog.log(this, "ui", "stop_edge_service_requested")
        stopService(Intent(this, EdgeRuntimeService::class.java))
        Toast.makeText(this, "端侧服务已停止，不再监听文件", Toast.LENGTH_SHORT).show()
    }

    private fun isEdgeRuntimeEnabled(): Boolean {
        return getSharedPreferences(RUNTIME_PREFS, MODE_PRIVATE)
            .getBoolean(RUNTIME_ENABLED, true)
    }

    private fun openOrCopy(context: Context, fileUri: String, title: String) {
        val resolved = resolveOpenUri(context, fileUri)
        if (resolved == null) {
            RuntimeEventLog.log(context, "ui", "open_fallback_clipboard", "title=$title uri=$fileUri")
            copyUriToClipboard(context, fileUri, title)
            return
        }
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(resolved, "*/*")
            flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
        }
        try {
            context.startActivity(intent)
            RuntimeEventLog.log(context, "ui", "open_success", "title=$title uri=$resolved")
        } catch (_: Exception) {
            RuntimeEventLog.log(context, "ui", "open_failed_fallback_clipboard", "title=$title uri=$resolved")
            copyUriToClipboard(context, fileUri, title)
        }
    }

    private fun resolveOpenUri(context: Context, fileUri: String): Uri? {
        val uri = fileUri.trim()
        if (uri.isBlank()) return null

        return when {
            uri.startsWith("content://") || uri.startsWith("http://") || uri.startsWith("https://") -> Uri.parse(uri)
            uri.startsWith("file://") -> Uri.parse(uri)
            uri.startsWith("/") -> {
                val file = File(uri)
                if (!file.exists()) {
                    null
                } else {
                    FileProvider.getUriForFile(
                        context,
                        "${context.packageName}.fileprovider",
                        file,
                    )
                }
            }
            uri.contains(":") && !uri.startsWith("http://") && !uri.startsWith("https://") -> {
                null
            }
            else -> {
                // 兼容某些返回为裸路径
                val file = File(uri)
                if (!file.exists()) {
                    null
                } else {
                    FileProvider.getUriForFile(
                        context,
                        "${context.packageName}.fileprovider",
                        file,
                    )
                }
            }
        }
    }

    private fun copyUriToClipboard(context: Context, fileUri: String, title: String) {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("file-uri", fileUri))
        Toast.makeText(context, "无法直接打开：$title，已复制地址", Toast.LENGTH_LONG).show()
    }

    private fun copyLogs(context: Context, text: String) {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("runtime_logs", text))
        RuntimeEventLog.log(context, "ui", "logs_copied", "chars=" + text.length)
        Toast.makeText(context, "日志已复制", Toast.LENGTH_SHORT).show()
    }

    private fun shareLogs(context: Context, logFile: File) {
        if (!logFile.exists()) {
            RuntimeEventLog.log(context, "ui", "logs_share_skipped", "file_missing")
            Toast.makeText(context, "暂无日志文件", Toast.LENGTH_SHORT).show()
            return
        }
        // 日志现落盘在 noBackupFilesDir（见 edge-runtime-data-layout.md 约定），
        // 不在 FileProvider 的 file_paths.xml 覆盖范围内（只有 files/cache/external），
        // 直接 getUriForFile 会抛 IllegalArgumentException。先复制到 cacheDir 再分享。
        val shareCopy = File(context.cacheDir, logFile.name)
        val copied = runCatching { logFile.copyTo(shareCopy, overwrite = true) }.isSuccess
        if (!copied) {
            RuntimeEventLog.log(context, "ui", "logs_share_copy_failed", logFile.absolutePath)
            Toast.makeText(context, "日志复制失败，请改用复制按钮", Toast.LENGTH_LONG).show()
            return
        }
        val uri = FileProvider.getUriForFile(context, context.packageName + ".fileprovider", shareCopy)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, "文件记忆助手运行日志")
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        runCatching {
            context.startActivity(Intent.createChooser(intent, "分享运行日志"))
            RuntimeEventLog.log(context, "ui", "logs_share_sheet_opened")
        }.onFailure {
            RuntimeEventLog.logException(context, "ui", "logs_share_failed", it)
            Toast.makeText(context, "无法打开分享面板", Toast.LENGTH_LONG).show()
        }
    }

    private fun shareFromPayload(context: Context, payload: Map<String, String>?, fallbackText: String) {
        val to = payload?.get("to") ?: "同事"
        val fileUri = payload?.get("file_uri").orEmpty()
        val text = if (fileUri.isBlank()) {
            fallbackText
        } else {
            "$fallbackText\n文件：$fileUri"
        }

        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TITLE, "给 $to 的文件")
            putExtra(Intent.EXTRA_TEXT, text)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        try {
            startActivity(Intent.createChooser(intent, "选择分享方式"))
            RuntimeEventLog.log(context, "ui", "share_sheet_opened", "to=$to")
        } catch (_: ActivityNotFoundException) {
            RuntimeEventLog.log(context, "ui", "share_failed_no_app", "to=$to")
            Toast.makeText(context, "没有可用分享应用", Toast.LENGTH_LONG).show()
        }
    }

    private fun openPermissionSetting() {
        try {
            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                data = Uri.parse("package:${packageName}")
            }
            startActivity(intent)
        } catch (_: Exception) {
            Toast.makeText(this, "请前往系统设置开启存储权限", Toast.LENGTH_LONG).show()
        }
    }

    companion object {
        private const val RUNTIME_PREFS = "edge_runtime_preferences"
        private const val RUNTIME_ENABLED = "runtime_enabled"
    }
}

@Composable
private fun ScreenContent(
    state: com.example.filememoryagent.ui.SearchUiState,
    onQueryChanged: (String) -> Unit,
    onSearch: () -> Unit,
    onClarifyChanged: (String) -> Unit,
    onClarify: () -> Unit,
    onExecute: (String, String, String?, String?, String?) -> Unit,
    edgeRuntimeLogs: List<String>,
    onRebuild: () -> Unit,
    onShareToChanged: (String) -> Unit,
    onNoteChanged: (String) -> Unit,
    onCopyLogs: () -> Unit,
    onShareLogs: () -> Unit,
    onClearLogs: () -> Unit,
    onStopService: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val rsp = state.response
    val isNeedClarify = rsp?.needsDisambiguation == true
    var shareTo by remember(rsp?.sessionId) { mutableStateOf("老王") }
    var note by remember(rsp?.sessionId) { mutableStateOf("已确认，留作证据") }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        OutlinedTextField(
            value = state.query,
            onValueChange = onQueryChanged,
            label = { Text("模糊线索") },
            placeholder = { Text("上周群里发的那张聚餐照片") },
            enabled = !state.loading,
            modifier = Modifier.fillMaxWidth(),
        )

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Button(onClick = onSearch, enabled = !state.loading) {
                Text(text = "搜索")
            }
            OutlinedButton(onClick = onRebuild, enabled = !state.loading) {
                Text(text = "重建索引")
            }
            OutlinedButton(onClick = onStopService) {
                Text(text = "停止监听")
            }
        }

        state.error?.let { error ->
            Text(
                text = error,
                color = MaterialTheme.colorScheme.error,
            )
        }

        if (state.loading) {
            androidx.compose.material3.LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        if (rsp != null) {
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(text = "会话", fontWeight = FontWeight.SemiBold)
                Spacer(modifier = Modifier.width(4.dp))
                Text(text = rsp.sessionId, fontSize = 12.sp)
                Spacer(modifier = Modifier.weight(1f))
                Text(text = "状态：${rsp.state}")
            }
            if (!rsp.question.isNullOrBlank()) {
                Text(text = rsp.question, fontWeight = FontWeight.Medium)
            }

            if (isNeedClarify) {
                OutlinedTextField(
                    value = state.clarifyInput,
                    onValueChange = onClarifyChanged,
                    label = { Text("补充线索或直接选择候选") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Button(onClick = onClarify, enabled = !state.loading && state.clarifyInput.isNotBlank()) {
                    Text("继续确认")
                }
            }

            HorizontalDivider(modifier = Modifier.height(8.dp))
            Text(text = "候选结果（${rsp.candidates.size}）", fontWeight = FontWeight.Bold)

            LazyColumn(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                items(rsp.candidates) { c ->
                    CandidateCard(
                        candidate = c,
                        onOpen = { onExecute(c.fileId, "open", null, null, null) },
                        onShare = { onExecute(c.fileId, "share", shareTo.ifBlank { "老王" }, null, null) },
                        onArchive = { onExecute(c.fileId, "archive", null, null, null) },
                        onAnnotate = {
                            onExecute(c.fileId, "annotate", null, null, note.ifBlank { "已确认" })
                        },
                        onCompare = {
                            val peer = rsp.candidates.firstOrNull { it.fileId != c.fileId }
                            if (peer != null) {
                                onExecute(c.fileId, "compare", null, peer.fileId, null)
                            }
                        },
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedTextField(
                    value = shareTo,
                    onValueChange = {
                        shareTo = it
                        onShareToChanged(it)
                    },
                    label = { Text("默认分享对象") },
                    modifier = Modifier.weight(1f),
                )
                OutlinedTextField(
                    value = note,
                    onValueChange = {
                        note = it
                        onNoteChanged(it)
                    },
                    label = { Text("默认备注") },
                    modifier = Modifier.weight(1f),
                )
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(text = "运行日志", fontWeight = FontWeight.Medium)
            Spacer(modifier = Modifier.weight(1f))
            OutlinedButton(onClick = onCopyLogs) { Text("复制") }
            OutlinedButton(onClick = onShareLogs) { Text("分享") }
            OutlinedButton(onClick = onClearLogs) { Text("清空") }
        }
        val logText = remember(state.logs) {
            state.logs.takeLast(30).joinToString("\n") { item ->
                item.text
            }
        }
        Text(
            text = logText.ifBlank { "暂无日志" },
            fontSize = 12.sp,
            modifier = Modifier.fillMaxWidth(),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Text(text = "端侧运行日志", fontWeight = FontWeight.Medium)
        val runtimeLogText = remember(edgeRuntimeLogs) {
            edgeRuntimeLogs.takeLast(30).joinToString("\n")
        }
        Text(
            text = runtimeLogText.ifBlank { "暂无端侧日志" },
            fontSize = 12.sp,
            modifier = Modifier.fillMaxWidth(),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun CandidateCard(
    candidate: FileSearchCandidate,
    onOpen: () -> Unit,
    onShare: () -> Unit,
    onArchive: () -> Unit,
    onAnnotate: () -> Unit,
    onCompare: () -> Unit,
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(text = candidate.title, fontWeight = FontWeight.SemiBold)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(text = "来源：${candidate.sourceApp}", fontSize = 12.sp)
                Text(text = "类型：${candidate.docType}", fontSize = 12.sp)
                candidate.capturedAt?.let {
                    Text(text = "时间：$it", fontSize = 12.sp)
                }
            }
            Text(text = candidate.evidence)
            Text(text = "匹配线索：${candidate.matchedClues.joinToString()}", fontSize = 12.sp)
            Text(text = "视觉线索：${candidate.visualHints.joinToString()}", fontSize = 12.sp)

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedButton(onClick = onOpen) {
                    Text(text = "打开")
                }
                OutlinedButton(onClick = onShare) {
                    Text(text = "分享")
                }
                OutlinedButton(onClick = onArchive) {
                    Text(text = "归档")
                }
                OutlinedButton(onClick = onAnnotate) {
                    Text(text = "备注")
                }
                OutlinedButton(onClick = onCompare) {
                    Text(text = "对比")
                }
            }
        }
    }
}
