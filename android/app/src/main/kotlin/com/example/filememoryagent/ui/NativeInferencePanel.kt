package com.example.filememoryagent.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.filememoryagent.runtime.data.EdgeDataPaths
import com.example.filememoryagent.runtime.native.NativeRuntime
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Actual on-device inference, accepting tokenizer-produced token IDs. */
@Composable
fun NativeInferencePanel() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var busy by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf(NativeRuntime.status()) }
    var modelReady by remember { mutableStateOf(status.startsWith("model_loaded")) }
    var prompt by remember { mutableStateOf("") }
    var blockWidth by remember { mutableStateOf("4") }
    var output by remember { mutableStateOf("") }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch {
            busy = true
            status = "正在导入模型…"
            val result = runCatching {
                withContext(Dispatchers.IO) {
                    val models = EdgeDataPaths(context).models
                    check(models.isDirectory || models.mkdirs()) { "无法创建模型目录" }
                    val temp = File(models, "importing.tqwen")
                    val dest = File(models, "model.tqwen")
                    try {
                        val stream = context.contentResolver.openInputStream(uri)
                            ?: error("无法读取选中的模型")
                        stream.use { input -> temp.outputStream().use { input.copyTo(it, 1024 * 1024) } }
                        check(temp.length() > 0) { "模型文件为空" }
                        check(!dest.exists() || dest.delete()) { "无法替换旧模型" }
                        check(temp.renameTo(dest)) { "无法完成模型导入" }
                        NativeRuntime.loadModel(dest)
                    } finally {
                        temp.delete()
                    }
                }
            }
            status = result.getOrElse { "导入/加载失败：${it.message}" }
            modelReady = status.startsWith("ok:")
            output = ""
            busy = false
        }
    }

    val draftPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch {
            busy = true
            status = "正在导入 DFlash 草稿…"
            val result = runCatching {
                withContext(Dispatchers.IO) {
                    val models = EdgeDataPaths(context).models
                    val target = File(models, "model.tqwen")
                    check(target.isFile) { "请先导入 target 模型" }
                    val width = blockWidth.toIntOrNull() ?: error("块宽必须是 2..8 的整数")
                    val temp = File(models, "importing_draft.tqwen")
                    val dest = File(models, "draft.tqwen")
                    try {
                        val stream = context.contentResolver.openInputStream(uri)
                            ?: error("无法读取选中的草稿")
                        stream.use { input -> temp.outputStream().use { input.copyTo(it, 1024 * 1024) } }
                        check(temp.length() > 0) { "草稿文件为空" }
                        check(!dest.exists() || dest.delete()) { "无法替换旧草稿" }
                        check(temp.renameTo(dest)) { "无法完成草稿导入" }
                        NativeRuntime.loadDFlashModel(target, dest, width)
                    } finally {
                        temp.delete()
                    }
                }
            }
            status = result.getOrElse { "DFlash 加载失败：${it.message}" }
            modelReady = status.startsWith("ok:")
            output = ""
            busy = false
        }
    }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("端侧推理（CPU / Vulkan DFlash · .tqwen）")
        Text("模型保存在应用私有目录。当前运行时不含分词器；请填由同一模型 tokenizer 生成的 prompt token IDs。")
        OutlinedButton(onClick = { picker.launch(arrayOf("*/*")) }, enabled = !busy && NativeRuntime.isLoaded()) {
            Text("导入并加载 target .tqwen（CPU）")
        }
        OutlinedTextField(
            value = blockWidth,
            onValueChange = { blockWidth = it },
            label = { Text("DFlash 块宽（2..8）") },
            modifier = Modifier.fillMaxWidth(),
            enabled = !busy,
        )
        OutlinedButton(onClick = { draftPicker.launch(arrayOf("*/*")) },
            enabled = !busy && NativeRuntime.isLoaded()) {
            Text("导入草稿并加载 Vulkan DFlash")
        }
        Text(status)
        OutlinedTextField(
            value = prompt,
            onValueChange = { prompt = it },
            label = { Text("Prompt token IDs（逗号或空格分隔）") },
            modifier = Modifier.fillMaxWidth(),
            enabled = !busy,
        )
        Button(onClick = {
            val ids = runCatching {
                prompt.trim().split(Regex("[,\\s]+"))
                    .filter { it.isNotBlank() }.map { it.toInt() }.toIntArray()
                    .also { require(it.isNotEmpty()) { "请输入至少一个 token ID" } }
            }
            if (ids.isFailure) {
                status = "输入错误：${ids.exceptionOrNull()?.message}"
            } else {
                scope.launch {
                    busy = true
                    status = "端侧生成中…"
                    val result = runCatching { withContext(Dispatchers.IO) { NativeRuntime.generate(ids.getOrThrow()) } }
                    result.onSuccess {
                        output = it.joinToString(" ")
                        val stats = NativeRuntime.stats()
                        status = "生成 ${it.size} 个 token" +
                            if (stats == "no_stats") "" else "; $stats"
                    }
                        .onFailure { status = "推理失败：${it.message}" }
                    busy = false
                }
            }
        }, enabled = !busy && modelReady) { Text("生成 32 token") }
        Text("输出 token IDs：${output.ifBlank { "暂无" }}")
        OutlinedButton(onClick = {
            NativeRuntime.unload()
            modelReady = false
            status = NativeRuntime.status()
        }, enabled = !busy && modelReady) {
            Text("卸载模型")
        }
    }
}
