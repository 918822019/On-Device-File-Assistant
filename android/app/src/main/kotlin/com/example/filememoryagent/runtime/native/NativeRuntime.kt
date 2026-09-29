package com.example.filememoryagent.runtime.native

import android.util.Log
import java.io.File

/** Native .tqwen model session. Tokenization is supplied by the caller. */
object NativeRuntime {
    private var loaded = false

    init {
        runCatching { System.loadLibrary("edge_runtime") }
            .onSuccess { loaded = true }
            .onFailure { Log.e("NativeRuntime", "native runtime unavailable", it) }
    }

    fun isLoaded(): Boolean = loaded

    fun status(): String = if (loaded) runtimeInfo() else "native_unavailable"

    fun loadModel(file: File, maxContext: Int = 512): String {
        check(loaded) { "端侧推理库未加载" }
        require(file.isFile && file.extension == "tqwen") { "请选择已导入的 .tqwen 模型" }
        return nativeLoad(file.absolutePath, maxContext)
    }

    fun loadDFlashModel(target: File, draft: File, width: Int = 4,
                        maxContext: Int = 512): String {
        check(loaded) { "端侧推理库未加载" }
        require(target.isFile && target.extension == "tqwen") { "请先导入 target 模型" }
        require(draft.isFile && draft.extension == "tqwen") { "请选择 DFlash .tqwen 草稿" }
        require(width in 2..8) { "块宽必须在 2..8 之间" }
        return nativeLoadDFlash(target.absolutePath, draft.absolutePath, maxContext, width)
    }

    fun stats(): String = if (loaded) nativeStats() else "native_unavailable"

    fun generate(promptTokenIds: IntArray, maxNewTokens: Int = 32): IntArray {
        check(loaded) { "端侧推理库未加载" }
        return nativeGenerate(promptTokenIds, maxNewTokens)
    }

    fun unload() {
        if (loaded) nativeUnload()
    }

    private external fun runtimeInfo(): String
    private external fun nativeLoad(path: String, maxContext: Int): String
    private external fun nativeLoadDFlash(targetPath: String, draftPath: String,
                                          maxContext: Int, width: Int): String
    private external fun nativeGenerate(prompt: IntArray, maxNewTokens: Int): IntArray
    private external fun nativeStats(): String
    private external fun nativeUnload()
}
