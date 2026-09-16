package com.example.filememoryagent.runtime.native

import android.util.Log

/** C++ 端侧推理运行时的稳定入口，当前仅提供状态探针。 */
object NativeRuntime {
    private const val TAG = "NativeRuntime"
    private var loaded = false

    init {
        runCatching {
            System.loadLibrary("edge_runtime")
            loaded = true
        }.onFailure { Log.i(TAG, "native runtime not packaged yet: " + it.message) }
    }

    fun isLoaded(): Boolean = loaded

    fun status(): String {
        if (!loaded) return "native_unavailable"
        return runCatching { runtimeInfo() }
            .getOrElse { "native_error:" + it.message.orEmpty() }
    }

    private external fun runtimeInfo(): String
}
