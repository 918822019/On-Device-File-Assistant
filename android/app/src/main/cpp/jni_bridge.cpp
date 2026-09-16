#include <jni.h>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_filememoryagent_runtime_native_NativeRuntime_runtimeInfo(
    JNIEnv* env,
    jobject /* thiz */
) {
    return env->NewStringUTF(
        "edge_runtime_native_ready; model_backend=not_loaded; backend=tinyllm-placeholder"
    );
}
