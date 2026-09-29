#include <jni.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#include "dispatch.h"
#include "dflash_model.h"
#include "dflash_execution_graph.h"
#include "speculative_decoder.h"
#include "edge_setup.h"
#include "model_loader.h"
#include "profiler.h"
#include "qwen_model.h"

namespace {

struct Session {
    tinyqwen::ModelFile file;
    tinyqwen::ModelFile draft_file;
    tinyqwen::Profiler profiler{false};
    std::unique_ptr<tinyqwen::QwenModel> model;
    std::unique_ptr<tinyqwen::DFlashModel> draft;
    std::unique_ptr<tinyqwen::DFlashExecutionGraph> graph;
    tinyqwen::SpeculativeStats last_stats;
    bool has_stats = false;
    int block_width = 4;
    int max_context = 0;
};

std::mutex session_mutex;
std::unique_ptr<Session> session;

void fail(JNIEnv *env, const std::string &message) {
    jclass type = env->FindClass("java/lang/IllegalStateException");
    if (type) env->ThrowNew(type, message.c_str());
}

uint64_t available_memory_bytes() {
    std::ifstream meminfo("/proc/meminfo");
    std::string key, unit;
    uint64_t kb = 0;
    while (meminfo >> key >> kb >> unit) {
        if (key == "MemAvailable:") return kb * 1024;
    }
    return 0; // unknown: do not pretend that the device has room for a model
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_filememoryagent_runtime_native_NativeRuntime_runtimeInfo(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(session_mutex);
    return env->NewStringUTF(session ? (session->graph ? "model_loaded; backend=vulkan-dflash"
                                                    : "model_loaded; backend=cpu") :
                                      "native_ready; model_backend=not_loaded");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_filememoryagent_runtime_native_NativeRuntime_nativeLoad(
    JNIEnv *env, jobject, jstring path, jint max_context) {
    std::lock_guard<std::mutex> lock(session_mutex);
    if (!path || max_context <= 0 || max_context > 4096)
        return env->NewStringUTF("error: invalid path or context length (1..4096)");
    const char *utf = env->GetStringUTFChars(path, nullptr);
    if (!utf) return nullptr;
    const std::string model_path(utf);
    env->ReleaseStringUTFChars(path, utf);

    // Loading a second model while keeping the first one alive can double peak RAM.
    session.reset();
    try {
        uint64_t need = 0;
        std::string err;
        if (!tinyqwen::ModelFile::estimate_resident_bytes(model_path, false, &need, &err))
            return env->NewStringUTF(("error: " + err).c_str());
        const uint64_t available = available_memory_bytes();
        const uint64_t margin = 256ull * 1024 * 1024;
        if (!available || need > available - std::min(available, available / 10 + margin))
            return env->NewStringUTF("error: insufficient available RAM for model and KV cache");

        auto candidate = std::make_unique<Session>();
        if (!candidate->file.load(model_path, &err))
            return env->NewStringUTF(("error: " + err).c_str());
        if (candidate->file.config().is_moe())
            return env->NewStringUTF("error: MoE models require expert offload; unsupported in app");
        const auto dtype = static_cast<tinyqwen::Dtype>(candidate->file.header().dtype);
        const char *impl = dtype == tinyqwen::Dtype::kI4 ? "sdot4_mt" :
                           dtype == tinyqwen::Dtype::kF16 ? "neon_mt_kv_nt" : "neon_mt_kv_nt";
        if (tinyqwen::select_matvec_registries(candidate->file, impl) != 0 ||
            !tinyqwen::set_ops_impl_by_name("neon"))
            return env->NewStringUTF("error: ARM NEON kernels unavailable");
        if (!tinyqwen::QwenModel::create(candidate->file, max_context, candidate->profiler,
                                         &err, &candidate->model))
            return env->NewStringUTF(("error: " + err).c_str());
        candidate->max_context = max_context;
        session = std::move(candidate);
        return env->NewStringUTF("ok: model_loaded; backend=cpu");
    } catch (const std::exception &exc) {
        return env->NewStringUTF((std::string("error: ") + exc.what()).c_str());
    }
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_filememoryagent_runtime_native_NativeRuntime_nativeGenerate(
    JNIEnv *env, jobject, jintArray prompt, jint max_new_tokens) {
    std::lock_guard<std::mutex> lock(session_mutex);
    if (!session || !session->model) {
        fail(env, "load a .tqwen model before generating");
        return nullptr;
    }
    session->has_stats = false;
    const jsize count = prompt ? env->GetArrayLength(prompt) : 0;
    if (count <= 0 || max_new_tokens <= 0 || max_new_tokens > 256 ||
        count + max_new_tokens > session->max_context) {
        fail(env, "prompt must be nonempty; max output 256; prompt + output must fit context");
        return nullptr;
    }
    std::vector<jint> input(count);
    env->GetIntArrayRegion(prompt, 0, count, input.data());
    if (env->ExceptionCheck()) return nullptr;
    const int vocab = static_cast<int>(session->file.config().vocab_size);
    for (int id : input) {
        if (id < 0 || id >= vocab) {
            fail(env, "prompt contains a token ID outside the model vocabulary");
            return nullptr;
        }
    }
    try {
        if (session->graph) {
            tinyqwen::SpeculativeConfig config;
            config.max_new_tokens = max_new_tokens;
            config.draft_tokens = session->block_width;
            config.eos_token_id = static_cast<int>(session->file.config().eos_token_id);
            tinyqwen::SpeculativeResult generated;
            std::string err;
            const std::vector<int> ids(input.begin(), input.end());
            if (!tinyqwen::dflash_speculative_generate(*session->model, *session->draft,
                                                      *session->graph, ids, config,
                                                      &generated, &err)) {
                fail(env, "Vulkan DFlash generation failed: " + err);
                return nullptr;
            }
            session->last_stats = generated.stats;
            session->has_stats = true;
            jintArray result = env->NewIntArray(
                static_cast<jsize>(generated.generated_ids.size()));
            if (result) env->SetIntArrayRegion(result, 0,
                static_cast<jsize>(generated.generated_ids.size()),
                generated.generated_ids.data());
            return result;
        }
        const auto prefill_start = std::chrono::steady_clock::now();
        auto &model = *session->model;
        model.reset();
        model.set_prompt_len(count);
        int next = -1;
        for (jsize i = 0; i < count; ++i)
            next = model.forward_token(input[i], nullptr, 0, i == count - 1);
        const auto decode_start = std::chrono::steady_clock::now();
        session->last_stats.prefill_ms =
            std::chrono::duration<double, std::milli>(decode_start - prefill_start).count();
        std::vector<jint> output;
        output.reserve(max_new_tokens);
        const int eos = static_cast<int>(session->file.config().eos_token_id);
        for (int i = 0; i < max_new_tokens; ++i) {
            if (next < 0 || next >= vocab) {
                fail(env, "model returned an invalid token ID");
                return nullptr;
            }
            output.push_back(next);
            if (eos > 0 && next == eos) break;
            if (i + 1 < max_new_tokens) next = model.forward_token(next, nullptr, 0);
        }
        session->last_stats.decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decode_start).count();
        session->has_stats = true;
        jintArray result = env->NewIntArray(static_cast<jsize>(output.size()));
        if (result) env->SetIntArrayRegion(result, 0, static_cast<jsize>(output.size()), output.data());
        return result;
    } catch (const std::exception &exc) {
        fail(env, exc.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_filememoryagent_runtime_native_NativeRuntime_nativeLoadDFlash(
    JNIEnv *env, jobject, jstring target_path, jstring draft_path,
    jint max_context, jint width) {
    std::lock_guard<std::mutex> lock(session_mutex);
    if (!target_path || !draft_path || max_context <= 0 || max_context > 4096 ||
        width < 2 || width > 8)
        return env->NewStringUTF("error: invalid target/draft path, context or block width");
    const char *target_utf = env->GetStringUTFChars(target_path, nullptr);
    if (!target_utf) return nullptr;
    const std::string target_name(target_utf);
    env->ReleaseStringUTFChars(target_path, target_utf);
    const char *draft_utf = env->GetStringUTFChars(draft_path, nullptr);
    if (!draft_utf) return nullptr;
    const std::string draft_name(draft_utf);
    env->ReleaseStringUTFChars(draft_path, draft_utf);

    // Loading two model files alongside their GPU copies must not double an
    // existing session's peak RSS. A failed load leaves no stale GPU state.
    session.reset();
    try {
        uint64_t target_need = 0, draft_need = 0;
        std::string err;
        if (!tinyqwen::ModelFile::estimate_resident_bytes(target_name, false,
                                                            &target_need, &err) ||
            !tinyqwen::ModelFile::estimate_resident_bytes(draft_name, false,
                                                            &draft_need, &err))
            return env->NewStringUTF(("error: " + err).c_str());
        const uint64_t available = available_memory_bytes();
        const uint64_t margin = available / 10 + 256ull * 1024 * 1024;
        if (available <= margin || target_need > (available - margin) / 2 ||
            draft_need > (available - margin) / 2 - target_need)
            return env->NewStringUTF("error: insufficient RAM for target, draft and GPU weights");

        auto candidate = std::make_unique<Session>();
        if (!candidate->file.load(target_name, &err) ||
            !candidate->draft_file.load(draft_name, &err))
            return env->NewStringUTF(("error: " + err).c_str());
        if (candidate->file.header().dtype != static_cast<uint32_t>(tinyqwen::Dtype::kF16) ||
            candidate->draft_file.header().dtype != static_cast<uint32_t>(tinyqwen::Dtype::kF16))
            return env->NewStringUTF("error: app Vulkan DFlash requires FP16 target and draft");
        if (tinyqwen::select_matvec_registries(candidate->file, "neon_mt_kv_nt") != 0 ||
            !tinyqwen::set_ops_impl_by_name("neon"))
            return env->NewStringUTF("error: ARM NEON kernels unavailable");
        if (!tinyqwen::QwenModel::create(candidate->file, max_context, candidate->profiler,
                                          &err, &candidate->model) ||
            !tinyqwen::DFlashModel::create(candidate->draft_file, max_context,
                                            *candidate->model, &err, &candidate->draft))
            return env->NewStringUTF(("error: " + err).c_str());
        if (width > candidate->draft->block_size())
            return env->NewStringUTF("error: requested width exceeds the draft block size");
        if (!candidate->draft->enable_vulkan(&err) ||
            !tinyqwen::DFlashExecutionGraph::create(candidate->file, candidate->draft_file,
                                                      *candidate->model, *candidate->draft,
                                                      nullptr, &err, &candidate->graph))
            return env->NewStringUTF(("error: Vulkan DFlash: " + err).c_str());
        candidate->max_context = max_context;
        candidate->block_width = width;
        session = std::move(candidate);
        return env->NewStringUTF("ok: model_loaded; backend=vulkan-dflash");
    } catch (const std::exception &exc) {
        return env->NewStringUTF((std::string("error: ") + exc.what()).c_str());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_filememoryagent_runtime_native_NativeRuntime_nativeStats(JNIEnv *env, jobject) {
    std::lock_guard<std::mutex> lock(session_mutex);
    if (!session || !session->has_stats) return env->NewStringUTF("no_stats");
    const auto &s = session->last_stats;
    char buffer[256];
    if (session->graph) {
        std::snprintf(buffer, sizeof(buffer),
                      "vulkan-dflash prefill=%.1fms decode=%.1fms draft=%.1fms "
                      "verify=%.1fms blocks=%d accepted=%d/%d",
                      s.prefill_ms, s.decode_ms, s.draft_ms, s.target_verify_ms,
                      s.blocks, s.draft_accepted, s.draft_proposed);
    } else {
        std::snprintf(buffer, sizeof(buffer), "cpu prefill=%.1fms decode=%.1fms",
                      s.prefill_ms, s.decode_ms);
    }
    return env->NewStringUTF(buffer);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_filememoryagent_runtime_native_NativeRuntime_nativeUnload(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(session_mutex);
    session.reset();
}
