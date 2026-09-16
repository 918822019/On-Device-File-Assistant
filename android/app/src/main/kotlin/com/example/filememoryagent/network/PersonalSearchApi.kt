package com.example.filememoryagent.network

import com.example.filememoryagent.model.FileActionRequest
import com.example.filememoryagent.model.FileSearchRequest
import com.example.filememoryagent.model.FileSearchResponse
import com.example.filememoryagent.model.FileSearchSessionRequest
import com.example.filememoryagent.model.FileActionResponse
import com.example.filememoryagent.model.RebuildIndexResponse
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST
import java.util.UUID
import java.util.concurrent.TimeUnit

interface PersonalSearchApi {
    @POST("/v1/search-agent/search")
    suspend fun search(@Body request: FileSearchRequest): FileSearchResponse

    @POST("/v1/search-agent/clarify")
    suspend fun clarify(@Body request: FileSearchSessionRequest): FileSearchResponse

    @POST("/v1/search-agent/execute")
    suspend fun execute(@Body request: FileActionRequest): FileActionResponse

    @POST("/v1/search-agent/rebuild-index")
    suspend fun rebuildIndex(): RebuildIndexResponse
}

object ApiClient {
    private const val TAG = "PersonalSearchApi"

    fun create(baseUrl: String): PersonalSearchApi {
        val normalizedBaseUrl = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
        val logger = HttpLoggingInterceptor { msg ->
            com.example.filememoryagent.runtime.logging.RuntimeEventLog.log(null, TAG, "http", msg)
        }.apply {
            level = HttpLoggingInterceptor.Level.BASIC
        }

        val httpClient = OkHttpClient.Builder()
            .addInterceptor { chain ->
                val traceId = UUID.randomUUID().toString()
                val startedAt = System.nanoTime()
                val request = chain.request().newBuilder()
                    .addHeader("x-trace-id", traceId)
                    .build()
                com.example.filememoryagent.runtime.logging.RuntimeEventLog.log(
                    null,
                    TAG,
                    "request_start",
                    "trace=" + traceId + " method=" + request.method + " path=" + request.url.encodedPath,
                )
                runCatching { chain.proceed(request) }.onFailure {
                    com.example.filememoryagent.runtime.logging.RuntimeEventLog.logException(
                        null, TAG, "request_failed", it, "trace=" + traceId,
                    )
                }.getOrThrow().also { response ->
                    val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000
                    com.example.filememoryagent.runtime.logging.RuntimeEventLog.log(
                        null,
                        TAG,
                        "response",
                        "trace=" + traceId + " code=" + response.code + " elapsed_ms=" + elapsedMs,
                    )
                }
            }
            .addInterceptor(logger)
            .connectTimeout(12, TimeUnit.SECONDS)
            .readTimeout(25, TimeUnit.SECONDS)
            .writeTimeout(25, TimeUnit.SECONDS)
            .build()

        return Retrofit.Builder()
            .baseUrl(normalizedBaseUrl.ifBlank { "http://10.0.2.2:9000/" })
            .client(httpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(PersonalSearchApi::class.java)
    }
}
