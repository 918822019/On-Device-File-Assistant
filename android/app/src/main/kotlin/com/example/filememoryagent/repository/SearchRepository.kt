package com.example.filememoryagent.repository

import com.example.filememoryagent.BuildConfig
import com.example.filememoryagent.model.*
import com.example.filememoryagent.network.ApiClient
import com.example.filememoryagent.network.PersonalSearchApi
import com.example.filememoryagent.runtime.logging.RuntimeEventLog
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class SearchRepository {
    private val api: PersonalSearchApi = ApiClient.create(BuildConfig.API_BASE_URL)
    private val _logs = MutableStateFlow(
        listOf(
            LogEvent(
                ts = System.currentTimeMillis(),
                text = "repo_ready: base_url=${BuildConfig.API_BASE_URL}",
            ),
        ),
    )
    val logs: StateFlow<List<LogEvent>> = _logs

    private fun addLog(message: String) {
        val formatter = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        val row = LogEvent(
            ts = System.currentTimeMillis(),
            text = "[${formatter.format(Date())}] $message",
        )
        RuntimeEventLog.log(null, "api_client", "repo_log", message)
        _logs.value = _logs.value + row
    }

    suspend fun search(query: String, topK: Int, forceDisambiguation: Boolean): FileSearchResponse {
        val req = FileSearchRequest(query = query, topK = topK, forceDisambiguation = forceDisambiguation)
        addLog("search_start query_len=${query.length} top_k=$topK")
        val rsp = api.search(req)
        addLog("search_done session=${rsp.sessionId} state=${rsp.state} candidates=${rsp.candidates.size}")
        return rsp
    }

    suspend fun clarify(sessionId: String, selectedFileId: String?, reply: String?, topK: Int): FileSearchResponse {
        val req = FileSearchSessionRequest(
            sessionId = sessionId,
            selectedFileId = selectedFileId,
            reply = reply,
            topK = topK,
        )
        addLog("clarify_start session=$sessionId selected=$selectedFileId reply=${reply.orEmpty()}")
        val rsp = api.clarify(req)
        addLog("clarify_done session=${rsp.sessionId} state=${rsp.state} candidates=${rsp.candidates.size}")
        return rsp
    }

    suspend fun execute(
        sessionId: String?,
        fileId: String,
        action: String,
        shareTo: String? = null,
        peerFileId: String? = null,
        note: String? = null,
    ): FileActionResponse {
        val req = FileActionRequest(
            sessionId = sessionId,
            fileId = fileId,
            action = action,
            shareTo = shareTo,
            peerFileId = peerFileId,
            note = note,
        )
        addLog("execute_start file=$fileId action=$action")
        val rsp = api.execute(req)
        addLog("execute_done file=${rsp.fileId} status=${rsp.status}")
        return rsp
    }

    suspend fun rebuildIndex(): RebuildIndexResponse {
        addLog("rebuild_start")
        val rsp = api.rebuildIndex()
        addLog("rebuild_done scanned=${rsp.scanned} imported=${rsp.imported}")
        return rsp
    }
}
