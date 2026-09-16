package com.example.filememoryagent.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.filememoryagent.model.FileActionResponse
import com.example.filememoryagent.model.FileSearchCandidate
import com.example.filememoryagent.model.FileSearchResponse
import com.example.filememoryagent.model.LogEvent
import com.example.filememoryagent.repository.SearchRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class SearchUiState(
    val query: String = "",
    val clarifyInput: String = "",
    val topK: Int = 6,
    val shareTo: String = "",
    val note: String = "",
    val loading: Boolean = false,
    val response: FileSearchResponse? = null,
    val logs: List<LogEvent> = emptyList(),
    val error: String? = null,
)

sealed interface ActionResult {
    data class Open(val uri: String, val title: String) : ActionResult
    data class Share(val payload: Map<String, String>?, val fallbackText: String) : ActionResult
    data class Message(val title: String, val message: String) : ActionResult
    data class Error(val message: String) : ActionResult
}

class MainViewModel(
    private val repository: SearchRepository = SearchRepository(),
) : ViewModel() {

    private val _state = MutableStateFlow(SearchUiState())
    val state: StateFlow<SearchUiState> = _state
    private val _actionEvents = MutableStateFlow<ActionResult?>(null)
    val actionEvents: StateFlow<ActionResult?> = _actionEvents

    fun setQuery(value: String) {
        _state.update { it.copy(query = value) }
    }

    fun setClarifyInput(value: String) {
        _state.update { it.copy(clarifyInput = value) }
    }

    fun setShareTo(value: String) {
        _state.update { it.copy(shareTo = value) }
    }

    fun setNote(value: String) {
        _state.update { it.copy(note = value) }
    }

    fun search() {
        val query = _state.value.query.trim()
        if (query.isBlank()) {
            _state.update { it.copy(error = "查询不能为空") }
            return
        }
        _state.update { it.copy(loading = true, error = null, response = null, clarifyInput = "") }
        viewModelScope.launch {
            try {
                val result = repository.search(query, _state.value.topK, forceDisambiguation = false)
                _state.update {
                    it.copy(
                        loading = false,
                        response = result,
                        clarifyInput = "",
                        logs = repository.logs.value,
                    )
                }
            } catch (e: Exception) {
                _state.update { it.copy(loading = false, error = e.message ?: "搜索失败") }
            }
        }
    }

    fun clarify() {
        val current = _state.value.response ?: return
        val reply = _state.value.clarifyInput.trim()
        if (reply.isBlank()) {
            _state.update { it.copy(error = "请输入要补充的线索，或直接点候选文件") }
            return
        }
        _state.update { it.copy(loading = true, error = null) }
        viewModelScope.launch {
            try {
                val result = repository.clarify(current.sessionId, current.selectedFileId, reply, _state.value.topK)
                _state.update {
                    it.copy(
                        loading = false,
                        response = result,
                        clarifyInput = "",
                        logs = repository.logs.value,
                    )
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(loading = false, error = e.message ?: "追问失败")
                }
            }
        }
    }

    fun execute(fileId: String, action: String, shareTo: String? = null, peerFileId: String? = null, note: String? = null) {
        val sessionId = _state.value.response?.sessionId
        viewModelScope.launch {
            try {
                val result = repository.execute(sessionId, fileId, action, shareTo, peerFileId, note)
                _state.update {
                    it.copy(logs = repository.logs.value)
                }
                _actionEvents.value = when (action) {
                    "open" -> ActionResult.Open(result.fileUri.orEmpty(), result.fileTitle.orEmpty())
                    "share" -> ActionResult.Share(result.sharePayload, result.message)
                    else -> ActionResult.Message(result.fileTitle.orEmpty(), result.message)
                }
                if (result.status != "ok") {
                    _actionEvents.value = ActionResult.Error("动作执行失败：${result.message}")
                }
            } catch (e: Exception) {
                _actionEvents.value = ActionResult.Error(e.message ?: "执行失败")
            }
        }
    }

    fun selectCandidate(candidate: FileSearchCandidate) {
        execute(candidate.fileId, action = "open")
    }

    fun markActionEventHandled() {
        _actionEvents.value = null
    }

    fun rebuildIndex() {
        _state.update { it.copy(loading = true, error = null) }
        viewModelScope.launch {
            try {
                repository.rebuildIndex()
                _state.update {
                    it.copy(loading = false, logs = repository.logs.value)
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(loading = false, error = e.message ?: "重建索引失败")
                }
            }
        }
    }
}
