package com.example.filememoryagent.model

import com.google.gson.annotations.SerializedName

data class FileSearchRequest(
    val query: String,
    @SerializedName("top_k")
    val topK: Int = 6,
    @SerializedName("force_disambiguation")
    val forceDisambiguation: Boolean = false,
)

data class FileSearchSessionRequest(
    @SerializedName("session_id")
    val sessionId: String,
    @SerializedName("top_k")
    val topK: Int = 6,
    @SerializedName("selected_file_id")
    val selectedFileId: String? = null,
    val reply: String? = null,
)

data class FileSearchCandidate(
    @SerializedName("file_id")
    val fileId: String,
    val title: String,
    @SerializedName("source_app")
    val sourceApp: String,
    @SerializedName("doc_type")
    val docType: String,
    @SerializedName("mime_type")
    val mimeType: String? = null,
    @SerializedName("file_uri")
    val fileUri: String? = null,
    @SerializedName("captured_at")
    val capturedAt: String? = null,
    val score: Double,
    val evidence: String,
    val preview: String,
    @SerializedName("visual_hints")
    val visualHints: List<String> = emptyList(),
    @SerializedName("matched_clues")
    val matchedClues: List<String> = emptyList(),
)

data class FileSearchResponse(
    val query: String,
    @SerializedName("session_id")
    val sessionId: String,
    val state: String,
    @SerializedName("needs_disambiguation")
    val needsDisambiguation: Boolean,
    val question: String? = null,
    val candidates: List<FileSearchCandidate> = emptyList(),
    @SerializedName("selected_file_id")
    val selectedFileId: String? = null,
    @SerializedName("next_action_suggestions")
    val nextActionSuggestions: List<String> = emptyList(),
)

data class FileActionRequest(
    @SerializedName("session_id")
    val sessionId: String? = null,
    @SerializedName("file_id")
    val fileId: String,
    val action: String,
    @SerializedName("share_to")
    val shareTo: String? = null,
    @SerializedName("peer_file_id")
    val peerFileId: String? = null,
    val note: String? = null,
)

data class FileActionResponse(
    val action: String,
    val status: String,
    @SerializedName("file_id")
    val fileId: String,
    @SerializedName("file_title")
    val fileTitle: String? = null,
    @SerializedName("file_uri")
    val fileUri: String? = null,
    val message: String,
    @SerializedName("share_payload")
    val sharePayload: Map<String, String>? = null,
    @SerializedName("compare_payload")
    val comparePayload: Map<String, String>? = null,
    val annotations: String? = null,
    val archived: Boolean = false,
    @SerializedName("next_action_suggestions")
    val nextActionSuggestions: List<String> = emptyList(),
)

data class RebuildIndexResponse(
    val scanned: Int,
    val imported: Int,
    val skipped: Int,
    val errors: Int,
    val materialIds: List<String> = emptyList(),
)

data class LogEvent(
    val ts: Long,
    val text: String,
)

