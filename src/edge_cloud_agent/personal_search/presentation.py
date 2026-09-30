"""个人文件搜索结果的公共响应组装与下一步建议。"""

from .relevance import SearchCandidate
from .schemas import FileSearchCandidate, FileSearchResponse


def to_search_candidate(candidate: SearchCandidate) -> FileSearchCandidate:
    """内部候选结构体转对外 schema，截断预览文本。"""

    item = candidate.item
    preview = item.summary or item.raw_text[:60]
    if not preview:
        preview = "无可检索文本，保留文件名与来源线索"
    return FileSearchCandidate(
        file_id=item.file_id,
        title=item.title,
        source_app=item.source_app,
        doc_type=item.doc_type,
        mime_type=item.mime_type,
        file_uri=item.file_uri,
        captured_at=item.captured_at,
        score=round(candidate.score, 4),
        evidence=candidate.evidence,
        preview=preview[:180],
        visual_hints=item.visual_hints,
        matched_clues=candidate.matched_clues,
    )


def first_resolved_file_id(candidates: list[FileSearchCandidate], state: str) -> str | None:
    """首次搜索确定结果时默认选中第一条；澄清阶段不自动选中。"""

    if state != "resolved" or not candidates:
        return None
    return candidates[0].file_id


def _action_suggestions(state: str, has_candidates: bool, resolved_file: bool) -> list[str]:
    if state == "resolved" and has_candidates:
        suggestions = ["打开原件", "分享给别人", "加备注", "归档"]
        if resolved_file:
            suggestions.append("对比版本")
        return suggestions
    if state == "needs_clarification":
        return ["补一个线索继续确认", "直接选一个候选"]
    if state == "not_found":
        return ["补充时间/来源/场景再搜"]
    return []


def build_search_response(
    *,
    query: str,
    session_id: str,
    state: str,
    needs_disambiguation: bool,
    question: str | None,
    candidates: list[FileSearchCandidate],
    selected_file_id: str | None,
) -> FileSearchResponse:
    """保持首次搜索和继续澄清的字段及建议动作口径一致。"""

    return FileSearchResponse(
        query=query,
        session_id=session_id,
        state=state,
        needs_disambiguation=needs_disambiguation,
        question=question,
        candidates=candidates,
        selected_file_id=selected_file_id,
        next_action_suggestions=_action_suggestions(state, bool(candidates), selected_file_id is not None),
    )
