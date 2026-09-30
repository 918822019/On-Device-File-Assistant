"""首轮文件检索用例：检索、节流刷新与复盘埋点的公共入口。

聊天 Agent 和 /v1/search-agent/search 走同一条路径，避免路由互相调用，
也保证「触发后台刷新 + 记录首轮状态」的口径一致。
"""

from ..analytics import record_safe
from .presentation import build_search_response, first_resolved_file_id
from .refresh import IndexRefreshScheduler
from .schemas import FileSearchResponse
from .service import PersonalFileSearchService

# 与 schemas.FileSearchRequest / FileSearchSessionRequest 的 le=20 约束保持一致。
TOP_K_HARD_LIMIT = 20
DEFAULT_TOP_K = 6


def normalize_top_k(top_k: int) -> int:
    """按 schema 声明的允许区间钳制 top_k。

    top_k_default 是客户端未指定时的缺省值，不应充当上限：显式传 20 就该返回 20。
    """

    return max(1, min(top_k, TOP_K_HARD_LIMIT))


def start_file_search(
    service: PersonalFileSearchService,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    force_disambiguation: bool = False,
    trace_id: str | None = None,
    refresher: IndexRefreshScheduler | None = None,
    metrics: object | None = None,
) -> FileSearchResponse:
    """发起首轮检索会话，返回可直接对外的响应结构。"""

    if refresher is not None:
        refresher.trigger(trace_id)
    top_k = normalize_top_k(top_k)
    session_id, candidates, state, should_disambiguate, question = service.start_search(
        query,
        top_k=top_k,
        force_disambiguation=force_disambiguation,
        trace_id=trace_id,
    )
    selected_file_id = first_resolved_file_id(candidates, state)
    # 复盘埋点：会话首轮状态（追问收敛率漏斗的入口）
    record_safe(metrics, "record_search_state", session_id, state, "search")
    return build_search_response(
        query=query,
        session_id=session_id,
        state=state,
        needs_disambiguation=should_disambiguate,
        question=question,
        candidates=candidates,
        selected_file_id=selected_file_id,
    )
