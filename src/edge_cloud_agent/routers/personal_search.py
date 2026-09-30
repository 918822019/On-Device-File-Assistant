"""Personal file search APIs for first-stage闭环。

提供三类端点：
- `search`：首次检索并创建 session
- `clarify`：基于回复/选中条目继续消歧
- `execute`：在命中后做打开/分享/备注/归档/对比
同时所有动作都带 trace_id，用于关联日志链路。
"""

from __future__ import annotations

import logging
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request

from ..personal_search.actions import FileActionError, execute_file_action
from ..personal_search.coverage import build_index_status
from ..personal_search.ingest import run_once
from ..personal_search.presentation import build_search_response
from ..personal_search.schemas import (
    FileActionRequest,
    FileActionResponse,
    FileSearchRequest,
    FileSearchResponse,
    FileSearchSessionRequest,
    IndexStatusResponse,
    RebuildIndexResponse,
)
from ..personal_search.search_flow import normalize_top_k, start_file_search
from . import deps

router = APIRouter()
_LOGGER = logging.getLogger("agent_server.personal_search.router")


@router.post("/v1/search-agent/search", response_model=FileSearchResponse)
def search(req: FileSearchRequest, request: Request):
    """首次搜索：生成 session 并返回 candidates/状态/澄清问题。"""

    service = deps.personal_file_service(request)
    request_trace_id = deps.trace_id(request)
    top_k = normalize_top_k(req.top_k)
    _LOGGER.info(
        "personal-search-api-search-requested",
        extra={
            "event": "api.search.requested",
            "trace_id": request_trace_id,
            "query_len": len(req.query),
            "top_k": top_k,
            "force_disambiguation": req.force_disambiguation,
            "path": "/v1/search-agent/search",
        },
    )
    started = perf_counter()

    response = start_file_search(
        service,
        req.query,
        top_k=top_k,
        force_disambiguation=req.force_disambiguation,
        trace_id=request_trace_id,
        refresher=deps.personal_file_refresher(request),
        metrics=deps.metrics(request),
    )

    _LOGGER.info(
        "personal-search-api-search-completed",
        extra={
            "event": "api.search.completed",
            "trace_id": request_trace_id,
            "search_id": response.session_id,
            "state": response.state,
            "candidate_count": len(response.candidates),
            "selected_file_id": response.selected_file_id,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "path": "/v1/search-agent/search",
        },
    )
    return response


@router.post("/v1/search-agent/clarify", response_model=FileSearchResponse)
def clarify(req: FileSearchSessionRequest, request: Request):
    """继续澄清：支持用户直接选ID或追加线索文本。"""

    service = deps.personal_file_service(request)
    request_trace_id = deps.trace_id(request)
    top_k = normalize_top_k(req.top_k)
    started = perf_counter()

    _LOGGER.info(
        "personal-search-api-clarify-requested",
        extra={
            "event": "api.clarify.requested",
            "trace_id": request_trace_id,
            "session_id": req.session_id,
            "top_k": top_k,
            "reply_len": len(req.reply or ""),
            "path": "/v1/search-agent/clarify",
        },
    )

    try:
        session_id, original_query, candidates, state, should_disambiguate, question, selected_file_id = (
            service.continue_search(
                session_id=req.session_id,
                selected_file_id=req.selected_file_id,
                reply=req.reply,
                top_k=top_k,
                trace_id=request_trace_id,
            )
        )
    except ValueError as exc:
        _LOGGER.warning(
            "personal-search-api-clarify-session-not-found",
            extra={
                "event": "api.clarify.session_not_found",
                "trace_id": request_trace_id,
                "session_id": req.session_id,
            },
        )
        raise HTTPException(status_code=404, detail="会话已失效，请重新发起搜索") from exc

    _LOGGER.info(
        "personal-search-api-clarify-completed",
        extra={
            "event": "api.clarify.completed",
            "trace_id": request_trace_id,
            "search_id": session_id,
            "state": state,
            "candidate_count": len(candidates),
            "selected_file_id": selected_file_id,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "path": "/v1/search-agent/clarify",
        },
    )

    # 复盘埋点：追问轮状态（任一轮 resolved 即计入收敛）
    deps.record_metric(request, "record_search_state", session_id, state, "clarify")

    return build_search_response(
        query=original_query,
        session_id=session_id,
        state=state,
        needs_disambiguation=should_disambiguate,
        question=question,
        candidates=candidates,
        selected_file_id=selected_file_id,
    )


@router.post("/v1/search-agent/execute", response_model=FileActionResponse)
def execute(req: FileActionRequest, request: Request):
    """命中结果后的动作执行（复盘埋点包装层）。

    动作逻辑在 personal_search.actions；本层只负责把 (session_id, action, status)
    记入指标事件流。HTTPException（400/404 等）不构成"执行动作"，不记录。
    """

    try:
        resp = execute_file_action(req, deps.personal_file_service(request), deps.trace_id(request))
    except FileActionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    deps.record_metric(request, "record_search_action", req.session_id, resp.action, resp.status)
    return resp


@router.get("/v1/search-agent/index-status", response_model=IndexStatusResponse)
def index_status(request: Request) -> IndexStatusResponse:
    """Expose index coverage without returning document contents or filenames."""

    return build_index_status(deps.personal_file_service(request), deps.personal_file_config(request))


@router.post("/v1/search-agent/rebuild-index", response_model=RebuildIndexResponse)
def rebuild_index(request: Request):
    """手工触发一次个人文件索引重建（扫描+向量重建）。"""

    service = deps.personal_file_service(request)
    cfg = deps.personal_file_config(request)
    request_trace_id = deps.trace_id(request)
    started = perf_counter()
    _LOGGER.info(
        "personal-search-api-rebuild-requested",
        extra={
            "event": "api.rebuild.requested",
            "trace_id": request_trace_id,
            "path": "/v1/search-agent/rebuild-index",
        },
    )
    try:
        # force_rehash=True：手工重建要做权威的 hash 校验，绕过 size+mtime 快路径。
        # 云同步/备份恢复可能把 mtime 还原成旧值，那种变更只有重算 hash 能发现，
        # 这个按钮就是它的兜底出口。
        result = run_once(service, cfg, trace_id=request_trace_id, force_rehash=True)
    except Exception as exc:  # pragma: no cover
        _LOGGER.exception(
            "personal-search-api-rebuild-failed",
            extra={
                "event": "api.rebuild.failed",
                "trace_id": request_trace_id,
                "path": "/v1/search-agent/rebuild-index",
            },
        )
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc

    _LOGGER.info(
        "personal-search-api-rebuild-completed",
        extra={
            "event": "api.rebuild.completed",
            "trace_id": request_trace_id,
            "scan_count": result.scanned,
            "imported": result.imported,
            "skipped": result.skipped,
            "errors": result.errors,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "path": "/v1/search-agent/rebuild-index",
        },
    )
    return RebuildIndexResponse(
        scanned=result.scanned,
        imported=result.imported,
        skipped=result.skipped,
        errors=result.errors,
        removed=result.removed,
        # wire 字段名被 Android 的 @SerializedName("material_ids") 钉住，不能改；
        # 内部名已是 imported_ids（这里装的是 file_id，不是 material_id）
        material_ids=result.imported_ids,
    )
