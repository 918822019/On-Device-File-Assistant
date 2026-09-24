"""Personal file search APIs for first-stage闭环。

提供三类端点：
- `search`：首次检索并创建 session
- `clarify`：基于回复/选中条目继续消歧
- `execute`：在命中后做打开/分享/备注/归档/对比
同时所有动作都带 trace_id，用于关联日志链路。
"""

from __future__ import annotations

import logging
import threading
import time
from fastapi import APIRouter, HTTPException, Request
from time import perf_counter

from ..analytics import record_safe
from ..config import PersonalFileConfig
from ..path_utils import is_wsl, wsl_to_windows_path
from ..personal_search.ingest import run_once
from ..personal_search.schemas import (
    FileActionRequest,
    FileActionResponse,
    RebuildIndexResponse,
    FileSearchResponse,
    FileSearchRequest,
    FileSearchSessionRequest,
)
from ..personal_search.service import PersonalFileSearchService

router = APIRouter()
_LOGGER = logging.getLogger("agent_server.personal_search.router")

# 节流异步刷新的全局状态：防止并发请求同时触发多个后台扫描。
_REFRESH_LOCK = threading.Lock()


def _maybe_refresh_index(request: Request, service: PersonalFileSearchService, cfg: PersonalFileConfig, trace_id: str | None) -> None:
    """带节流的异步增量刷新。

    此前 /search 每次请求都同步 run_once（全量目录扫描 + 读文件 + embedding +
    FAISS 重建），文件量上来后首查延迟不可接受，且与后台 watch 线程重复劳动。
    现在改为：每 scan_interval_seconds 窗口内最多触发一次后台扫描，请求立即返回；
    新鲜度由「启动 warm scan + 后台 watch loop + 本节流刷新」共同保证。
    """

    state = request.app.state
    now = time.monotonic()
    window = max(5, cfg.scan_interval_seconds)
    with _REFRESH_LOCK:
        last = getattr(state, "personal_refresh_ts", 0.0)
        running = getattr(state, "personal_refresh_running", False)
        if running or now - last < window:
            return
        state.personal_refresh_ts = now
        state.personal_refresh_running = True

    def _worker() -> None:
        try:
            run_once(service, cfg, trace_id=trace_id)
        except Exception:
            _LOGGER.warning(
                "personal-search-api-refresh-failed",
                extra={
                    "event": "api.search.refresh_failed",
                    "trace_id": trace_id,
                    "path": "/v1/search-agent/search",
                },
            )
        finally:
            with _REFRESH_LOCK:
                state.personal_refresh_running = False

    threading.Thread(target=_worker, name="personal-file-refresh", daemon=True).start()


def _get_service(request: Request) -> PersonalFileSearchService:
    """从 app.state 读取已初始化好的 service；未就绪直接返回 503。"""

    service: PersonalFileSearchService | None = getattr(
        request.app.state,
        "personal_file_service",
        None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="个人文件检索服务未就绪，请配置 FILE_MEMORY_SOURCE_DIR 后重启服务。",
        )
    return service


def _service_config(request: Request) -> PersonalFileConfig:
    """读取 PersonalFileConfig；若未设置则走默认配置。"""

    return getattr(request.app.state, "personal_file_cfg", PersonalFileConfig())


# 与 schemas.FileSearchRequest / FileSearchSessionRequest 的 le=20 约束保持一致。
_TOP_K_HARD_LIMIT = 20


def _normalize_top_k(top_k: int) -> int:
    """按 schema 声明的允许区间钳制 top_k。

    此前钳到 cfg.top_k_default（默认 8）：用户显式传 top_k=20 会被静默截到 8，
    与 schema 的 le=20 相互矛盾。top_k_default 是"客户端未指定"时的缺省值，
    不应充当上限。
    """

    return max(1, min(top_k, _TOP_K_HARD_LIMIT))


def _action_suggestions(state: str, has_candidates: bool, resolved_file: bool) -> list[str]:
    """根据状态给前端返回建议动作，避免用户不知道下一步。"""

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


def _trace_id(request: Request) -> str | None:
    """从 middleware 注入的 request.state 中读取 trace_id。"""

    return getattr(request.state, "trace_id", None)


def _first_resolved_file_id(candidates: list, state: str) -> str | None:
    """若已 resolved 且有候选，直接返回第一条作为默认落点。"""

    if state != "resolved" or not candidates:
        return None
    return candidates[0].file_id


@router.post("/v1/search-agent/search", response_model=FileSearchResponse)
def search(req: FileSearchRequest, request: Request):
    """首次搜索：生成 session 并返回 candidates/状态/澄清问题。"""

    service = _get_service(request)
    cfg: PersonalFileConfig = _service_config(request)
    request_trace_id = _trace_id(request)
    top_k = _normalize_top_k(req.top_k)
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

    if cfg.source_dir:
        # 节流异步刷新：不再阻塞本次请求（同步全量扫描会让首查延迟随文件数爆炸）。
        _maybe_refresh_index(request, service, cfg, request_trace_id)
    session_id, candidates, state, should_disambiguate, question = service.start_search(
        req.query,
        top_k=top_k,
        force_disambiguation=req.force_disambiguation,
        trace_id=request_trace_id,
    )
    selected_file_id = _first_resolved_file_id(candidates, state)
    _LOGGER.info(
        "personal-search-api-search-completed",
        extra={
            "event": "api.search.completed",
            "trace_id": request_trace_id,
            "search_id": session_id,
            "state": state,
            "candidate_count": len(candidates),
            "selected_file_id": selected_file_id,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
            "path": "/v1/search-agent/search",
        },
    )

    # 复盘埋点：会话首轮状态（追问收敛率漏斗的入口）
    record_safe(
        getattr(request.app.state, "metrics", None),
        "record_search_state",
        session_id,
        state,
        "search",
    )

    return FileSearchResponse(
        query=req.query,
        session_id=session_id,
        state=state,
        needs_disambiguation=should_disambiguate,
        question=question,
        candidates=candidates,
        selected_file_id=selected_file_id,
        next_action_suggestions=_action_suggestions(state, bool(candidates), selected_file_id is not None),
    )


@router.post("/v1/search-agent/clarify", response_model=FileSearchResponse)
def clarify(req: FileSearchSessionRequest, request: Request):
    """继续澄清：支持用户直接选ID或追加线索文本。"""

    service = _get_service(request)
    cfg: PersonalFileConfig = _service_config(request)
    request_trace_id = _trace_id(request)
    top_k = _normalize_top_k(req.top_k)
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
        session_id, original_query, candidates, state, should_disambiguate, question, selected_file_id = service.continue_search(
            session_id=req.session_id,
            selected_file_id=req.selected_file_id,
            reply=req.reply,
            top_k=top_k,
            trace_id=request_trace_id,
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
    record_safe(
        getattr(request.app.state, "metrics", None),
        "record_search_state",
        session_id,
        state,
        "clarify",
    )

    return FileSearchResponse(
        query=original_query,
        session_id=session_id,
        state=state,
        needs_disambiguation=should_disambiguate,
        question=question,
        candidates=candidates,
        selected_file_id=selected_file_id,
        next_action_suggestions=_action_suggestions(state, bool(candidates), selected_file_id is not None),
    )


@router.post("/v1/search-agent/execute", response_model=FileActionResponse)
def execute(req: FileActionRequest, request: Request):
    """命中结果后的动作执行（复盘埋点包装层）。

    动作逻辑在 _execute_impl；本层只负责把 (session_id, action, status)
    记入指标事件流。HTTPException（400/404 等）不构成"执行动作"，不记录。
    """

    resp = _execute_impl(req, request)
    record_safe(
        getattr(request.app.state, "metrics", None),
        "record_search_action",
        req.session_id,
        resp.action,
        resp.status,
    )
    return resp


def _execute_impl(req: FileActionRequest, request: Request) -> FileActionResponse:
    """命中结果后的动作执行。

动作包括：open/share/annotate/archive/compare。
每个动作都会输出独立 event，后续可据此做审计与埋点。
    """

    service = _get_service(request)
    request_trace_id = _trace_id(request)
    started = perf_counter()

    file_item = service.get_file(req.file_id)
    if file_item is None:
        _LOGGER.warning(
            "personal-search-api-execute-missing-file",
            extra={
                "event": "api.execute.file_not_found",
                "trace_id": request_trace_id,
                "file_id": req.file_id,
                "action": req.action,
                "path": "/v1/search-agent/execute",
            },
        )
        raise HTTPException(status_code=404, detail="未找到目标文件")

    action = req.action
    _LOGGER.info(
        "personal-search-api-execute-requested",
        extra={
            "event": "api.execute.requested",
            "trace_id": request_trace_id,
            "file_id": req.file_id,
            "action": action,
            "share_to": req.share_to,
            "path": "/v1/search-agent/execute",
        },
    )

    if action == "open":
        if not file_item.file_uri:
            _LOGGER.warning(
                "personal-search-api-execute-open-missing-uri",
                extra={
                    "event": "api.execute.open_missing_uri",
                    "trace_id": request_trace_id,
                    "file_id": req.file_id,
                    "path": "/v1/search-agent/execute",
                },
            )
            return FileActionResponse(
                action=action,
                status="failed",
                file_id=req.file_id,
                file_title=file_item.title,
                file_uri=file_item.file_uri,
                message="文件缺少可直接打开的 URI",
                archived=service.is_archived(req.file_id),
                annotations=service.get_annotation(req.file_id),
                next_action_suggestions=["分享", "加备注", "归档"],
            )
        # WSL 部署：file:///mnt/c/... 对 Windows 浏览器无意义，附带映射后的
        # Windows 路径（C:\...）供前端展示/复制。非 WSL 环境为 None，行为不变。
        windows_path = (
            wsl_to_windows_path(file_item.file_path) if is_wsl() else None
        )
        _LOGGER.info(
            "personal-search-api-execute-open-ok",
            extra={
                "event": "api.execute.open_ok",
                "trace_id": request_trace_id,
                "file_id": req.file_id,
                "windows_path_mapped": windows_path is not None,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "path": "/v1/search-agent/execute",
            },
        )
        return FileActionResponse(
            action=action,
            status="ok",
            file_id=req.file_id,
            file_title=file_item.title,
            file_uri=file_item.file_uri,
            windows_path=windows_path,
            message=f"已定位到原件：{file_item.title}",
            archived=service.is_archived(req.file_id),
            annotations=service.get_annotation(req.file_id),
            next_action_suggestions=["分享", "加备注", "归档", "对比版本"],
        )

    if action == "share":
        if not req.share_to:
            raise HTTPException(status_code=400, detail="share 动作需要填写 share_to")
        _LOGGER.info(
            "personal-search-api-execute-share",
            extra={
                "event": "api.execute.share",
                "trace_id": request_trace_id,
                "file_id": req.file_id,
                "share_to": req.share_to,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "path": "/v1/search-agent/execute",
            },
        )
        return FileActionResponse(
            action=action,
            status="ok",
            file_id=req.file_id,
            file_title=file_item.title,
            file_uri=file_item.file_uri,
            message=f"已准备将“{file_item.title}”分享给“{req.share_to}”",
            share_payload={
                "to": req.share_to,
                "file_uri": file_item.file_uri,
            },
            archived=service.is_archived(req.file_id),
            annotations=service.get_annotation(req.file_id),
            next_action_suggestions=["打开原件", "加备注", "归档"],
        )

    if action == "annotate":
        if not req.note:
            raise HTTPException(status_code=400, detail="annotate 动作需要 note")
        success = service.add_annotation(req.file_id, req.note)
        if not success:
            raise HTTPException(status_code=404, detail="未找到目标文件")
        _LOGGER.info(
            "personal-search-api-execute-annotate",
            extra={
                "event": "api.execute.annotate",
                "trace_id": request_trace_id,
                "file_id": req.file_id,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "path": "/v1/search-agent/execute",
            },
        )
        return FileActionResponse(
            action=action,
            status="ok",
            file_id=req.file_id,
            file_title=file_item.title,
            file_uri=file_item.file_uri,
            message=f"备注已保存：{req.note}",
            annotations=req.note,
            archived=service.is_archived(req.file_id),
            next_action_suggestions=["打开原件", "分享", "归档"],
        )

    if action == "compare":
        if not req.peer_file_id:
            raise HTTPException(status_code=400, detail="compare 动作需要 peer_file_id")
        if req.peer_file_id == req.file_id:
            raise HTTPException(status_code=400, detail="对比文件不能与自己一致")
        try:
            payload = service.compare_payload(req.file_id, req.peer_file_id)
        except ValueError as exc:
            _LOGGER.warning(
                "personal-search-api-execute-compare-not-found",
                extra={
                    "event": "api.execute.compare_payload_not_found",
                    "trace_id": request_trace_id,
                    "file_id": req.file_id,
                    "peer_file_id": req.peer_file_id,
                    "path": "/v1/search-agent/execute",
                },
            )
            raise HTTPException(status_code=404, detail="对比对象未找到") from exc

        _LOGGER.info(
            "personal-search-api-execute-compare",
            extra={
                "event": "api.execute.compare",
                "trace_id": request_trace_id,
                "file_id": req.file_id,
                "peer_file_id": req.peer_file_id,
                "path": "/v1/search-agent/execute",
            },
        )

        return FileActionResponse(
            action=action,
            status="ok",
            file_id=req.file_id,
            file_title=file_item.title,
            file_uri=file_item.file_uri,
            message="已输出对比建议",
            compare_payload=payload,
            archived=service.is_archived(req.file_id),
            annotations=service.get_annotation(req.file_id),
            next_action_suggestions=["打开原件", "加备注", "归档"],
        )

    if action == "archive":
        ok = service.archive_file(req.file_id)
        if not ok:
            raise HTTPException(status_code=404, detail="未找到目标文件")
        _LOGGER.info(
            "personal-search-api-execute-archive",
            extra={
                "event": "api.execute.archive",
                "trace_id": request_trace_id,
                "file_id": req.file_id,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
                "path": "/v1/search-agent/execute",
            },
        )
        return FileActionResponse(
            action=action,
            status="ok",
            file_id=req.file_id,
            file_title=file_item.title,
            file_uri=file_item.file_uri,
            message="已将文件加入归档标记",
            archived=True,
            annotations=service.get_annotation(req.file_id),
            next_action_suggestions=["打开原件", "分享", "加备注"],
        )

    _LOGGER.warning(
        "personal-search-api-execute-unsupported",
        extra={
            "event": "api.execute.unsupported",
            "trace_id": request_trace_id,
            "file_id": req.file_id,
            "action": action,
            "path": "/v1/search-agent/execute",
        },
    )
    return FileActionResponse(
        action=action,
        status="unsupported",
        file_id=req.file_id,
        file_title=file_item.title,
        file_uri=file_item.file_uri,
        message="不支持该动作",
        archived=service.is_archived(req.file_id),
        annotations=service.get_annotation(req.file_id),
        next_action_suggestions=["打开原件", "分享", "加备注"],
    )


@router.post("/v1/search-agent/rebuild-index", response_model=RebuildIndexResponse)
def rebuild_index(request: Request):
    """手工触发一次个人文件索引重建（扫描+向量重建）。"""

    service = _get_service(request)
    cfg: PersonalFileConfig = _service_config(request)
    request_trace_id = _trace_id(request)
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
        result = run_once(service, cfg, trace_id=request_trace_id)
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
        material_ids=result.material_ids,
    )
