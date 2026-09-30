"""个人文件动作编排：与 HTTP 框架无关。"""

import logging
from time import perf_counter

from ..common.path_utils import to_windows_path, uri_to_path
from ..personal_search.schemas import FileActionRequest, FileActionResponse
from ..personal_search.service import PersonalFileSearchService

_LOGGER = logging.getLogger("agent_server.personal_search.router")


class FileActionError(Exception):
    """动作失败，由 HTTP 适配层映射为原有状态码和响应。"""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def execute_file_action(
    req: FileActionRequest,
    service: PersonalFileSearchService,
    trace_id: str | None = None,
) -> FileActionResponse:
    """命中结果后执行 open/share/annotate/archive/compare 并记录独立事件。"""

    request_trace_id = trace_id
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
        raise FileActionError(status_code=404, detail="未找到目标文件")

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
        # WSL / Windows 原生部署：file URI 对 Windows 浏览器无意义或含前导斜杠，
        # 附带映射后的 Windows 路径（C:\...）供前端展示/复制。
        # macOS / Linux 为 None，行为不变。
        windows_path = to_windows_path(
            file_item.file_path or uri_to_path(file_item.file_uri)
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
            raise FileActionError(status_code=400, detail="share 动作需要填写 share_to")
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
            raise FileActionError(status_code=400, detail="annotate 动作需要 note")
        success = service.add_annotation(req.file_id, req.note)
        if not success:
            raise FileActionError(status_code=404, detail="未找到目标文件")
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
            raise FileActionError(status_code=400, detail="compare 动作需要 peer_file_id")
        if req.peer_file_id == req.file_id:
            raise FileActionError(status_code=400, detail="对比文件不能与自己一致")
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
            raise FileActionError(status_code=404, detail="对比对象未找到") from exc

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
            raise FileActionError(status_code=404, detail="未找到目标文件")
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

