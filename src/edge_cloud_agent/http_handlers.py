"""注册请求追踪中间件和统一异常响应。"""

import logging
import uuid
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from .routers.schemas import ErrorResponse

logger = logging.getLogger("agent_server")


def _build_error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    detail: object | None = None,
) -> JSONResponse:
    """统一错误响应结构，保留 trace_id 与 path，便于调用方排障。"""

    trace_id = getattr(request.state, "trace_id", None)
    payload = ErrorResponse(
        code=code,
        message=message,
        status=status_code,
        trace_id=trace_id,
        path=request.url.path,
        detail=detail,
    )
    return JSONResponse(
        status_code=status_code,
        content={"error": payload.model_dump()},
    )


def register_http_handlers(app: FastAPI) -> None:
    """给应用安装 HTTP 追踪、统一错误处理和静态资源缓存规则。"""

    @app.middleware("http")
    async def add_trace_id(request: Request, call_next):
        """请求级中间件：注入 trace_id、统计耗时、回填响应头。"""

        started = perf_counter()
        trace_id = request.headers.get("x-trace-id") or request.headers.get("X-Trace-Id")
        if not trace_id:
            trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id

        response = await call_next(request)
        duration_ms = round((perf_counter() - started) * 1000, 2)
        logger.info(
            "personal-search-http-request",
            extra={
                "event": "http.request",
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["x-trace-id"] = trace_id
        if request.url.path.startswith("/web/"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """统一处理 HTTP 异常并映射业务 code。"""

        message = str(exc.detail)
        code = "http_error"
        if exc.status_code == 404:
            code = "not_found"
        elif exc.status_code == 503:
            code = "service_unavailable"
        elif exc.status_code >= 500:
            code = "internal_error"
        elif exc.status_code == 400:
            code = "bad_request"
        elif exc.status_code == 401:
            code = "unauthorized"
        elif exc.status_code == 403:
            code = "forbidden"
        elif exc.status_code == 429:
            code = "rate_limited"

        trace_id = getattr(request.state, "trace_id", None)
        logger.warning(
            "HTTP exception",
            extra={
                "event": "request.http_exception",
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": exc.status_code,
            },
        )
        return _build_error_response(
            request=request,
            status_code=exc.status_code,
            code=code,
            message=message,
            detail=None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """参数校验异常返回 422，并保留错误细节。"""

        trace_id = getattr(request.state, "trace_id", None)
        logger.warning(
            "Request validation error",
            extra={
                "event": "request.validation_error",
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 422,
            },
        )
        return _build_error_response(
            request=request,
            status_code=422,
            code="validation_error",
            message="请求参数校验失败",
            detail={"errors": exc.errors(), "body": exc.body},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """兜底异常处理：避免栈外抛出导致连接中断。"""

        trace_id = getattr(request.state, "trace_id", None)
        logger.error(
            "Unhandled exception",
            extra={
                "event": "request.unhandled_exception",
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
            },
            exc_info=True,
        )
        return _build_error_response(
            request=request,
            status_code=500,
            code="internal_error",
            message="服务内部错误",
            detail=str(exc),
        )
