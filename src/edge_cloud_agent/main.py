"""FastAPI entrypoint for edge-cloud hybrid agent.

本文件承担三件事：
- 配置统一日志输出（event + trace_id）
- 全局异常处理，返回统一错误格式
- 应用生命周期里初始化各子服务并启动后台监听线程
"""

import logging
import os
import uuid
import threading
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, RedirectResponse

from .runtime.agent import EdgeCloudOrchestrator
from .analytics.service import MetricsService
from .analytics.storage import MetricsEventStore
from .config import (
    CloudConfig,
    EdgeConfig,
    EmbeddingConfig,
    ExpenseConfig,
    MetricsConfig,
    PersonalFileConfig,
    RouteConfig,
)
from .expense.service import ExpenseService
from .expense.storage import ExpenseStore
from .runtime.embedding_runtime import EdgeEmbeddingRuntime
from .personal_search.service import PersonalFileSearchService
from .personal_search.storage import PersonalFileStore
from .routers.chat import router as chat_router
from .routers.embeddings import router as embeddings_router
from .routers.expense import router as expense_router
from .routers.metrics import router as metrics_router
from .routers.personal_search import router as personal_search_router
from .routers.schemas import ErrorResponse
from .expense.ingest import start_watch_loop
from .personal_search.ingest import run_once as run_personal_scan_once, start_watch_loop as start_personal_watch_loop

logger = logging.getLogger("agent_server")


class _DefaultLogContext(logging.Filter):
    """为结构化日志补齐字段，避免 formatter 因缺字段抛异常。"""

    _defaults = {
        "event": "-",
        "trace_id": "-",
        "method": "-",
        "path": "-",
        "status_code": "-",
        "duration_ms": "-",
        "query_digest": "-",
        "search_id": "-",
        "session_id": "-",
        "file_id": "-",
        "action": "-",
        "top_k": "-",
        "state": "-",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self._defaults.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def _configure_logging() -> None:
    """读取 APP_LOG_LEVEL 并配置根 logger。

输出格式要求：每条日志都有 event、trace_id、method、path、status 等固定字段。
"""

    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s event=%(event)s trace_id=%(trace_id)s "
        "method=%(method)s path=%(path)s status=%(status_code)s dur_ms=%(duration_ms)s %(message)s"
    )

    handlers = root_logger.handlers or [logging.StreamHandler()]
    if not root_logger.handlers:
        root_logger.addHandler(handlers[0])

    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler.addFilter(_DefaultLogContext())


_configure_logging()


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


def create_app() -> FastAPI:
    """创建应用：路由绑定、日志中间件、异常处理、服务初始化。"""

    app = FastAPI(title="Edge-Cloud Hybrid Agent")
    app.include_router(chat_router)
    app.include_router(embeddings_router)
    app.include_router(expense_router)
    app.include_router(personal_search_router)
    app.include_router(metrics_router)

    # Web UI：同源静态托管（repo/web/）。同源意味着零 CORS 配置；
    # WSL2 下 Windows 浏览器经 localhost 端口转发直达，见 docs/WEB_UI.md。
    # 注册在所有 API 路由之后，不会遮蔽 /v1/* 与 /health。
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.is_dir():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

        @app.get("/", include_in_schema=False)
        def _web_root() -> RedirectResponse:
            return RedirectResponse(url="/web/")

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

    @app.on_event("startup")
    def startup() -> None:
        """服务启动：初始化配置、模型运行时、storage/service 并启动扫描线程。"""

        app.state.edge_cfg = EdgeConfig()
        app.state.cloud_cfg = CloudConfig()
        app.state.route_cfg = RouteConfig()
        app.state.embedding_cfg = EmbeddingConfig()
        app.state.expense_cfg = ExpenseConfig()

        # 复盘指标：装配失败不致命（埋点侧对 None 全部静默跳过）
        app.state.metrics_cfg = MetricsConfig()
        try:
            app.state.metrics = MetricsService(
                store=MetricsEventStore(app.state.metrics_cfg.store_path),
                revisit_window_days=app.state.metrics_cfg.revisit_window_days,
                followup_window_hours=app.state.metrics_cfg.followup_window_hours,
            )
            logger.info("Metrics service ready: %s", app.state.metrics_cfg.store_path)
        except Exception as exc:  # pragma: no cover
            app.state.metrics = None
            logger.warning("Metrics service unavailable: %s", exc)

        app.state.orchestrator = EdgeCloudOrchestrator(
            app.state.edge_cfg,
            app.state.cloud_cfg,
            app.state.route_cfg,
        )

        try:
            app.state.embedding_runtime = EdgeEmbeddingRuntime(app.state.embedding_cfg)
            logger.info("Embedding runtime loaded: %s", app.state.embedding_cfg.model_id)
        except Exception as exc:  # pragma: no cover
            app.state.embedding_runtime = None
            logger.warning("Embedding runtime unavailable: %s", exc)

        try:
            app.state.expense_service = ExpenseService(
                config=app.state.expense_cfg,
                store=ExpenseStore(app.state.expense_cfg.store_path),
                embedding_runtime=app.state.embedding_runtime,
            )
        except Exception as exc:  # pragma: no cover
            app.state.expense_service = None
            logger.warning("Expense service unavailable: %s", exc)

        app.state.personal_file_cfg = PersonalFileConfig()
        try:
            app.state.personal_file_service = PersonalFileSearchService(
                config=app.state.personal_file_cfg,
                store=PersonalFileStore(app.state.personal_file_cfg.store_path),
                embedding_runtime=app.state.embedding_runtime,
            )
        except Exception as exc:  # pragma: no cover
            app.state.personal_file_service = None
            logger.warning("Personal file search service unavailable: %s", exc)

        personal_file_cfg: PersonalFileConfig = getattr(app.state, "personal_file_cfg", PersonalFileConfig())
        if getattr(app.state, "personal_file_service", None) is not None and personal_file_cfg.source_dir:
            try:
                run_personal_scan_once(app.state.personal_file_service, personal_file_cfg)
            except Exception as exc:  # pragma: no cover
                logger.warning("Personal file warm scan failed: %s", exc)
            app.state.personal_file_watch_stop = threading.Event()
            app.state.personal_file_watch_thread = threading.Thread(
                target=start_personal_watch_loop,
                args=(
                    app.state.personal_file_service,
                    personal_file_cfg,
                    app.state.personal_file_watch_stop,
                ),
                daemon=True,
            )
            app.state.personal_file_watch_thread.start()

        expense_watch_dir = app.state.expense_cfg.watch_dir
        if app.state.expense_service is not None and expense_watch_dir:
            app.state.expense_watch_stop = threading.Event()
            app.state.expense_watch_thread = threading.Thread(
                target=start_watch_loop,
                args=(
                    app.state.expense_service,
                    app.state.expense_cfg,
                    app.state.expense_watch_stop,
                ),
                daemon=True,
            )
            app.state.expense_watch_thread.start()

        logger.info(
            "Hybrid agent started: source=%s, tiny-first=%s, cloud-enabled=%s",
            app.state.edge_cfg.source,
            app.state.route_cfg.use_tiny_first,
            app.state.cloud_cfg.enabled,
        )

    @app.on_event("shutdown")
    def shutdown() -> None:
        """优雅停机：触发后台 watch loop 停止事件。"""

        stop_event = getattr(app.state, "expense_watch_stop", None)
        if stop_event is not None:
            stop_event.set()

        personal_stop_event = getattr(app.state, "personal_file_watch_stop", None)
        if personal_stop_event is not None:
            personal_stop_event.set()

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    return app


app = create_app()
