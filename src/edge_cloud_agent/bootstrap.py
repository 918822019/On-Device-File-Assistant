"""应用生命周期中的服务装配与后台扫描线程管理。"""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

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
from .engines.embedding_runtime import EdgeEmbeddingRuntime
from .expense.ingest import start_watch_loop
from .expense.service import ExpenseService
from .expense.storage import ExpenseStore
from .llm.orchestrator import EdgeCloudOrchestrator
from .personal_search.ingest import run_once as run_personal_scan_once
from .personal_search.ingest import start_watch_loop as start_personal_watch_loop
from .personal_search.refresh import IndexRefreshScheduler
from .personal_search.service import PersonalFileSearchService
from .personal_search.storage import PersonalFileStore

logger = logging.getLogger("agent_server")


def initialize_services(app: FastAPI) -> None:
    """初始化配置、模型运行时、storage/service 并启动扫描线程。"""

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

    personal_file_cfg: PersonalFileConfig = app.state.personal_file_cfg
    if app.state.personal_file_service is not None and personal_file_cfg.source_dir:
        app.state.personal_file_refresher = IndexRefreshScheduler(app.state.personal_file_service, personal_file_cfg)
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


def stop_services(app: FastAPI) -> None:
    """触发后台 watch loop 停止事件。"""

    stop_event = getattr(app.state, "expense_watch_stop", None)
    if stop_event is not None:
        stop_event.set()

    personal_stop_event = getattr(app.state, "personal_file_watch_stop", None)
    if personal_stop_event is not None:
        personal_stop_event.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan：启动装配服务，退出时停止后台扫描线程。"""

    initialize_services(app)
    try:
        yield
    finally:
        stop_services(app)
