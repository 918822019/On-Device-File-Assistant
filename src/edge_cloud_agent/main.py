"""FastAPI 应用入口：装配路由、静态页面与 lifespan。"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse

from .bootstrap import lifespan
from .http_handlers import register_http_handlers
from .logging_config import configure_logging
from .routers.chat import router as chat_router
from .routers.embeddings import router as embeddings_router
from .routers.expense import router as expense_router
from .routers.metrics import router as metrics_router
from .routers.personal_search import router as personal_search_router

configure_logging()


def create_app() -> FastAPI:
    """创建应用并挂载 HTTP 适配器与服务生命周期。"""

    app = FastAPI(title="Edge-Cloud Hybrid Agent", lifespan=lifespan)
    app.include_router(chat_router)
    app.include_router(embeddings_router)
    app.include_router(expense_router)
    app.include_router(personal_search_router)
    app.include_router(metrics_router)

    # 注册在所有 API 路由之后，不会遮蔽 /v1/* 与 /health。
    web_dir = Path(__file__).resolve().parents[2] / "web"
    if web_dir.is_dir():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

        @app.get("/", include_in_schema=False)
        def _web_root() -> RedirectResponse:
            return RedirectResponse(url="/web/")

    register_http_handlers(app)

    @app.get("/health")
    async def health() -> dict:
        """存活探针 + 降级可见性。

        必须是 async def：本项目 14 个端点里, 除本函数外的 13 个全部是同步 def,
        FastAPI 会把它们丢进同一个 anyio 线程池（默认 40 个 token）。两个运行时加了推理锁后，并发的
        /v1/chat 会排队持锁，极端情况下把线程池占满；若 /health 也是同步的，
        它会被一起饿死，deploy.sh 与 service.sh 的健康门禁随即失败，systemd 会把
        一个「健康但繁忙」的服务重启掉。本函数只读 app.state，没有任何阻塞 IO，
        跑在事件循环上即可永不被业务负载阻塞。

        契约：只要进程能服务就返回 HTTP 200 且 `ok=true`。deploy.sh 与
        service.sh 用 `curl -fsS ... | grep -q '"ok"'` 做启动门禁（-f 会把
        503 当成失败并死等到超时），web/js/core.js 用 `resp.ok && data.ok`
        驱动状态点，因此这里不能因为降级而改状态码或翻转 ok。

        但 bootstrap 的装配是「每个服务失败都吞成 None + 一条 warning」，
        配合原先恒绿的 /health，后端可以整体坏掉而外部毫无察觉（典型：
        embedding_runtime 加载失败后，搜索静默退化为纯关键词匹配，
        仍然返回 200 和看起来正常的结果）。故补充 degraded 与 services
        明细，让静默降级至少在一次 curl 里可见。

        degraded 只统计「本该初始化却失败」的服务；cloud_enabled 是配置
        选择（默认 false），不计入降级，否则会恒定告警。
        索引就绪度不在此处，见 /v1/search-agent/index-status。
        """

        state = app.state
        orchestrator = getattr(state, "orchestrator", None)
        services = {
            "edge_llm": getattr(orchestrator, "edge", None) is not None,
            "embedding_runtime": getattr(state, "embedding_runtime", None) is not None,
            "personal_file_service": getattr(state, "personal_file_service", None) is not None,
            "expense_service": getattr(state, "expense_service", None) is not None,
            "metrics": getattr(state, "metrics", None) is not None,
        }
        return {
            "ok": True,
            "degraded": not all(services.values()),
            "services": services,
            "cloud_enabled": bool(getattr(getattr(orchestrator, "cloud_cfg", None), "enabled", False)),
        }

    return app


app = create_app()
