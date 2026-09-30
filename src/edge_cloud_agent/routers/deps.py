"""路由层公共依赖：服务就绪检查、trace_id 与复盘埋点入口。"""

from fastapi import HTTPException, Request

from ..analytics import record_safe
from ..config import PersonalFileConfig
from ..expense.service import ExpenseService
from ..personal_search.refresh import IndexRefreshScheduler
from ..personal_search.service import PersonalFileSearchService

PERSONAL_SEARCH_UNAVAILABLE = "个人文件检索服务未就绪，请配置 FILE_MEMORY_SOURCE_DIR 后重启服务。"
EXPENSE_UNAVAILABLE = "报销服务未就绪，请检查数据存储/配置后重启服务。"


def trace_id(request: Request) -> str | None:
    """从 trace 中间件注入的 request.state 中读取 trace_id。"""

    return getattr(request.state, "trace_id", None)


def metrics(request: Request):
    """复盘指标服务；启动装配失败时为 None，调用方静默跳过。"""

    return getattr(request.app.state, "metrics", None)


def record_metric(request: Request, method_name: str, *args) -> None:
    """埋点包装：服务未装配或记录失败都不影响业务响应。"""

    record_safe(metrics(request), method_name, *args)


def personal_file_service(request: Request) -> PersonalFileSearchService:
    """个人文件检索服务；未就绪直接返回 503。"""

    service: PersonalFileSearchService | None = getattr(request.app.state, "personal_file_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail=PERSONAL_SEARCH_UNAVAILABLE)
    return service


def optional_personal_file_service(request: Request) -> PersonalFileSearchService | None:
    """个人文件检索服务；不可用时返回 None，由调用方降级处理。"""

    return getattr(request.app.state, "personal_file_service", None)


def personal_file_config(request: Request) -> PersonalFileConfig:
    """读取 PersonalFileConfig；若未装配则走默认配置。"""

    return getattr(request.app.state, "personal_file_cfg", PersonalFileConfig())


def personal_file_refresher(request: Request) -> IndexRefreshScheduler | None:
    """搜索请求触发的节流刷新调度器；未配置扫描目录时为 None。"""

    return getattr(request.app.state, "personal_file_refresher", None)


def expense_service(request: Request) -> ExpenseService:
    """报销服务；未就绪直接返回 503。"""

    service: ExpenseService | None = getattr(request.app.state, "expense_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail=EXPENSE_UNAVAILABLE)
    return service
