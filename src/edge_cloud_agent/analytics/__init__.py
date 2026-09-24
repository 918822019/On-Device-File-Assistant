"""复盘指标采集与计算（V1 产品目标：用户再次回来）。"""

from __future__ import annotations


def record_safe(metrics, method_name: str, *args) -> None:
    """埋点失败保护：metrics 未装配或记录异常时静默跳过，绝不影响业务响应。

    路由层统一经此调用 MetricsService.record_*，避免各 router 重复 try/except。
    """

    if metrics is None:
        return
    try:
        getattr(metrics, method_name)(*args)
    except Exception:  # pragma: no cover
        pass
