"""复盘指标只读端点。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/v1/metrics")
def get_metrics(request: Request):
    """返回报销回访/补齐/纠正与文件搜索追问收敛的复盘指标。

    指标服务不可用（启动装配失败）时返回 503，不影响其他端点。
    比率字段在分母为 0 时为 null（样本不足），不以 0.0 冒充。
    """

    metrics = getattr(request.app.state, "metrics", None)
    if metrics is None:
        raise HTTPException(status_code=503, detail="指标服务未就绪")
    return metrics.compute()
