"""Expense APIs for reimbursement workflow MVP."""

from __future__ import annotations

from datetime import datetime, timezone
from time import time

from fastapi import APIRouter, HTTPException, Request

from ..config import ExpenseConfig
from ..expense.schemas import (
    ExpenseCollectRequest,
    ExpenseCollectResponse,
    ExpenseRebuildIndexResponse,
    ExpenseExportRequest,
    ExpenseExportResponse,
    ExpenseExportMaterial,
    ExpenseSearchRequest,
    ExpenseSearchResponse,
    ExpenseSearchHit,
)
from ..expense.ingest import run_once
from ..expense.service import ExpenseService
from ..expense.storage import ExpenseMaterial


router = APIRouter()


def _extraction_confidence(material: ExpenseMaterial) -> float:
    """金额/日期/商户三个抽取字段的命中占比（0~1）。

    旧实现 `1.0 if material.summary else 0.0` 恒为 1.0（summary 永不为空），
    是假指标；现在反映真实抽取质量，也是复盘指标"字段纠正次数"的分母参照。
    """

    signals = (
        material.extracted_amount is not None,
        bool(material.extracted_date),
        bool(material.merchant),
    )
    return round(sum(signals) / len(signals), 2)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


@router.post("/v1/expense/collect", response_model=ExpenseCollectResponse)
def collect(req: ExpenseCollectRequest, request: Request):
    service: ExpenseService | None = getattr(request.app.state, "expense_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="报销服务未就绪，请检查数据存储/配置后重启服务。")

    cfg: ExpenseConfig = request.app.state.expense_cfg
    claim_id = req.claim_id or cfg.default_claim_id

    try:
        material = service.collect(req, claim_id=claim_id)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"收集材料失败: {exc}") from exc

    claim_total, claim_count, missing_types = service.build_claim_summary(claim_id=material.claim_id)
    return ExpenseCollectResponse(
        material_id=material.material_id,
        claim_id=material.claim_id,
        title=material.title,
        doc_type=material.doc_type,
        extracted_amount=material.extracted_amount,
        extracted_date=material.extracted_date,
        merchant=material.merchant,
        summary=material.summary,
        extraction_confidence=_extraction_confidence(material),
        needs_follow_up=bool(missing_types),
        missing_required_types=missing_types,
        claim_material_count=claim_count,
        claim_total_amount=claim_total,
    )


@router.post("/v1/expense/search", response_model=ExpenseSearchResponse)
def search(req: ExpenseSearchRequest, request: Request):
    service: ExpenseService | None = getattr(request.app.state, "expense_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="报销服务未就绪，请检查数据存储/配置后重启服务。")
    try:
        items = service.search(req)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"检索材料失败: {exc}") from exc

    hits = [
        ExpenseSearchHit(
            material_id=item.material.material_id,
            claim_id=item.material.claim_id,
            title=item.material.title,
            doc_type=item.material.doc_type,
            source_app=item.material.source_app,
            summary=item.material.summary,
            extracted_amount=item.material.extracted_amount,
            extracted_date=item.material.extracted_date,
            merchant=item.material.merchant,
            score=item.score,
        )
        for item in items
    ]

    missing_types: list[str] = []
    if req.claim_id:
        _, _, missing_types = service.build_claim_summary(req.claim_id)
    return ExpenseSearchResponse(
        keyword=req.keyword,
        claim_id=req.claim_id,
        total=len(hits),
        items=hits,
        missing_required_types=missing_types,
    )


def _build_export_material(material: ExpenseMaterial) -> ExpenseExportMaterial:
    return ExpenseExportMaterial(
        material_id=material.material_id,
        claim_id=material.claim_id,
        title=material.title,
        doc_type=material.doc_type,
        extracted_amount=material.extracted_amount,
        extracted_date=material.extracted_date,
        merchant=material.merchant,
        file_uri=material.file_uri,
    )


@router.post("/v1/expense/export", response_model=ExpenseExportResponse)
def export(req: ExpenseExportRequest, request: Request):
    service: ExpenseService | None = getattr(request.app.state, "expense_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="报销服务未就绪，请检查数据存储/配置后重启服务。")
    claim_id = req.claim_id
    manifest, materials = service.export_bundle(req)
    if not materials:
        raise HTTPException(status_code=404, detail="未找到可导出的材料")
    export_id = f"exp_{int(time())}"
    if not claim_id and materials:
        claim_id = materials[0].claim_id

    return ExpenseExportResponse(
        export_id=export_id,
        claim_id=claim_id,
        material_count=len(materials),
        export_filename=f"expense_export_{claim_id or 'manual'}_{export_id}.txt",
        # 旧值取 materials[0].updated_at（材料更新时间），与字段语义不符；改为真实导出时刻
        export_time=_now_iso(),
        manifest=manifest,
        materials=[_build_export_material(material) for material in materials],
    )


@router.post("/v1/expense/rebuild-index", response_model=ExpenseRebuildIndexResponse)
def rebuild_index(request: Request):
    service: ExpenseService | None = getattr(request.app.state, "expense_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="报销服务未就绪，请检查数据存储/配置后重启服务。")

    cfg: ExpenseConfig = request.app.state.expense_cfg
    try:
        result = run_once(service, cfg)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc

    return ExpenseRebuildIndexResponse(
        scanned=result.scanned,
        imported=result.imported,
        skipped=result.skipped,
        errors=result.errors,
        material_ids=result.material_ids,
    )
