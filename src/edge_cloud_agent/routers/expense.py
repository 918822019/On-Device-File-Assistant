"""Expense APIs for reimbursement workflow MVP."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..common.time_utils import now_iso
from ..config import ExpenseConfig
from ..expense.ingest import run_once
from ..expense.presentation import export_material, extraction_confidence
from ..expense.schemas import (
    ExpenseCollectRequest,
    ExpenseCollectResponse,
    ExpenseCorrectRequest,
    ExpenseCorrectResponse,
    ExpenseExportRequest,
    ExpenseExportResponse,
    ExpenseRebuildIndexResponse,
    ExpenseSearchHit,
    ExpenseSearchRequest,
    ExpenseSearchResponse,
)
from . import deps

router = APIRouter()


@router.post("/v1/expense/collect", response_model=ExpenseCollectResponse)
def collect(req: ExpenseCollectRequest, request: Request):
    service = deps.expense_service(request)

    cfg: ExpenseConfig = request.app.state.expense_cfg
    claim_id = req.claim_id or cfg.default_claim_id

    try:
        material = service.collect(req, claim_id=claim_id)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"收集材料失败: {exc}") from exc

    claim_total, claim_count, missing_types = service.build_claim_summary(claim_id=material.claim_id)
    deps.record_metric(
        request,
        "record_expense_collect",
        material.claim_id,
        material.material_id,
        bool(missing_types),
        missing_types,
    )
    return ExpenseCollectResponse(
        material_id=material.material_id,
        claim_id=material.claim_id,
        title=material.title,
        doc_type=material.doc_type,
        extracted_amount=material.extracted_amount,
        extracted_date=material.extracted_date,
        merchant=material.merchant,
        summary=material.summary,
        extraction_confidence=extraction_confidence(material),
        needs_follow_up=bool(missing_types),
        missing_required_types=missing_types,
        claim_material_count=claim_count,
        claim_total_amount=claim_total,
    )


@router.post("/v1/expense/search", response_model=ExpenseSearchResponse)
def search(req: ExpenseSearchRequest, request: Request):
    service = deps.expense_service(request)
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
    deps.record_metric(request, "record_expense_search", req.claim_id)
    return ExpenseSearchResponse(
        keyword=req.keyword,
        claim_id=req.claim_id,
        total=len(hits),
        items=hits,
        missing_required_types=missing_types,
    )


@router.post("/v1/expense/export", response_model=ExpenseExportResponse)
def export(req: ExpenseExportRequest, request: Request):
    service = deps.expense_service(request)
    claim_id = req.claim_id
    export_id, manifest, materials = service.export_bundle(req)
    if not materials:
        raise HTTPException(status_code=404, detail="未找到可导出的材料")
    if not claim_id:
        claim_id = materials[0].claim_id

    deps.record_metric(request, "record_expense_export", claim_id, len(materials))
    return ExpenseExportResponse(
        export_id=export_id,
        claim_id=claim_id,
        material_count=len(materials),
        export_filename=f"expense_export_{claim_id or 'manual'}_{export_id}.txt",
        # 真实导出时刻；刻意不复用材料的 updated_at
        export_time=now_iso(),
        manifest=manifest,
        materials=[export_material(material) for material in materials],
    )


@router.post("/v1/expense/correct", response_model=ExpenseCorrectResponse)
def correct(req: ExpenseCorrectRequest, request: Request):
    """人工纠正抽取字段；实际改动计入复盘指标「字段纠正次数」。"""

    service = deps.expense_service(request)
    try:
        material, changed = service.correct(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="材料不存在") from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"纠正失败: {exc}") from exc

    if changed:
        deps.record_metric(request, "record_expense_correct", material.claim_id, material.material_id, changed)
    return ExpenseCorrectResponse(
        material_id=material.material_id,
        claim_id=material.claim_id,
        corrected_fields=changed,
        title=material.title,
        extracted_amount=material.extracted_amount,
        extracted_date=material.extracted_date,
        merchant=material.merchant,
        updated_at=material.updated_at,
    )


@router.post("/v1/expense/rebuild-index", response_model=ExpenseRebuildIndexResponse)
def rebuild_index(request: Request):
    service = deps.expense_service(request)

    cfg: ExpenseConfig = request.app.state.expense_cfg
    try:
        # force_rehash=True：与 personal_search 的「重建索引」同一语义 —— 手工
        # 重建要做权威的 hash 校验，绕过 size+mtime 快路径。云同步/备份恢复可能
        # 把 mtime 还原成旧值，那种变更只有重算 hash 能发现。
        result = run_once(service, cfg, trace_id=deps.trace_id(request), force_rehash=True)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"重建索引失败: {exc}") from exc

    return ExpenseRebuildIndexResponse(
        scanned=result.scanned,
        imported=result.imported,
        skipped=result.skipped,
        errors=result.errors,
        removed=result.removed,
        material_ids=result.imported_ids,
    )
