"""Schemas for the reimbursement workflow APIs."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class ExpenseCollectRequest(BaseModel):
    """Collect one material piece into a reimbursement claim."""

    claim_id: str | None = Field(default=None, description="报销单号，不传则使用系统默认报销单")
    source_app: str = Field(default="manual", description="来源 App，如 微信/邮件/图库")
    doc_type: str = Field(default="receipt", description="材料类型：invoice/receipt/bank_transfer/approval")
    title: str | None = Field(default=None, description="用户自定义标题")
    raw_text: str = Field(..., min_length=1, description="OCR/分享摘要文本")
    file_uri: str | None = Field(default=None, description="可选：文件路径/对象链接")
    file_hash: str | None = Field(default=None, description="可选：文件内容 hash，用于增量判重")
    file_size_bytes: int | None = Field(default=None, description="可选：文件大小（字节），用于增量判重")
    captured_at: str | None = Field(default=None, description="可选：原始截图/文件时间（ISO 字符串）")
    notes: str | None = Field(default=None, description="可选：额外备注")


class ExpenseCollectResponse(BaseModel):
    material_id: str
    claim_id: str
    title: str
    doc_type: str
    extracted_amount: float | None = None
    extracted_date: str | None = None
    merchant: str | None = None
    summary: str
    extraction_confidence: float
    needs_follow_up: bool
    missing_required_types: list[str]
    claim_material_count: int
    claim_total_amount: float | None = None


class ExpenseSearchRequest(BaseModel):
    """Search within a claim or all collected materials."""

    claim_id: str | None = Field(default=None, description="限定某个报销单")
    keyword: str | None = Field(default=None, description="关键词，如 商店名/发票号")
    doc_types: list[str] = Field(default_factory=list, description="可选类型过滤")
    from_date: str | None = Field(default=None, description="起始日期，ISO 格式")
    to_date: str | None = Field(default=None, description="结束日期，ISO 格式")
    min_amount: float | None = Field(default=None, ge=0, description="金额下限")
    max_amount: float | None = Field(default=None, ge=0, description="金额上限")
    limit: int = Field(default=20, ge=1, le=100, description="返回数量上限")


class ExpenseSearchHit(BaseModel):
    material_id: str
    claim_id: str
    title: str
    doc_type: str
    source_app: str
    summary: str
    extracted_amount: float | None = None
    extracted_date: str | None = None
    merchant: str | None = None
    score: float


class ExpenseSearchResponse(BaseModel):
    keyword: str | None
    claim_id: str | None
    total: int
    items: list[ExpenseSearchHit]
    missing_required_types: list[str]


class ExpenseExportRequest(BaseModel):
    """Export by claim or by selected ids."""

    claim_id: str | None = Field(default=None, description="要导出的报销单")
    material_ids: list[str] = Field(default_factory=list, description="或按 material_id 导出")
    include_raw_text: bool = Field(default=False, description="导出是否携带 raw_text")

    @model_validator(mode="after")
    def _validate_scope(self):
        if not self.claim_id and not self.material_ids:
            raise ValueError("claim_id 和 material_ids 至少需要填写一个")
        return self


class ExpenseExportMaterial(BaseModel):
    material_id: str
    claim_id: str
    title: str
    doc_type: str
    extracted_amount: float | None = None
    extracted_date: str | None = None
    merchant: str | None = None
    file_uri: str | None = None


class ExpenseExportResponse(BaseModel):
    export_id: str
    claim_id: str | None
    material_count: int
    export_filename: str
    export_time: str
    manifest: str
    materials: list[ExpenseExportMaterial]


class ExpenseRebuildIndexResponse(BaseModel):
    scanned: int = 0
    imported: int = 0
    skipped: int = 0
    errors: int = 0
    removed: int = 0
    material_ids: list[str] = Field(default_factory=list)
