"""Persistence layer for local reimbursement materials."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..common.jsonl_store import JsonlSnapshotStore


@dataclass(frozen=True)
class ExpenseMaterial:
    """Single collected material line item."""

    material_id: str
    claim_id: str
    title: str
    doc_type: str
    source_app: str
    raw_text: str
    summary: str
    extracted_amount: float | None = None
    extracted_date: str | None = None
    merchant: str | None = None
    captured_at: str | None = None
    file_uri: str | None = None
    file_hash: str | None = None
    file_size_bytes: int | None = None
    # st_mtime_ns，与 PersonalFileItem.file_mtime_ns 同一用途：watch 循环的廉价
    # 变更预筛。size 与 mtime 都没变就不必读文件算 hash。旧记录没有该字段
    # （from_dict 回落 None），会被当作「未知」而重算一次 hash 并就地回填。
    file_mtime_ns: int | None = None
    notes: str | None = None
    keywords: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    embedding: list[float] | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> ExpenseMaterial:
        """容错构造：历史 JSONL 中多出/缺失字段时不整行崩溃（与 personal 侧对齐）。"""

        return cls(
            material_id=payload.get("material_id", ""),
            claim_id=payload.get("claim_id", ""),
            title=payload.get("title", ""),
            doc_type=payload.get("doc_type", "receipt"),
            source_app=payload.get("source_app", "manual"),
            raw_text=payload.get("raw_text", ""),
            summary=payload.get("summary", ""),
            extracted_amount=payload.get("extracted_amount"),
            extracted_date=payload.get("extracted_date"),
            merchant=payload.get("merchant"),
            captured_at=payload.get("captured_at"),
            file_uri=payload.get("file_uri"),
            file_hash=payload.get("file_hash"),
            file_size_bytes=payload.get("file_size_bytes"),
            file_mtime_ns=payload.get("file_mtime_ns"),
            notes=payload.get("notes"),
            keywords=payload.get("keywords", []) or [],
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            embedding=payload.get("embedding"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class ExpenseStore(JsonlSnapshotStore[ExpenseMaterial]):
    """报销材料存储：主键 ``material_id``，二级索引 ``file_uri``（可空）。

    通用的快照读写、原子落盘、二级索引维护都在 ``JsonlSnapshotStore``；这里只
    留报销业务自己的查询（按报销单聚合）。

    注意 ``ExpenseMaterial.file_uri`` 是 ``str | None``：手工录入的材料没有
    来源文件，基类对假值不建索引项，因此这类行只能按 ``material_id`` 取。
    """

    row_type = ExpenseMaterial
    id_attr = "material_id"
    index_attr = "file_uri"

    def get_by_file_uri(self, file_uri: str) -> ExpenseMaterial | None:
        return self.get_by_index(file_uri)

    def delete_by_file_uri(self, file_uri: str, persist: bool = True) -> bool:
        return self.delete_by_index(file_uri, persist=persist)

    def list_by_claim(self, claim_id: str) -> list[ExpenseMaterial]:
        return [m for m in self._items.values() if m.claim_id == claim_id]

    def all_claim_ids(self) -> list[str]:
        return sorted({m.claim_id for m in self._items.values()})
