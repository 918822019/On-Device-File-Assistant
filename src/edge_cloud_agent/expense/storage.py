"""Persistence layer for local reimbursement materials."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Iterable


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
    notes: str | None = None
    keywords: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    embedding: list[float] | None = None

    @classmethod
    def from_dict(cls, payload: dict) -> "ExpenseMaterial":
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)


class ExpenseStore:
    """JSONL-backed local store (single file), intentionally simple and local-only."""

    def __init__(self, path: str) -> None:
        self.path = Path(path).resolve()
        self._items: dict[str, ExpenseMaterial] = {}
        self._lock = Lock()
        self._load()

    def add_or_update(self, material: ExpenseMaterial) -> None:
        with self._lock:
            self._items[material.material_id] = material
            self._persist()

    def get(self, material_id: str) -> ExpenseMaterial | None:
        return self._items.get(material_id)

    def get_by_file_uri(self, file_uri: str) -> ExpenseMaterial | None:
        for material in self._items.values():
            if material.file_uri == file_uri:
                return material
        return None

    def list_all(self) -> list[ExpenseMaterial]:
        return list(self._items.values())

    def list_by_claim(self, claim_id: str) -> list[ExpenseMaterial]:
        return [m for m in self._items.values() if m.claim_id == claim_id]

    def get_many(self, material_ids: Iterable[str]) -> list[ExpenseMaterial]:
        result: list[ExpenseMaterial] = []
        for material_id in material_ids:
            material = self._items.get(material_id)
            if material is not None:
                result.append(material)
        return result

    def all_claim_ids(self) -> list[str]:
        return sorted({m.claim_id for m in self._items.values()})

    def _load(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return

        with self.path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                row = raw_line.strip()
                if not row:
                    continue
                try:
                    payload = json.loads(row)
                    material = ExpenseMaterial.from_dict(payload)
                    self._items[material.material_id] = material
                except Exception:
                    # 向后兼容：一行坏数据直接跳过，避免影响服务启动
                    continue

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for material in self._items.values():
                file.write(json.dumps(material.to_dict(), ensure_ascii=False))
                file.write("\n")
