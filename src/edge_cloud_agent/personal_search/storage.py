"""Persistence layer for the personal file memory index."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Iterable


@dataclass(frozen=True)
class PersonalFileItem:
    """One indexed personal file item."""

    file_id: str
    title: str
    file_uri: str
    source_app: str
    doc_type: str
    mime_type: str
    raw_text: str
    summary: str
    file_path: str
    captured_at: str | None = None
    updated_at: str | None = None
    file_size_bytes: int | None = None
    file_hash: str | None = None
    visual_hints: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    created_at: str = ""
    last_scanned_at: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "PersonalFileItem":
        return cls(
            file_id=payload.get("file_id", ""),
            title=payload.get("title", ""),
            file_uri=payload.get("file_uri", ""),
            source_app=payload.get("source_app", "unknown"),
            doc_type=payload.get("doc_type", "file"),
            mime_type=payload.get("mime_type", ""),
            raw_text=payload.get("raw_text", ""),
            summary=payload.get("summary", ""),
            file_path=payload.get("file_path", ""),
            captured_at=payload.get("captured_at"),
            updated_at=payload.get("updated_at"),
            file_size_bytes=payload.get("file_size_bytes"),
            file_hash=payload.get("file_hash"),
            visual_hints=payload.get("visual_hints", []) or [],
            tags=payload.get("tags", []) or [],
            embedding=payload.get("embedding"),
            created_at=payload.get("created_at", ""),
            last_scanned_at=payload.get("last_scanned_at", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class PersonalFileStore:
    """Simple JSONL store for local file index."""

    def __init__(self, path: str) -> None:
        self.path = Path(path).resolve()
        self._items: dict[str, PersonalFileItem] = {}
        self._by_file_uri: dict[str, str] = {}
        self._lock = Lock()
        self._load()

    def add_or_update(self, item: PersonalFileItem) -> None:
        with self._lock:
            self._items[item.file_id] = item
            self._by_file_uri[item.file_uri] = item.file_id
            self._persist()

    def get(self, file_id: str) -> PersonalFileItem | None:
        return self._items.get(file_id)

    def get_by_file_uri(self, file_uri: str) -> PersonalFileItem | None:
        file_id = self._by_file_uri.get(file_uri)
        if file_id is None:
            return None
        return self._items.get(file_id)

    def list_all(self) -> list[PersonalFileItem]:
        return list(self._items.values())

    def get_many(self, file_ids: Iterable[str]) -> list[PersonalFileItem]:
        result: list[PersonalFileItem] = []
        for file_id in file_ids:
            material = self._items.get(file_id)
            if material is not None:
                result.append(material)
        return result

    def delete_by_file_uri(self, file_uri: str) -> bool:
        with self._lock:
            file_id = self._by_file_uri.get(file_uri)
            if file_id is None:
                return False
            del self._by_file_uri[file_uri]
            self._items.pop(file_id, None)
            self._persist()
            return True

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
                    item = PersonalFileItem.from_dict(payload)
                    if item.file_id:
                        self._items[item.file_id] = item
                        self._by_file_uri[item.file_uri] = item.file_id
                except Exception:
                    continue

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for item in self._items.values():
                file.write(json.dumps(item.to_dict(), ensure_ascii=False))
                file.write("\n")
