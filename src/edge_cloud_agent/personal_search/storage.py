"""Persistence layer for the personal file memory index."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock

from ..common.file_io import atomic_write_text
from ..common.jsonl_store import JsonlSnapshotStore


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
    # st_mtime_ns，用作周期扫描的廉价变更预筛：size 与 mtime 都没变就不再读文件
    # 算 hash。旧记录没有该字段（from_dict 回落 None），会被当作「未知」而重算
    # 一次 hash 并就地回填，不会触发内容重读或重新 embedding。
    file_mtime_ns: int | None = None
    visual_hints: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    created_at: str = ""
    last_scanned_at: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> PersonalFileItem:
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
            file_mtime_ns=payload.get("file_mtime_ns"),
            visual_hints=payload.get("visual_hints", []) or [],
            tags=payload.get("tags", []) or [],
            embedding=payload.get("embedding"),
            created_at=payload.get("created_at", ""),
            last_scanned_at=payload.get("last_scanned_at", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class PersonalFileStore(JsonlSnapshotStore[PersonalFileItem]):
    """文件索引存储：主键 ``file_id``，二级索引 ``file_uri``。

    通用的快照读写、原子落盘、二级索引维护都在 ``JsonlSnapshotStore``；这里只
    提供本业务线的命名别名（``file_uri`` 是这条线的既有公开 API，被 ingest /
    service / 路由直接调用，不随基类改名）。
    """

    row_type = PersonalFileItem
    id_attr = "file_id"
    index_attr = "file_uri"

    def get_by_file_uri(self, file_uri: str) -> PersonalFileItem | None:
        return self.get_by_index(file_uri)

    def delete_by_file_uri(self, file_uri: str, persist: bool = True) -> bool:
        return self.delete_by_index(file_uri, persist=persist)


class FileStateStore:
    """备注与归档标记的持久化存储（JSON 单文件，原子写）。

    此前 annotations/archived 只存在 service 的内存 dict/set 里，重启即丢。
    文件格式：{"annotations": {file_id: note}, "archived": [file_id, ...]}
    损坏时从空状态起步，不阻塞服务启动。
    """

    def __init__(self, path: str) -> None:
        self.path = Path(path).resolve()
        self._lock = Lock()
        self._annotations: dict[str, str] = {}
        self._archived: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            annotations = payload.get("annotations") or {}
            if isinstance(annotations, dict):
                self._annotations = {str(k): str(v) for k, v in annotations.items()}
            archived = payload.get("archived") or []
            if isinstance(archived, list):
                self._archived = {str(item) for item in archived}
        except Exception:
            self._annotations = {}
            self._archived = set()

    def _persist(self) -> None:
        payload = {
            "annotations": self._annotations,
            "archived": sorted(self._archived),
        }
        atomic_write_text(self.path, json.dumps(payload, ensure_ascii=False, indent=1))

    def set_annotation(self, file_id: str, note: str) -> None:
        with self._lock:
            self._annotations[file_id] = note
            self._persist()

    def get_annotation(self, file_id: str) -> str | None:
        return self._annotations.get(file_id)

    def set_archived(self, file_id: str) -> None:
        with self._lock:
            self._archived.add(file_id)
            self._persist()

    def is_archived(self, file_id: str) -> bool:
        return file_id in self._archived

    def prune(self, valid_file_ids: Iterable[str]) -> int:
        """清理已不在索引中的 file_id 的备注与归档标记，返回清理条数。

        幽灵清理只删 store 里的条目、不会动这里，所以被删文件的备注/归档会永久
        残留。这不只是无界增长：file_id 由路径 md5 派生（``fm_ + md5(path)[:14]``），
        同一路径永远得到同一个 id —— 文件被删除后若该路径上出现一个**内容不同**的
        新文件，旧备注会凭空贴到新文件上，用户看到的是一条与内容无关的历史标记。

        无变化时不落盘，避免每轮扫描都重写一次 JSON。
        """

        valid = set(valid_file_ids)
        with self._lock:
            stale_annotations = [fid for fid in self._annotations if fid not in valid]
            stale_archived = self._archived - valid
            if not stale_annotations and not stale_archived:
                return 0
            for fid in stale_annotations:
                del self._annotations[fid]
            self._archived -= stale_archived
            self._persist()
            return len(stale_annotations) + len(stale_archived)
