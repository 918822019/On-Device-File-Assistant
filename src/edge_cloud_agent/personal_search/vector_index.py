"""FAISS-backed vector index for personal file search."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..common.file_io import atomic_write_text
from ..config import PersonalFileConfig
from .storage import PersonalFileItem

_LOGGER = logging.getLogger("agent_server.personal_search.vector_index")


@dataclass(frozen=True)
class SearchHit:
    file_id: str
    score: float


class _FaissBackend:
    """Small adapter over optional faiss dependency."""

    def __init__(self, index_path: Path):
        self._index_path = index_path
        self._ids_path = index_path.with_suffix(index_path.suffix + ".ids.json")
        self._faiss = None
        self._index = None
        self._ids: list[str] = []
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore

            self._faiss = faiss
            self._np = np
        except Exception:
            self._faiss = None
            self._np = None

    @property
    def available(self) -> bool:
        return self._faiss is not None and self._np is not None

    @property
    def ids(self) -> list[str]:
        return self._ids

    def is_ready(self) -> bool:
        if not self.available:
            return False
        if self._index is None or not self._ids:
            return False
        # ids 与向量是按位置一一对应的（第 i 个向量 -> _ids[i]）。两者长度不一致时
        # search() 会把命中的向量下标映射到错误的 file_id 上 —— 返回的是别的文件，
        # 而且不报错。故把「长度必须相等」作为就绪的硬条件。
        return len(self._ids) == self._index.ntotal

    def _normalize(self, vectors: "np.ndarray") -> "np.ndarray":
        norms = self._np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = self._np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _persist_ids(self) -> None:
        # 原子写口径由 common.file_io 统一提供。此前这里是直接写目标文件，崩溃
        # 会留下半截 JSON（读取端能容错跳过，但真正危险的是下面 _load_index
        # 处理的错位场景）。
        atomic_write_text(self._ids_path, json.dumps({"ids": self._ids}, ensure_ascii=False))

    def _load_ids(self) -> None:
        if not self._ids_path.exists():
            self._ids = []
            return
        try:
            payload = json.loads(self._ids_path.read_text(encoding="utf-8"))
            ids = payload.get("ids", [])
            self._ids = ids if isinstance(ids, list) else []
        except Exception:
            self._ids = []

    def _load_index(self) -> bool:
        if not self.available or not self._index_path.exists():
            return False
        try:
            self._index = self._faiss.read_index(str(self._index_path))
            self._load_ids()
            if not self.is_ready():
                # index 与 ids.json 是两次独立写入，中间崩溃会留下错位的组合
                # （典型：index 已更新为 N 个向量，ids.json 仍是上一轮的 M 条）。
                # 这种状态**不能**当作就绪：search() 会按下标把新向量映射到旧
                # file_id 上，静默返回错误的文件。判为未就绪即可，下一轮扫描
                # 会从 store（唯一事实源）全量重建，FAISS 只是可丢弃的缓存。
                if self._index is not None and self._ids and len(self._ids) != self._index.ntotal:
                    _LOGGER.warning(
                        "personal-search-faiss-ids-mismatch",
                        extra={
                            "event": "faiss.ids_mismatch",
                            "index_vectors": int(self._index.ntotal),
                            "ids_count": len(self._ids),
                            "index_path": str(self._index_path),
                        },
                    )
                self._index = None
                self._ids = []
                return False
            return True
        except Exception:
            self._index = None
            self._ids = []
            return False

    def build(self, items: Sequence[PersonalFileItem]) -> None:
        if not self.available:
            return
        embeddings: list[list[float]] = []
        ids: list[str] = []
        for item in items:
            if item.embedding:
                embeddings.append(item.embedding)
                ids.append(item.file_id)

        if not embeddings:
            self._index = None
            self._ids = []
            self._persist_ids()
            return

        vectors = self._np.asarray(embeddings, dtype=self._np.float32)
        vectors = self._normalize(vectors)
        dim = vectors.shape[1]
        index = self._faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index
        self._ids = ids
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(index, str(self._index_path))
        self._persist_ids()

    def load(self) -> None:
        if not self.available:
            return
        loaded = self._load_index()
        if not loaded:
            self._index = None
            self._ids = []

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        if not self.available or not self.is_ready():
            return []

        vector = self._np.asarray([query_embedding], dtype=self._np.float32)
        vector = self._normalize(vector)
        if vector.size == 0:
            return []
        if vector.shape[1] != self._index.d:
            return []

        k = max(1, min(top_k, len(self._ids)))
        scores, indices = self._index.search(vector, k)
        if scores.size == 0:
            return []
        results: list[SearchHit] = []
        for idx, score in zip(indices[0].tolist(), scores[0].tolist(), strict=False):
            if idx < 0 or idx >= len(self._ids):
                continue
            results.append(SearchHit(file_id=self._ids[idx], score=max(0.0, min(1.0, float(score)))))
        return results


class PersonalFileVectorIndex:
    """Facade for personal file vector retrieval."""

    def __init__(self, config: PersonalFileConfig):
        self.config = config
        self.enabled = config.enable_faiss
        self._backend = _FaissBackend(Path(config.faiss_index_path))
        if self.enabled:
            self._backend.load()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def is_ready(self) -> bool:
        if not self.enabled:
            return False
        return self._backend.is_ready()

    def is_available(self) -> bool:
        return self._backend.available

    def rebuild(self, items: list[PersonalFileItem]) -> None:
        if not self.enabled:
            return
        if self._backend is None:
            return
        self._backend.build(items)

    def refresh(self, items: list[PersonalFileItem]) -> None:
        self.rebuild(items)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        if not self.enabled:
            return []
        return self._backend.search(query_embedding, top_k)
