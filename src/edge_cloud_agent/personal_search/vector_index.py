"""FAISS-backed vector index for personal file search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import PersonalFileConfig
from .storage import PersonalFileItem


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
        return self._index is not None and bool(self._ids)

    def _normalize(self, vectors: "np.ndarray") -> "np.ndarray":
        norms = self._np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = self._np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def _persist_ids(self) -> None:
        self._ids_path.parent.mkdir(parents=True, exist_ok=True)
        with self._ids_path.open("w", encoding="utf-8") as file:
            json.dump({"ids": self._ids}, file, ensure_ascii=False)

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
            return self.is_ready()
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
