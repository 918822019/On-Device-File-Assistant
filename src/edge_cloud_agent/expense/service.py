"""Expense workflow service: collect -> find -> export."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, NamedTuple
from uuid import uuid4


class _ExtractedFields(NamedTuple):
    """collect() 字段抽取结果，替代每次调用都动态创建的匿名类。"""
    amount: float | None
    date: str | None
    merchant: str | None

from ..config import ExpenseConfig
from ..embedding_runtime import EdgeEmbeddingRuntime
from ..text_utils import tokenize
from .schemas import (
    ExpenseCollectRequest,
    ExpenseExportRequest,
    ExpenseSearchRequest,
)
from .storage import ExpenseMaterial, ExpenseStore


_TOKEN_RE = re.compile(r"[\s,，。；;:：!！?？\\-_/\\|()（）【】、]+")
_AMOUNT_PATTERNS = [
    re.compile(r"金额[:：]?\s*([0-9]+(?:\.[0-9]{1,2})?)"),
    re.compile(r"(?:¥|￥)\s*([0-9]+(?:\.[0-9]{1,2})?)"),
    re.compile(r"([0-9]+(?:\.[0-9]{1,2})?)\s*元"),
    re.compile(r"共计[:：]?\s*([0-9]+(?:\.[0-9]{1,2})?)"),
]
_DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12][0-9]|3[01])"),
    re.compile(r"(\d{1,2})月(\d{1,2})日"),
]
_MERCHANT_PATTERNS = [
    # 惰性捕获商户名，遇到以下任一即截止：连续空白（字段间常用多空格分隔）、
    # 标点、常见后续字段关键词、换行、串尾。
    # 旧模式为贪婪匹配且只排除标点/冒号，"商户: 京东  金额: 128" 会抽出
    # "京东  金额"，把下一个字段吞进商户名。
    # 标签后必须跟冒号或空白：句中恰好包含"商户/单位"等词时（如"没有商户字段"）
    # 不会误把后续文本当成商户名抽出。
    re.compile(
        r"(?:商户|收款方|单位|医院|门店|机构)[:：\s]\s*"
        r"([^\n，,。；;:：]+?)"
        r"(?=\s{2,}|\s*[，,。；;:：]|\s*(?:金额|日期|时间|单号|订单|发票号|合计)|[\r\n]|\s*$)"
    ),
]


def _now_iso() -> str:
    # utcnow() 在 3.12+ 已弃用；保持 naive UTC 输出格式与存量数据一致
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _safe_float(value: str) -> float | None:
    try:
        parsed = float(value.replace(",", ""))
        if math.isfinite(parsed):
            return parsed
    except Exception:
        return None
    return None


def _safe_parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _required_doc_types(config: ExpenseConfig) -> list[str]:
    return [item.strip() for item in config.required_doc_types.split(",") if item.strip()]


@dataclass(frozen=True)
class SearchResult:
    material: ExpenseMaterial
    score: float


class ExpenseService:
    def __init__(self, config: ExpenseConfig, store: ExpenseStore, embedding_runtime: EdgeEmbeddingRuntime | None) -> None:
        self.config = config
        self.store = store
        self.embedding_runtime = embedding_runtime

    def collect(self, req: ExpenseCollectRequest, claim_id: str, persist: bool = True) -> ExpenseMaterial:
        now = _now_iso()
        extracted = self._extract_fields(req.raw_text)
        keywords = self._extract_keywords(req.raw_text, req.doc_type, req.source_app, req.notes)
        title = req.title or self._build_title(req.doc_type, extracted, keywords)
        embedding = self._embed_text(req.raw_text) if self.config.enable_embedding_search else None

        material = ExpenseMaterial(
            material_id=uuid4().hex,
            claim_id=claim_id,
            title=title,
            doc_type=self._normalize_doc_type(req.doc_type),
            source_app=req.source_app.strip() or "manual",
            raw_text=req.raw_text.strip(),
            summary=self._build_summary(extracted, req.raw_text),
            extracted_amount=extracted.amount,
            extracted_date=extracted.date,
            merchant=extracted.merchant,
            captured_at=req.captured_at,
            file_uri=req.file_uri,
            file_hash=req.file_hash,
            file_size_bytes=req.file_size_bytes,
            notes=req.notes,
            keywords=keywords,
            created_at=now,
            updated_at=now,
            embedding=embedding,
        )
        self.store.add_or_update(material, persist=persist)
        return material

    def search(self, req: ExpenseSearchRequest) -> list[SearchResult]:
        keyword = (req.keyword or "").strip().lower()
        query_embedding: list[float] | None = None
        if keyword and self.config.enable_embedding_search and self.embedding_runtime is not None:
            try:
                query_embedding = self.embedding_runtime.embed([keyword]).embeddings[0]
            except Exception:
                query_embedding = None

        results: list[SearchResult] = []
        for material in self.store.list_all():
            if req.claim_id and material.claim_id != req.claim_id:
                continue
            if req.doc_types and material.doc_type not in req.doc_types:
                continue
            if not self._within_amount(material.extracted_amount, req.min_amount, req.max_amount):
                continue
            if not self._within_date(material.extracted_date, req.from_date, req.to_date):
                continue

            score = self._score(material, keyword, query_embedding)
            if keyword and score <= 0:
                continue
            results.append(SearchResult(material=material, score=score))

        if not keyword and req.claim_id:
            # 无关键字时按时间倒序展示，提高“我上次找的材料先看到”的体验
            results.sort(key=lambda x: x.material.updated_at, reverse=True)
        else:
            results.sort(key=lambda x: x.score, reverse=True)

        limit = min(req.limit or self.config.max_search_limit, self.config.max_search_limit)
        return results[:limit]

    def build_claim_summary(self, claim_id: str) -> tuple[float | None, int, list[str]]:
        materials = self.store.list_by_claim(claim_id)
        total_amount = self._sum_amount(materials)
        present_types = {item.doc_type for item in materials}
        required_types = set(_required_doc_types(self.config))
        missing = sorted(required_types - present_types)
        return total_amount, len(materials), list(missing)

    def export_bundle(self, req: ExpenseExportRequest) -> tuple[str, str, list[ExpenseMaterial]]:
        """返回 (export_id, manifest_text, materials)，export_id 与 manifest 文本中的 ID 一致。"""
        if req.material_ids:
            materials = self.store.get_many(req.material_ids)
        else:
            assert req.claim_id is not None
            materials = self.store.list_by_claim(req.claim_id)

        if not materials:
            materials = []
        export_id = uuid4().hex
        lines = [f"报销材料导出清单 {export_id}"]
        lines.append(f"生成时间：{_now_iso()}")
        lines.append(f"材料数量：{len(materials)}")
        lines.append("")

        for index, material in enumerate(materials, start=1):
            lines.append(f"{index}. {material.title}")
            lines.append(f"  - ID: {material.material_id}")
            lines.append(f"  - 报销单: {material.claim_id}")
            lines.append(f"  - 类型: {material.doc_type}")
            lines.append(f"  - 来源: {material.source_app}")
            if material.extracted_amount is not None:
                lines.append(f"  - 金额: {material.extracted_amount}")
            if material.extracted_date:
                lines.append(f"  - 日期: {material.extracted_date}")
            if material.merchant:
                lines.append(f"  - 机构/商户: {material.merchant}")
            lines.append(f"  - 摘要: {material.summary}")
            if req.include_raw_text:
                lines.append(f"  - 原始文本: {material.raw_text}")
            if material.file_uri:
                lines.append(f"  - 文件: {material.file_uri}")
            lines.append("")

        return export_id, "\n".join(lines), materials

    def _embed_text(self, text: str) -> list[float] | None:
        if self.embedding_runtime is None:
            return None

        if not self.config.enable_embedding_search:
            return None

        try:
            result = self.embedding_runtime.embed([text])
            embedding = result.embeddings[0]
            return embedding if embedding else None
        except Exception:
            return None

    def _extract_fields(self, text: str) -> _ExtractedFields:
        return _ExtractedFields(
            amount=self._extract_amount(text),
            date=self._extract_date(text),
            merchant=self._extract_merchant(text),
        )

    @staticmethod
    def _extract_amount(text: str) -> float | None:
        for pattern in _AMOUNT_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = match.group(1)
            parsed = _safe_float(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _extract_date(text: str) -> str | None:
        for pattern in _DATE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            if len(match.groups()) == 3:
                year, month, day = match.groups()
                return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            if len(match.groups()) == 2:
                month, day = match.groups()
                now = datetime.now()
                return f"{now.year:04d}-{int(month):02d}-{int(day):02d}"
        return None

    @staticmethod
    def _extract_merchant(text: str) -> str | None:
        for pattern in _MERCHANT_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _build_title(doc_type: str, fields, keywords: list[str]) -> str:
        if fields.merchant:
            return f"{doc_type}:{fields.merchant}"
        if keywords:
            return f"{doc_type}:{keywords[0]}"
        if fields.amount is not None:
            return f"{doc_type}: {fields.amount} 元"
        return f"{doc_type} 材料"

    @staticmethod
    def _build_summary(fields, text: str) -> str:
        chunks = []
        if fields.amount is not None:
            chunks.append(f"金额 {fields.amount} 元")
        if fields.date:
            chunks.append(f"日期 {fields.date}")
        if fields.merchant:
            chunks.append(f"商户 {fields.merchant}")
        if not chunks:
            chunks.append(text[:40])
        return "；".join(chunks)

    @staticmethod
    def _normalize_doc_type(doc_type: str) -> str:
        normalized = doc_type.strip().lower().replace(" ", "-")
        return normalized or "receipt"

    @staticmethod
    def _extract_keywords(*parts: str | None) -> list[str]:
        text = " ".join(part for part in parts if part)
        tokens = [t for t in _TOKEN_RE.split(text.lower()) if t and len(t) > 1]
        return sorted(set(tokens))

    @staticmethod
    def _within_amount(amount: float | None, minimum: float | None, maximum: float | None) -> bool:
        if amount is None:
            return minimum is None and maximum is None
        if minimum is not None and amount < minimum:
            return False
        if maximum is not None and amount > maximum:
            return False
        return True

    @staticmethod
    def _within_date(candidate: str | None, from_date: str | None, to_date: str | None) -> bool:
        candidate_dt = _safe_parse_datetime(candidate)
        if candidate_dt is None:
            return not from_date and not to_date
        begin = _safe_parse_datetime(from_date)
        end = _safe_parse_datetime(to_date)
        if begin and candidate_dt < begin:
            return False
        if end and candidate_dt > end:
            return False
        return True

    @staticmethod
    def _sum_amount(materials: Iterable[ExpenseMaterial]) -> float | None:
        total = 0.0
        count = 0
        for material in materials:
            if material.extracted_amount is None:
                continue
            total += material.extracted_amount
            count += 1
        return round(total, 2) if count else None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        size = min(len(a), len(b))
        if size == 0:
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for i in range(size):
            dot += a[i] * b[i]
            norm_a += a[i] ** 2
            norm_b += b[i] ** 2
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / math.sqrt(norm_a * norm_b)

    def _score(self, material: ExpenseMaterial, keyword: str, query_embedding: list[float] | None) -> float:
        if not keyword and query_embedding is None:
            return 0.0
        if not keyword:
            return 0.0

        haystack = " ".join(
            [
                material.title.lower(),
                material.summary.lower(),
                material.claim_id.lower(),
                material.doc_type.lower(),
                material.source_app.lower(),
                material.raw_text.lower(),
            ]
        )
        # 用共享分词器替代 keyword.split()：中文查询通常不带空格，
        # 按空格切会退化成单个整 token（如"京东发票"），命中率显著偏低。
        tokens = tokenize(keyword)
        keyword_score = 0.0
        for token in tokens:
            if token and token in haystack:
                keyword_score += 1.0

        embed_score = 0.0
        if query_embedding is not None and material.embedding:
            embed_score = self._cosine_similarity(query_embedding, material.embedding)

        return round((keyword_score / max(1, len(tokens))) * 0.7 + embed_score * 0.3, 4)
