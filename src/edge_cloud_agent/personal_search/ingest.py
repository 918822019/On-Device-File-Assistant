"""Personal file ingestion and incremental rebuild loop.

职责：
- 扫描指定目录得到候选文件
- 从文件内容构造 PersonalFileItem
- 去重/更新已存在记录
- 触发向量索引重建

本模块不改变检索策略，仅负责将新文件持续变更落入索引输入。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from threading import Event
from time import sleep
from typing import Sequence
from datetime import datetime

from .schemas import RebuildIndexResponse
from .service import PersonalFileSearchService
from .storage import PersonalFileItem
from .service import _now_iso
from ..config import PersonalFileConfig


_LOGGER = logging.getLogger("agent_server.personal_search.ingest")


@dataclass
class IngestResult:
    scanned: int
    imported: int
    skipped: int
    errors: int
    material_ids: list[str]


_FILE_SUFFIXES_COMMON = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".pdf", ".txt", ".md", ".json", ".csv", ".log",
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".mp4", ".mov", ".m4a", ".mp3", ".wav",
}


def _normalize_root_suffix_set(cfg: PersonalFileConfig) -> set[str]:
    """计算本次扫描使用的后缀白名单；空配置时退回默认集合。"""

    suffixes = {item.strip().lower() for item in cfg.scan_file_suffixes.split(",") if item.strip()}
    if not suffixes:
        return _FILE_SUFFIXES_COMMON
    return suffixes


def _discover_files(cfg: PersonalFileConfig) -> list[Path]:
    """遍历 source_dir，返回满足后缀过滤的文件清单。"""

    source_dir = Path(cfg.source_dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        return []

    suffix_set = _normalize_root_suffix_set(cfg)
    iterator = source_dir.rglob("*") if cfg.scan_recursive else source_dir.iterdir()

    files: list[Path] = []
    for item in iterator:
        if not item.is_file():
            continue
        if item.suffix.lower() not in suffix_set:
            continue
        files.append(item)
    return files


def _read_text_content(path: Path) -> tuple[str, str | None, str]:
    """抽取可用于检索的文本内容与文档类型。"""

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".csv", ".log"}:
        try:
            text = path.read_text(encoding="utf-8")
            return text, path.suffix.lower().strip("."), f"text/{path.suffix.lower().strip('.')}"
        except Exception:
            pass

    fallback = _extract_filename_semantics(path)
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        return fallback, "image", "image/png"
    if suffix in {".pdf"}:
        return fallback, "document", "application/pdf"
    if suffix in {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}:
        return fallback, "document", "application/ms-office"
    if suffix in {".mp4", ".mov", ".m4a", ".mp3", ".wav"}:
        return fallback, "media", "audio/video"

    return fallback, "file", "application/octet-stream"


def _extract_filename_semantics(path: Path) -> str:
    """文件名语义化兜底：当内容不可读时保留文件名线索。"""

    stem = path.stem.replace("_", " ").replace("-", " ")
    return f"文件名: {stem}"


def _infer_doc_type(suffix: str) -> str:
    """按后缀返回粗分类，用于前端展示与检索线索拼接。"""

    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        return "image"
    if suffix == ".pdf" or suffix in {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}:
        return "document"
    if suffix in {".mp4", ".mov", ".m4a", ".mp3", ".wav"}:
        return "media"
    return "text"


def _as_file_uri(path: Path) -> str:
    """统一文件 URI 形式，避免不同系统路径差异导致的展示抖动。"""

    return f"file://{path.as_posix()}"


def _short_sha1(path: Path) -> str:
    """简化 hash：用于判重；只读取前 1MB 以降低 IO 成本。"""

    try:
        data = path.read_bytes()[:1024 * 1024]
        return md5(data).hexdigest()
    except Exception:
        return ""


def _build_item(service: PersonalFileSearchService, cfg: PersonalFileConfig, path: Path) -> PersonalFileItem:
    """基于单个文件构建 PersonalFileItem。

包括：原文摘要、标签、视觉线索、来源识别、embedding。
"""

    raw_text, doc_type_guess, mime_type = _read_text_content(path)
    suffix = path.suffix.lower()
    source_app = service.detect_source_app(path)
    now = _now_iso()
    file_id = f"fm_{service.next_file_id(path)}"
    doc_type = doc_type_guess

    summary = service.make_summary(raw_text)
    tags = service.extract_tags(path, raw_text, source_app)
    visual_hints = service.extract_visual_hints(path, raw_text)
    captured_at = _now_iso()
    return PersonalFileItem(
        file_id=file_id,
        title=path.stem,
        file_uri=_as_file_uri(path),
        source_app=source_app,
        doc_type=doc_type,
        mime_type=mime_type,
        raw_text=raw_text[:service.config.max_search_query_len] if raw_text else raw_text,
        summary=summary,
        file_path=path.as_posix(),
        captured_at=captured_at,
        updated_at=now,
        file_size_bytes=path.stat().st_size if path.exists() else None,
        file_hash=_short_sha1(path),
        visual_hints=visual_hints,
        tags=tags,
        embedding=service.embed_text(raw_text),
        created_at=now,
        last_scanned_at=now,
    )


def run_once(
    service: PersonalFileSearchService,
    cfg: PersonalFileConfig,
    trace_id: str | None = None,
) -> IngestResult:
    """执行一次扫描/更新周期。

行为特征：
- 未配置 source_dir 时返回全 0；
- 文件 hash/size 不变则跳过；
- hash 变更则先删后加。
"""

    if not cfg.source_dir:
        _LOGGER.warning(
            "personal-search-ingest-disabled",
            extra={
                "event": "ingest.disabled",
                "source_dir": cfg.source_dir,
                "trace_id": trace_id,
            },
        )
        return IngestResult(scanned=0, imported=0, skipped=0, errors=0, material_ids=[])

    scanned = 0
    imported = 0
    skipped = 0
    errors = 0
    material_ids: list[str] = []
    run_started = datetime.now()

    for file_path in _discover_files(cfg):
        scanned += 1
        try:
            item = _build_item(service, cfg, file_path)
            if not item.raw_text:
                skipped += 1
                continue

            exists = service.store.get_by_file_uri(item.file_uri)
            if exists is not None and exists.file_hash == item.file_hash and exists.file_size_bytes == item.file_size_bytes:
                skipped += 1
                continue

            if exists is not None and exists.file_hash != item.file_hash:
                service.remove_by_uri(item.file_uri)

            service.store.add_or_update(item)
            material_ids.append(item.file_id)
            imported += 1
        except Exception as exc:
            errors += 1
            _LOGGER.warning(
                "personal-search-ingest-item-error",
                extra={
                    "event": "ingest.item_failed",
                    "file_path": str(file_path),
                    "exception": type(exc).__name__,
                    "trace_id": trace_id,
                },
            )

    if imported > 0 or not service.vector_index_ready:
        try:
            service.rebuild_vector_index()
        except Exception:
            pass

    _LOGGER.info(
        "personal-search-ingest-finished",
        extra={
            "event": "ingest.finished",
            "source_dir": str(cfg.source_dir),
            "scanned": scanned,
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "cost_ms": round((datetime.now() - run_started).total_seconds() * 1000, 2),
            "index_ready": service.vector_index_ready,
            "sample_file_ids": material_ids[:5],
            "trace_id": trace_id,
        },
    )

    return IngestResult(
        scanned=scanned,
        imported=imported,
        skipped=skipped,
        errors=errors,
        material_ids=material_ids,
    )


def start_watch_loop(
    service: PersonalFileSearchService,
    cfg: PersonalFileConfig,
    stop_event: Event,
    interval: int | None = None,
) -> None:
    """后台循环：按配置间隔持续扫描。"""

    delay = interval if interval is not None else cfg.scan_interval_seconds
    while not stop_event.is_set():
        try:
            run_once(service, cfg)
        except Exception:
            pass
        stop_event.wait(delay)
