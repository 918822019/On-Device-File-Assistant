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
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from threading import Event, RLock
from time import sleep
from typing import Sequence
from datetime import datetime, timezone

from .schemas import RebuildIndexResponse
from .service import PersonalFileSearchService
from .storage import PersonalFileItem
from .service import _now_iso
from ..config import PersonalFileConfig


_LOGGER = logging.getLogger("agent_server.personal_search.ingest")

# 扫描串行锁：run_once 可能被后台 watch 线程、/search 节流刷新、/rebuild-index
# 并发触发；全局锁保证同一时刻只有一个扫描在改写 store 与 FAISS 索引文件。
_SCAN_LOCK = RLock()


@dataclass
class IngestResult:
    scanned: int
    imported: int
    skipped: int
    errors: int
    removed: int = 0
    material_ids: list[str] = field(default_factory=list)


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


def _file_captured_at(path: Path) -> str:
    """取文件自身修改时间（mtime）作为 captured_at。

    captured_at 是时间线索检索（"昨天/上周/最近N天"）的锚点。
    此前这里取扫描时刻，导致所有新入库文件都假命中近期时间词、
    老文件永远无法按真实拍摄/修改时间被搜到。
    与 _now_iso 保持一致：naive UTC ISO 字符串。
    """

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _now_iso()
    if not mtime:
        return _now_iso()
    return (
        datetime.fromtimestamp(mtime, tz=timezone.utc)
        .replace(tzinfo=None)
        .isoformat(timespec="seconds")
    )


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
    captured_at = _file_captured_at(path)
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


def _sweep_deleted_files(service: PersonalFileSearchService, cfg: PersonalFileConfig) -> int:
    """清理磁盘上已被删除、但索引里仍存在的幽灵记录。

    仅在 source_dir 目录本身仍存在时执行：目录未挂载/暂不可达时全量扫描
    会得到空列表，此时清索引会误删全部数据。删除只改内存，落盘由调用方
    统一 flush。
    """

    source_root = Path(cfg.source_dir).expanduser().resolve()
    if not source_root.is_dir():
        return 0

    removed = 0
    for item in service.store.list_all():
        if not item.file_path:
            continue
        if Path(item.file_path).exists():
            continue
        service.store.delete_by_file_uri(item.file_uri, persist=False)
        removed += 1
    return removed


def run_once(
    service: PersonalFileSearchService,
    cfg: PersonalFileConfig,
    trace_id: str | None = None,
) -> IngestResult:
    """执行一次扫描/更新周期。

行为特征：
- 未配置 source_dir 时返回全 0；
- hash/size 判重前置到构建 item 之前：未变更文件不再读内容、不再重算
  embedding（此前每个扫描周期都会对全部文件重跑 300M 向量模型）；
- hash 变更则先删后加；
- 磁盘上已删除的文件会被清出索引（计入 removed）；
- 批量导入期间 store 只在结束时落盘一次；
- 全程持有 _SCAN_LOCK，多入口触发时串行执行。
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
        return IngestResult(scanned=0, imported=0, skipped=0, errors=0)

    scanned = 0
    imported = 0
    skipped = 0
    errors = 0
    removed = 0
    material_ids: list[str] = []
    run_started = datetime.now()

    with _SCAN_LOCK:
        for file_path in _discover_files(cfg):
            scanned += 1
            try:
                file_uri = _as_file_uri(file_path)
                file_hash = _short_sha1(file_path)
                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    file_size = None

                exists = service.store.get_by_file_uri(file_uri)
                if (
                    exists is not None
                    and exists.file_hash == file_hash
                    and exists.file_size_bytes == file_size
                ):
                    skipped += 1
                    continue

                # 确认是新增或变更后，才做内容读取 / embedding / item 构建。
                item = _build_item(service, cfg, file_path)
                if not item.raw_text:
                    skipped += 1
                    continue

                if exists is not None:
                    service.store.delete_by_file_uri(file_uri, persist=False)

                service.store.add_or_update(item, persist=False)
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

        removed = _sweep_deleted_files(service, cfg)

        if imported > 0 or removed > 0:
            service.store.flush()

        if imported > 0 or removed > 0 or not service.vector_index_ready:
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
            "removed": removed,
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
        removed=removed,
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
