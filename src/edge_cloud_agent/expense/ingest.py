"""Automatic file ingest helpers for expense index build."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from threading import Event

from .schemas import ExpenseCollectRequest
from .service import ExpenseService
from ..config import ExpenseConfig


@dataclass
class IngestResult:
    scanned: int
    imported: int
    skipped: int
    errors: int
    removed: int = 0
    material_ids: list[str] = field(default_factory=list)


def _normalize_claim_from_path(root: str, file_path: Path, default_claim_id: str) -> str:
    root_path = Path(root).resolve()
    parent = file_path.parent
    try:
        if parent == root_path:
            return default_claim_id
        if parent.is_relative_to(root_path):
            relative = parent.relative_to(root_path)
            if str(relative) and relative.parts:
                return str(relative.parts[0])
    except Exception:
        # Python 版本兼容（3.11）或路径异常兜底
        pass
    return default_claim_id


def _as_file_uri(path: Path) -> str:
    return f"file://{path.as_posix()}"


def _short_hash(path: Path) -> str:
    """只读前 1MB 计算 md5，与 personal_search/ingest.py 口径一致。"""
    try:
        data = path.read_bytes()[:1024 * 1024]
        return md5(data).hexdigest()
    except Exception:
        return ""


def _discover_files(cfg: ExpenseConfig) -> list[Path]:
    watch_dir = Path(cfg.watch_dir).expanduser().resolve()
    if not watch_dir.exists() or not watch_dir.is_dir():
        return []

    suffix_set = {item.strip().lower() for item in cfg.watch_file_suffixes.split(",") if item.strip()}
    if not suffix_set:
        suffix_set = {".txt"}

    if not cfg.watch_recursive:
        iterator = watch_dir.iterdir()
    else:
        iterator = watch_dir.rglob("*")

    files: list[Path] = []
    for item in iterator:
        if not item.is_file():
            continue
        if item.suffix.lower() in suffix_set:
            files.append(item)
    return files


def _read_text_file(path: Path) -> str | None:
    if path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _sweep_deleted_materials(service: ExpenseService, cfg: ExpenseConfig) -> int:
    """清理 watch 目录中已删除、但 store 里仍残留的材料记录。

    仅在 watch_dir 本身仍存在时执行：目录不可达时全量扫描会得到空列表，
    此时清理会误删全部数据（与 personal_search/ingest.py 同一防护逻辑）。
    """
    watch_root = Path(cfg.watch_dir).expanduser().resolve()
    if not watch_root.is_dir():
        return 0

    removed = 0
    for material in service.store.list_all():
        if not material.file_uri:
            continue
        # file_uri 格式为 file:///path/to/file
        path_str = material.file_uri.removeprefix("file://")
        if Path(path_str).exists():
            continue
        service.store.delete_by_file_uri(material.file_uri, persist=False)
        removed += 1
    return removed


def run_once(service: ExpenseService, cfg: ExpenseConfig) -> IngestResult:
    """Run a single ingest pass and return statistics."""
    if not cfg.watch_dir:
        return IngestResult(scanned=0, imported=0, skipped=0, errors=0)

    scanned = 0
    imported = 0
    skipped = 0
    errors = 0
    material_ids: list[str] = []
    watch_root = Path(cfg.watch_dir).expanduser().resolve()

    for file_path in _discover_files(cfg):
        scanned += 1
        file_uri = _as_file_uri(file_path)
        file_hash = _short_hash(file_path)
        try:
            file_size = file_path.stat().st_size
        except OSError:
            file_size = None

        existing = service.store.get_by_file_uri(file_uri)
        if existing is not None:
            if existing.file_hash == file_hash and existing.file_size_bytes == file_size:
                skipped += 1
                continue
            # 内容变更：先删旧记录再重新导入
            service.store.delete_by_file_uri(file_uri, persist=False)

        raw_text = _read_text_file(file_path)
        if not raw_text:
            # 白名单内的 pdf/图片本就读不出文本，属于预期跳过
            skipped += 1
            continue

        claim_id = _normalize_claim_from_path(str(watch_root), file_path, cfg.default_claim_id)
        doc_type = _infer_doc_type(file_path.suffix.lower())

        request = ExpenseCollectRequest(
            claim_id=claim_id,
            source_app="auto-watch",
            doc_type=doc_type,
            title=file_path.stem,
            raw_text=raw_text.strip(),
            file_uri=file_uri,
            file_hash=file_hash,
            file_size_bytes=file_size,
            captured_at=None,
            notes=f"watch-dir: {file_path.as_posix()}",
        )

        try:
            material = service.collect(request, claim_id=claim_id, persist=False)
            material_ids.append(material.material_id)
            imported += 1
        except Exception:
            errors += 1

    removed = _sweep_deleted_materials(service, cfg)

    if imported > 0 or removed > 0:
        service.store.flush()

    return IngestResult(
        scanned=scanned,
        imported=imported,
        skipped=skipped,
        errors=errors,
        removed=removed,
        material_ids=material_ids,
    )


def _infer_doc_type(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in {".pdf"}:
        return "invoice"
    if suffix in {".jpg", ".jpeg", ".png"}:
        return "receipt"
    if suffix in {".csv", ".log"}:
        return "approval"
    return "receipt"


def start_watch_loop(service: ExpenseService, cfg: ExpenseConfig, stop_event: Event, interval: int | None = None) -> None:
    delay = interval if interval is not None else cfg.watch_interval_seconds
    while not stop_event.is_set():
        try:
            run_once(service, cfg)
        except Exception:
            # 循环内错误不打断守护线程
            pass
        stop_event.wait(delay)
