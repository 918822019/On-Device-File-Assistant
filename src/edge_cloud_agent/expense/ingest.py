"""Automatic file ingest helpers for expense index build."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
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
    material_ids: list[str]


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


def run_once(service: ExpenseService, cfg: ExpenseConfig) -> IngestResult:
    """Run a single ingest pass and return statistics."""
    if not cfg.watch_dir:
        return IngestResult(scanned=0, imported=0, skipped=0, errors=0, material_ids=[])

    scanned = 0
    imported = 0
    skipped = 0
    errors = 0
    material_ids: list[str] = []
    watch_root = Path(cfg.watch_dir).expanduser().resolve()

    for file_path in _discover_files(cfg):
        scanned += 1
        file_uri = _as_file_uri(file_path)
        if service.store.get_by_file_uri(file_uri) is not None:
            skipped += 1
            continue

        raw_text = _read_text_file(file_path)
        if not raw_text:
            # 白名单内的 pdf/图片本就读不出文本，属于预期跳过；
            # 此前同时计入 skipped 和 errors，导致监控里全是假告警。
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
            captured_at=None,
            notes=f"watch-dir: {file_path.as_posix()}",
        )

        try:
            material = service.collect(request, claim_id=claim_id)
            material_ids.append(material.material_id)
            imported += 1
        except Exception:
            errors += 1

    return IngestResult(
        scanned=scanned,
        imported=imported,
        skipped=skipped,
        errors=errors,
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
