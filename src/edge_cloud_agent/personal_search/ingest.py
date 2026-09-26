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
import os
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
from ..common.file_io import DEFAULT_ENCODINGS, parse_encodings, read_text_with_fallback
from ..common.path_utils import as_file_uri


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


def source_roots(cfg: PersonalFileConfig) -> list[Path]:
    """解析多根源目录：逗号分隔，expanduser + resolve，去重保序。

    WSL 部署下 Windows 侧文件天然分散（Desktop/Downloads/Pictures/微信目录），
    单根不够用；单根配置是本函数的退化情形，行为不变。
    """

    roots: list[Path] = []
    seen: set[str] = set()
    for chunk in (cfg.source_dir or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        path = Path(chunk).expanduser().resolve()
        # normcase：Windows 文件系统大小写不敏感，c:\x 与 C:\x 是同一个根；
        # POSIX 上 normcase 为恒等变换，行为不变
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return roots


def _exclude_dir_names(cfg: PersonalFileConfig) -> set[str]:
    return {c.strip().lower() for c in (cfg.scan_exclude_dirs or "").split(",") if c.strip()}


# Windows 云占位符文件属性位（OneDrive 等「仅在线」文件）：读取会触发静默
# 全量下载，扫描阶段直接排除出扫描面。非 Windows 平台无 st_file_attributes
# （或恒为 0），判定自然短路为 False。
_FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000


def _is_windows() -> bool:
    """当前进程是否 Windows 原生。

    独立间接层而非直接读 os.name：全局 patch os.name 会让 pathlib 在
    非 Windows 机器上实例化 WindowsPath 崩溃，单测只能 patch 本函数。
    """

    return os.name == "nt"


def _safe_stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None


def _is_cloud_placeholder(st) -> bool:
    """判断 stat 结果是否为 Windows 云占位符文件（st 为 None / 非 Windows → False）。"""

    if st is None:
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & (_FILE_ATTRIBUTE_RECALL_ON_OPEN | _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS))


def _discover_files(cfg: PersonalFileConfig) -> list[Path]:
    """遍历全部源根目录，返回满足后缀过滤的文件清单（排除目录剪枝）。"""

    suffix_set = _normalize_root_suffix_set(cfg)
    excludes = _exclude_dir_names(cfg)
    # 云占位符检测只在 Windows 上有意义；先按平台门控，
    # 避免其他平台每个文件多一次 stat
    skip_placeholders = bool(getattr(cfg, "skip_cloud_placeholders", True)) and _is_windows()

    def _wanted(path: Path) -> bool:
        if path.suffix.lower() not in suffix_set:
            return False
        if skip_placeholders and _is_cloud_placeholder(_safe_stat(path)):
            return False
        return True

    files: list[Path] = []
    for root in source_roots(cfg):
        if not root.is_dir():
            continue
        if not cfg.scan_recursive:
            try:
                entries = list(root.iterdir())
            except OSError:
                continue
            for item in entries:
                if item.is_file() and _wanted(item):
                    files.append(item)
            continue
        # os.walk + 原地剪枝。此前用 rglob("*")：无法跳过整棵子树，WSL 下
        # 跨 9P 扫 /mnt/c 时 node_modules/AppData 级目录会让扫描成本爆炸。
        # followlinks=False 防符号链接环。
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if excludes:
                dirnames[:] = [d for d in dirnames if d.lower() not in excludes]
            for name in filenames:
                path = Path(dirpath) / name
                if _wanted(path):
                    files.append(path)
    files.sort()
    return files


def _read_text_content(
    path: Path,
    encodings: Sequence[str] | None = None,
) -> tuple[str, str | None, str]:
    """抽取可用于检索的文本内容与文档类型。

    文本后缀走编码降级读取（默认 utf-8-sig → gb18030，见 file_io）：
    Windows 常见 GBK/ANSI 文件不再被静默跳过，BOM 不再混入正文。
    全部解码失败（或二进制）才降级为文件名语义。
    """

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".csv", ".log"}:
        decoded = read_text_with_fallback(path, encodings=encodings or DEFAULT_ENCODINGS)
        if decoded is not None:
            text = decoded[0]
            return text, suffix.strip("."), f"text/{suffix.strip('.')}"

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
    """统一文件 URI 形式（委托 path_utils.as_file_uri，单一事实源）。

    POSIX 与历史 ``f"file://{as_posix()}"`` 逐字一致（存量索引零失配）；
    Windows 盘符路径产出合法的 ``file:///C:/...``（旧形式 ``file://C:/...``
    会把盘符落在 authority 位）。
    """

    return as_file_uri(path)


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

    raw_text, doc_type_guess, mime_type = _read_text_content(
        path,
        encodings=parse_encodings(getattr(cfg, "text_encodings", "")),
    )
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
        # 截断长度用独立配置：此前误复用 max_search_query_len（查询截断 120 字符），
        # 文档正文可检索面被限死在开头一小段。已入库的旧记录在文件变更或
        # 删除 store 重建后刷新。
        raw_text=raw_text[: service.config.raw_text_max_chars] if raw_text else raw_text,
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


def _owning_root(path: Path, roots: list[Path]) -> Path | None:
    """返回 path 所属的源根目录（不在任何根下时返回 None）。"""

    for root in roots:
        try:
            path.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def _sweep_deleted_files(service: PersonalFileSearchService, cfg: PersonalFileConfig) -> int:
    """清理磁盘上已被删除、但索引里仍存在的幽灵记录（多根按根保护）。

    防误删语义从「单根可达」推广为「按根可达」：
    - 全部根不可达 → 完全不清理（原保护逻辑）；
    - 记录归属的根暂不可达（如 /mnt/c 未就绪）→ 保留该记录；
    - 记录归属的根可达但文件已不存在，或记录不属于任何配置根
      （源目录被移出配置的历史残留）→ 清理。
    删除只改内存，落盘由调用方统一 flush。
    """

    roots = source_roots(cfg)
    if not roots:
        return 0
    # normcase 与 source_roots 的去重口径一致（Windows 大小写不敏感）
    reachable = {os.path.normcase(str(r)) for r in roots if r.is_dir()}
    if not reachable:
        return 0

    removed = 0
    for item in service.store.list_all():
        if not item.file_path:
            continue
        path = Path(item.file_path)
        if path.exists():
            continue
        owner = _owning_root(path, roots)
        if owner is not None and os.path.normcase(str(owner)) not in reachable:
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
