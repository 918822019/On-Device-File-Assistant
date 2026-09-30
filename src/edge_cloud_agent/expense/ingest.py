"""报销材料的 watch 目录摄取：扫描 → 增量判重 → 导入 → 幽灵清理。

与 ``personal_search/ingest.py`` 共用 ``common/fs_scan``（目录遍历、指纹、幽灵
判定）与 ``common/watch_loop``（循环骨架、结果 DTO）。本模块只保留报销业务自己
的部分：从路径推断报销单号、按后缀推断材料类型、调用 ``service.collect``。

此前这里是 personal 侧的一份**旧快照**：同样的骨架，但 personal 侧后来修的
四项都没有跟过来 —— ``rglob("*")`` 无法剪枝整棵子树、每个文件每轮都读头部
1 MB 算 hash、幽灵清理逐条 ``stat``、``run_once`` 无串行锁（watch 线程与
``/rebuild-index`` 可以并发改写同一个 store）。详见 docs/KNOWN_ISSUES.md R29。
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event, RLock

from ..common.file_io import content_hash, read_text_with_fallback
from ..common.fs_scan import fingerprint, is_ghost, is_unchanged, iter_files, reachable_roots
from ..common.path_utils import as_file_uri, uri_to_path
from ..common.time_utils import now_iso
from ..common.watch_loop import IngestResult, run_watch_loop
from ..config import ExpenseConfig
from .schemas import ExpenseCollectRequest
from .service import ExpenseService
from .storage import ExpenseMaterial

_LOGGER = logging.getLogger("agent_server.expense.ingest")

# 扫描串行锁：run_once 有 watch 线程与 /rebuild-index 两个入口，并发跑会让两个
# 线程同时改写同一个 store（add_or_update / delete_by_file_uri 各自持 store 锁，
# 但「读 existing → 判定 → 写」这个复合操作不是原子的）。与 personal_search 同口径。
_SCAN_LOCK = RLock()

# 白名单里读不出文本的后缀：没有 OCR，属于预期跳过而非错误
_UNREADABLE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}

_DEFAULT_SUFFIXES = {".txt"}


def watch_root(cfg: ExpenseConfig) -> Path:
    """watch 目录的规范化路径（expanduser + resolve）。"""

    return Path(cfg.watch_dir).expanduser().resolve()


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


def _watch_suffixes(cfg: ExpenseConfig) -> set[str]:
    suffixes = {item.strip().lower() for item in cfg.watch_file_suffixes.split(",") if item.strip()}
    return suffixes or _DEFAULT_SUFFIXES


def _discover_files(cfg: ExpenseConfig) -> list[Path]:
    """遍历 watch 目录，返回后缀白名单内的文件（已排序、可整棵子树剪枝）。"""

    return iter_files(
        [watch_root(cfg)],
        suffixes=_watch_suffixes(cfg),
        recursive=cfg.watch_recursive,
    )


def _read_text_file(path: Path) -> str | None:
    if path.suffix.lower() in _UNREADABLE_SUFFIXES:
        return None
    # 编码降级读取（utf-8-sig → gb18030）：Windows 常见 GBK 文件不再静默跳过
    decoded = read_text_with_fallback(path)
    return decoded[0] if decoded is not None else None


def _sweep_deleted_materials(
    service: ExpenseService,
    cfg: ExpenseConfig,
    known_existing: set[str] | None = None,
) -> int:
    """清理 watch 目录中已删除、但 store 里仍残留的材料记录。

    仅在 watch_dir 本身仍存在时执行：目录不可达时全量扫描会得到空列表，
    此时清理会误删全部数据（与 personal_search 同一防护逻辑，判定本体共用
    ``common/fs_scan.is_ghost``）。

    ``known_existing`` 是本轮已枚举到的路径集合（normcase 后）。命中即代表文件
    存在，可跳过 ``stat``；此前每条已入库记录每轮都要一次 ``stat``。
    """

    root = watch_root(cfg)
    if not root.is_dir():
        return 0
    roots = [root]
    reachable = reachable_roots(roots)
    if not reachable:
        return 0

    removed = 0
    for material in service.store.list_all():
        if not material.file_uri:
            continue
        # URI → 路径用 uri_to_path 兼容新旧格式（file:///C:/x 与 file://C:/x、
        # file:///abs）。此前 removeprefix("file://") 在 Windows 新格式下会得到
        # "/C:/x"，exists() 恒 False → 全量误删。
        path_str = uri_to_path(material.file_uri)
        if not path_str:
            continue
        if is_ghost(
            Path(path_str),
            roots=roots,
            reachable=reachable,
            known_existing=known_existing,
        ):
            service.store.delete_by_file_uri(material.file_uri, persist=False)
            removed += 1
    return removed


def _with_mtime(material: ExpenseMaterial, file_mtime_ns: int | None) -> ExpenseMaterial:
    """回填 file_mtime_ns 与 updated_at，返回新实例（ExpenseMaterial 是 frozen）。"""

    return replace(material, file_mtime_ns=file_mtime_ns, updated_at=now_iso())


def run_once(
    service: ExpenseService,
    cfg: ExpenseConfig,
    trace_id: str | None = None,
    force_rehash: bool = False,
) -> IngestResult:
    """执行一轮 watch 目录摄取。

    - 未配置 ``watch_dir`` 时返回全 0
    - **size + mtime 快路径**：两者都没变的文件连 hash 都不算，只做一次 stat。
      此前每个文件每轮都要读头部 1 MB 算 md5，稳态 IO = O(watch 目录总字节)/轮，
      而 watch 循环默认每 120 秒跑一次
    - ``force_rehash=True`` 绕过快路径做全量权威校验（``/rebuild-index`` 用）：
      云同步/备份恢复可能把 mtime 还原成旧值，那种变更只有 hash 能发现
    - hash 相同但 mtime 记录缺失/不一致时就地回填元数据（不重读内容、不重算 embedding）
    - 幽灵清理复用本轮枚举结果
    - 批量导入期间 store 只在结束时落盘一次
    - 全程持有 ``_SCAN_LOCK``
    - 单个文件出错只计入 ``errors`` 并记日志，不中断整轮
    """

    if not cfg.watch_dir:
        _LOGGER.warning(
            "expense-ingest-disabled",
            extra={"event": "expense.ingest.disabled", "trace_id": trace_id},
        )
        return IngestResult(scanned=0, imported=0, skipped=0, errors=0)

    scanned = 0
    imported = 0
    skipped = 0
    errors = 0
    rehashed = 0
    mtime_backfilled = 0
    imported_ids: list[str] = []
    seen_paths: set[str] = set()
    run_started = datetime.now()
    root = watch_root(cfg)

    with _SCAN_LOCK:
        for file_path in _discover_files(cfg):
            scanned += 1
            try:
                file_uri = as_file_uri(file_path)
                file_size, file_mtime_ns = fingerprint(file_path)
                # normcase 与幽灵清理的命中集口径一致（Windows 大小写不敏感）
                seen_paths.add(os.path.normcase(str(file_path)))

                existing = service.store.get_by_file_uri(file_uri)

                if not force_rehash and existing is not None and is_unchanged(
                    existing.file_size_bytes, existing.file_mtime_ns, file_size, file_mtime_ns
                ):
                    skipped += 1
                    continue

                file_hash = content_hash(file_path)
                rehashed += 1

                if existing is not None:
                    if existing.file_hash == file_hash and existing.file_size_bytes == file_size:
                        skipped += 1
                        # 内容没变但 mtime 记录缺失或对不上（升级前的旧记录没有
                        # file_mtime_ns，或云同步动过 mtime）。就地回填，否则这些
                        # 文件每轮都会掉进慢路径重算 hash。只 replace 元数据：
                        # 不重读内容、不重算 embedding。
                        if existing.file_mtime_ns != file_mtime_ns:
                            service.store.add_or_update(
                                _with_mtime(existing, file_mtime_ns),
                                persist=False,
                            )
                            mtime_backfilled += 1
                        continue
                    # 内容变更：先删旧记录再重新导入。
                    # 顺序与改动前一致 —— 若把删除挪到「读不出文本」的分支之后，
                    # 一个变成不可读的文件会留下过期记录。
                    service.store.delete_by_file_uri(file_uri, persist=False)

                raw_text = _read_text_file(file_path)
                if not raw_text:
                    # 白名单内的 pdf/图片本就读不出文本，属于预期跳过
                    skipped += 1
                    continue

                claim_id = _normalize_claim_from_path(str(root), file_path, cfg.default_claim_id)
                request = ExpenseCollectRequest(
                    claim_id=claim_id,
                    source_app="auto-watch",
                    doc_type=_infer_doc_type(file_path.suffix.lower()),
                    title=file_path.stem,
                    raw_text=raw_text.strip(),
                    file_uri=file_uri,
                    file_hash=file_hash,
                    file_size_bytes=file_size,
                    file_mtime_ns=file_mtime_ns,
                    captured_at=None,
                    notes=f"watch-dir: {file_path.as_posix()}",
                )
                material = service.collect(request, claim_id=claim_id, persist=False)
                imported_ids.append(material.material_id)
                imported += 1
            except Exception as exc:
                # 单个文件失败不中断整轮。此前只有 collect() 被 try 包住，
                # stat / hash / 清理阶段的异常会直接掀掉整轮扫描。
                errors += 1
                _LOGGER.warning(
                    "expense-ingest-item-error",
                    extra={
                        "event": "expense.ingest.item_failed",
                        "file_path": str(file_path),
                        # 只记异常类型不记 message：消息里常带完整文件路径，
                        # 与 personal_search 的 ingest.item_failed 同口径
                        "exception": type(exc).__name__,
                        "trace_id": trace_id,
                    },
                )

        removed = _sweep_deleted_materials(service, cfg, known_existing=seen_paths)

        # mtime 回填只改内存，同样需要落盘，否则下轮又要重算 hash
        if imported > 0 or removed > 0 or mtime_backfilled > 0:
            service.store.flush()

    _LOGGER.info(
        "expense-ingest-finished",
        extra={
            "event": "expense.ingest.finished",
            "watch_dir": str(cfg.watch_dir),
            "scanned": scanned,
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "removed": removed,
            # 稳态下 rehashed 应接近 0；若长期等于 scanned，说明快路径没生效
            "rehashed": rehashed,
            "mtime_backfilled": mtime_backfilled,
            "force_rehash": force_rehash,
            "cost_ms": round((datetime.now() - run_started).total_seconds() * 1000, 2),
            "sample_material_ids": imported_ids[:5],
            "trace_id": trace_id,
        },
    )

    return IngestResult(
        scanned=scanned,
        imported=imported,
        skipped=skipped,
        errors=errors,
        removed=removed,
        imported_ids=imported_ids,
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


def start_watch_loop(
    service: ExpenseService,
    cfg: ExpenseConfig,
    stop_event: Event,
    interval: int | None = None,
) -> None:
    """后台循环：按配置间隔持续扫描 watch 目录。

    循环骨架在 ``common/watch_loop``。此前的 ``except Exception: pass`` 会把
    守护线程里的持续失败完全吞掉，连一行日志都没有。
    """

    run_watch_loop(
        lambda: run_once(service, cfg),
        stop_event,
        interval=cfg.watch_interval_seconds if interval is None else interval,
        logger=_LOGGER,
        loop_name="expense-watch",
        event_prefix="expense.watch",
    )
