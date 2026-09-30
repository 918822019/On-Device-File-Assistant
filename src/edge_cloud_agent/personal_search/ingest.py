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
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock

from ..common.file_io import DEFAULT_ENCODINGS, content_hash, parse_encodings, read_text_with_fallback
from ..common.fs_scan import fingerprint, is_ghost, is_unchanged, iter_files, reachable_roots
from ..common.path_utils import as_file_uri
from ..common.time_utils import now_iso
from ..common.watch_loop import IngestResult, run_watch_loop
from ..config import PersonalFileConfig
from .service import PersonalFileSearchService
from .storage import PersonalFileItem

_LOGGER = logging.getLogger("agent_server.personal_search.ingest")

# 扫描串行锁：run_once 可能被后台 watch 线程、/search 节流刷新、/rebuild-index
# 并发触发；全局锁保证同一时刻只有一个扫描在改写 store 与 FAISS 索引文件。
# 锁留在本模块而不在 common/watch_loop：它保护的是 store 与向量索引的写入，
# 而 run_once 有多个入口，只有包在 run_once 里才覆盖得到全部。
_SCAN_LOCK = RLock()


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


def _discover_files(cfg: PersonalFileConfig) -> list[Path]:
    """遍历全部源根目录，返回满足后缀过滤的文件清单（排除目录整棵剪枝）。

    遍历本体在 ``common/fs_scan.iter_files``，与 expense 侧共用同一份实现：
    ``os.walk`` + 原地剪枝、``followlinks=False`` 防符号链接环、结果排序、
    Windows 云占位符门控。平台判定统一走 ``path_utils.is_windows_native()``
    （本模块此前的私有 ``_is_windows()`` 是第三份拷贝，且不支持
    ``FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP`` 覆盖 —— 那是非 Windows 机器上
    测试 Windows 分支的唯一入口）。
    """

    return iter_files(
        source_roots(cfg),
        suffixes=_normalize_root_suffix_set(cfg),
        exclude_dirs=_exclude_dir_names(cfg),
        recursive=cfg.scan_recursive,
        skip_cloud_placeholders=bool(getattr(cfg, "skip_cloud_placeholders", True)),
    )


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


def _file_captured_at(path: Path) -> str:
    """取文件自身修改时间（mtime）作为 captured_at。

    captured_at 是时间线索检索（"昨天/上周/最近N天"）的锚点。
    此前这里取扫描时刻，导致所有新入库文件都假命中近期时间词、
    老文件永远无法按真实拍摄/修改时间被搜到。
    与 now_iso 保持一致：naive UTC ISO 字符串。
    """

    try:
        mtime = path.stat().st_mtime
    except OSError:
        return now_iso()
    if not mtime:
        return now_iso()
    return (
        datetime.fromtimestamp(mtime, tz=timezone.utc)
        .replace(tzinfo=None)
        .isoformat(timespec="seconds")
    )


def _build_item(
    service: PersonalFileSearchService,
    cfg: PersonalFileConfig,
    path: Path,
    *,
    file_hash: str | None = None,
    file_size: int | None = None,
    file_mtime_ns: int | None = None,
) -> PersonalFileItem:
    """基于单个文件构建 PersonalFileItem。

    包括：原文摘要、标签、视觉线索、来源识别、embedding。
    doc_type 与 mime_type 由 _read_text_content 一并给出（它知道内容是否真的
    可读，比单纯按后缀猜更准），故此处不再单独推断后缀分类。

    file_hash / file_size / file_mtime_ns 可由调用方传入，以复用已经做过的
    stat 与 hash：run_once 在判定「有变更」之前就已经算过这两样，若不传进来，
    每个变更文件都会被 stat 两次、hash 两次（等于多读一遍头部 1MB）。
    不传时自行计算，单独调用（如测试）的行为保持不变。
    """

    raw_text, doc_type_guess, mime_type = _read_text_content(
        path,
        encodings=parse_encodings(getattr(cfg, "text_encodings", "")),
    )
    source_app = service.detect_source_app(path)
    now = now_iso()
    file_id = f"fm_{service.next_file_id(path)}"
    doc_type = doc_type_guess

    summary = service.make_summary(raw_text)
    tags = service.extract_tags(path, raw_text, source_app)
    visual_hints = service.extract_visual_hints(path, raw_text)
    captured_at = _file_captured_at(path)

    if file_size is None or file_mtime_ns is None:
        # 一次 stat 同时取 size 与 mtime_ns；此前是 exists() + stat() 两次系统调用
        stat_size, stat_mtime_ns = fingerprint(path)
        if file_size is None:
            file_size = stat_size
        if file_mtime_ns is None:
            file_mtime_ns = stat_mtime_ns
    if file_hash is None:
        file_hash = content_hash(path)

    return PersonalFileItem(
        file_id=file_id,
        title=path.stem,
        file_uri=as_file_uri(path),
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
        file_size_bytes=file_size,
        file_hash=file_hash,
        file_mtime_ns=file_mtime_ns,
        visual_hints=visual_hints,
        tags=tags,
        embedding=service.embed_text(raw_text),
        created_at=now,
        last_scanned_at=now,
    )


def _sweep_deleted_files(
    service: PersonalFileSearchService,
    cfg: PersonalFileConfig,
    known_existing: set[str] | None = None,
) -> int:
    """清理磁盘上已被删除、但索引里仍存在的幽灵记录（多根按根保护）。

    防误删语义从「单根可达」推广为「按根可达」：
    - 全部根不可达 → 完全不清理（原保护逻辑）；
    - 记录归属的根暂不可达（如 /mnt/c 未就绪）→ 保留该记录；
    - 记录归属的根可达但文件已不存在，或记录不属于任何配置根
      （源目录被移出配置的历史残留）→ 清理。
    删除只改内存，落盘由调用方统一 flush。

    判定本体在 ``common/fs_scan.is_ghost``（含 known_existing 命中集跳过 stat、
    未命中回落 path.exists()、按根可达保护），与 expense 侧共用同一份实现。
    删除只改内存，落盘由调用方统一 flush。
    """

    roots = source_roots(cfg)
    if not roots:
        return 0
    reachable = reachable_roots(roots)
    if not reachable:
        return 0

    removed = 0
    for item in service.store.list_all():
        if not item.file_path:
            continue
        if is_ghost(
            Path(item.file_path),
            roots=roots,
            reachable=reachable,
            known_existing=known_existing,
        ):
            service.store.delete_by_file_uri(item.file_uri, persist=False)
            removed += 1
    return removed


def run_once(
    service: PersonalFileSearchService,
    cfg: PersonalFileConfig,
    trace_id: str | None = None,
    force_rehash: bool = False,
) -> IngestResult:
    """执行一次扫描/更新周期。

    行为特征：
    - 未配置 source_dir 时返回全 0；
    - **size + mtime 快路径**：两者都没变的文件连 hash 都不算，只做一次 stat。
      此前每个文件每轮都要读头部 1MB 算 md5，稳态 IO = O(语料总字节)/轮；
    - ``force_rehash=True`` 时绕过快路径，对全部文件做权威的 hash 校验。
      周期扫描传 False，UI 的「重建索引」传 True —— 云同步/备份恢复可能把
      mtime 还原成旧值，那种变更只有 hash 能发现，需要一个显式的兜底出口；
    - 判重前置到构建 item 之前：未变更文件不读内容、不重算 embedding
      （此前每个周期都会对全部文件重跑 300M 向量模型）；
    - hash 变更则先删后加；
    - hash 相同但 mtime 记录缺失/不一致时就地回填元数据（不重读内容），
      避免旧记录永久停留在慢路径；
    - 磁盘上已删除的文件会被清出索引（计入 removed），清理复用本轮枚举结果，
      不再对每个已入库条目单独 stat；
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
    rehashed = 0
    mtime_backfilled = 0
    state_pruned = 0
    imported_ids: list[str] = []
    run_started = datetime.now()
    # 本轮实际枚举到的路径，交给幽灵清理复用，省掉「每个已入库条目一次 stat」
    seen_paths: set[str] = set()

    with _SCAN_LOCK:
        for file_path in _discover_files(cfg):
            scanned += 1
            try:
                file_uri = as_file_uri(file_path)
                # 一次 stat 同时取 size 与 mtime_ns。stat 远比读文件头部 1MB 便宜，
                # 在 WSL 的 9P 跨系统调用上尤其明显。
                file_size, file_mtime_ns = fingerprint(file_path)
                # normcase 与 source_roots / 幽灵清理的去重口径一致（Windows 大小写不敏感）
                seen_paths.add(os.path.normcase(str(file_path)))

                exists = service.store.get_by_file_uri(file_uri)

                # 快路径：size 与 mtime 都没变，内容就不可能变，连 hash 都不必算。
                # 此前每个文件每轮都要读头部 1MB 算 md5，稳态 IO = O(语料总字节)/轮。
                # force_rehash 时绕过快路径做全量权威校验 —— 云同步/备份恢复可能把
                # mtime 还原成旧值（本项目恰好以 OneDrive/iCloud/微信目录为主要目标），
                # 那种情况下只有 hash 能发现变更，故 UI 的「重建索引」走 force 路径。
                # is_unchanged 在 mtime_ns 任一侧为 None 时返回 False，
                # 「未知」不等于「没变」，故不再需要单独的 file_mtime_ns 判空。
                if (
                    not force_rehash
                    and exists is not None
                    and is_unchanged(
                        exists.file_size_bytes, exists.file_mtime_ns, file_size, file_mtime_ns
                    )
                ):
                    skipped += 1
                    continue

                file_hash = content_hash(file_path)
                rehashed += 1

                if (
                    exists is not None
                    and exists.file_hash == file_hash
                    and exists.file_size_bytes == file_size
                ):
                    skipped += 1
                    # 内容没变，但 mtime 记录缺失或对不上（升级前的旧记录没有
                    # file_mtime_ns，或云同步动过 mtime）。就地回填，否则这些文件
                    # 每轮都会掉进慢路径重算 hash。只 replace 元数据：
                    # 不重读内容、不重算 embedding。
                    if exists.file_mtime_ns != file_mtime_ns:
                        service.store.add_or_update(
                            replace(
                                exists,
                                file_mtime_ns=file_mtime_ns,
                                last_scanned_at=now_iso(),
                            ),
                            persist=False,
                        )
                        mtime_backfilled += 1
                    continue

                # 确认是新增或变更后，才做内容读取 / embedding / item 构建。
                # 把已算好的 hash/size/mtime 传下去，避免 _build_item 再算一遍。
                item = _build_item(
                    service,
                    cfg,
                    file_path,
                    file_hash=file_hash,
                    file_size=file_size,
                    file_mtime_ns=file_mtime_ns,
                )
                if not item.raw_text:
                    skipped += 1
                    continue

                if exists is not None:
                    service.store.delete_by_file_uri(file_uri, persist=False)

                service.store.add_or_update(item, persist=False)
                imported_ids.append(item.file_id)
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

        removed = _sweep_deleted_files(service, cfg, known_existing=seen_paths)

        # 幽灵清理只动 store；备注/归档标记要单独剪枝，否则孤儿 file_id 永久残留。
        # force_rehash（手工重建）时也跑一次，让升级前积累的存量孤儿有确定的清理时机。
        if removed > 0 or force_rehash:
            try:
                state_pruned = service.prune_file_state()
            except Exception:
                # 状态剪枝失败不该影响扫描结果，下轮还会再试
                state_pruned = 0
                _LOGGER.warning(
                    "personal-search-state-prune-failed",
                    extra={"event": "ingest.state_prune_failed", "trace_id": trace_id},
                )

        # mtime 回填只改内存，同样需要落盘，否则下轮又要重算 hash
        if imported > 0 or removed > 0 or mtime_backfilled > 0:
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
            # rehashed 是本轮真正读了文件头部算 hash 的数量。稳态下应接近 0
            # （绝大多数文件走 size+mtime 快路径）；若长期等于 scanned，
            # 说明快路径没生效（例如 file_mtime_ns 一直回填不上）。
            "rehashed": rehashed,
            "mtime_backfilled": mtime_backfilled,
            "state_pruned": state_pruned,
            "force_rehash": force_rehash,
            "cost_ms": round((datetime.now() - run_started).total_seconds() * 1000, 2),
            "index_ready": service.vector_index_ready,
            "sample_file_ids": imported_ids[:5],
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


def start_watch_loop(
    service: PersonalFileSearchService,
    cfg: PersonalFileConfig,
    stop_event: Event,
    interval: int | None = None,
) -> None:
    """后台循环：按配置间隔持续扫描。

    循环骨架（调度、停止、失败可见）在 ``common/watch_loop``。此前的
    ``except Exception: pass`` 会把守护线程里的持续失败完全吞掉，连一行日志都
    没有 —— watch 是索引保持新鲜的唯一自动机制，它坏了外部只能看到「新文件
    搜不到」，与 R14「/health 恒绿」是同一类可观测性幻觉。
    """

    run_watch_loop(
        lambda: run_once(service, cfg),
        stop_event,
        interval=cfg.scan_interval_seconds if interval is None else interval,
        logger=_LOGGER,
        loop_name="personal-search-watch",
        event_prefix="ingest.watch",
    )
