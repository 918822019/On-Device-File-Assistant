"""个人文件 ingest：captured_at=mtime、增量跳过、变更重导、删除清理。"""

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from edge_cloud_agent.config import PersonalFileConfig
from edge_cloud_agent.personal_search.ingest import _build_item, _file_captured_at, run_once
from edge_cloud_agent.personal_search.service import PersonalFileSearchService
from edge_cloud_agent.personal_search.storage import PersonalFileStore


@pytest.fixture()
def env(tmp_path):
    src_dir = tmp_path / "files"
    src_dir.mkdir()
    cfg = PersonalFileConfig(
        store_path=str(tmp_path / "store.jsonl"),
        state_path=str(tmp_path / "state.json"),
        source_dir=str(src_dir),
        faiss_index_path=str(tmp_path / "idx.index"),
        enable_faiss=False,
    )
    store = PersonalFileStore(cfg.store_path)
    service = PersonalFileSearchService(config=cfg, store=store, embedding_runtime=None)
    return cfg, service, src_dir


def test_captured_at_uses_file_mtime(env):
    """captured_at 必须是文件自身 mtime，而不是扫描时刻。"""

    cfg, service, src_dir = env
    f = src_dir / "old_photo.txt"
    f.write_text("聚餐 photo", encoding="utf-8")
    three_days_ago = time.time() - 3 * 86400
    os.utime(f, (three_days_ago, three_days_ago))

    captured = _file_captured_at(f)
    dt = datetime.fromisoformat(captured)
    expected = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=3)
    assert abs((dt - expected).total_seconds()) < 120, f"captured_at={captured} 应接近 3 天前"

    item = _build_item(service, cfg, f)
    assert item.captured_at == captured


def test_run_once_import_skip_update_remove(env):
    cfg, service, src_dir = env
    f = src_dir / "note.txt"
    f.write_text("会议纪要", encoding="utf-8")

    r1 = run_once(service, cfg)
    assert (r1.scanned, r1.imported, r1.skipped, r1.errors) == (1, 1, 0, 0)
    assert len(service.store.list_all()) == 1

    # 未变更：应跳过，不重复导入
    r2 = run_once(service, cfg)
    assert (r2.scanned, r2.imported, r2.skipped) == (1, 0, 1)

    # 内容变更：先删后加，重新导入
    f.write_text("会议纪要 v2 更新了内容", encoding="utf-8")
    r3 = run_once(service, cfg)
    assert r3.imported == 1
    assert len(service.store.list_all()) == 1

    # 磁盘删除：索引应清理幽灵记录
    f.unlink()
    r4 = run_once(service, cfg)
    assert r4.removed == 1
    assert service.store.list_all() == []


def test_run_once_without_source_dir_noop(tmp_path):
    cfg = PersonalFileConfig(
        store_path=str(tmp_path / "store.jsonl"),
        state_path=str(tmp_path / "state.json"),
        source_dir="",
        enable_faiss=False,
    )
    service = PersonalFileSearchService(
        config=cfg,
        store=PersonalFileStore(cfg.store_path),
        embedding_runtime=None,
    )
    result = run_once(service, cfg)
    assert (result.scanned, result.imported, result.removed) == (0, 0, 0)


def test_sweep_skipped_when_source_dir_missing(tmp_path):
    """source_dir 不可达（如未挂载）时不得清空索引。"""

    cfg = PersonalFileConfig(
        store_path=str(tmp_path / "store.jsonl"),
        state_path=str(tmp_path / "state.json"),
        source_dir=str(tmp_path / "not_mounted"),
        enable_faiss=False,
    )
    service = PersonalFileSearchService(
        config=cfg,
        store=PersonalFileStore(cfg.store_path),
        embedding_runtime=None,
    )
    # source_dir 不存在时 run_once 直接早退（不扫描、不清理）
    result = run_once(service, cfg)
    assert result.scanned == 0
    assert result.removed == 0


# ---------------- WSL 文件索引层：多根 / 剪枝 / 按根保护 ----------------


def _make_service(tmp_path, source_dir: str, name: str = "multi"):
    cfg = PersonalFileConfig(
        store_path=str(tmp_path / f"{name}_store.jsonl"),
        state_path=str(tmp_path / f"{name}_state.json"),
        source_dir=source_dir,
        faiss_index_path=str(tmp_path / f"{name}_idx.index"),
        enable_faiss=False,
    )
    service = PersonalFileSearchService(
        config=cfg,
        store=PersonalFileStore(cfg.store_path),
        embedding_runtime=None,
    )
    return cfg, service


def test_multi_root_scan_imports_from_all_roots(tmp_path):
    """FILE_MEMORY_SOURCE_DIR 逗号分隔多根：两个根下的文件都要入库。"""

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    (root_a / "桌面笔记.txt").write_text("desktop note", encoding="utf-8")
    (root_b / "下载发票.txt").write_text("invoice from downloads", encoding="utf-8")

    cfg, service = _make_service(tmp_path, f"{root_a},{root_b}")
    result = run_once(service, cfg)
    assert (result.scanned, result.imported, result.errors) == (2, 2, 0)
    titles = {item.title for item in service.store.list_all()}
    assert titles == {"桌面笔记", "下载发票"}


def test_multi_root_dedupe_and_blank_chunks(tmp_path):
    """重复根去重；首尾多余逗号/空格不影响解析。"""

    root_a = tmp_path / "a"
    root_a.mkdir()
    (root_a / "only.txt").write_text("single root repeated", encoding="utf-8")

    cfg, service = _make_service(tmp_path, f" , {root_a} , {root_a}, ")
    result = run_once(service, cfg)
    assert (result.scanned, result.imported) == (1, 1)


def test_exclude_dirs_pruned(tmp_path):
    """排除目录整棵剪枝：node_modules 下的文件不进扫描面。"""

    root = tmp_path / "root"
    (root / "node_modules" / "dep").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "node_modules" / "dep" / "lib.txt").write_text("junk", encoding="utf-8")
    (root / ".git" / "config.txt").write_text("junk", encoding="utf-8")
    (root / "keep.txt").write_text("real content", encoding="utf-8")

    cfg, service = _make_service(tmp_path, str(root))
    result = run_once(service, cfg)
    assert result.scanned == 1
    assert result.imported == 1
    assert service.store.list_all()[0].title == "keep"


def test_exclude_dirs_can_be_disabled(tmp_path):
    """FILE_MEMORY_SCAN_EXCLUDE_DIRS 置空 = 关闭剪枝。"""

    root = tmp_path / "root"
    (root / "node_modules").mkdir(parents=True)
    (root / "node_modules" / "lib.txt").write_text("junk", encoding="utf-8")

    cfg = PersonalFileConfig(
        store_path=str(tmp_path / "s.jsonl"),
        state_path=str(tmp_path / "st.json"),
        source_dir=str(root),
        scan_exclude_dirs="",
        enable_faiss=False,
    )
    service = PersonalFileSearchService(
        config=cfg, store=PersonalFileStore(cfg.store_path), embedding_runtime=None
    )
    result = run_once(service, cfg)
    assert result.scanned == 1


def test_sweep_per_root_protection(tmp_path):
    """按根保护：不可达根（如 /mnt/c 未挂载）下的记录保留，可达根下的幽灵清理。"""

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    file_a = root_a / "a.txt"
    file_b = root_b / "b.txt"
    file_a.write_text("content a", encoding="utf-8")
    file_b.write_text("content b", encoding="utf-8")

    cfg, service = _make_service(tmp_path, f"{root_a},{root_b}")
    run_once(service, cfg)
    assert len(service.store.list_all()) == 2

    # 模拟：B 根整个不可达（未挂载），同时 A 根下文件被删除
    root_b.rename(tmp_path / "b_unmounted")
    file_a.unlink()
    result = run_once(service, cfg)

    assert result.removed == 1, "只清 A 根下的幽灵，B 根不可达须保留"
    remaining = service.store.list_all()
    assert len(remaining) == 1
    assert remaining[0].title == "b"

    # B 根恢复可达后，b.txt 仍在磁盘 → 记录继续保留且判重跳过
    (tmp_path / "b_unmounted").rename(root_b)
    result2 = run_once(service, cfg)
    assert (result2.imported, result2.removed, result2.skipped) == (0, 0, 1)


def test_sweep_all_roots_unreachable_no_cleanup(tmp_path):
    """全部根不可达时完全不清理（原单根保护语义的多根推广）。"""

    root_a = tmp_path / "a"
    root_a.mkdir()
    (root_a / "a.txt").write_text("content", encoding="utf-8")
    cfg, service = _make_service(tmp_path, str(root_a))
    run_once(service, cfg)
    assert len(service.store.list_all()) == 1

    root_a.rename(tmp_path / "a_gone")
    result = run_once(service, cfg)
    assert result.removed == 0
    assert len(service.store.list_all()) == 1
