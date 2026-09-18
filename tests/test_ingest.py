"""个人文件 ingest：captured_at=mtime、增量跳过、变更重导、删除清理。"""

import os
import time
from datetime import datetime, timedelta

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
    expected = datetime.utcnow() - timedelta(days=3)
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
