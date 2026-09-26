"""个人文件 ingest：captured_at=mtime、增量跳过、变更重导、删除清理。"""

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from edge_cloud_agent.config import PersonalFileConfig
from edge_cloud_agent.personal_search.ingest import _build_item, _file_captured_at, run_once
from edge_cloud_agent.personal_search.service import (
    PersonalFileSearchService,
    _infer_source_from_path,
)
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


def test_macos_volume_junk_pruned(tmp_path):
    """macOS 卷元数据目录（Spotlight/fseventsd 等）默认剪枝。"""

    root = tmp_path / "vol"
    (root / ".Spotlight-V100").mkdir(parents=True)
    (root / ".fseventsd").mkdir()
    (root / ".Spotlight-V100" / "index.txt").write_text("junk", encoding="utf-8")
    (root / ".fseventsd" / "log.txt").write_text("junk", encoding="utf-8")
    (root / "real.txt").write_text("real file", encoding="utf-8")

    cfg, service = _make_service(tmp_path, str(root), name="macjunk")
    result = run_once(service, cfg)
    assert result.scanned == 1
    assert service.store.list_all()[0].title == "real"


def test_infer_source_macos_screenshots():
    """macOS 截图文件名（中/英）应识别为 camera 来源。"""

    assert _infer_source_from_path(Path("/Users/x/Desktop/截屏2026-09-24 下午3.00.00.png")) == "camera"
    assert _infer_source_from_path(Path("/Users/x/Desktop/Screen Shot 2026-09-24 at 3.00 PM.png")) == "camera"
    assert _infer_source_from_path(Path("/Users/x/Desktop/Screenshot 2026-09-24.png")) == "camera"
    # 微信沙盒路径（macOS 容器布局）仍应命中 wechat
    wx = Path("/Users/x/Library/Containers/com.tencent.xinWeChat/Data/.../MessageTemp/abc/IMG_001.png")
    assert _infer_source_from_path(wx) == "wechat"


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


# ---------------- 跨平台：编码降级 / URI 规范化 / 云占位符 ----------------

from edge_cloud_agent.path_utils import as_file_uri  # noqa: E402
from edge_cloud_agent.personal_search import ingest as ingest_mod  # noqa: E402
from edge_cloud_agent.personal_search.ingest import (  # noqa: E402
    _as_file_uri,
    _is_cloud_placeholder,
)


def test_gb18030_file_indexed_end_to_end(tmp_path):
    """Windows 常见 GBK 编码文本：整链路（run_once）后正文可检索。"""

    root = tmp_path / "gbk_root"
    root.mkdir()
    f = root / "会议纪要.txt"
    f.write_bytes("会议纪要 报销单 金额128元".encode("gb18030"))

    cfg, service = _make_service(tmp_path, str(root), name="gbk")
    result = run_once(service, cfg)
    assert (result.imported, result.errors) == (1, 0)
    item = service.store.list_all()[0]
    assert "报销单" in item.raw_text


def test_utf8_bom_not_in_raw_text(tmp_path):
    root = tmp_path / "bom_root"
    root.mkdir()
    f = root / "note.txt"
    f.write_bytes("\ufeff正文内容".encode("utf-8"))

    cfg, service = _make_service(tmp_path, str(root), name="bom")
    run_once(service, cfg)
    item = service.store.list_all()[0]
    # utf-8-sig 剥掉 BOM：正文首字符不得残留 \ufeff
    assert item.raw_text == "正文内容"
    assert "\ufeff" not in item.raw_text


def test_build_item_encodings_from_cfg(tmp_path):
    """cfg.text_encodings 可限定编码链：仅 ascii 时中文文件降级为文件名语义。"""

    f = tmp_path / "中文笔记.txt"
    f.write_bytes("正文内容".encode("utf-8"))
    cfg = PersonalFileConfig(
        store_path=str(tmp_path / "s.jsonl"),
        state_path=str(tmp_path / "st.json"),
        source_dir=str(tmp_path),
        text_encodings="ascii",
        enable_faiss=False,
    )
    service = PersonalFileSearchService(
        config=cfg, store=PersonalFileStore(cfg.store_path), embedding_runtime=None
    )
    item = _build_item(service, cfg, f)
    assert item.raw_text.startswith("文件名:")


def test_as_file_uri_delegates_to_path_utils():
    """两处 _as_file_uri 与 path_utils.as_file_uri 单一事实源（防再漂移）。"""

    from edge_cloud_agent.expense.ingest import _as_file_uri as expense_uri

    samples = [Path("/Users/x/a.png"), Path("/mnt/c/Users/x/a.png"), Path("C:/Users/x/a.png")]
    for p in samples:
        assert _as_file_uri(p) == as_file_uri(p)
        assert expense_uri(p) == as_file_uri(p)
    # Windows 盘符：合法三斜杠；POSIX：与历史格式一致
    assert as_file_uri(Path("C:/x/a.png")) == "file:///C:/x/a.png"
    assert as_file_uri(Path("/tmp/x.txt")) == "file:///tmp/x.txt"


def test_store_add_or_update_drops_stale_uri_key(tmp_path):
    """URI 变更后旧键不得残留在 _by_file_uri（防悬空键积累）。"""

    from edge_cloud_agent.personal_search.storage import PersonalFileItem

    store = PersonalFileStore(str(tmp_path / "s.jsonl"))
    base = dict(
        file_id="fm_x",
        title="t",
        source_app="unknown",
        doc_type="file",
        mime_type="",
        raw_text="r",
        summary="s",
        file_path="C:/x/a.png",
    )
    store.add_or_update(PersonalFileItem(file_uri="file://C:/x/a.png", **base), persist=False)
    store.add_or_update(PersonalFileItem(file_uri="file:///C:/x/a.png", **base), persist=False)

    assert store.get_by_file_uri("file://C:/x/a.png") is None
    assert store.get_by_file_uri("file:///C:/x/a.png") is not None


def test_is_cloud_placeholder_detection():
    """云占位符判定为纯函数：duck-typing stat 结果。"""

    class _St:
        def __init__(self, attrs):
            self.st_file_attributes = attrs

    assert _is_cloud_placeholder(_St(0x400000)) is True   # RECALL_ON_DATA_ACCESS
    assert _is_cloud_placeholder(_St(0x40000)) is True    # RECALL_ON_OPEN
    assert _is_cloud_placeholder(_St(0x20)) is False      # 普通归档位
    assert _is_cloud_placeholder(None) is False

    class _PlainSt:  # 非 Windows：无 st_file_attributes 属性
        pass

    assert _is_cloud_placeholder(_PlainSt()) is False


def test_cloud_placeholder_skipped_on_windows(tmp_path, monkeypatch):
    """Windows 下 OneDrive「仅在线」占位符不进扫描面（防静默全量下载）。"""

    root = tmp_path / "onedrive"
    root.mkdir()
    real = root / "real.txt"
    cloud = root / "online.txt"
    real.write_text("local content", encoding="utf-8")
    cloud.write_text("placeholder", encoding="utf-8")

    # 不能 patch 全局 os.name（pathlib 会随之实例化 WindowsPath 崩溃），
    # 走 ingest 模块的 _is_windows 间接层
    monkeypatch.setattr(ingest_mod, "_is_windows", lambda: True)

    class _FakeStat:
        st_file_attributes = ingest_mod._FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS

    orig_safe_stat = ingest_mod._safe_stat
    monkeypatch.setattr(
        ingest_mod,
        "_safe_stat",
        lambda p: _FakeStat() if p.name == "online.txt" else orig_safe_stat(p),
    )

    cfg, _ = _make_service(tmp_path, str(root), name="cloud")
    assert ingest_mod._discover_files(cfg) == [real]

    # 关闭开关后占位符重新进扫描面
    cfg_off = PersonalFileConfig(
        store_path=str(tmp_path / "cloud_off.jsonl"),
        state_path=str(tmp_path / "cloud_off_state.json"),
        source_dir=str(root),
        skip_cloud_placeholders=False,
        enable_faiss=False,
    )
    assert sorted(p.name for p in ingest_mod._discover_files(cfg_off)) == ["online.txt", "real.txt"]


def test_windows_junk_dirs_pruned(tmp_path):
    """Windows 系统/回收站目录默认剪枝（$RECYCLE.BIN、System Volume Information）。"""

    root = tmp_path / "win_root"
    (root / "$RECYCLE.BIN").mkdir(parents=True)
    (root / "System Volume Information").mkdir()
    (root / "$RECYCLE.BIN" / "deleted.txt").write_text("junk", encoding="utf-8")
    (root / "System Volume Information" / "sys.txt").write_text("junk", encoding="utf-8")
    (root / "real.txt").write_text("real file", encoding="utf-8")

    cfg, service = _make_service(tmp_path, str(root), name="winjunk")
    result = run_once(service, cfg)
    assert result.scanned == 1
    assert service.store.list_all()[0].title == "real"
