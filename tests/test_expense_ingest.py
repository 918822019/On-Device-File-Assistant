"""expense ingest：URI 格式兼容的幽灵清理（sweep）回归。

最高危场景：Windows 原生下 file_uri 从历史 file://C:/... 变为 file:///C:/...，
旧实现 removeprefix("file://") 会得到 "/C:/..."，exists() 恒 False → 全量误删。
"""

import pytest

from edge_cloud_agent.config import ExpenseConfig
from edge_cloud_agent.expense import ingest as expense_ingest
from edge_cloud_agent.expense.service import ExpenseService
from edge_cloud_agent.expense.storage import ExpenseMaterial, ExpenseStore


@pytest.fixture()
def env(tmp_path):
    watch_dir = tmp_path / "watch"
    watch_dir.mkdir()
    cfg = ExpenseConfig(
        store_path=str(tmp_path / "expense.jsonl"),
        watch_dir=str(watch_dir),
        enable_embedding_search=False,
    )
    service = ExpenseService(config=cfg, store=ExpenseStore(cfg.store_path), embedding_runtime=None)
    return cfg, service, watch_dir


def _material(file_uri: str, material_id: str = "m1") -> ExpenseMaterial:
    return ExpenseMaterial(
        material_id=material_id,
        claim_id="c1",
        title="t",
        doc_type="receipt",
        source_app="auto-watch",
        raw_text="金额 128 元",
        summary="s",
        file_uri=file_uri,
    )


def test_sweep_keeps_live_file_with_new_uri_format(env, tmp_path):
    """新格式 file:///abs 且文件仍在磁盘 → 不得清理（旧 removeprefix 实现会误删）。"""

    cfg, service, watch_dir = env
    f = watch_dir / "invoice.txt"
    f.write_text("金额 128 元", encoding="utf-8")
    uri = expense_ingest._as_file_uri(f)
    assert uri.startswith("file:///")
    service.store.add_or_update(_material(uri))

    removed = expense_ingest._sweep_deleted_materials(service, cfg)
    assert removed == 0
    assert len(service.store.list_all()) == 1


def test_sweep_keeps_live_file_with_legacy_uri_format(env):
    """历史格式 file://abs（POSIX 存量 JSONL）仍须识别为存活。"""

    cfg, service, watch_dir = env
    f = watch_dir / "legacy.txt"
    f.write_text("金额 66 元", encoding="utf-8")
    legacy_uri = f"file://{f.as_posix()}"
    service.store.add_or_update(_material(legacy_uri))

    removed = expense_ingest._sweep_deleted_materials(service, cfg)
    assert removed == 0


def test_sweep_removes_gone_file(env):
    cfg, service, watch_dir = env
    f = watch_dir / "gone.txt"
    f.write_text("金额 1 元", encoding="utf-8")
    uri = expense_ingest._as_file_uri(f)
    service.store.add_or_update(_material(uri))
    f.unlink()

    removed = expense_ingest._sweep_deleted_materials(service, cfg)
    assert removed == 1
    assert service.store.list_all() == []


def test_sweep_removes_windows_shaped_uri_when_file_gone(env):
    """Windows 形状 URI（盘符）文件不存在时同样清理；本用例验证 URI 解析分支。

    在 POSIX 机器上 C:/... 不存在 → exists() False → 应清理（解析出的路径
    形状正确性由 test_path_utils 的 uri_to_path 用例钉死）。
    """

    cfg, service, _ = env
    service.store.add_or_update(_material("file:///C:/Users/x/gone.txt"))
    removed = expense_ingest._sweep_deleted_materials(service, cfg)
    assert removed == 1


def test_sweep_skips_non_file_uri(env):
    """file_uri 为空/异常形态时跳过该条，不误删。"""

    cfg, service, _ = env
    service.store.add_or_update(_material("", material_id="m_empty"))
    removed = expense_ingest._sweep_deleted_materials(service, cfg)
    assert removed == 0
    assert len(service.store.list_all()) == 1


def test_sweep_skipped_when_watch_dir_missing(env, tmp_path):
    """watch_dir 不可达时完全不清理（原有防护语义不变）。"""

    cfg, service, watch_dir = env
    f = watch_dir / "live.txt"
    f.write_text("金额 2 元", encoding="utf-8")
    service.store.add_or_update(_material(expense_ingest._as_file_uri(f)))
    watch_dir.rename(tmp_path / "watch_gone")

    removed = expense_ingest._sweep_deleted_materials(service, cfg)
    assert removed == 0
    assert len(service.store.list_all()) == 1


def test_read_text_file_gb18030(tmp_path):
    """expense 侧文本读取同样走编码降级：GBK 文件不再被跳过。"""

    f = tmp_path / "gbk.txt"
    f.write_bytes("报销单 金额128元".encode("gb18030"))
    assert expense_ingest._read_text_file(f) == "报销单 金额128元"


def test_read_text_file_pdf_returns_none(tmp_path):
    f = tmp_path / "invoice.pdf"
    f.write_bytes(b"%PDF-1.4 fake")
    assert expense_ingest._read_text_file(f) is None
