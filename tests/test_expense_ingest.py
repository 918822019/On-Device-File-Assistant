"""expense ingest：URI 格式兼容的幽灵清理（sweep）回归。

最高危场景：Windows 原生下 file_uri 从历史 file://C:/... 变为 file:///C:/...，
旧实现 removeprefix("file://") 会得到 "/C:/..."，exists() 恒 False → 全量误删。
"""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from edge_cloud_agent.common.path_utils import as_file_uri, uri_to_path
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
    uri = as_file_uri(f)
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
    uri = as_file_uri(f)
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
    service.store.add_or_update(_material(as_file_uri(f)))
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


# ---------------------------------------------------------------------------
# 快路径 / 串行锁 / 幽灵清理复用扫描结果
#
# 这四项此前只存在于 personal_search 侧：expense 的 run_once 是它的一份旧快照，
# 每个文件每轮都读头部 1 MB 算 hash、幽灵清理逐条 stat、run_once 无锁、
# 目录遍历用无法剪枝的 rglob。抽取 common/fs_scan 之后两侧共用同一份实现，
# 下面的用例钉住 expense 侧确实拿到了这些行为。
# ---------------------------------------------------------------------------



@pytest.fixture()
def hash_calls(monkeypatch):
    """统计 content_hash 的真实调用次数 —— 快路径是否生效的唯一可观测指标。"""

    calls: list[str] = []
    real = expense_ingest.content_hash

    def _counting(path, *args, **kwargs):
        calls.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(expense_ingest, "content_hash", _counting)
    return calls


def _write(watch_dir, name: str, text: str = "金额 128 元"):
    f = watch_dir / name
    f.write_text(text, encoding="utf-8")
    return f


def test_first_scan_hashes_then_steady_state_hashes_nothing(env, hash_calls):
    cfg, service, watch_dir = env
    for i in range(5):
        _write(watch_dir, f"m{i}.txt", f"金额 {i} 元")

    first = expense_ingest.run_once(service, cfg)
    assert (first.scanned, first.imported) == (5, 5)
    assert len(hash_calls) == 5

    hash_calls.clear()
    second = expense_ingest.run_once(service, cfg)
    assert (second.scanned, second.imported, second.skipped) == (5, 0, 5)
    assert hash_calls == [], "稳态仍在算 hash —— size+mtime 快路径没生效"


def test_content_change_is_still_detected(env, hash_calls):
    cfg, service, watch_dir = env
    f = _write(watch_dir, "m.txt", "金额 1 元")
    expense_ingest.run_once(service, cfg)

    hash_calls.clear()
    f.write_text("金额 999 元 商户 某某公司", encoding="utf-8")
    result = expense_ingest.run_once(service, cfg)

    assert len(hash_calls) == 1, "变更文件必须重算 hash"
    assert result.imported == 1
    assert len(service.store.list_all()) == 1, "旧记录应先删后加，不留重复"
    assert "999" in service.store.list_all()[0].raw_text


def test_imported_record_carries_file_mtime_ns(env):
    """没有 file_mtime_ns，下一轮就没有快路径可走。"""

    cfg, service, watch_dir = env
    f = _write(watch_dir, "m.txt")
    expense_ingest.run_once(service, cfg)
    material = service.store.list_all()[0]
    assert material.file_mtime_ns == f.stat().st_mtime_ns


def test_force_rehash_bypasses_the_fast_path(env, hash_calls):
    """手工「重建索引」必须做权威 hash 校验（云同步会还原 mtime）。"""

    cfg, service, watch_dir = env
    for i in range(3):
        _write(watch_dir, f"m{i}.txt", f"金额 {i} 元")
    expense_ingest.run_once(service, cfg)

    hash_calls.clear()
    result = expense_ingest.run_once(service, cfg, force_rehash=True)
    assert len(hash_calls) == 3
    assert (result.imported, result.skipped) == (0, 3)


def test_mtime_restored_change_needs_force_rehash(env, hash_calls):
    """钉住已知取舍：size 与 mtime 都不变的变更只有 force_rehash 能发现。"""

    cfg, service, watch_dir = env
    f = _write(watch_dir, "m.txt", "AAAA")
    expense_ingest.run_once(service, cfg)
    st = f.stat()

    f.write_text("BBBB", encoding="utf-8")  # 同长度
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))  # 还原 mtime

    hash_calls.clear()
    assert expense_ingest.run_once(service, cfg).imported == 0
    assert hash_calls == []
    assert expense_ingest.run_once(service, cfg, force_rehash=True).imported == 1


def test_legacy_record_without_mtime_is_backfilled_then_uses_fast_path(env, hash_calls):
    """升级路径：旧记录没有 file_mtime_ns，重算一次 hash 后就地回填，不重读内容。"""

    cfg, service, watch_dir = env
    f = _write(watch_dir, "m.txt", "金额 7 元")
    expense_ingest.run_once(service, cfg)

    legacy = replace(service.store.list_all()[0], file_mtime_ns=None)
    service.store.add_or_update(legacy)

    hash_calls.clear()
    result = expense_ingest.run_once(service, cfg)
    assert len(hash_calls) == 1
    assert result.imported == 0, "内容没变，不得重新导入（那会重算 embedding）"

    material = service.store.list_all()[0]
    assert material.file_mtime_ns == f.stat().st_mtime_ns
    assert material.raw_text == "金额 7 元", "回填只动元数据，不得改内容"

    hash_calls.clear()
    assert expense_ingest.run_once(service, cfg).imported == 0
    assert hash_calls == [], "回填后下一轮就该走快路径"


def test_sweep_reuses_the_scan_result_instead_of_stat_per_record(env, monkeypatch):
    """已入库记录若本轮扫到过，就不必再 stat 一次。"""

    cfg, service, watch_dir = env
    for i in range(6):
        _write(watch_dir, f"m{i}.txt", f"金额 {i} 元")
    expense_ingest.run_once(service, cfg)

    exists_calls: list[str] = []
    real_exists = Path.exists

    def counting_exists(self):
        if str(self).startswith(str(watch_dir)):
            exists_calls.append(str(self))
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", counting_exists)
    result = expense_ingest.run_once(service, cfg)
    monkeypatch.setattr(Path, "exists", real_exists)

    assert result.removed == 0
    assert exists_calls == [], f"本轮扫到的文件又被逐条 exists(): {exists_calls}"


def test_sweep_keeps_existing_file_that_left_the_scan_scope(env):
    """复用扫描结果不得误删「文件仍在磁盘、只是本轮没被枚举到」的记录。

    known_existing 只覆盖本轮枚举到的路径，所以未命中时**必须回落 exists()**。
    没有这个回落，把后缀从白名单里去掉（或文件被移进排除目录）就会让一个
    好端端的文件从索引里消失。

    注意与「改名」区分开：把 m.txt 改名成 m.bin 之后，记录里的旧 URI 指向的
    路径**确实不存在了**，清理它是正确行为，不是这个用例要防的。
    """

    cfg, service, watch_dir = env
    f = _write(watch_dir, "m.txt", "金额 5 元")
    expense_ingest.run_once(service, cfg)
    assert len(service.store.list_all()) == 1
    assert f.exists()

    # 换一份后缀白名单：m.txt 仍在磁盘上，但本轮不会被枚举到
    narrowed = replace(cfg, watch_file_suffixes=".md")
    result = expense_ingest.run_once(service, narrowed)

    assert result.scanned == 0, "白名单里没有 .txt，本轮不该扫到任何文件"
    assert result.removed == 0, "文件仍在磁盘上，不得因为没被扫到就清掉"
    assert len(service.store.list_all()) == 1


def test_sweep_removes_record_whose_path_really_is_gone(env):
    """对照组：改名之后旧 URI 指向的路径确实不存在 → 应当清理。"""

    cfg, service, watch_dir = env
    f = _write(watch_dir, "m.txt", "金额 5 元")
    expense_ingest.run_once(service, cfg)

    f.rename(watch_dir / "m.bin")
    result = expense_ingest.run_once(service, cfg)
    assert result.removed == 1
    assert service.store.list_all() == []


def test_one_bad_file_does_not_abort_the_whole_pass(env, monkeypatch):
    """单个文件出错只计 errors，不掀掉整轮。

    此前只有 service.collect() 被 try 包住，stat / hash 阶段的异常会直接冒到
    watch 循环里被 pass 掉 —— 一整轮扫描的结果全丢。
    """

    cfg, service, watch_dir = env
    good = _write(watch_dir, "good.txt", "金额 1 元")
    bad = _write(watch_dir, "bad.txt", "金额 2 元")

    real_hash = expense_ingest.content_hash

    def flaky(path, *a, **kw):
        if path == bad:
            raise OSError("permission denied")
        return real_hash(path, *a, **kw)

    monkeypatch.setattr(expense_ingest, "content_hash", flaky)
    result = expense_ingest.run_once(service, cfg)

    assert (result.scanned, result.imported, result.errors) == (2, 1, 1)
    indexed = {Path(uri_to_path(m.file_uri)).name for m in service.store.list_all()}
    assert indexed == {good.name}


def test_run_once_holds_a_module_level_scan_lock():
    """watch 线程与 /rebuild-index 是两个入口，必须串行。

    源码级断言：锁对象在 import 时创建，用 helper 绕过 __init__ 的功能测试
    抓不到「那行被删了」。
    """

    import inspect

    src = inspect.getsource(expense_ingest)
    assert "_SCAN_LOCK = RLock()" in src
    assert "with _SCAN_LOCK:" in src
    assert inspect.getsource(expense_ingest.run_once).count("with _SCAN_LOCK:") == 1


def test_watch_loop_delegates_to_the_shared_skeleton():
    """不得再出现 `except Exception: pass` 那种把守护线程失败完全吞掉的写法。

    用 AST 而不是字符串搜索：函数 docstring 里为了说明这段历史引用了
    ``except Exception: pass`` 的字面量，字符串匹配会把散文误判成代码。
    """

    import ast
    import inspect

    for mod in (expense_ingest, _personal_ingest()):
        tree = ast.parse(inspect.getsource(mod.start_watch_loop))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)], (
            f"{mod.__name__}.start_watch_loop 里还有 try/except —— "
            "循环骨架应当完全委托给 common.watch_loop"
        )
        called = [
            n.func.id for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        assert "run_watch_loop" in called


def _personal_ingest():
    from edge_cloud_agent.personal_search import ingest as personal_ingest

    return personal_ingest


def test_file_mtime_ns_survives_a_jsonl_roundtrip(env, hash_calls):
    """快路径必须跨进程成立 —— 也就是必须落盘。

    只在内存里断言 file_mtime_ns 是抓不到 from_dict 漏读这个字段的：那种情况下
    同一个进程内的第二轮扫描照样走快路径（内存里的对象是完整的），只有重启后
    的第一轮会退回慢路径。而「重启后每轮都重算全部 hash」正是这次要消灭的
    稳态 IO，它不会报错，只会让 watch 目录一直忙。
    """

    cfg, service, watch_dir = env
    for i in range(3):
        _write(watch_dir, f"m{i}.txt", f"金额 {i} 元")
    expense_ingest.run_once(service, cfg)
    assert len(hash_calls) == 3

    # 模拟重启：从同一个路径重新载入 store
    fresh_store = ExpenseStore(cfg.store_path)
    assert all(m.file_mtime_ns is not None for m in fresh_store.list_all()), (
        "file_mtime_ns 没有落盘或 from_dict 漏读"
    )
    fresh_service = ExpenseService(
        config=cfg, store=fresh_store, embedding_runtime=None
    )

    hash_calls.clear()
    result = expense_ingest.run_once(fresh_service, cfg)
    assert (result.scanned, result.imported, result.skipped) == (3, 0, 3)
    assert hash_calls == [], "重启后第一轮就退回慢路径 —— file_mtime_ns 没能跨进程生效"
