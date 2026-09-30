"""增量扫描的 size+mtime 快路径与幽灵清理的路径复用。

这两处都是纯性能优化，因此测试的重点不是「功能还能跑」，而是**语义没有变**：

- 快路径省掉的是 hash 计算，不能因此漏掉真实变更；
- 已知其代价：size 与 mtime 都不变的变更（云同步/备份恢复还原 mtime）只有
  force_rehash 能发现 —— 这个取舍必须被测试钉住，而不是留给读者猜；
- 幽灵清理复用扫描结果省掉的是 stat，不能因此误删「仍存在但已不在扫描范围」
  的文件。
"""

import os
import time
from dataclasses import replace

import pytest

from edge_cloud_agent.config import PersonalFileConfig
from edge_cloud_agent.personal_search import ingest as ingest_mod
from edge_cloud_agent.personal_search.ingest import run_once
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
    service = PersonalFileSearchService(
        config=cfg, store=PersonalFileStore(cfg.store_path), embedding_runtime=None
    )
    return cfg, service, src_dir


@pytest.fixture()
def hash_calls(monkeypatch):
    """统计 content_hash 的真实调用次数 —— 快路径是否生效的唯一可观测指标。"""

    calls: list[str] = []
    real = ingest_mod.content_hash

    def _counting(path, *args, **kwargs):
        calls.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ingest_mod, "content_hash", _counting)
    return calls


# ------------------------------------------------------------------ 快路径


def test_first_scan_imports_and_hashes(env, hash_calls):
    cfg, service, src_dir = env
    (src_dir / "a.txt").write_text("会议纪要", encoding="utf-8")

    result = run_once(service, cfg)

    assert (result.scanned, result.imported) == (1, 1)
    assert len(hash_calls) == 1, "首轮导入必须算一次 hash"


def test_unchanged_file_skips_hashing_entirely(env, hash_calls):
    """核心优化：未变更文件只做一次 stat，不再读头部 1MB 算 md5。"""

    cfg, service, src_dir = env
    (src_dir / "a.txt").write_text("会议纪要", encoding="utf-8")
    run_once(service, cfg)
    hash_calls.clear()

    result = run_once(service, cfg)

    assert (result.scanned, result.imported, result.skipped) == (1, 0, 1)
    assert hash_calls == [], f"稳态下不该再算 hash，实际算了 {len(hash_calls)} 次"


def test_force_rehash_hashes_every_file(env, hash_calls):
    """手工重建走全量权威校验，必须绕过快路径。"""

    cfg, service, src_dir = env
    (src_dir / "a.txt").write_text("会议纪要", encoding="utf-8")
    (src_dir / "b.txt").write_text("聚餐照片", encoding="utf-8")
    run_once(service, cfg)
    hash_calls.clear()

    run_once(service, cfg)
    assert hash_calls == [], "非 force 时不该算 hash"

    run_once(service, cfg, force_rehash=True)
    assert len(hash_calls) == 2, "force_rehash 必须对每个文件都算 hash"


def test_content_change_detected_via_mtime(env, hash_calls):
    """内容变了 mtime 也会变 —— 快路径不能漏掉这种最常见的变更。"""

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("原始内容", encoding="utf-8")
    run_once(service, cfg)
    hash_calls.clear()

    time.sleep(0.01)
    f.write_text("原始内容改过了", encoding="utf-8")
    result = run_once(service, cfg)

    assert result.imported == 1, "内容变更必须被重新导入"
    assert len(hash_calls) == 1, "变更后应走慢路径算 hash"
    assert service.store.list_all()[0].raw_text.startswith("原始内容改过了")


def test_same_size_change_detected_via_mtime(env):
    """等长改写（size 不变）仍须被发现 —— 靠的是 mtime，不是 size。"""

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("AAAA", encoding="utf-8")
    run_once(service, cfg)

    time.sleep(0.01)
    f.write_text("BBBB", encoding="utf-8")  # 长度完全相同
    assert f.stat().st_size == 4

    result = run_once(service, cfg)
    assert result.imported == 1, "等长改写漏检会导致索引内容永久过期"
    assert service.store.list_all()[0].raw_text.startswith("BBBB")


def test_mtime_restored_change_needs_force_rehash(env, hash_calls):
    """钉住已知取舍：size 与 mtime 都不变时，只有 force_rehash 能发现变更。

    这正是云同步/备份恢复的行为（本项目以 OneDrive/iCloud/微信目录为主要目标），
    也是 UI 上「重建索引」按钮必须传 force_rehash=True 的原因。
    """

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("AAAA", encoding="utf-8")
    run_once(service, cfg)
    hash_calls.clear()  # 首轮导入必然算一次 hash，清掉才能只看后续轮次

    stamp = f.stat()
    f.write_text("BBBB", encoding="utf-8")
    # 必须用 ns= 精确还原：os.utime 传浮点秒会丢纳秒精度，还原后的 st_mtime_ns
    # 与原值不同，快路径就不会命中，测出来的就不是「mtime 被还原」这个场景了。
    os.utime(f, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert f.stat().st_mtime_ns == stamp.st_mtime_ns, "mtime 未精确还原，用例前提不成立"

    result = run_once(service, cfg)
    assert result.imported == 0, "快路径按设计不会发现 mtime 被还原的变更"
    assert hash_calls == []

    forced = run_once(service, cfg, force_rehash=True)
    assert forced.imported == 1, "force_rehash 是这种变更的唯一兜底出口"
    assert service.store.list_all()[0].raw_text.startswith("BBBB")


# ------------------------------------------------------- 旧记录 mtime 回填


def test_legacy_record_without_mtime_is_backfilled_then_uses_fast_path(env, hash_calls):
    """升级前的记录没有 file_mtime_ns，必须回填一次，否则永久停留在慢路径。"""

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("会议纪要", encoding="utf-8")
    run_once(service, cfg)

    # 模拟升级前的存量记录：抹掉 file_mtime_ns
    item = service.store.list_all()[0]
    service.store.add_or_update(replace(item, file_mtime_ns=None))
    assert service.store.list_all()[0].file_mtime_ns is None
    hash_calls.clear()

    first = run_once(service, cfg)
    assert len(hash_calls) == 1, "mtime 未知时必须重算 hash 才能判断有没有变"
    assert first.imported == 0, "内容没变，不该重新导入（不重读内容、不重算 embedding）"
    backfilled = service.store.list_all()[0]
    assert backfilled.file_mtime_ns == f.stat().st_mtime_ns
    assert backfilled.file_hash == item.file_hash, "回填不应改动 hash"
    assert backfilled.embedding == item.embedding, "回填不应触发重新 embedding"

    hash_calls.clear()
    run_once(service, cfg)
    assert hash_calls == [], "回填之后应回到快路径"


def test_backfilled_record_survives_restart(env, tmp_path):
    """回填必须落盘，否则重启后又是一轮全量重算。"""

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("会议纪要", encoding="utf-8")
    run_once(service, cfg)

    reloaded = PersonalFileStore(cfg.store_path)
    items = reloaded.list_all()
    assert len(items) == 1
    assert items[0].file_mtime_ns == f.stat().st_mtime_ns, "file_mtime_ns 未持久化"


# ------------------------------------------------------------ 幽灵清理复用


def test_sweep_still_removes_deleted_file(env):
    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("会议纪要", encoding="utf-8")
    run_once(service, cfg)

    f.unlink()
    result = run_once(service, cfg)

    assert result.removed == 1
    assert service.store.list_all() == []


def test_sweep_keeps_existing_file_that_left_scan_scope(env):
    """复用扫描结果后仍不能误删：文件还在，只是不再被 _discover_files 返回。

    _discover_files 只返回白名单后缀且不在排除目录里的文件，所以「仍存在但不在
    本轮 seen_paths 里」是正常情况，必须回落到 path.exists() 而不是直接判定为幽灵。
    """

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("会议纪要", encoding="utf-8")
    run_once(service, cfg)
    item = service.store.list_all()[0]

    # 把记录的 file_path 指向一个真实存在、但后缀不在白名单里的文件
    out_of_scope = src_dir / "still_here.xyz"
    out_of_scope.write_text("data", encoding="utf-8")
    service.store.add_or_update(replace(item, file_path=out_of_scope.as_posix()))

    result = run_once(service, cfg)

    kept = [i for i in service.store.list_all() if i.file_path == out_of_scope.as_posix()]
    assert kept, "文件仍存在于磁盘，不该被当成幽灵清理掉"
    assert result.removed == 0


def test_sweep_protects_item_under_unreachable_root(tmp_path):
    """按根可达性保护不能因为复用 seen_paths 而失效。

    需要多根配置：某一根暂时不可达（如 WSL 下 /mnt/c 未挂载）时，归属该根的
    记录必须保留。注意「不属于任何配置根」是另一条语义 —— 那种记录会被清理，
    所以这里必须让 ghost_root 确实出现在 source_dir 里。
    """

    live_root = tmp_path / "live"
    live_root.mkdir()
    (live_root / "a.txt").write_text("会议纪要", encoding="utf-8")
    ghost_root = tmp_path / "mnt_c_not_mounted"  # 刻意不创建，模拟未挂载

    cfg = PersonalFileConfig(
        store_path=str(tmp_path / "store.jsonl"),
        state_path=str(tmp_path / "state.json"),
        source_dir=f"{live_root},{ghost_root}",
        faiss_index_path=str(tmp_path / "idx.index"),
        enable_faiss=False,
    )
    service = PersonalFileSearchService(
        config=cfg, store=PersonalFileStore(cfg.store_path), embedding_runtime=None
    )
    run_once(service, cfg)

    live_item = service.store.list_all()[0]
    service.store.add_or_update(
        replace(
            live_item,
            file_id="fm_ghost01",
            file_uri="file:///mnt/c/gone/doc.txt",
            file_path=str(ghost_root / "doc.txt"),
        )
    )

    run_once(service, cfg)

    assert any(i.file_id == "fm_ghost01" for i in service.store.list_all()), (
        "归属根不可达的记录必须保留，否则 /mnt/c 未挂载时会误删整批索引"
    )
    assert any(i.file_path.endswith("a.txt") for i in service.store.list_all())


def test_stat_is_not_called_for_files_seen_this_round(env, monkeypatch):
    """直接断言优化本身：本轮已枚举到的路径，清理阶段不应再 stat。"""

    cfg, service, src_dir = env
    for name in ("a.txt", "b.txt", "c.txt"):
        (src_dir / name).write_text(f"内容 {name}", encoding="utf-8")
    run_once(service, cfg)

    stat_targets: list[str] = []
    real_exists = ingest_mod.Path.exists

    def _counting_exists(self):
        stat_targets.append(str(self))
        return real_exists(self)

    monkeypatch.setattr(ingest_mod.Path, "exists", _counting_exists)
    run_once(service, cfg)

    assert stat_targets == [], (
        f"所有文件本轮都已被枚举，清理阶段不该再 exists()/stat()，实际调用了 {stat_targets}"
    )


# ------------------------------------------------- 备注/归档孤儿剪枝（数据）


def test_annotation_of_deleted_file_is_pruned(env):
    """幽灵清理后，被删文件的备注必须一起清掉。"""

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("会议纪要", encoding="utf-8")
    run_once(service, cfg)
    item = service.store.list_all()[0]

    assert service.add_annotation(item.file_id, "这条要留证据")
    assert service.archive_file(item.file_id)
    assert service.get_annotation(item.file_id) == "这条要留证据"

    f.unlink()
    result = run_once(service, cfg)

    assert result.removed == 1
    assert service.store.list_all() == []
    assert service.get_annotation(item.file_id) is None, "孤儿备注未被剪枝"
    assert service.is_archived(item.file_id) is False, "孤儿归档标记未被剪枝"


def test_new_file_at_same_path_does_not_inherit_old_annotation(env):
    """file_id 由路径 md5 派生，同路径必然同 id —— 旧备注不能凭空贴到新文件上。"""

    cfg, service, src_dir = env
    f = src_dir / "report.txt"
    f.write_text("2025 年 Q3 报表", encoding="utf-8")
    run_once(service, cfg)
    old_id = service.store.list_all()[0].file_id
    service.add_annotation(old_id, "旧文件的备注：数字有误")

    f.unlink()
    run_once(service, cfg)

    # 同一路径写入完全不同的内容
    f.write_text("聚餐照片备份清单", encoding="utf-8")
    run_once(service, cfg)
    new_item = service.store.list_all()[0]

    assert new_item.file_id == old_id, "同路径应得到同一 file_id（用例前提）"
    assert service.get_annotation(new_item.file_id) is None, (
        "新文件继承了旧备注 —— 用户会看到一条与当前内容无关的历史标记"
    )


def test_prune_keeps_annotations_of_live_files(env):
    cfg, service, src_dir = env
    live = src_dir / "live.txt"
    gone = src_dir / "gone.txt"
    live.write_text("还在的文件", encoding="utf-8")
    gone.write_text("要删的文件", encoding="utf-8")
    run_once(service, cfg)

    by_title = {i.title: i.file_id for i in service.store.list_all()}
    service.add_annotation(by_title["live"], "保留我")
    service.add_annotation(by_title["gone"], "删掉我")

    gone.unlink()
    run_once(service, cfg)

    assert service.get_annotation(by_title["live"]) == "保留我"
    assert service.get_annotation(by_title["gone"]) is None


def test_prune_without_changes_does_not_rewrite_state_file(env):
    """无孤儿时不该重写 JSON —— 否则每轮扫描都产生一次无谓的磁盘写。"""

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("会议纪要", encoding="utf-8")
    run_once(service, cfg)
    service.add_annotation(service.store.list_all()[0].file_id, "备注")

    before = os.stat(cfg.state_path).st_mtime_ns
    time.sleep(0.01)
    pruned = service.prune_file_state()

    assert pruned == 0
    assert os.stat(cfg.state_path).st_mtime_ns == before, "无变化却重写了状态文件"


def test_force_rehash_prunes_pre_existing_orphans(env):
    """升级前积累的存量孤儿，应在手工重建时被确定性地清掉。"""

    cfg, service, src_dir = env
    f = src_dir / "a.txt"
    f.write_text("会议纪要", encoding="utf-8")
    run_once(service, cfg)

    # 直接往状态存储里塞一个索引中不存在的 file_id，模拟历史遗留孤儿
    service._state_store.set_annotation("fm_orphan0001", "早就删掉的文件")

    run_once(service, cfg)  # 常规扫描：没有文件被删，不触发剪枝
    assert service.get_annotation("fm_orphan0001") == "早就删掉的文件"

    run_once(service, cfg, force_rehash=True)
    assert service.get_annotation("fm_orphan0001") is None, "手工重建应清理存量孤儿"
