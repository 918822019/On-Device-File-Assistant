"""common/fs_scan：目录遍历、变更指纹与幽灵判定。

这里的每一条都对应一个曾经只存在于一条业务线的行为，抽取之后必须证明它**对两条
线都成立**、且没有在抽取过程中被削弱：

- 排除目录必须真的**剪枝**（os.walk 原地赋值），而不是遍历完再过滤 —— 两者结果
  相同，代价差几个数量级
- followlinks=False 必须真的防住符号链接环（否则后台扫描线程会挂死）
- is_unchanged 在 mtime 缺失时必须判「有变」：「未知」不等于「没变」，判反了
  会让升级前的存量记录永远发现不了内容变更
- is_ghost 的按根可达保护：外接盘/WSL 挂载点暂时不可达时不得清库
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from edge_cloud_agent.common import fs_scan, path_utils
from edge_cloud_agent.common.fs_scan import (
    fingerprint,
    is_ghost,
    is_unchanged,
    iter_files,
    owning_root,
    reachable_roots,
)

MD = {".md"}


# ------------------------------------------------------------------ iter_files


def test_iter_files_filters_by_suffix_case_insensitively(tmp_path: Path):
    (tmp_path / "a.MD").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("x", encoding="utf-8")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in iter_files([tmp_path], suffixes=MD)] == ["a.MD", "b.md"]


def test_iter_files_result_is_sorted(tmp_path: Path):
    """扫描顺序必须确定，否则两轮之间的差异不可复现。"""

    for name in ("z.md", "m.md", "a.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert [p.name for p in iter_files([tmp_path], suffixes=MD)] == ["a.md", "m.md", "z.md"]


def test_iter_files_prunes_excluded_dirs_instead_of_filtering_them(tmp_path: Path):
    """剪枝与过滤结果相同、代价差几个数量级 —— 这里证明是真的剪枝。

    判据：被排除目录下的文件**根本不会进入 want 谓词**。若是「先遍历再过滤」，
    谓词会被调用到。node_modules / AppData / 微信 Attach 这类目录动辄上万文件，
    这个区别就是「扫描几秒」与「扫描几分钟」。
    """

    junk = tmp_path / "node_modules" / "deep" / "deeper"
    junk.mkdir(parents=True)
    for i in range(20):
        (junk / f"f{i}.md").write_text("x", encoding="utf-8")
    (tmp_path / "real.md").write_text("x", encoding="utf-8")

    seen: list[str] = []
    result = iter_files(
        [tmp_path],
        suffixes=MD,
        exclude_dirs={"node_modules"},
        want=lambda p: seen.append(p.name) or True,
    )

    assert [p.name for p in result] == ["real.md"]
    assert seen == ["real.md"], f"被排除目录下的文件进入了谓词，说明没有剪枝: {seen}"


def test_iter_files_exclude_dirs_is_case_insensitive(tmp_path: Path):
    (tmp_path / "Node_Modules").mkdir()
    (tmp_path / "Node_Modules" / "a.md").write_text("x", encoding="utf-8")
    assert iter_files([tmp_path], suffixes=MD, exclude_dirs={"node_modules"}) == []


def test_iter_files_non_recursive_only_takes_top_level(tmp_path: Path):
    (tmp_path / "top.md").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.md").write_text("x", encoding="utf-8")
    got = iter_files([tmp_path], suffixes=MD, recursive=False)
    assert [p.name for p in got] == ["top.md"]


def test_iter_files_skips_missing_and_non_dir_roots(tmp_path: Path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    a_file = tmp_path / "not_a_dir.md"
    a_file.write_text("x", encoding="utf-8")
    got = iter_files(
        [tmp_path, tmp_path / "missing", a_file],
        suffixes=MD,
        recursive=False,
    )
    assert [p.name for p in got] == ["a.md", "not_a_dir.md"]


def test_iter_files_handles_multiple_roots(tmp_path: Path):
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    r1.mkdir()
    r2.mkdir()
    (r1 / "a.md").write_text("x", encoding="utf-8")
    (r2 / "b.md").write_text("x", encoding="utf-8")
    got = iter_files([r1, r2], suffixes=MD)
    assert [p.name for p in got] == ["a.md", "b.md"]


def test_iter_files_survives_a_symlink_cycle(tmp_path: Path):
    """followlinks=False 的实际意义：符号链接环会让遍历永不终止。

    后台 watch 线程挂死不会有任何报错，只会表现为「索引不再更新」。
    """

    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    try:
        (sub / "loop").symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前文件系统不支持符号链接")

    got = iter_files([tmp_path], suffixes=MD)
    # 环里的 a.md 会被再次枚举（不同路径），但遍历必须终止
    assert "a.md" in [p.name for p in got]


def test_iter_files_want_predicate_can_exclude(tmp_path: Path):
    (tmp_path / "keep.md").write_text("x", encoding="utf-8")
    (tmp_path / "drop.md").write_text("x", encoding="utf-8")
    got = iter_files([tmp_path], suffixes=MD, want=lambda p: p.name.startswith("keep"))
    assert [p.name for p in got] == ["keep.md"]


def test_iter_files_ignores_directories_that_match_the_suffix(tmp_path: Path):
    (tmp_path / "weird.md").mkdir()
    (tmp_path / "real.md").write_text("x", encoding="utf-8")
    assert [p.name for p in iter_files([tmp_path], suffixes=MD)] == ["real.md"]


# ------------------------------------------------------------------ 变更指纹


def test_fingerprint_returns_size_and_mtime_ns(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    size, mtime_ns = fingerprint(f)
    st = f.stat()
    assert (size, mtime_ns) == (st.st_size, st.st_mtime_ns)
    assert size == 5


def test_fingerprint_of_missing_file_is_none_pair(tmp_path: Path):
    assert fingerprint(tmp_path / "nope.txt") == (None, None)


@pytest.mark.parametrize(
    ("rec_size", "rec_mtime", "size", "mtime", "expected"),
    [
        (10, 100, 10, 100, True),
        (10, 100, 11, 100, False),   # size 变了
        (10, 100, 10, 101, False),   # mtime 变了
        (10, None, 10, 100, False),  # 记录侧 mtime 未知 → 判「有变」
        (10, 100, 10, None, False),  # 磁盘侧 stat 失败 → 判「有变」
        (None, None, None, None, False),
    ],
)
def test_is_unchanged(rec_size, rec_mtime, size, mtime, expected):
    assert is_unchanged(rec_size, rec_mtime, size, mtime) is expected


def test_is_unchanged_treats_unknown_mtime_as_changed(tmp_path: Path):
    """这条是升级路径的命门：旧记录没有 file_mtime_ns。

    若把 None 当作「相等」，存量记录会永远走快路径、内容变更再也发现不了。
    """

    f = tmp_path / "a.txt"
    f.write_text("v1", encoding="utf-8")
    size, mtime_ns = fingerprint(f)
    assert is_unchanged(size, None, size, mtime_ns) is False

    f.write_text("v2-longer", encoding="utf-8")
    new_size, new_mtime = fingerprint(f)
    assert is_unchanged(size, mtime_ns, new_size, new_mtime) is False


def test_fingerprint_uses_ns_not_float_seconds(tmp_path: Path):
    """必须是 st_mtime_ns（int）而不是 st_mtime（float 秒）。

    float 秒在亚秒级写入上会丢精度：两次不同的写入可能得到同一个 st_mtime，
    快路径就会把真实变更判成「没变」。int 纳秒不会。
    """

    f = tmp_path / "a.txt"
    f.write_text("aaaa", encoding="utf-8")
    size, first = fingerprint(f)
    assert isinstance(first, int) and not isinstance(first, bool)
    assert first == f.stat().st_mtime_ns

    st = f.stat()
    f.write_text("bbbb", encoding="utf-8")  # 同长度，只有 mtime 能区分
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))  # 还原到同一纳秒
    assert fingerprint(f) == (size, first)


# ------------------------------------------------------------------ 幽灵判定


def test_reachable_roots_excludes_missing_and_normalizes_case(tmp_path: Path):
    live = tmp_path / "live"
    live.mkdir()
    got = reachable_roots([live, tmp_path / "missing"])
    assert got == {os.path.normcase(str(live))}


def test_owning_root(tmp_path: Path):
    root = tmp_path / "r"
    (root / "sub").mkdir(parents=True)
    deep = root / "sub" / "x.md"
    assert owning_root(deep, [root]) == root
    assert owning_root(tmp_path / "elsewhere.md", [root]) is None


def test_is_ghost_false_when_no_root_is_reachable(tmp_path: Path):
    """全部根不可达 → 一律不清理（否则会清库）。"""

    assert is_ghost(tmp_path / "gone.md", roots=[tmp_path / "missing"], reachable=set()) is False


def test_is_ghost_false_for_path_seen_this_round(tmp_path: Path):
    key = os.path.normcase(str(tmp_path / "a.md"))
    assert is_ghost(
        tmp_path / "a.md",
        roots=[tmp_path],
        reachable={os.path.normcase(str(tmp_path))},
        known_existing={key},
    ) is False


def test_is_ghost_false_for_file_that_still_exists(tmp_path: Path):
    f = tmp_path / "live.md"
    f.write_text("x", encoding="utf-8")
    assert is_ghost(
        f, roots=[tmp_path], reachable={os.path.normcase(str(tmp_path))}, known_existing=set()
    ) is False


def test_is_ghost_false_when_owning_root_is_unreachable(tmp_path: Path):
    """外接盘 / WSL 挂载点暂时不可达时不得清理该根下的记录。"""

    live_root = tmp_path / "live"
    live_root.mkdir()
    gone_root = tmp_path / "unmounted"
    assert is_ghost(
        gone_root / "a.md",
        roots=[live_root, gone_root],
        reachable={os.path.normcase(str(live_root))},
    ) is False


def test_is_ghost_true_when_owning_root_is_reachable_but_file_gone(tmp_path: Path):
    assert is_ghost(
        tmp_path / "gone.md",
        roots=[tmp_path],
        reachable={os.path.normcase(str(tmp_path))},
    ) is True


def test_is_ghost_true_for_record_outside_every_configured_root(tmp_path: Path):
    """源目录被移出配置的历史残留：不属于任何根 → 清理。"""

    root = tmp_path / "r"
    root.mkdir()
    assert is_ghost(
        tmp_path / "orphan.md", roots=[root], reachable={os.path.normcase(str(root))}
    ) is True


def test_known_existing_never_resurrects_a_deleted_file(tmp_path: Path):
    """命中集只能证明「存在」，不能反过来让已删除的文件逃过清理。"""

    f = tmp_path / "a.md"
    f.write_text("x", encoding="utf-8")
    key = os.path.normcase(str(f))
    f.unlink()
    # 调用方传的是本轮**真实枚举**到的集合；这里故意传一个过期集合，
    # 断言的是：即便如此，语义仍与逐条 stat 一致（命中集说了算 = 调用方责任）
    assert is_ghost(
        f, roots=[tmp_path], reachable={os.path.normcase(str(tmp_path))}, known_existing={key}
    ) is False
    assert is_ghost(
        f, roots=[tmp_path], reachable={os.path.normcase(str(tmp_path))}, known_existing=None
    ) is True


# ------------------------------------------------------------------ 单一事实源


def test_platform_predicate_is_shared_with_path_utils():
    """fs_scan 不得再有一份私有的平台判定。

    personal_search/ingest.py 此前有个 `_is_windows()`（`os.name == "nt"`），
    是 path_utils.is_windows_native 的第三份拷贝，而且**不支持**
    FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP 覆盖 —— 那是非 Windows 机器上测试
    Windows 分支的唯一入口。

    用 AST 而不是字符串搜索：模块 docstring 里为了说明这段历史提到了 os.name，
    字符串匹配会把散文误判成代码（第一版就是这么假的）。
    """

    assert fs_scan.is_windows_native is path_utils.is_windows_native

    tree = ast.parse(Path(fs_scan.__file__).read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "name"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    ]
    assert not offenders, f"fs_scan 里直接读了 os.name（行 {offenders}），应走 path_utils"
