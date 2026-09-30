"""文件系统扫描与增量变更检测原语（与业务类型无关，只依赖 stdlib）。

两条业务线的摄取循环（``personal_search/ingest.py`` 与 ``expense/ingest.py``）
此前各自实现了一份目录遍历、一份幽灵清理、一份变更判定。差别不在业务，而在于
**修复只落在一边**：

- personal 侧用 ``os.walk`` + 原地剪枝；expense 侧还在用 ``rglob("*")`` ——
  后者无法跳过整棵子树，personal 侧的注释里正是为此改掉它的（WSL 下跨 9P 扫
  ``/mnt/c`` 时 ``node_modules`` / ``AppData`` 级目录会让扫描成本爆炸）
- personal 侧的幽灵清理复用本轮枚举结果、省掉逐条 ``stat``；expense 侧每条一次
- personal 侧有 size + mtime 快路径；expense 侧每个文件每轮都读头部 1 MB 算 hash
- personal 侧的平台判定是私有 ``_is_windows()``（``os.name == "nt"``），
  而 ``common/path_utils`` 早就有 ``is_windows_native()`` —— 后者还支持
  ``FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP`` 覆盖，是非 Windows 机器上测试
  Windows 分支的唯一入口。这里统一用后者。

本模块只提供文件系统原语，**不知道任何业务类型**；「扫到文件之后做什么」仍由
各业务线的 ``run_once`` 决定。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from .path_utils import is_windows_native

# Windows 云占位符文件属性位（OneDrive 等「仅在线」文件）：读取会触发静默
# 全量下载，扫描阶段直接排除出扫描面。非 Windows 平台无 st_file_attributes
# （或恒为 0），判定自然短路为 False。
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000

# 变更检测的指纹：(st_size, st_mtime_ns)。用 st_mtime_ns 而非 st_mtime，
# 因为 float 秒在亚秒级写入上会丢精度，两次不同的写入可能得到同一个 mtime。
Fingerprint = tuple[int | None, int | None]


def safe_stat(path: Path):
    """``path.stat()``，失败返回 None（不可读/已删除/权限不足都不该中断扫描）。"""

    try:
        return path.stat()
    except OSError:
        return None


def is_cloud_placeholder(st) -> bool:
    """stat 结果是否为 Windows 云占位符文件（st 为 None / 非 Windows → False）。"""

    if st is None:
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    return bool(attrs & (FILE_ATTRIBUTE_RECALL_ON_OPEN | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS))


def fingerprint(path: Path) -> Fingerprint:
    """取 ``(size, mtime_ns)``；取不到时两位都是 None。

    两位都返回而不是抛异常，是为了让调用方能区分「stat 失败」（指纹为
    ``(None, None)``，快路径必然不命中 → 回落到权威的 hash 校验）与
    「stat 成功且未变」（跳过）。
    """

    st = safe_stat(path)
    if st is None:
        return None, None
    return st.st_size, st.st_mtime_ns


def is_unchanged(
    recorded_size: int | None,
    recorded_mtime_ns: int | None,
    size: int | None,
    mtime_ns: int | None,
) -> bool:
    """size 与 mtime_ns 都逐位相等才算未变。

    ``recorded_mtime_ns is None`` 时**必须**返回 False：旧记录没有这个字段，
    「未知」不等于「没变」，否则升级后的存量记录会永远走不到 hash 校验、
    内容变更再也发现不了。调用方拿到 False 后会算一次 hash 并就地回填。

    这个显式守卫在逻辑上被后面的相等比较覆盖（``None == <int>`` 恒为 False），
    因此删掉它不会改变任何输入的输出 —— 保留是为了把这条不变量写在代码里，
    而不是留给读者从 ``None`` 的比较语义推出来；万一将来有人把比较改成只看
    size，守卫仍然挡得住。
    """

    if recorded_mtime_ns is None or mtime_ns is None:
        return False
    return recorded_size == size and recorded_mtime_ns == mtime_ns


def iter_files(
    roots: Iterable[Path],
    *,
    suffixes: Sequence[str],
    exclude_dirs: Iterable[str] = (),
    recursive: bool = True,
    skip_cloud_placeholders: bool = True,
    want: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """遍历全部源根，返回满足条件的文件清单（已排序）。

    - ``suffixes`` 需已小写化（含点，如 ``".md"``）；比较时对路径做 ``.lower()``
    - ``exclude_dirs`` 是**目录名**（不含路径）的小写集合，命中即整棵子树剪掉。
      这是 ``os.walk`` 相对 ``rglob("*")`` 的关键优势：后者只能逐个文件过滤，
      已经付掉了遍历整棵子树的代价
    - ``followlinks=False``：防符号链接环（``rglob`` 的跟随行为跨版本不一致）
    - ``skip_cloud_placeholders`` 只在 Windows 上有意义，先按平台门控，
      避免其他平台每个文件多一次 ``stat``
    - ``want`` 是业务侧的追加谓词（返回 False 即排除），用于放不进上述参数
      的规则；不要用它做后缀过滤，那会失去集合查找的短路
    - 排序是为了让扫描顺序确定：同一份语料两轮之间的 diff 才可读，
      出错时也能复现
    """

    suffix_set = {s.lower() for s in suffixes}
    excludes = {d.lower() for d in exclude_dirs}
    # 平台门控只算一次：is_windows_native() 每次调用都读环境变量
    placeholders_matter = skip_cloud_placeholders and is_windows_native()
    extra = want

    def _wanted(path: Path) -> bool:
        if path.suffix.lower() not in suffix_set:
            return False
        if placeholders_matter and is_cloud_placeholder(safe_stat(path)):
            return False
        if extra is not None and not extra(path):
            return False
        return True

    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        if not recursive:
            try:
                entries = list(root.iterdir())
            except OSError:
                continue
            for item in entries:
                try:
                    if item.is_file() and _wanted(item):
                        files.append(item)
                except OSError:
                    continue
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if excludes:
                # 原地赋值才会真正剪枝；重新绑定局部变量对 os.walk 无效
                dirnames[:] = [d for d in dirnames if d.lower() not in excludes]
            for name in filenames:
                path = Path(dirpath) / name
                if _wanted(path):
                    files.append(path)
    files.sort()
    return files


def reachable_roots(roots: Iterable[Path]) -> set[str]:
    """返回当前可达（``is_dir()``）的根，已按 ``normcase`` 归一。

    ``normcase`` 与源根去重的口径一致：Windows 文件系统大小写不敏感，
    ``c:\\x`` 与 ``C:\\x`` 是同一个根；POSIX 上是恒等变换。
    """

    return {os.path.normcase(str(r)) for r in roots if r.is_dir()}


def owning_root(path: Path, roots: Sequence[Path]) -> Path | None:
    """返回 path 所属的源根目录；不属于任何根时返回 None。"""

    for root in roots:
        try:
            path.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def is_ghost(
    path: Path,
    *,
    roots: Sequence[Path],
    reachable: set[str],
    known_existing: set[str] | None = None,
) -> bool:
    """判断一条已入库记录是否应当被清理（文件已从磁盘消失）。

    防误删语义是**按根可达**而非全局可达：

    - 全部根不可达 → 调用方应完全跳过清理（``reachable`` 为空集时本函数一律
      返回 False）
    - 记录归属的根暂不可达（如 WSL 下 ``/mnt/c`` 未就绪、外接盘未挂载）→ 保留
    - 记录归属的根可达但文件已不存在，或记录不属于任何配置根（源目录被移出
      配置的历史残留）→ 清理

    ``known_existing`` 是本轮扫描已枚举到的路径集合（``normcase`` 后的字符串）。
    命中即代表文件确实存在，可跳过 ``stat`` —— 否则每个已入库条目每轮都要一次
    ``stat``，在 WSL 9P 上与全量 hash 一样昂贵。

    未命中时**仍回落到 ``path.exists()``**，这不是冗余：``iter_files`` 只返回
    白名单后缀且不在排除目录里的文件，一个仍然存在、只是被移出扫描范围的文件
    不该因此被误删。有了这个回落，传入 ``known_existing`` 与逐条 ``stat`` 的
    判定结果完全一致，只是快得多。
    """

    if not reachable:
        return False
    if known_existing is not None and os.path.normcase(str(path)) in known_existing:
        return False
    if path.exists():
        return False
    owner = owning_root(path, roots)
    return owner is None or os.path.normcase(str(owner)) in reachable
