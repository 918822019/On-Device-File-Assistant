"""跨平台文件索引源目录探测（scripts/sources.py 的逻辑层）。

在 Windows 原生 / WSL / macOS / Linux 四种部署下探测常见个人目录与 IM
文件目录，产出候选清单供交互多选，并把结果写入 .env 的
FILE_MEMORY_SOURCE_DIR（逗号分隔多根，personal_search/ingest.source_roots
的消费口径）。

可测性设计：目录存在性（exists）、可读性（readable）、OneDrive 环境值、
xdg-user-dir 查询、微信深层 glob 全部作为**参数注入**，单测不需要真实
Windows/macOS 环境；本模块只依赖 stdlib（不触发 torch/transformers 导入）。

原 scripts/wsl_sources.sh 与 scripts/macos_sources.sh 的候选清单在本模块
逐项对齐（tests/test_source_discovery.py 有防漂移用例），两个 bash 脚本
已改为薄包装转发到 scripts/sources.py。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

Exists = Callable[[str], bool]
Readable = Callable[[str], bool]


@dataclass(frozen=True)
class SourceCandidate:
    """一个候选源目录。

    kind: user_dir（标准个人目录）| im（微信/QQ/钉钉等）| cloud（iCloud/OneDrive）
    default_selected: `sources.py --all` 非交互模式下自动勾选的子集（只含
    标准个人目录，IM/云盘目录体量与隐私敏感度更高，需显式选择）。
    """

    path: str
    label: str
    kind: str
    exists: bool
    readable: bool
    default_selected: bool


def _default_exists(path: str) -> bool:
    return os.path.isdir(path)


def _default_readable(path: str) -> bool:
    """listdir 成功 ≈ 有读权限（macOS TCC 拦截时抛 OSError）。"""

    try:
        os.listdir(path)
        return True
    except OSError:
        return False


def _join(home: str, *parts: str) -> str:
    return str(Path(home).joinpath(*parts))


def _make(path: str, label: str, kind: str, exists: Exists, readable: Readable,
          default_selected: bool = False) -> SourceCandidate | None:
    """目录存在才成为候选；存在时进一步探测可读性（TCC/权限）。"""

    if not exists(path):
        return None
    return SourceCandidate(
        path=path,
        label=label,
        kind=kind,
        exists=True,
        readable=readable(path),
        default_selected=default_selected,
    )


# ---------------- Windows 原生 ----------------

# 与 wsl_sources.sh 的子目录清单逐项对齐（防漂移用例见测试）
_WINDOWS_USER_SUBDIRS: tuple[tuple[str, str, bool], ...] = (
    # (相对 home 的子路径, 标签, 是否默认勾选)
    ("Desktop", "桌面", True),
    ("Downloads", "下载", True),
    ("Documents", "文档", True),
    ("Pictures", "图片", True),
    ("Videos", "视频", False),
    ("Music", "音乐", False),
    ("Pictures/Screenshots", "截图", False),
    ("Pictures/Camera Roll", "相机胶卷", False),
)

_WINDOWS_IM_SUBDIRS: tuple[tuple[str, str], ...] = (
    ("Documents/WeChat Files", "微信 3.x (WeChat Files)"),
    ("Documents/xwechat_files", "微信 4.x (xwechat_files)"),
    ("AppData/Roaming/Tencent/WeChat", "微信缓存 (AppData)"),
    ("Documents/Tencent Files", "QQ (Tencent Files)"),
    ("Documents/WXWork", "企业微信 (WXWork)"),
    ("Documents/DingDing", "钉钉 (DingDing)"),
    ("AppData/Roaming/DingTalk", "钉钉缓存 (AppData)"),
)


def windows_candidates(
    home: str,
    *,
    exists: Exists = _default_exists,
    readable: Readable = _default_readable,
    onedrive_dirs: Sequence[str] = (),
) -> list[SourceCandidate]:
    """Windows 原生候选：标准个人目录 + IM 目录 + OneDrive。

    onedrive_dirs 由调用方注入（CLI 传 [os.environ["OneDrive"]] 或
    ~/OneDrive、~/OneDrive - * glob 结果），保持本函数纯逻辑。
    """

    candidates: list[SourceCandidate] = []
    for sub, label, default in _WINDOWS_USER_SUBDIRS:
        item = _make(_join(home, *sub.split("/")), label, "user_dir", exists, readable, default)
        if item is not None:
            candidates.append(item)
    for sub, label in _WINDOWS_IM_SUBDIRS:
        item = _make(_join(home, *sub.split("/")), label, "im", exists, readable)
        if item is not None:
            candidates.append(item)
    for directory in onedrive_dirs:
        item = _make(directory, "OneDrive", "cloud", exists, readable)
        if item is not None:
            candidates.append(item)
    return _dedupe(candidates)


# ---------------- WSL ----------------

_WSL_SKIP_ACCOUNTS = {"public", "default", "all users", "desktop.ini", "administrator"}

# 与 wsl_sources.sh 逐项相同（含大小写）
_WSL_SUBDIRS: tuple[str, ...] = (
    "Desktop", "Downloads", "Documents", "Pictures", "Videos", "Music",
    "Documents/WeChat Files", "Documents/xwechat_files",
    "Documents/Tencent Files", "Documents/DingDing",
    "Pictures/Camera Roll", "Pictures/Screenshots",
    "AppData/Roaming/Tencent/WeChat",
)


def wsl_user_dirs(mount_root: str, listdir: Callable[[str], list[str]]) -> list[str]:
    """列出 /mnt/c/Users 下的真实用户目录（跳过系统账户）。"""

    try:
        names = listdir(mount_root)
    except OSError:
        return []
    result = []
    for name in sorted(names):
        low = name.lower()
        if low in _WSL_SKIP_ACCOUNTS or low.startswith("default"):
            continue
        result.append(_join(mount_root, name))
    return result


def wsl_candidates(
    users_root: str = "/mnt/c/Users",
    *,
    exists: Exists = _default_exists,
    readable: Readable = _default_readable,
    listdir: Callable[[str], list[str]] = os.listdir,
) -> list[SourceCandidate]:
    """WSL 候选：/mnt/c/Users/* 下与 wsl_sources.sh 相同的子目录清单。"""

    candidates: list[SourceCandidate] = []
    for user_dir in wsl_user_dirs(users_root, listdir):
        if not exists(user_dir):
            continue
        account = Path(user_dir).name
        for sub in _WSL_SUBDIRS:
            label = f"{account} · {sub.rsplit('/', 1)[-1]}"
            kind = "im" if ("WeChat" in sub or "Tencent" in sub or "DingDing" in sub) else "user_dir"
            default = sub in {"Desktop", "Downloads", "Documents", "Pictures"}
            item = _make(f"{user_dir}/{sub}", label, kind, exists, readable, default)
            if item is not None:
                candidates.append(item)
    return _dedupe(candidates)


# ---------------- macOS ----------------

_MACOS_USER_SUBDIRS: tuple[tuple[str, str, bool], ...] = (
    ("Desktop", "桌面", True),
    ("Downloads", "下载", True),
    ("Documents", "文档", True),
    ("Pictures", "图片", True),
    ("Movies", "影片", False),
    ("Music", "音乐", False),
    ("Pictures/Screenshots", "截图", False),
)

_MACOS_ICLOUD = "Library/Mobile Documents/com~apple~CloudDocs"
_MACOS_WECHAT_AS = (
    "Library/Containers/com.tencent.xinWeChat/Data/Library/"
    "Application Support/com.tencent.xinWeChat"
)


def _default_wechat_glob(root: str) -> list[str]:
    """在微信 Application Support 下找 MessageTemp（限深 5，与 find -maxdepth 5 对齐）。"""

    found: list[str] = []
    root_depth = len(Path(root).parts)
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        if depth >= 5:
            dirnames[:] = []
            continue
        if Path(dirpath).name == "MessageTemp":
            found.append(dirpath)
            dirnames[:] = []  # 命中后不再深入
    return sorted(found)


def macos_candidates(
    home: str,
    *,
    exists: Exists = _default_exists,
    readable: Readable = _default_readable,
    wechat_glob: Callable[[str], list[str]] | None = None,
) -> list[SourceCandidate]:
    """macOS 候选：标准个人目录 + iCloud Drive + IM 沙盒（含 TCC 可读性）。

    微信优先更深的 MessageTemp（避开缓存子树），深层 glob 不中时退回
    Application Support 根，由排除目录剪枝兜底 —— 与 macos_sources.sh 一致。
    """

    candidates: list[SourceCandidate] = []
    for sub, label, default in _MACOS_USER_SUBDIRS:
        item = _make(_join(home, *sub.split("/")), label, "user_dir", exists, readable, default)
        if item is not None:
            candidates.append(item)

    icloud = _make(_join(home, *_MACOS_ICLOUD.split("/")), "iCloud Drive", "cloud", exists, readable)
    if icloud is not None:
        candidates.append(icloud)

    wechat_as = _join(home, *_MACOS_WECHAT_AS.split("/"))
    if exists(wechat_as):
        glob = wechat_glob or _default_wechat_glob
        deep = glob(wechat_as)
        if deep:
            for directory in deep:
                item = _make(directory, "微信 MessageTemp", "im", exists, readable)
                if item is not None:
                    candidates.append(item)
        else:
            item = _make(wechat_as, "微信 (Application Support)", "im", exists, readable)
            if item is not None:
                candidates.append(item)

    im_fixed = (
        ("Library/Containers/com.tencent.qq/Data/Documents", "QQ 沙盒"),
        ("Library/Containers/com.tencent.WeWorkMac/Data/Library/Application Support/WXWork", "企业微信 (WXWork)"),
        ("Library/Application Support/DingTalk", "钉钉 (DingTalk)"),
        ("Library/Application Support/iDingTalk", "钉钉 (iDingTalk)"),
    )
    for sub, label in im_fixed:
        item = _make(_join(home, *sub.split("/")), label, "im", exists, readable)
        if item is not None:
            candidates.append(item)
    return _dedupe(candidates)


# ---------------- Linux ----------------

_LINUX_FALLBACK_SUBDIRS: tuple[tuple[str, str, bool], ...] = (
    ("Desktop", "桌面", True),
    ("Downloads", "下载", True),
    ("Documents", "文档", True),
    ("Pictures", "图片", True),
    ("Videos", "视频", False),
    ("Music", "音乐", False),
)

_XDG_KEYS: tuple[tuple[str, str, bool], ...] = (
    ("DESKTOP", "桌面", True),
    ("DOWNLOAD", "下载", True),
    ("DOCUMENTS", "文档", True),
    ("PICTURES", "图片", True),
    ("VIDEOS", "视频", False),
    ("MUSIC", "音乐", False),
)

_USER_DIRS_LINE_RE = re.compile(r'^XDG_(\w+)_DIR="?([^"\n]+)"?\s*$')


def parse_user_dirs_file(text: str, home: str) -> dict[str, str]:
    """解析 ~/.config/user-dirs.dirs：XDG_DESKTOP_DIR="$HOME/桌面" → 展开路径。

    返回 {"DESKTOP": "/home/x/桌面", ...}；注释/损坏行跳过。
    """

    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _USER_DIRS_LINE_RE.match(line)
        if not match:
            continue
        key, raw = match.group(1), match.group(2)
        if raw.startswith("$HOME"):
            raw = home + raw[len("$HOME"):]
        result[key] = raw
    return result


def linux_candidates(
    home: str,
    *,
    exists: Exists = _default_exists,
    readable: Readable = _default_readable,
    user_dirs_text: str | None = None,
    xdg_lookup: Callable[[str], str | None] | None = None,
) -> list[SourceCandidate]:
    """Linux 候选，三级回落：user-dirs.dirs 文本 → xdg-user-dir 命令 → ~/子目录探测。"""

    candidates: list[SourceCandidate] = []
    mapping: dict[str, str] = {}
    if user_dirs_text:
        mapping = parse_user_dirs_file(user_dirs_text, home)
    if not mapping and xdg_lookup is not None:
        for key, _, _ in _XDG_KEYS:
            value = xdg_lookup(key)
            if value:
                mapping[key] = value

    if mapping:
        for key, label, default in _XDG_KEYS:
            path = mapping.get(key)
            if not path:
                continue
            item = _make(path, label, "user_dir", exists, readable, default)
            if item is not None:
                candidates.append(item)
    if not candidates:
        for sub, label, default in _LINUX_FALLBACK_SUBDIRS:
            item = _make(_join(home, sub), label, "user_dir", exists, readable, default)
            if item is not None:
                candidates.append(item)
    return _dedupe(candidates)


# ---------------- 平台分发 / 通用工具 ----------------


def candidates_for_platform(
    platform: str | None = None,
    *,
    home: str | None = None,
    **kwargs,
) -> list[SourceCandidate]:
    """按平台产出候选清单；platform=None 时自动 detect_platform()。

    kwargs 透传给各平台函数（exists/readable/onedrive_dirs/wechat_glob/
    user_dirs_text/xdg_lookup/listdir 等注入点）。
    """

    if platform is None:
        from .path_utils import detect_platform

        platform = detect_platform()
    home = home or os.path.expanduser("~")

    if platform == "windows":
        return windows_candidates(home, **kwargs)
    if platform == "wsl":
        return wsl_candidates(**kwargs)
    if platform == "macos":
        return macos_candidates(home, **kwargs)
    return linux_candidates(home, **kwargs)


def _dedupe(candidates: Iterable[SourceCandidate]) -> list[SourceCandidate]:
    seen: set[str] = set()
    result: list[SourceCandidate] = []
    for item in candidates:
        key = os.path.normcase(item.path)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def parse_selection(answer: str, count: int) -> list[int] | None:
    """解析多选输入："1 3,4" → [0,2,3]；"a" → 全选；"q" → None；非法 → ValueError。"""

    answer = (answer or "").strip()
    if answer.lower() == "q":
        return None
    if answer.lower() == "a":
        return list(range(count))
    selected: list[int] = []
    for token in answer.replace(",", " ").split():
        if not token.isdigit():
            raise ValueError(f"无效编号: {token}")
        idx = int(token) - 1
        if not (0 <= idx < count):
            raise ValueError(f"无效编号: {token}")
        if idx not in selected:
            selected.append(idx)
    if not selected:
        raise ValueError("未选择任何目录")
    return selected


def render_source_dir(paths: Sequence[str]) -> str:
    """逗号拼接多根（路径本身可含空格，分隔符只认逗号，与 bash 版一致不加引号）。"""

    return ",".join(paths)


def update_env_file(env_path: str | Path, key: str, value: str) -> bool:
    """把 KEY=VALUE 写入 .env：命中 ^\\s*KEY= 的每一行都替换，无命中则追加。

    与 deploy.sh / wsl_sources.sh 的 env_set 语义一致（避免"首个/末个生效"
    在 run.sh 与 python-dotenv 之间打架）。值不加引号（Windows 反斜杠在
    dotenv 双引号值里会被转义）。原子写：tmp + os.replace。
    返回是否发生了替换（False = 追加或新建）。
    """

    env_path = Path(env_path)
    pattern = re.compile(rf"^(\s*){re.escape(key)}=(.*)$")
    replaced = False
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if pattern.match(line):
            lines[i] = f"{key}={value}"
            replaced = True
    if not replaced:
        lines.append(f"{key}={value}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = env_path.with_name(env_path.name + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp_path, env_path)
    return replaced
