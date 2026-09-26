"""跨平台路径 / file URI 工具。

职责（全部为纯函数或仅依赖环境变量/系统常量，可跨平台单测）：
1. 平台判定：Windows 原生 / WSL / macOS / Linux；
2. file URI 规范化：POSIX 路径保持历史 ``file://{as_posix()}`` 格式**字面不变**
   （存量索引 JSONL 不失配、不重导）；Windows 盘符路径产出合法的
   ``file:///C:/...``（旧实现会产出 ``file://C:/...``，盘符落在 authority 位）；
3. URI → 路径还原：兼容新旧格式（含历史 ``file://C:/...`` 形态）；
4. open 动作的 Windows 路径回传：WSL（``/mnt/c/...``）与 Windows 原生
   （``C:/...``）两种部署都映射成 ``C:\\...`` 交前端展示/复制。

刻意**不做**百分号编码（不用 ``Path.as_uri()``）：as_uri 会对空格与中文
percent-encode，导致存量 URI（大量含 "WeChat Files/微信/聚餐.jpg" 之类）
全量失配重导。全链路把 file_uri 当作不透明去重键 + 展示串使用。

WSL 检测顺序（is_wsl）：
1. `FILE_MEMORY_WSL_PATH_MAP` 显式覆盖（true/false，供非 WSL 环境测试与排障）
2. `/proc/version` 含 microsoft/WSL 字样（WSL 官方推荐判据）

Windows 原生检测（is_windows_native）：
1. `FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP` 显式覆盖（供非 Windows 环境测试）
2. `os.name == "nt"`（不加 lru_cache：系统常量，测试无需清缓存）
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

# /mnt/<盘符>/<路径>；WSL 自动挂载点固定为 /mnt/x（x 为单个字母盘符）
_WSL_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])(/.*)?$")

# Windows 盘符路径形状（C:/ 或 C:\）。按路径形状而非 os.name 判定，
# 使 as_file_uri/uri_to_path 成为纯函数：macOS/Linux 上也能测 Windows 分支。
# 已知权衡：POSIX 上真存在名为 "C:" 的目录会被误判，实际几乎不会出现。
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[/\\]")


def _env_flag(name: str) -> bool | None:
    """解析三态环境变量覆盖：true/false/未设置(None)。"""

    override = os.getenv(name, "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    return None


@lru_cache(maxsize=1)
def is_wsl() -> bool:
    """当前进程是否运行在 WSL 内（可被环境变量强制覆盖）。"""

    override = _env_flag("FILE_MEMORY_WSL_PATH_MAP")
    if override is not None:
        return override
    try:
        with open("/proc/version", encoding="ascii", errors="ignore") as fh:
            version = fh.read().lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


def is_windows_native() -> bool:
    """当前进程是否运行在 Windows 原生（非 WSL）。

    可被 FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP 强制覆盖（供测试与排障）。
    """

    override = _env_flag("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP")
    if override is not None:
        return override
    return os.name == "nt"


def detect_platform() -> str:
    """返回部署平台：'windows' | 'wsl' | 'macos' | 'linux'。

    windows 先判：两个覆盖变量同时置 true 时以显式的 Windows 原生为准。
    调用时读取（不缓存），便于测试 monkeypatch。
    """

    if is_windows_native():
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if is_wsl():
        return "wsl"
    return "linux"


def wsl_to_windows_path(path_str: str | None) -> str | None:
    """把 WSL 挂载路径转成 Windows 路径。

    `/mnt/c/Users/x/a.png` → `C:\\Users\\x\\a.png`；`/mnt/d` → `D:\\`。
    非 /mnt/<盘符> 路径（WSL 原生文件系统、已转换过的路径）返回 None。
    """

    if not path_str:
        return None
    match = _WSL_MOUNT_RE.match(path_str)
    if not match:
        return None
    drive = match.group(1).upper()
    rest = (match.group(2) or "").replace("/", "\\")
    return f"{drive}:{rest or chr(92)}"


def as_file_uri(path: Path) -> str:
    """路径 → file URI（单一事实源，两条业务线的 ingest 共用）。

    - 盘符路径：`C:/Users/x/a.png` → `file:///C:/Users/x/a.png`（合法三斜杠）
    - POSIX 路径：`/mnt/c/x` → `file:///mnt/c/x`，与历史 `f"file://{as_posix()}"`
      字面完全一致（as_posix 以 / 开头），存量索引零失配
    - UNC 路径：`//server/share/x` → `file:////server/share/x`（恰好合法）
    """

    posix = path.as_posix()
    if _WIN_DRIVE_RE.match(posix):
        return f"file:///{posix}"
    return f"file://{posix}"


def uri_to_path(file_uri: str | None) -> str | None:
    """file URI → 路径字符串，兼容历史与新格式。

    - `file:///Users/x`     → `/Users/x`
    - `file:///C:/x`        → `C:/x`（新 Windows 格式）
    - `file://C:/x`         → `C:/x`（历史 Windows 格式）
    - `file:////server/sh`  → `//server/sh`（UNC）
    - 非 file:// 前缀原样返回；空 → None
    """

    if not file_uri:
        return None
    if not file_uri.startswith("file://"):
        return file_uri
    rest = file_uri[len("file://"):]
    # file:///C:/x → rest = "/C:/x"，剥掉前导斜杠还原盘符路径
    if rest.startswith("/") and _WIN_DRIVE_RE.match(rest[1:]):
        return rest[1:]
    return rest


def to_windows_path(path_str: str | None) -> str | None:
    """open 动作统一入口：把命中文件路径映射成 Windows 侧可展示/复制的形式。

    - Windows 原生：`C:/Users/x/a.png`（或 `c:\\...`）→ `C:\\Users\\x\\a.png`
      （分隔符归一 + 盘符大写）；非盘符路径返回 None
    - WSL：委托 wsl_to_windows_path（/mnt/<盘符> → C:\\...）
    - macOS / Linux：None（前端沿用 file_uri 展示）
    """

    if not path_str:
        return None
    if is_windows_native():
        if not _WIN_DRIVE_RE.match(path_str.replace("\\", "/")):
            return None
        normalized = path_str.replace("/", "\\")
        return normalized[0].upper() + normalized[1:]
    if is_wsl():
        return wsl_to_windows_path(path_str)
    return None
