"""WSL / Windows 路径工具。

后端跑在 WSL 下时，文件索引层覆盖的是 Windows 侧目录（/mnt/c/Users/...）。
`file:///mnt/c/...` 形式的 URI 对 Windows 浏览器/资源管理器没有意义，open 等
动作需要把命中文件映射回 Windows 路径（C:\\...）交给前端展示与复制。

检测顺序：
1. `FILE_MEMORY_WSL_PATH_MAP` 显式覆盖（true/false，供非 WSL 环境测试与排障）
2. `/proc/version` 含 microsoft/WSL 字样（WSL 官方推荐判据）
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

# /mnt/<盘符>/<路径>；WSL 自动挂载点固定为 /mnt/x（x 为单个字母盘符）
_WSL_MOUNT_RE = re.compile(r"^/mnt/([a-zA-Z])(/.*)?$")


@lru_cache(maxsize=1)
def is_wsl() -> bool:
    """当前进程是否运行在 WSL 内（可被环境变量强制覆盖）。"""

    override = os.getenv("FILE_MEMORY_WSL_PATH_MAP", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    try:
        with open("/proc/version", encoding="ascii", errors="ignore") as fh:
            version = fh.read().lower()
    except OSError:
        return False
    return "microsoft" in version or "wsl" in version


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
