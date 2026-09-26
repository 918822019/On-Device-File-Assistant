#!/usr/bin/env python3
"""跨平台文件索引源目录配置助手（统一入口，取代 wsl_sources.sh / macos_sources.sh）。

按当前平台（Windows 原生 / WSL / macOS / Linux）探测常见个人目录与 IM 文件
目录，交互多选后把逗号分隔的多根写入 .env 的 FILE_MEMORY_SOURCE_DIR。

用法：
  python scripts/sources.py                     # 自动识别平台 → 交互选择
  python scripts/sources.py --print             # 只列候选（✓/⚠️），不写 .env
  python scripts/sources.py --platform wsl      # 强制平台 auto|windows|wsl|macos|linux
  python scripts/sources.py --all               # 非交互：只写默认勾选（标准个人目录）
  python scripts/sources.py /path/a /path/b     # 直接写入给定目录（任意平台）

说明：
- 多根用逗号分隔，路径本身可含空格（如 "WeChat Files"），不要往值里加引号
- 写入后需重启服务并触发一次重建索引（Web 页面「重建索引」按钮或
  POST /v1/search-agent/rebuild-index）
- 扫描排除目录由 FILE_MEMORY_SCAN_EXCLUDE_DIRS 控制（默认已含 node_modules、
  $RECYCLE.BIN、各类缓存目录等）
- ⚠️ 不可读目录：macOS 是 TCC 隐私拦截（系统设置 → 隐私与安全性 → 完全磁盘
  访问权限，把运行本脚本/后端的终端加进去）；Windows 多为权限受限目录
- Windows + OneDrive：「仅在线」占位符文件默认被扫描跳过
  （FILE_MEMORY_SKIP_CLOUD_PLACEHOLDERS），不会触发静默下载
"""

from __future__ import annotations

import argparse
import glob as globmod
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
_SRC = ROOT_DIR / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from edge_cloud_agent.common.path_utils import detect_platform  # noqa: E402
from edge_cloud_agent.common.source_discovery import (  # noqa: E402
    candidates_for_platform,
    parse_selection,
    render_source_dir,
    update_env_file,
)

_PREFIX = "[sources]"


def info(message: str) -> None:
    print(f"{_PREFIX} {message}")


def fail(message: str) -> "NoReturn":  # noqa: F821
    print(f"{_PREFIX}[ERROR] {message}", file=sys.stderr)
    sys.exit(1)


def _onedrive_dirs(home: str) -> list[str]:
    """Windows OneDrive 目录：环境变量优先，其次 ~\\OneDrive、~\\OneDrive - *。"""

    env_dir = os.environ.get("OneDrive", "").strip()
    if env_dir:
        return [env_dir]
    result: list[str] = []
    plain = os.path.join(home, "OneDrive")
    if os.path.isdir(plain):
        result.append(plain)
    result.extend(sorted(globmod.glob(os.path.join(home, "OneDrive - *"))))
    return result


def _xdg_lookup(key: str) -> str | None:
    """xdg-user-dir 命令查询（Linux）；命令不存在/失败返回 None。"""

    try:
        out = subprocess.run(
            ["xdg-user-dir", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (out.stdout or "").strip()
    return value or None


def collect_candidates(platform: str) -> list:
    home = os.path.expanduser("~")
    kwargs: dict = {}
    if platform == "windows":
        kwargs["onedrive_dirs"] = _onedrive_dirs(home)
    if platform == "linux":
        user_dirs_file = Path(home) / ".config" / "user-dirs.dirs"
        if user_dirs_file.is_file():
            try:
                kwargs["user_dirs_text"] = user_dirs_file.read_text(encoding="utf-8")
            except OSError:
                pass
        kwargs["xdg_lookup"] = _xdg_lookup
    return candidates_for_platform(platform, home=home, **kwargs)


def write_paths(paths: list[str]) -> None:
    env_file = ROOT_DIR / ".env"
    update_env_file(env_file, "FILE_MEMORY_SOURCE_DIR", render_source_dir(paths))
    info(f".env: FILE_MEMORY_SOURCE_DIR 已更新（{len(render_source_dir(paths))} 字符）")


def mode_direct(paths: list[str]) -> int:
    for p in paths:
        if not os.path.isdir(p):
            fail(f"目录不存在: {p}")
    write_paths(paths)
    info("重启服务后生效；建议随后触发一次重建索引")
    return 0


def print_candidates(platform: str, candidates: list) -> None:
    info(f"平台: {platform}；探测到 {len(candidates)} 个候选目录：")
    for i, cand in enumerate(candidates):
        mark = "✓" if cand.readable else "⚠️ 不可读"
        default = " [默认]" if cand.default_selected else ""
        print(f"  [{i + 1:2d}] {cand.path}  ({cand.label}, {cand.kind}) {mark}{default}")


def platform_hints(platform: str, candidates: list) -> None:
    if platform == "wsl":
        info("提示：跨 9P 扫 /mnt/c 首扫较慢，之后 hash/size 判重前置，增量成本低")
    if platform == "macos" and any(not c.readable for c in candidates):
        info("⚠️ 有目录不可读：macOS TCC 拦截了未授权访问。请到「系统设置 → 隐私与")
        info("   安全性 → 完全磁盘访问权限」添加运行本脚本与后端服务的终端 App，")
        info("   然后重启终端再试。iCloud 未下载文件（.icloud 占位符）会被自然跳过")
    if platform == "windows":
        info("提示：OneDrive「仅在线」占位符默认跳过，不会触发静默下载；")
        info("   深层路径超 260 字符的文件会计入 errors，可开启系统 LongPathsEnabled")


def mode_interactive(platform: str, candidates: list, select_all: bool, print_only: bool) -> int:
    if not candidates:
        fail(f"平台 {platform} 未探测到任何候选目录；可直接传路径：python scripts/sources.py /path/a /path/b")
    print_candidates(platform, candidates)
    platform_hints(platform, candidates)
    if print_only:
        return 0

    if select_all:
        selected = [c for c in candidates if c.default_selected] or candidates
    else:
        print()
        try:
            answer = input(f"{_PREFIX} 选择要纳入索引的编号（空格/逗号分隔，a=全选，q=退出）: ")
        except (EOFError, KeyboardInterrupt):
            info("已退出，未修改 .env")
            return 2
        try:
            indexes = parse_selection(answer, len(candidates))
        except ValueError as exc:
            fail(str(exc))
        if indexes is None:
            info("已退出，未修改 .env")
            return 2
        selected = [candidates[i] for i in indexes]

    paths = [c.path for c in selected]
    write_paths(paths)
    info(f"已选 {len(paths)} 个根目录：")
    for p in paths:
        info(f"  - {p}")
    print()
    info("后续步骤：")
    if platform == "windows":
        info("  1. 重启服务: python scripts\\run.py（或 scripts\\run.bat）")
    else:
        info("  1. 重启服务: bash scripts/service.sh restart（或前台 make run）")
    info("  2. 首扫在启动时自动执行；也可在 Web 页面点「重建索引」")
    info("  3. 想剪掉更多目录: 编辑 .env 的 FILE_MEMORY_SCAN_EXCLUDE_DIRS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sources.py", add_help=True)
    parser.add_argument("paths", nargs="*", help="直接写入的目录路径（跳过探测）")
    parser.add_argument("--print", dest="print_only", action="store_true", help="只列候选，不写 .env")
    parser.add_argument(
        "--platform",
        choices=("auto", "windows", "wsl", "macos", "linux"),
        default="auto",
        help="强制平台（默认自动识别）",
    )
    parser.add_argument("--all", dest="select_all", action="store_true", help="非交互：只写默认勾选目录")
    args = parser.parse_args(argv)

    if args.paths:
        return mode_direct(args.paths)

    platform = detect_platform() if args.platform == "auto" else args.platform
    if platform == "wsl" and not Path("/mnt/c").is_dir():
        # 与 wsl_sources.sh 相同的前置检查：/mnt/c 不存在时 WSL 探测无意义
        fail("未找到 /mnt/c：当前不是 WSL 环境。请直接传目录路径或换 --platform")
    candidates = collect_candidates(platform)
    return mode_interactive(platform, candidates, args.select_all, args.print_only)


if __name__ == "__main__":
    sys.exit(main())
