#!/usr/bin/env python3
"""跨平台服务启动器（Windows 原生 / macOS / Linux / WSL 通用）。

与 scripts/run.sh 语义对齐：
- 优先项目 .venv 解释器（Windows: .venv\\Scripts\\python.exe，POSIX:
  .venv\\bin\\python），其次已激活的虚拟环境；拒绝非 venv 解释器
  （--skip-venv-check 可绕过），防止 cpython-3xx 混乱缓存
- HOST/PORT 只从进程环境取（默认 0.0.0.0:9000）
- .env 无需在此解析：edge_cloud_agent/__init__.py 在包 import 时
  load_dotenv(override=False)，与 run.sh 手写循环语义一致

run.sh 保留不动（service.sh 的 nohup 降级与 systemd ExecStart 依赖它）；
Windows 原生推荐 scripts\\run.bat / run.ps1（均为本文件的薄包装）。

用法：
  python scripts/run.py [--skip-venv-check]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


class VenvNotFoundError(RuntimeError):
    pass


def _venv_layout_dirs() -> tuple[tuple[str, str], ...]:
    """(Scripts, python.exe) / (bin, python) 两种 venv 布局，Windows 优先。"""

    return (
        ("Scripts", "python.exe"),
        ("bin", "python"),
    )


def find_python(root: Path) -> Path:
    """定位虚拟环境解释器：项目 .venv → $VIRTUAL_ENV；找不到抛 VenvNotFoundError。"""

    search_roots = [root / ".venv"]
    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        search_roots.append(Path(virtual_env))

    for base in search_roots:
        for dirname, exe in _venv_layout_dirs():
            candidate = base / dirname / exe
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

    if os.name == "nt":
        hint = "python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt"
    else:
        hint = "python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    raise VenvNotFoundError(
        f"找不到虚拟环境（查找了 {', '.join(str(p) for p in search_roots)}）。请先执行: {hint}"
    )


def build_pythonpath(root: Path, existing: str = "") -> str:
    """src 目录 + os.pathsep + 既有 PYTHONPATH（修掉 run.sh 硬编码 ':' 的问题）。"""

    src = str(root / "src")
    return f"{src}{os.pathsep}{existing}" if existing else src


def is_venv(python: Path) -> bool:
    """子进程验证 sys.prefix != sys.base_prefix（与 run.sh 同款检查）。"""

    try:
        result = subprocess.run(
            [str(python), "-c", "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("--skip-venv-check", action="store_true", help="跳过 venv 解释器校验")
    args = parser.parse_args(argv)

    # data/*.jsonl 与 FAISS 索引都是相对路径，必须 cd 到仓库根再起（对齐 run.sh）
    os.chdir(ROOT_DIR)

    try:
        python = find_python(ROOT_DIR)
    except VenvNotFoundError as exc:
        print(f"[run.py] {exc}", file=sys.stderr)
        return 1

    if not args.skip_venv_check and not is_venv(python):
        print(f"[run.py] {python} 不是虚拟环境解释器，已中止（--skip-venv-check 可绕过）。", file=sys.stderr)
        return 1

    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "9000")

    env = dict(os.environ)
    env["PYTHONPATH"] = build_pythonpath(ROOT_DIR, env.get("PYTHONPATH", ""))

    version = subprocess.run([str(python), "-V"], capture_output=True, text=True)
    print(f"[run.py] python   = {(version.stdout or version.stderr).strip()}")
    print(f"[run.py] venv     = {python}")
    print(f"[run.py] PYTHONPATH = {env['PYTHONPATH']}")
    print(f"[run.py] listening = http://{host}:{port}")

    # 不用 os.execv：Windows 上不替换进程、退出码语义不同
    result = subprocess.run(
        [str(python), "-m", "uvicorn", "edge_cloud_agent.main:app", "--host", host, "--port", port],
        env=env,
        cwd=str(ROOT_DIR),
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
