"""scripts/run.py 跨平台启动器：解释器定位、PYTHONPATH 拼接、启动编排。"""

import importlib.util
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_run_module():
    """scripts/ 不是包，按文件路径加载（run.py 有 __main__ 守卫，import 安全）。"""

    spec = importlib.util.spec_from_file_location("run_script_under_test", _ROOT / "scripts" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def run_mod():
    return _load_run_module()


def _make_python(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_find_python_prefers_venv_scripts_layout(run_mod, tmp_path):
    """两种布局并存时 Scripts/python.exe 优先（Windows 布局在前）。"""

    _make_python(tmp_path / ".venv" / "bin" / "python")
    scripts_py = _make_python(tmp_path / ".venv" / "Scripts" / "python.exe")
    assert run_mod.find_python(tmp_path) == scripts_py


def test_find_python_posix_layout(run_mod, tmp_path):
    _make_python(tmp_path / ".venv" / "bin" / "python")
    assert run_mod.find_python(tmp_path) == tmp_path / ".venv" / "bin" / "python"


def test_find_python_falls_back_to_virtual_env(run_mod, tmp_path, monkeypatch):
    venv_py = _make_python(tmp_path / "custom_venv" / "bin" / "python")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "custom_venv"))
    assert run_mod.find_python(tmp_path / "no_project_venv") == venv_py


def test_find_python_missing_raises_with_platform_hint(run_mod, tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    with pytest.raises(run_mod.VenvNotFoundError) as excinfo:
        run_mod.find_python(tmp_path / "empty")
    message = str(excinfo.value)
    if os.name == "nt":
        assert ".venv\\Scripts" in message or "Scripts" in message
    else:
        assert ".venv/bin" in message or "bin" in message


def test_build_pythonpath_uses_os_pathsep(run_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(os, "pathsep", ";")
    result = run_mod.build_pythonpath(tmp_path, "X:\\other")
    assert result == f"{tmp_path / 'src'};X:\\other"
    assert run_mod.build_pythonpath(tmp_path, "") == str(tmp_path / "src")


def test_main_returns_1_without_venv(run_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod, "ROOT_DIR", tmp_path)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(run_mod.os, "chdir", lambda _p: None)
    assert run_mod.main([]) == 1


def test_main_chdir_to_repo_root_and_launches_uvicorn(run_mod, tmp_path, monkeypatch):
    """main 必须 cd 到仓库根（data/*.jsonl 是相对路径）并以 venv 解释器起 uvicorn。"""

    python = _make_python(tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(run_mod, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(run_mod.os, "chdir", lambda _p: None)
    monkeypatch.setattr(run_mod, "is_venv", lambda _p: True)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(returncode=0, stdout="Python 3.11.0\n", stderr="")

    monkeypatch.setattr(run_mod.subprocess, "run", fake_run)
    monkeypatch.setenv("PORT", "9100")

    assert run_mod.main([]) == 0

    # 最后一次调用是 uvicorn 启动
    uvicorn_cmd, uvicorn_kwargs = calls[-1]
    assert uvicorn_cmd[0] == str(python)
    assert "-m" in uvicorn_cmd and "uvicorn" in uvicorn_cmd
    assert "--port" in uvicorn_cmd and "9100" in uvicorn_cmd
    assert uvicorn_kwargs["cwd"] == str(tmp_path)
    assert uvicorn_kwargs["env"]["PYTHONPATH"].startswith(str(tmp_path / "src"))


def test_main_rejects_non_venv_interpreter(run_mod, tmp_path, monkeypatch):
    python = _make_python(tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(run_mod, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(run_mod.os, "chdir", lambda _p: None)
    monkeypatch.setattr(run_mod, "is_venv", lambda _p: False)
    assert run_mod.main([]) == 1
    # --skip-venv-check 可绕过（此时会真正尝试起 uvicorn，mock subprocess.run）
    calls = []
    monkeypatch.setattr(
        run_mod.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    assert run_mod.main(["--skip-venv-check"]) == 0
    assert any("uvicorn" in c for c in calls)
