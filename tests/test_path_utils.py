"""WSL 路径工具：/mnt/c → C:\ 映射与 is_wsl 检测覆盖。"""

import pytest

from edge_cloud_agent.path_utils import is_wsl, wsl_to_windows_path


@pytest.fixture(autouse=True)
def _clear_cache():
    """is_wsl 带 lru_cache，逐用例清缓存避免环境变量串扰。"""

    is_wsl.cache_clear()
    yield
    is_wsl.cache_clear()


def test_wsl_to_windows_path_basic():
    assert wsl_to_windows_path("/mnt/c/Users/x/a.png") == "C:\\Users\\x\\a.png"
    assert wsl_to_windows_path("/mnt/d/data/b.txt") == "D:\\data\\b.txt"


def test_wsl_to_windows_path_space_and_cjk():
    p = wsl_to_windows_path("/mnt/c/Users/x/Documents/WeChat Files/微信/聚餐.jpg")
    assert p == "C:\\Users\\x\\Documents\\WeChat Files\\微信\\聚餐.jpg"


def test_wsl_to_windows_path_bare_drive():
    assert wsl_to_windows_path("/mnt/c") == "C:\\"
    assert wsl_to_windows_path("/mnt/e/") == "E:\\"


def test_wsl_to_windows_path_non_mount_returns_none():
    assert wsl_to_windows_path("/home/x/a.txt") is None
    assert wsl_to_windows_path("/tmp/mnt/c/a.txt") is None  # 前缀不在开头，不匹配
    assert wsl_to_windows_path("") is None
    assert wsl_to_windows_path(None) is None


def test_is_wsl_env_override(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "true")
    assert is_wsl() is True
    is_wsl.cache_clear()
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "false")
    assert is_wsl() is False


def test_is_wsl_auto_on_non_wsl(monkeypatch):
    """无覆盖变量时按 /proc/version 判定；macOS/普通 Linux 应为 False。"""

    monkeypatch.delenv("FILE_MEMORY_WSL_PATH_MAP", raising=False)
    result = is_wsl()
    assert isinstance(result, bool)
    try:
        with open("/proc/version", encoding="ascii", errors="ignore") as fh:
            version = fh.read().lower()
        assert result == ("microsoft" in version or "wsl" in version)
    except OSError:
        assert result is False
