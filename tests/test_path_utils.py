"""跨平台路径工具：WSL 映射、Windows 原生判定、file URI 规范化与还原。"""

import os
import sys
from pathlib import Path

import pytest

from edge_cloud_agent.path_utils import (
    as_file_uri,
    detect_platform,
    is_windows_native,
    is_wsl,
    to_windows_path,
    uri_to_path,
    wsl_to_windows_path,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """is_wsl 带 lru_cache，逐用例清缓存避免环境变量串扰。"""

    is_wsl.cache_clear()
    yield
    is_wsl.cache_clear()


@pytest.fixture(autouse=True)
def _clear_platform_overrides(monkeypatch):
    """默认清掉两个平台覆盖变量，用例内按需显式设置。"""

    monkeypatch.delenv("FILE_MEMORY_WSL_PATH_MAP", raising=False)
    monkeypatch.delenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", raising=False)


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


def test_is_wsl_auto_on_non_wsl():
    """无覆盖变量时按 /proc/version 判定；macOS/普通 Linux 应为 False。"""

    result = is_wsl()
    assert isinstance(result, bool)
    try:
        with open("/proc/version", encoding="ascii", errors="ignore") as fh:
            version = fh.read().lower()
        assert result == ("microsoft" in version or "wsl" in version)
    except OSError:
        assert result is False


# ---------------- Windows 原生判定 / 平台识别 ----------------


def test_is_windows_native_env_override(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "true")
    assert is_windows_native() is True
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "false")
    assert is_windows_native() is False


def test_is_windows_native_default_matches_os_name():
    assert is_windows_native() == (os.name == "nt")


def test_detect_platform_windows_native(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "true")
    assert detect_platform() == "windows"


def test_detect_platform_windows_wins_over_wsl(monkeypatch):
    """两个覆盖变量同时为 true 时，以显式 Windows 原生为准（判定顺序钉死）。"""

    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "true")
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "true")
    assert detect_platform() == "windows"


def test_detect_platform_macos(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "false")
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "false")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert detect_platform() == "macos"


def test_detect_platform_wsl(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "false")
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "true")
    monkeypatch.setattr(sys, "platform", "linux")
    assert detect_platform() == "wsl"


def test_detect_platform_linux(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "false")
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "false")
    monkeypatch.setattr(sys, "platform", "linux")
    assert detect_platform() == "linux"


# ---------------- file URI 规范化 ----------------


def test_as_file_uri_posix_unchanged():
    """POSIX 输出必须与历史 f"file://{as_posix()}" 逐字一致（存量索引零失配）。"""

    for raw in ("/Users/x/a.png", "/mnt/c/Users/x/a.png", "/tmp/WeChat Files/微信 聚餐.jpg"):
        path = Path(raw)
        assert as_file_uri(path) == f"file://{path.as_posix()}"
    assert as_file_uri(Path("/Users/x/a.png")) == "file:///Users/x/a.png"


def test_as_file_uri_windows_drive_gets_triple_slash():
    # 盘符按路径形状判定：POSIX 机器上也能验证 Windows 分支
    assert as_file_uri(Path("C:/Users/x/a.png")) == "file:///C:/Users/x/a.png"
    assert as_file_uri(Path("c:/data/b.txt")) == "file:///c:/data/b.txt"


def test_as_file_uri_windows_unc_kept():
    posix = Path("//server/share/x.txt").as_posix()
    if posix.startswith("//"):  # POSIX 上 as_posix 保留双斜杠；Windows 上同样
        assert as_file_uri(Path("//server/share/x.txt")) == f"file://{posix}"


def test_as_file_uri_space_and_cjk_not_percent_encoded():
    """钉死决策：不用 Path.as_uri()，空格/中文不做百分号编码。"""

    uri = as_file_uri(Path("/Users/x/WeChat Files/聚餐 照片.jpg"))
    assert uri == "file:///Users/x/WeChat Files/聚餐 照片.jpg"
    assert "%" not in uri


def test_uri_to_path_posix_form():
    assert uri_to_path("file:///Users/x/a.png") == "/Users/x/a.png"
    assert uri_to_path("file:///mnt/c/Users/x/a.png") == "/mnt/c/Users/x/a.png"


def test_uri_to_path_windows_new_triple_slash():
    assert uri_to_path("file:///C:/Users/x/a.png") == "C:/Users/x/a.png"


def test_uri_to_path_windows_legacy_double_slash():
    assert uri_to_path("file://C:/Users/x/a.png") == "C:/Users/x/a.png"


def test_uri_to_path_unc():
    assert uri_to_path("file:////server/share/x.txt") == "//server/share/x.txt"


def test_uri_to_path_non_file_uri_passthrough():
    assert uri_to_path("https://example.com/a.png") == "https://example.com/a.png"
    assert uri_to_path("/plain/path.txt") == "/plain/path.txt"


def test_uri_to_path_empty_none():
    assert uri_to_path("") is None
    assert uri_to_path(None) is None


def test_uri_path_roundtrip_all_shapes():
    for raw in ("/Users/x/a.png", "/mnt/c/x.txt", "C:/Users/x/a.png", "d:/data/b.csv"):
        path = Path(raw)
        assert uri_to_path(as_file_uri(path)) == path.as_posix()


# ---------------- open 动作 Windows 路径回传 ----------------


def test_to_windows_path_native_normalizes_separators(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "true")
    assert to_windows_path("C:/Users/x/a.png") == "C:\\Users\\x\\a.png"
    assert to_windows_path("c:\\users\\x\\b.txt") == "C:\\users\\x\\b.txt"


def test_to_windows_path_native_rejects_posix_path(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "true")
    assert to_windows_path("/home/x/a.txt") is None


def test_to_windows_path_wsl_delegates(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "false")
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "true")
    is_wsl.cache_clear()
    assert to_windows_path("/mnt/c/Users/x/a.png") == "C:\\Users\\x\\a.png"
    assert to_windows_path("/home/x/a.txt") is None


def test_to_windows_path_windows_wins_over_wsl_override(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "true")
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "true")
    assert to_windows_path("C:/x/a.png") == "C:\\x\\a.png"


def test_to_windows_path_none_on_macos_linux(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "false")
    monkeypatch.setenv("FILE_MEMORY_WSL_PATH_MAP", "false")
    is_wsl.cache_clear()
    assert to_windows_path("/Users/x/a.png") is None
    assert to_windows_path("C:/x/a.png") is None


def test_to_windows_path_empty_none():
    assert to_windows_path("") is None
    assert to_windows_path(None) is None
