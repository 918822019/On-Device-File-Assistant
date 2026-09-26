"""source_discovery：四平台候选探测、多选解析、.env 写入。

全部通过依赖注入（exists/readable/listdir/wechat_glob/xdg_lookup）模拟
文件系统与平台差异，不需要真实 Windows/WSL 环境。路径断言统一把反斜杠
归一为正斜杠，保证本测试文件在 Windows 上也能跑。
"""

import pytest

from edge_cloud_agent import source_discovery as sd


def _norm(path: str) -> str:
    return path.replace("\\", "/")


def _fake_fs(dirs, unreadable=()):
    """按字符串集合模拟 isdir/listdir 可读性。"""

    dirset = {_norm(d) for d in dirs}
    unreadable_set = {_norm(d) for d in unreadable}

    def exists(path):
        return _norm(path) in dirset

    def readable(path):
        return _norm(path) not in unreadable_set

    return exists, readable


# ---------------- Windows 原生 ----------------

_WIN_HOME = "C:/Users/x"


def test_windows_candidates_cover_user_dirs_and_im():
    dirs = [
        f"{_WIN_HOME}/Desktop",
        f"{_WIN_HOME}/Downloads",
        f"{_WIN_HOME}/Documents",
        f"{_WIN_HOME}/Pictures",
        f"{_WIN_HOME}/Documents/WeChat Files",
        f"{_WIN_HOME}/Documents/xwechat_files",
    ]
    exists, readable = _fake_fs(dirs)
    candidates = sd.windows_candidates(_WIN_HOME, exists=exists, readable=readable)

    by_path = {_norm(c.path): c for c in candidates}
    assert len(candidates) == 6
    desktop = by_path[f"{_WIN_HOME}/Desktop"]
    assert (desktop.kind, desktop.default_selected, desktop.readable) == ("user_dir", True, True)
    wechat = by_path[f"{_WIN_HOME}/Documents/WeChat Files"]
    assert (wechat.kind, wechat.default_selected) == ("im", False)
    assert by_path[f"{_WIN_HOME}/Documents/xwechat_files"].kind == "im"


def test_windows_candidates_onedrive_from_env():
    dirs = [f"{_WIN_HOME}/Desktop", "C:/Users/x/OneDrive"]
    exists, readable = _fake_fs(dirs)
    candidates = sd.windows_candidates(
        _WIN_HOME, exists=exists, readable=readable, onedrive_dirs=["C:/Users/x/OneDrive"]
    )
    kinds = {_norm(c.path): c.kind for c in candidates}
    assert kinds["C:/Users/x/OneDrive"] == "cloud"


def test_windows_candidates_skip_missing_dirs():
    exists, readable = _fake_fs([f"{_WIN_HOME}/Desktop"])
    candidates = sd.windows_candidates(_WIN_HOME, exists=exists, readable=readable)
    assert [_norm(c.path) for c in candidates] == [f"{_WIN_HOME}/Desktop"]


def test_windows_candidates_readable_flag():
    """权限受限目录：仍是候选，但 readable=False（CLI 打 ⚠️）。"""

    dirs = [f"{_WIN_HOME}/Desktop", f"{_WIN_HOME}/Documents"]
    exists, readable = _fake_fs(dirs, unreadable=[f"{_WIN_HOME}/Documents"])
    candidates = sd.windows_candidates(_WIN_HOME, exists=exists, readable=readable)
    flags = {_norm(c.path): c.readable for c in candidates}
    assert flags[f"{_WIN_HOME}/Desktop"] is True
    assert flags[f"{_WIN_HOME}/Documents"] is False


def test_windows_im_subdirs_align_with_wsl_script():
    """防漂移：Windows 原生 IM 子目录清单应覆盖 wsl_sources.sh 的同名项。"""

    im_subs = {sub for sub, _ in sd._WINDOWS_IM_SUBDIRS}
    wsl_im_subs = {s for s in sd._WSL_SUBDIRS if "WeChat" in s or "Tencent" in s or "DingDing" in s}
    # AppData/Roaming/Tencent/WeChat 两边都有；Windows 侧额外多 WXWork/DingTalk 缓存
    assert wsl_im_subs <= im_subs


# ---------------- WSL ----------------


def test_wsl_user_dirs_skip_system_accounts():
    names = ["alice", "Public", "Default", "DefaultUser0", "All Users", "desktop.ini", "Administrator", "bob"]
    result = sd.wsl_user_dirs("/mnt/c/Users", lambda _p: names)
    assert result == ["/mnt/c/Users/alice", "/mnt/c/Users/bob"]


def test_wsl_user_dirs_listdir_error_returns_empty():
    def boom(_path):
        raise OSError("not mounted")

    assert sd.wsl_user_dirs("/mnt/c/Users", boom) == []


def test_wsl_candidates_subdirs_match_legacy_script():
    """防漂移：_WSL_SUBDIRS 必须与旧 wsl_sources.sh 的候选清单逐项一致。"""

    assert sd._WSL_SUBDIRS == (
        "Desktop", "Downloads", "Documents", "Pictures", "Videos", "Music",
        "Documents/WeChat Files", "Documents/xwechat_files",
        "Documents/Tencent Files", "Documents/DingDing",
        "Pictures/Camera Roll", "Pictures/Screenshots",
        "AppData/Roaming/Tencent/WeChat",
    )


def test_wsl_candidates_probe():
    alice = "/mnt/c/Users/alice"
    dirs = [alice, f"{alice}/Desktop", f"{alice}/Documents/WeChat Files"]
    exists, readable = _fake_fs(dirs)
    candidates = sd.wsl_candidates(
        "/mnt/c/Users", exists=exists, readable=readable, listdir=lambda _p: ["alice"]
    )
    by_path = {_norm(c.path): c for c in candidates}
    assert set(by_path) == {f"{alice}/Desktop", f"{alice}/Documents/WeChat Files"}
    assert by_path[f"{alice}/Desktop"].kind == "user_dir"
    assert by_path[f"{alice}/Documents/WeChat Files"].kind == "im"


# ---------------- macOS ----------------

_MAC_HOME = "/Users/x"
_WECHAT_AS = (
    "/Users/x/Library/Containers/com.tencent.xinWeChat/Data/Library/"
    "Application Support/com.tencent.xinWeChat"
)


def test_macos_candidates_include_im_sandbox_and_icloud():
    message_temp = f"{_WECHAT_AS}/2.0b4.0.9/MessageTemp"
    dirs = [
        f"{_MAC_HOME}/Desktop",
        f"{_MAC_HOME}/Library/Mobile Documents/com~apple~CloudDocs",
        _WECHAT_AS,
        message_temp,
        f"{_MAC_HOME}/Library/Application Support/DingTalk",
    ]
    exists, readable = _fake_fs(dirs)
    candidates = sd.macos_candidates(
        _MAC_HOME, exists=exists, readable=readable, wechat_glob=lambda root: [message_temp]
    )
    by_path = {_norm(c.path): c for c in candidates}
    assert by_path[message_temp].kind == "im"
    # 深层命中 MessageTemp 后不再退回 Application Support 根
    assert _WECHAT_AS not in by_path
    assert by_path[f"{_MAC_HOME}/Library/Mobile Documents/com~apple~CloudDocs"].kind == "cloud"
    assert by_path[f"{_MAC_HOME}/Library/Application Support/DingTalk"].kind == "im"


def test_macos_candidates_wechat_fallback_to_app_support_root():
    dirs = [f"{_MAC_HOME}/Desktop", _WECHAT_AS]
    exists, readable = _fake_fs(dirs)
    candidates = sd.macos_candidates(_MAC_HOME, exists=exists, readable=readable, wechat_glob=lambda root: [])
    paths = {_norm(c.path) for c in candidates}
    assert _WECHAT_AS in paths


def test_macos_candidates_tcc_unreadable():
    dirs = [f"{_MAC_HOME}/Desktop", f"{_MAC_HOME}/Documents"]
    exists, readable = _fake_fs(dirs, unreadable=[f"{_MAC_HOME}/Documents"])
    candidates = sd.macos_candidates(_MAC_HOME, exists=exists, readable=readable)
    flags = {c.path: c.readable for c in candidates}
    assert flags[f"{_MAC_HOME}/Documents"] is False


# ---------------- Linux ----------------


def test_parse_user_dirs_file_expands_home():
    text = (
        '# 注释行\n'
        'XDG_DESKTOP_DIR="$HOME/桌面"\n'
        'XDG_DOWNLOAD_DIR="$HOME/Downloads"\n'
        'XDG_DOCUMENTS_DIR=/data/docs\n'
        'broken line without equals\n'
    )
    result = sd.parse_user_dirs_file(text, "/home/u")
    assert result == {
        "DESKTOP": "/home/u/桌面",
        "DOWNLOAD": "/home/u/Downloads",
        "DOCUMENTS": "/data/docs",
    }


def test_linux_candidates_from_user_dirs_file():
    exists, readable = _fake_fs(["/home/u/桌面", "/home/u/Downloads"])
    candidates = sd.linux_candidates(
        "/home/u",
        exists=exists,
        readable=readable,
        user_dirs_text='XDG_DESKTOP_DIR="$HOME/桌面"\nXDG_DOWNLOAD_DIR="$HOME/Downloads"\n',
    )
    assert {_norm(c.path) for c in candidates} == {"/home/u/桌面", "/home/u/Downloads"}


def test_linux_candidates_xdg_lookup():
    exists, readable = _fake_fs(["/home/u/Desktop"])
    candidates = sd.linux_candidates(
        "/home/u",
        exists=exists,
        readable=readable,
        xdg_lookup=lambda key: "/home/u/Desktop" if key == "DESKTOP" else None,
    )
    assert [_norm(c.path) for c in candidates] == ["/home/u/Desktop"]


def test_linux_candidates_fallback_to_home_subdirs():
    exists, readable = _fake_fs(["/home/u/Desktop", "/home/u/Music"])
    candidates = sd.linux_candidates("/home/u", exists=exists, readable=readable)
    assert {_norm(c.path) for c in candidates} == {"/home/u/Desktop", "/home/u/Music"}


# ---------------- 平台分发 ----------------


def test_candidates_for_platform_dispatch():
    dirs = [f"{_WIN_HOME}/Desktop", f"{_MAC_HOME}/Desktop", "/home/u/Desktop"]
    exists, readable = _fake_fs(dirs)
    win = sd.candidates_for_platform("windows", home=_WIN_HOME, exists=exists, readable=readable)
    mac = sd.candidates_for_platform("macos", home=_MAC_HOME, exists=exists, readable=readable)
    linux = sd.candidates_for_platform("linux", home="/home/u", exists=exists, readable=readable)
    assert [_norm(c.path) for c in win] == [f"{_WIN_HOME}/Desktop"]
    assert [_norm(c.path) for c in mac] == [f"{_MAC_HOME}/Desktop"]
    assert [_norm(c.path) for c in linux] == ["/home/u/Desktop"]


def test_candidates_for_platform_auto_detect(monkeypatch):
    monkeypatch.setenv("FILE_MEMORY_WINDOWS_NATIVE_PATH_MAP", "true")
    exists, readable = _fake_fs([f"{_WIN_HOME}/Desktop"])
    candidates = sd.candidates_for_platform(None, home=_WIN_HOME, exists=exists, readable=readable)
    assert [_norm(c.path) for c in candidates] == [f"{_WIN_HOME}/Desktop"]


# ---------------- 交互解析 / .env 写入 ----------------


def test_parse_selection_tokens():
    assert sd.parse_selection("1 3,4", 5) == [0, 2, 3]
    assert sd.parse_selection("2 2", 5) == [1]  # 去重
    assert sd.parse_selection("a", 3) == [0, 1, 2]
    assert sd.parse_selection("q", 3) is None
    assert sd.parse_selection("Q", 3) is None


def test_parse_selection_invalid():
    with pytest.raises(ValueError):
        sd.parse_selection("x", 3)
    with pytest.raises(ValueError):
        sd.parse_selection("9", 3)
    with pytest.raises(ValueError):
        sd.parse_selection("0", 3)
    with pytest.raises(ValueError):
        sd.parse_selection("", 3)


def test_render_source_dir_keeps_spaces_unquoted():
    assert sd.render_source_dir(["/a/WeChat Files", "/b"]) == "/a/WeChat Files,/b"


def test_update_env_file_replaces_all_matching_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "CLOUD_ENABLED=false\n"
        "FILE_MEMORY_SOURCE_DIR=/old/a\n"
        "# 注释\n"
        "FILE_MEMORY_SOURCE_DIR=/old/b\n",
        encoding="utf-8",
    )
    replaced = sd.update_env_file(env, "FILE_MEMORY_SOURCE_DIR", "/new/a,/new/b")
    assert replaced is True
    content = env.read_text(encoding="utf-8")
    assert content.count("FILE_MEMORY_SOURCE_DIR=") == 2
    assert "/new/a,/new/b" in content
    assert "/old" not in content
    # 其他键与注释保持原样
    assert "CLOUD_ENABLED=false" in content
    assert "# 注释" in content


def test_update_env_file_appends_when_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OTHER=1\n", encoding="utf-8")
    replaced = sd.update_env_file(env, "FILE_MEMORY_SOURCE_DIR", "/a")
    assert replaced is False
    assert env.read_text(encoding="utf-8") == "OTHER=1\nFILE_MEMORY_SOURCE_DIR=/a\n"


def test_update_env_file_creates_file(tmp_path):
    env = tmp_path / "sub" / ".env"
    sd.update_env_file(env, "FILE_MEMORY_SOURCE_DIR", "/a")
    assert env.read_text(encoding="utf-8") == "FILE_MEMORY_SOURCE_DIR=/a\n"


def test_update_env_file_writes_windows_path_raw(tmp_path):
    """值不加引号：Windows 反斜杠在 dotenv 双引号值里会被转义，必须裸写。"""

    env = tmp_path / ".env"
    sd.update_env_file(env, "FILE_MEMORY_SOURCE_DIR", "C:\\Users\\x\\Desktop,D:\\data")
    assert env.read_text(encoding="utf-8") == "FILE_MEMORY_SOURCE_DIR=C:\\Users\\x\\Desktop,D:\\data\n"
