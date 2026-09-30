"""Web 前端与后端契约的静态校验。

前端是原生 ES module、无构建步骤、无 JS 测试框架，因此这里用 Python 直接读源文件
做静态断言。目的是锁住那条曾经长期漂移的链路：

    index.html 的静态 maxlength  ==  .env.example  ==  config 默认值

后两者的相等已由 tests/test_config_env_parity.py 保证，故此处只需把 HTML/JS 接到
同一条链上。历史问题：前端硬编码 260、后端静默截断到 120，121~260 字符的尾部
线索被无声丢弃，用户侧只表现为「搜不准」且无任何提示。

运行期上限现在由 /v1/search-agent/index-status 的 max_query_len 下发（见
test_index_status.py），静态值只是首屏与后端不可达时的兜底。
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = REPO_ROOT / "web"


def _read(*parts: str) -> str:
    return (WEB.joinpath(*parts)).read_text(encoding="utf-8")


def _backend_default_query_len() -> int:
    """从 .env.example 取 FILE_MEMORY_MAX_QUERY_LEN，作为全链路的一致基准。"""

    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    match = re.search(r"^FILE_MEMORY_MAX_QUERY_LEN\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    assert match, ".env.example 未定义 FILE_MEMORY_MAX_QUERY_LEN"
    return int(match.group(1))


def test_search_input_static_maxlength_matches_backend_default():
    limit = _backend_default_query_len()
    html = _read("index.html")
    match = re.search(r'<input[^>]*id="sq"[^>]*maxlength="(\d+)"', html)
    assert match, "index.html 里的 #sq 输入框没有 maxlength，超长查询会被后端静默截断"
    assert int(match.group(1)) == limit, (
        f"#sq 的静态 maxlength={match.group(1)} 与后端默认上限 {limit} 不一致；"
        "超出部分会被 service.start_search 截断掉"
    )


def test_core_js_limits_default_matches_backend_default():
    limit = _backend_default_query_len()
    core = _read("js", "core.js")
    match = re.search(r"export const limits\s*=\s*\{\s*maxQueryLen:\s*(\d+)", core)
    assert match, "core.js 未导出 limits.maxQueryLen，前端将失去统一的上限来源"
    assert int(match.group(1)) == limit, (
        f"core.js 的 limits.maxQueryLen 初值 {match.group(1)} 与后端默认上限 {limit} 不一致"
    )


def test_search_js_applies_server_provided_limit():
    """上限必须以后端下发值为准，否则改 config 后前端会重新漂移。"""

    search = _read("js", "search.js")
    assert "max_query_len" in search, "search.js 未读取 index-status 下发的 max_query_len"
    assert "limits.maxQueryLen" in search, "search.js 未把下发值写入共享的 limits"
    assert re.search(r"maxLength\s*=\s*d\.max_query_len", search), (
        "search.js 未用下发值设置输入框 maxLength"
    )


def test_chat_js_uses_shared_limit_instead_of_hardcoding():
    """聊天转搜索曾经硬编码 260，与后端 120 长期不一致。"""

    chat = _read("js", "chat.js")
    assert "limits.maxQueryLen" in chat, "chat.js 未使用共享的 limits.maxQueryLen"
    assert not re.search(r"length\s*>\s*260", chat), "chat.js 仍在硬编码 260 字符上限"


def test_modules_import_core_bare_so_limits_is_a_single_instance():
    """所有模块必须裸引 ./core.js。

    若某个模块改成 "./core.js?v=x"，浏览器模块注册表会把它当成另一个模块，
    limits 就会出现两份实例 —— search.js 写入的上限 chat.js 看不到，
    而且 cache-busting 也并不会因此生效（服务端已对 /web/ 发 no-cache）。
    """

    for name in ("chat.js", "search.js", "expense.js", "metrics.js"):
        source = _read("js", name)
        for match in re.finditer(r'from\s+"(\./core\.js[^"]*)"', source):
            assert match.group(1) == "./core.js", (
                f"js/{name} 以 {match.group(1)!r} 引入 core.js；必须裸引 ./core.js，"
                "否则 limits 会出现双实例"
            )
