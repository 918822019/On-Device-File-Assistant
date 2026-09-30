"""README / docs 契约：文档不得引用不存在的东西，也不得漏掉存在的东西。

为什么值得用测试盯着：README 是这套代码唯一的入口文档，而它的失效模式是
**静默的** —— 链接指向被移走的文件、配置表里写着一个已经被改名的环境变量、
`docs/` 里多出一篇没人索引的文档，都不会有任何报错，只会让下一个人照着做然后
踩空。本轮重写 README 时实测到三处：`docs/gemma4-qat-gguf-inventory.md` 从未被
索引、配置表里写的是 `FILE_MEMORY_MAX_SEARCH_QUERY_LEN`（真名是
`FILE_MEMORY_MAX_QUERY_LEN`）、`main.py` 的 docstring 声称「22 个端点」
（实际 14 个）。

这里刻意只校验**可机械判定的事实**（存在性、名字、数量），不校验散文表述 ——
后者会随重构正常演化，钉死只会让测试变成阻碍。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DOCS = ROOT / "docs"
SRC = ROOT / "src"
COMMON = SRC / "edge_cloud_agent" / "common"

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_BACKTICK_ENV_RE = re.compile(r"`([A-Z][A-Z0-9_]{3,})`")


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _env_vars_read_by_code() -> set[str]:
    """扫 src/ 与 scripts/ 里所有被读取的环境变量名。

    用 AST 而不是 grep：`os.getenv(KEY, default)` 与项目自己的
    `_env_int/_env_float/_env_bool` 包装都要算，而 grep 会把注释和文档里的
    示例也算进来。
    """

    names: set[str] = set()
    getters = {"getenv", "environ", "_env_int", "_env_float", "_env_bool", "_env_str"}
    for path in [*SRC.rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if fname not in getters or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.add(first.value)
    return names


def _env_example_keys() -> set[str]:
    keys: set[str] = set()
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0].strip())
    return keys


# --------------------------------------------------------------------- 链接与索引


def test_every_relative_link_in_readme_resolves():
    """README 里的每个相对链接都必须指向真实存在的文件。"""

    broken = []
    for target in _LINK_RE.findall(_readme()):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0]
        if not path:
            continue
        if not (ROOT / path).exists():
            broken.append(target)
    assert not broken, f"README 里这些链接指向不存在的文件: {broken}"


def test_every_doc_is_indexed_in_readme():
    """docs/ 下的每一篇都必须在 README 的文档索引里出现。

    漏索引的文档等于不存在：没人会去 ls docs/。
    """

    text = _readme()
    linked = {t.split("#", 1)[0] for t in _LINK_RE.findall(text)}
    missing = sorted(
        f"docs/{p.name}" for p in DOCS.glob("*.md") if f"docs/{p.name}" not in linked
    )
    assert not missing, f"这些文档没有被 README 索引: {missing}"


def test_readme_index_has_no_dead_entries():
    """反向：索引里列出的 docs/ 条目都必须还存在（防止删文档忘了删索引）。"""

    linked = {t.split("#", 1)[0] for t in _LINK_RE.findall(_readme())}
    dead = sorted(t for t in linked if t.startswith("docs/") and not (ROOT / t).exists())
    assert not dead, f"README 索引里这些文档已不存在: {dead}"


# --------------------------------------------------------------------- 配置表


def test_env_vars_named_in_readme_are_real():
    """README 里反引号包起来的全大写标识符必须是真实的环境变量名。

    只校验形如 `FILE_MEMORY_*` / `EDGE_*` / `CLOUD_*` / `ROUTE_*` / `EXPENSE_*` /
    `APP_*` 的 token —— 这些前缀覆盖了 config.py 的全部命名空间，而其它全大写
    词（如 `README` 里提到的 `MAX_PATH`、`EXDEV`）不是本项目的配置项。
    """

    prefixes = ("FILE_MEMORY_", "EDGE_", "CLOUD_", "ROUTE_", "EXPENSE_", "APP_")
    known = _env_vars_read_by_code() | _env_example_keys()
    mentioned = {
        m for m in _BACKTICK_ENV_RE.findall(_readme()) if m.startswith(prefixes)
    }
    unknown = sorted(mentioned - known)
    assert not unknown, (
        f"README 提到但代码与 .env.example 都不认识的环境变量: {unknown}"
    )


def test_readme_config_table_covers_every_env_group():
    """config.py 里每个前缀分组都应在 README 的配置表里被提到（哪怕只列关键项）。

    不要求逐个变量都列（.env.example 才是权威参考），但整组消失意味着
    README 的配置章节已经跟不上代码。
    """

    prefixes = {"FILE_MEMORY_", "EDGE_", "CLOUD_", "ROUTE_", "EXPENSE_", "APP_"}
    text = _readme()
    covered = {p for p in prefixes if p in text}
    assert covered == prefixes, f"README 配置章节漏掉了这些分组: {sorted(prefixes - covered)}"


# --------------------------------------------------------------------- 模块地图


def test_common_module_table_matches_the_package():
    """README 的 common/ 表格与 common/ 里的实际模块必须一一对应。

    双向校验：表里列的模块必须存在（防改名后没同步），实际存在的模块必须在
    表里（防新增模块后忘了写文档 —— common/ 是全包最底层，它的清单就是
    「有哪些能力可以复用」的唯一索引）。
    """

    text = _readme()
    # 只取 common/ 那张表的行（首列是反引号包起来的模块名）
    section = text.split("### `common/`", 1)[1].split("\n---", 1)[0]
    listed = set(re.findall(r"^\| `([a-z_]+)` \|", section, re.MULTILINE))

    actual = {p.stem for p in COMMON.glob("*.py") if p.stem != "__init__"}
    assert listed, "没能从 README 解析出 common/ 模块表，表格格式可能变了"
    assert listed - actual == set(), f"README 列了但已不存在的模块: {sorted(listed - actual)}"
    assert actual - listed == set(), f"存在但 README 没写的模块: {sorted(actual - listed)}"


def test_modules_named_in_the_main_map_exist():
    """模块地图里点名的每个包/文件都必须存在。"""

    text = _readme()
    section = text.split("## 模块地图", 1)[1].split("### `common/`", 1)[0]
    named = set(re.findall(r"^\| `([a-z_/]+\.py|[a-z_]+/)` \|", section, re.MULTILINE))
    assert named, "没能从 README 解析出模块地图，表格格式可能变了"

    base = SRC / "edge_cloud_agent"
    missing = []
    for entry in sorted(named):
        rel = entry.rstrip("/")
        # 包内模块解析到 src/edge_cloud_agent/ 下；android/ 与 scripts/ 这类
        # 顶层目录解析到仓库根
        if not (base / rel).exists() and not (ROOT / rel).exists():
            missing.append(entry)
    assert not missing, f"README 模块地图里这些路径不存在: {missing}"


# --------------------------------------------------------------------- 数量口径


def _actual_endpoint_count() -> int:
    from edge_cloud_agent.main import app

    return len(app.openapi().get("paths", {}))


def test_readme_endpoint_count_is_accurate():
    """README 里写的端点数量必须与实际注册数一致。

    main.py 的 docstring 曾长期声称「22 个端点」而实际是 14 个 —— 这类数字
    一旦写进文档就会被后续作者当成事实引用（本项目里它被引用了两处）。
    """

    actual = _actual_endpoint_count()
    claimed = {int(n) for n in re.findall(r"(\d+)\s*个端点", _readme())}
    assert claimed, "README 里没有提到端点数量，若刻意如此请删除本用例"
    assert claimed == {actual}, f"README 声称端点数为 {claimed}，实际是 {actual}"


def test_main_docstring_endpoint_count_is_accurate():
    """同上，但盯的是 main.py 里 /health 的 docstring（它也被引用过）。"""

    source = (SRC / "edge_cloud_agent" / "main.py").read_text(encoding="utf-8")
    claimed = {int(n) for n in re.findall(r"(\d+)\s*个端点", source)}
    actual = _actual_endpoint_count()
    assert claimed, "main.py 里没有提到端点数量"
    assert claimed == {actual}, f"main.py docstring 声称 {claimed}，实际是 {actual}"


def test_readme_states_the_health_contract():
    """/health 的「永远 200 + ok:true」契约必须在 README 里写清楚。

    这条契约不显然且反直觉（降级了为什么不返回 503？），而 deploy.sh 与
    service.sh 的 `curl -fsS` 门禁依赖它。文档里不写明，下一个人「顺手修好」
    状态码就会让部署脚本死等到 900s 超时。

    断言范围限定在 `/health` 那一节，而不是全文搜关键词：全文搜 "degraded"
    只要那个词还在别处出现就会通过，等于什么都没测（第一版就是这么假的）。
    """

    text = _readme()
    marker = '"degraded"'
    assert marker in text, "README 里没有 /health 的响应示例"
    section = text.split(marker, 1)[1]
    # 截到下一个二/三级标题为止，得到「示例 + 紧随其后的解释」这一段
    section = re.split(r"\n#{2,3} ", section, maxsplit=1)[0]

    assert "200" in section, "/health 一节没有说明状态码始终是 200"
    assert "curl" in section or "-fsS" in section, (
        "/health 一节没有说明部署脚本依赖这个契约"
    )
    assert "degraded" in section


@pytest.mark.parametrize("doc", sorted(p.name for p in DOCS.glob("*.md")))
def test_docs_are_not_empty(doc: str):
    """哨兵：docs/ 里不得出现空文件（曾有过一个 0 字节的上游 README 害得
    只能自己去解析 GGUF 头部）。"""

    content = (DOCS / doc).read_text(encoding="utf-8").strip()
    assert len(content) > 200, f"docs/{doc} 只有 {len(content)} 字符，疑似占位空文件"
