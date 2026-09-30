"""架构层边界：依赖方向必须单向，且 Agent 不依赖 HTTP 也能被调用。

为什么用 AST 静态扫而不是靠 import 冒烟测试：`import` 冒烟只能发现「模块坏了」，
发现不了「模块依赖错了方向」。层边界一旦破了不会有任何报错 —— 只会在几个月后
表现为「改 common 要跑全量测试」「删不掉某个业务模块」这类说不清的成本。

依赖方向（见 docs/PROJECT_STRUCTURE.md）：

    routers → agents / personal_search / expense / analytics → llm / engines → common

下面的 ALLOWED_EDGES 把**当前实际存在**的边固化成白名单。新增一条边就会红，
这是刻意的：跨层依赖应该是一个需要显式决定的动作，而不是顺手 import 的副作用。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from edge_cloud_agent.agents.chat import ChatAgent
from edge_cloud_agent.personal_search.schemas import FileSearchCandidate, FileSearchResponse

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "edge_cloud_agent"

LAYERS = {
    "common",
    "engines",
    "llm",
    "agents",
    "personal_search",
    "expense",
    "analytics",
    "routers",
}

#: 允许存在的跨包依赖边（源 → 目标集合）。同包内部依赖不在此列。
ALLOWED_EDGES: dict[str, set[str]] = {
    # 最底层：只依赖 stdlib 与三方库
    "common": set(),
    "engines": set(),
    # 端云编排需要端侧引擎
    "llm": {"engines"},
    # 业务层
    "analytics": {"common"},
    "expense": {"common", "engines"},
    "personal_search": {"common", "engines", "analytics"},
    "agents": {"llm", "personal_search"},
    # HTTP 适配层可以看所有东西
    "routers": {"agents", "analytics", "common", "engines", "expense", "personal_search"},
}


def _layer_of(path: Path) -> str | None:
    parts = path.relative_to(SRC_ROOT).parts
    return parts[0] if len(parts) > 1 and parts[0] in LAYERS else None


def _resolve_relative(node: ast.ImportFrom, parts: tuple[str, ...]) -> str:
    """把相对导入还原成 edge_cloud_agent 下的绝对模块路径。"""

    base = list(parts[:-1])
    if node.level > 1:
        base = base[: -(node.level - 1)]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                mods.append(_resolve_relative(node, path.relative_to(SRC_ROOT.parent).parts))
            elif node.module:
                mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
    return mods


def _actual_edges() -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {layer: set() for layer in LAYERS}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        src = _layer_of(path)
        if src is None:
            continue
        for mod in _imported_modules(path):
            if not mod.startswith("edge_cloud_agent."):
                continue
            target = mod.split(".")[1]
            if target in LAYERS and target != src:
                edges[src].add(target)
    return edges


def test_no_dependency_points_upward_or_sideways():
    actual = _actual_edges()
    violations = {
        src: sorted(targets - ALLOWED_EDGES.get(src, set()))
        for src, targets in actual.items()
        if targets - ALLOWED_EDGES.get(src, set())
    }
    assert not violations, (
        f"出现了未登记的跨层依赖: {violations}。"
        f"若这条依赖是有意的，请更新 ALLOWED_EDGES 并同步 docs/PROJECT_STRUCTURE.md。"
    )


def test_common_depends_on_nothing_inside_the_package():
    """common/ 是全包最底层，任何向上的依赖都会让它无法被单独复用与测试。"""

    assert _actual_edges()["common"] == set()


def test_documented_edges_still_exist():
    """白名单里已消失的边要删掉，否则白名单会慢慢变成一份过期的架构图。"""

    actual = _actual_edges()
    stale = {
        src: sorted(allowed - actual.get(src, set()))
        for src, allowed in ALLOWED_EDGES.items()
        if allowed - actual.get(src, set())
    }
    assert not stale, f"这些依赖已不存在，请从 ALLOWED_EDGES 删除: {stale}"


def test_legacy_runtime_shim_is_gone():
    """runtime/ 兼容垫片已删除，不得再被重新引入。

    它曾是 6 个文件、30 行纯 re-export，src/ 与 scripts/ 零引用，唯一使用者是
    验证垫片自身可用的测试 —— 即一份只为让自己继续存在而存在的代码。分层重构
    尚未发布过，`edge_cloud_agent.runtime.*` 从来不是对外契约。
    """

    assert not (SRC_ROOT / "runtime").exists(), "runtime/ 兼容垫片已被删除，不要恢复"
    with pytest.raises(ImportError):
        __import__("edge_cloud_agent.runtime")


def test_chat_agent_coordinates_search_and_llm_without_http():
    """Agent 层必须能在没有 HTTP 的情况下被调用（否则等于和业务逻辑焊死在路由上）。"""

    calls = {"model": 0}

    def ask(**kwargs):
        calls["model"] += 1
        return SimpleNamespace(
            used_source="edge", escalated=False, reason="edge_ok", model="test-llm",
            edge_confidence=0.8, final_text="model answer",
        )

    candidate = FileSearchCandidate(
        file_id="fm_file001", title="MNN部署笔记", source_app="local",
        doc_type="document", mime_type=None, file_uri=None,
        captured_at=None, score=0.4, evidence="文件名命中", preview="", matched_clues=["mnn"],
    )

    def search(query):
        return FileSearchResponse(
            query=query, session_id="session-123", state="needs_clarification",
            needs_disambiguation=True, candidates=[candidate],
        )

    agent = ChatAgent(SimpleNamespace(ask=ask), search_files=search)
    assert agent.answer("MNN部署").search.candidates[0].title == "MNN部署笔记"
    assert agent.answer("什么是光合作用？").text == "model answer"
    assert calls["model"] == 1
