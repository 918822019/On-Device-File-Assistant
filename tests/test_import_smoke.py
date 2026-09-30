"""导入冒烟测试：任何模块 import 失败都必须立刻在此暴露。

为什么需要这条：项目未安装 linter，而 ruff/pyflakes 只做单文件分析，**不跨模块
解析导入** —— `from .service import _now_iso` 在语法上完全合法，静态检查看不出
service.py 里已经没有这个符号。真正能发现它的是「把每个模块都 import 一遍」，
或跑一次 pytest collection。

历史上正是这样漏掉的：一次拆分重构把 `_now_iso` 从 service.py 移到
common/time_utils.py，但 personal_search/ingest.py 的导入没跟着改，结果
`import edge_cloud_agent.main` 直接 ImportError —— **应用完全起不来**，而 10 个
模块连带失败、8 个测试文件无法 collection。因为 /health 之外的东西都没跑过，
这个状态在工作区里存活了整段时间。

本测试逐模块 import，把「某个模块坏了」变成一条指名道姓的失败，而不是让
调用方在 ImportError 的连锁里自己找根因。
"""

import importlib
import pkgutil

import pytest

import edge_cloud_agent


def _all_submodules() -> list[str]:
    names = [m.name for m in pkgutil.walk_packages(edge_cloud_agent.__path__, "edge_cloud_agent.")]
    assert names, "未发现任何子模块，walk_packages 的路径可能已失效"
    return sorted(names)


@pytest.mark.parametrize("module_name", _all_submodules())
def test_every_module_imports(module_name):
    """每个模块都必须能独立 import。

    用 parametrize 而非在单个测试里循环：失败时直接显示是哪个模块，
    且一个模块坏掉不会掩盖其他模块的状态。
    """

    importlib.import_module(module_name)


def test_app_object_is_constructible():
    """main.app 必须能装配出来 —— 这是 uvicorn 的入口，坏了就等于服务起不来。

    用 app.openapi()["paths"] 而不是遍历 app.routes：后者是内部结构，
    FastAPI 0.139 起 include_router 的结果被包成惰性 _IncludedRouter、
    不再有 .path 属性，直接遍历会漏掉全部 /v1 路由。openapi() 是公开契约，
    在 0.116 与 0.139 下都返回同样的 14 条路径。
    """

    main = importlib.import_module("edge_cloud_agent.main")
    assert main.app is not None
    paths = set(main.app.openapi().get("paths", {}))
    assert "/health" in paths, "应用未注册 /health，部署门禁会失败"
    v1 = {p for p in paths if p.startswith("/v1/")}
    assert v1, "应用未注册任何 /v1 业务路由"
    # 四条业务线的入口都必须在，少一条通常意味着 create_app 漏了 include_router
    for expected in ("/v1/chat", "/v1/embeddings", "/v1/search-agent/search", "/v1/metrics"):
        assert expected in paths, f"缺少关键路由 {expected}"


def test_module_inventory_is_not_shrinking():
    """模块数量骤降通常意味着重构删掉了包或 __init__ 导出断了。

    这个数字是个粗略的哨兵，不是精确契约：新增模块时把它调大即可。
    54 = 删除 runtime/ 兼容垫片（-6）、新增 common/vectors、common/jsonl_store、
    common/fs_scan、common/watch_loop（+4）之后的实际数量。
    """

    count = len(_all_submodules())
    assert count >= 54, f"只发现 {count} 个子模块，远少于预期，包结构可能已损坏"
