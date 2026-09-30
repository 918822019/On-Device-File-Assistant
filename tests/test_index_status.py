"""Index overview reports real coverage without exposing indexed content."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from edge_cloud_agent.personal_search.schemas import IndexStatusResponse
from edge_cloud_agent.routers.personal_search import router


def test_index_status_counts_nested_roots_once_and_reports_missing_root(tmp_path):
    root = tmp_path / "docs"
    nested = root / "plans"
    nested.mkdir(parents=True)
    missing = tmp_path / "missing"
    item = SimpleNamespace(file_path=str(nested / "private-note.md"), last_scanned_at="2026-09-29T12:00:00")
    store = SimpleNamespace(list_all=lambda: [item])
    service = SimpleNamespace(store=store, vector_index_ready=True)
    cfg = SimpleNamespace(
        source_dir=f"{root},{nested},{missing}",
        scan_recursive=True,
        scan_interval_seconds=120,
        max_search_query_len=120,
    )
    app = FastAPI()
    app.include_router(router)
    app.state.personal_file_service = service
    app.state.personal_file_cfg = cfg

    response = TestClient(app).get("/v1/search-agent/index-status")
    assert response.status_code == 200
    result = IndexStatusResponse.model_validate(response.json())
    assert result.indexed_files == 1
    assert [entry.indexed_files for entry in result.sources] == [0, 1, 0]
    assert [entry.available for entry in result.sources] == [True, True, False]
    assert result.last_indexed_at == item.last_scanned_at
    assert "private-note" not in response.text


def test_index_status_publishes_query_len_limit_from_config(tmp_path):
    """查询长度上限必须由 config 下发，前端据此设置 maxlength。

    此前前端硬编码 260、后端静默截断到 120：121~260 字符的尾部线索被无声丢弃，
    用户侧只表现为「搜不准」且无任何提示。让上限只有一个事实源即可根治漂移，
    所以这里断言它确实来自 cfg 而不是写死的常量。
    """

    store = SimpleNamespace(list_all=lambda: [])
    service = SimpleNamespace(store=store, vector_index_ready=False)
    cfg = SimpleNamespace(
        source_dir=str(tmp_path),
        scan_recursive=True,
        scan_interval_seconds=120,
        max_search_query_len=77,  # 刻意取一个不像默认值的数，防止断言被常量蒙对
    )
    app = FastAPI()
    app.include_router(router)
    app.state.personal_file_service = service
    app.state.personal_file_cfg = cfg

    result = IndexStatusResponse.model_validate(TestClient(app).get("/v1/search-agent/index-status").json())
    assert result.max_query_len == 77
