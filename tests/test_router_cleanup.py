"""路由清理后保持搜索选择语义和报销服务错误码。"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edge_cloud_agent.personal_search.schemas import FileSearchCandidate
from edge_cloud_agent.routers.expense import router as expense_router
from edge_cloud_agent.routers.personal_search import router as search_router


def test_search_and_clarify_keep_distinct_default_selection():
    candidate = FileSearchCandidate(
        file_id="file-123456", title="note", source_app="local", doc_type="file",
        mime_type=None, file_uri="file:///note", captured_at=None, score=0.9,
        evidence="match", preview="note",
    )

    class SearchService:
        def start_search(self, query, **kwargs):
            return "session-123456", [candidate], "resolved", False, None

        def continue_search(self, **kwargs):
            return "session-123456", "note", [candidate], "resolved", False, None, None

    app = FastAPI()
    app.include_router(search_router)
    app.state.personal_file_service = SearchService()
    client = TestClient(app)

    first = client.post("/v1/search-agent/search", json={"query": "note"})
    clarified = client.post("/v1/search-agent/clarify", json={"session_id": "session-123456", "reply": "note"})
    assert first.status_code == clarified.status_code == 200
    assert first.json()["selected_file_id"] == candidate.file_id
    assert clarified.json()["selected_file_id"] is None
    assert first.json()["next_action_suggestions"][-1] == "对比版本"
    assert "对比版本" not in clarified.json()["next_action_suggestions"]


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("collect", {"raw_text": "receipt"}),
        ("search", {}),
        ("export", {"claim_id": "claim-1"}),
        ("correct", {"material_id": "item-1"}),
        ("rebuild-index", {}),
    ],
)
def test_expense_endpoints_return_503_when_service_is_unavailable(path, payload):
    app = FastAPI()
    app.include_router(expense_router)
    response = TestClient(app).post(f"/v1/expense/{path}", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"] == "报销服务未就绪，请检查数据存储/配置后重启服务。"
