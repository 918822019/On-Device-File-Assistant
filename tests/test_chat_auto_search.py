"""Chat route must return real file evidence instead of a generic model answer."""

from types import SimpleNamespace

import pytest

from edge_cloud_agent.personal_search.schemas import FileSearchCandidate, FileSearchResponse
from edge_cloud_agent.routers import chat as chat_router
from edge_cloud_agent.routers.schemas import ChatRequest

# 默认参数只在函数定义时求值一次，写成 service=object() 会让所有调用共享同一个
# 实例（B008）。这里只需要一个「非 None 的占位服务」，故提到模块级把意图写明。
_PLACEHOLDER_SERVICE = object()


def _result(query: str, score: float = 0.41, clues: list[str] | None = None):
    return FileSearchResponse(
        query=query,
        session_id="test-session-123",
        state="needs_clarification",
        needs_disambiguation=True,
        candidates=[FileSearchCandidate(
            file_id="fm_test123", title="MNN部署笔记", source_app="local",
            doc_type="document", mime_type="text/plain", file_uri="file:///tmp/mnn.txt",
            captured_at=None, score=score, evidence="文件名命中 MNN",
            preview="", matched_clues=clues if clues is not None else ["mnn"],
        )],
    )


def _search_stub(calls, factory=_result):
    """替代 search_flow.start_file_search，记录调用次数并返回固定候选。"""

    def search(service, query, **kwargs):
        calls["search"] += 1
        return factory(query)

    return search


def _ask_stub(calls):
    def ask(**kwargs):
        calls["model"] += 1
        return SimpleNamespace(
            used_source="edge", escalated=False, reason="edge_ok",
            model="test-model", edge_confidence=0.8, final_text="model reply",
        )

    return ask


def _request(orchestrator, service=_PLACEHOLDER_SERVICE):
    return SimpleNamespace(
        state=SimpleNamespace(trace_id="trace-chat"),
        app=SimpleNamespace(state=SimpleNamespace(
            personal_file_service=service,
            personal_file_refresher=None,
            metrics=None,
            orchestrator=orchestrator,
        )),
    )


@pytest.fixture
def backend(monkeypatch):
    calls = {"search": 0, "model": 0}
    monkeypatch.setattr(chat_router, "start_file_search", _search_stub(calls))
    return _request(SimpleNamespace(ask=_ask_stub(calls))), calls


def test_short_file_topic_returns_indexed_files_without_model(backend):
    request, calls = backend
    response = chat_router.chat(ChatRequest(message="MNN部署"), request)
    assert response.source == "file_search"
    assert response.search.candidates[0].title == "MNN部署笔记"
    assert calls == {"search": 1, "model": 0}


def test_general_question_and_force_cloud_keep_model_route(backend):
    request, calls = backend
    for message, force_cloud in [("什么是光合作用？", False), ("MNN部署", True)]:
        response = chat_router.chat(ChatRequest(message=message, force_cloud=force_cloud), request)
        assert response.source == "edge"
        assert response.search is None
    assert calls == {"search": 0, "model": 2}


def test_weak_match_falls_back_but_explicit_file_request_never_hallucinates(backend, monkeypatch):
    request, calls = backend
    monkeypatch.setattr(
        chat_router,
        "start_file_search",
        _search_stub(calls, lambda query: _result(query, 0.1, [])),
    )
    assert chat_router.chat(ChatRequest(message="端侧推理"), request).source == "edge"
    response = chat_router.chat(ChatRequest(message="帮我找端侧推理的文件"), request)
    assert response.source == "file_search"
    assert response.search is not None
    assert calls["model"] == 1


def test_missing_index_degrades_to_model_route(monkeypatch):
    """索引未装配时跳过检索走模型，不报 503。"""

    calls = {"search": 0, "model": 0}
    monkeypatch.setattr(chat_router, "start_file_search", _search_stub(calls))
    request = _request(SimpleNamespace(ask=_ask_stub(calls)), service=None)

    response = chat_router.chat(ChatRequest(message="MNN部署"), request)
    assert response.source == "edge"
    assert response.search is None
    assert calls == {"search": 0, "model": 1}
