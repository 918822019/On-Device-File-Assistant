"""应用入口拆分后锁定统一错误格式、请求追踪、静态资源缓存与 lifespan。"""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from edge_cloud_agent import bootstrap
from edge_cloud_agent.main import create_app


def _stub_lifespan(monkeypatch, initialize=None):
    """把 lifespan 的装配/停止换成桩，避免测试触发真实模型加载。"""

    monkeypatch.setattr(bootstrap, "initialize_services", initialize or (lambda app: None))
    monkeypatch.setattr(bootstrap, "stop_services", lambda app: None)


def _healthy_state(app):
    app.state.orchestrator = SimpleNamespace(edge=object(), cloud_cfg=SimpleNamespace(enabled=False))
    app.state.embedding_runtime = object()
    app.state.personal_file_service = object()
    app.state.expense_service = object()
    app.state.metrics = object()


def test_lifespan_initializes_services_before_serving_and_stops_them_on_exit(monkeypatch):
    events = []
    monkeypatch.setattr(bootstrap, "initialize_services", lambda app: events.append("start"))
    monkeypatch.setattr(bootstrap, "stop_services", lambda app: events.append("stop"))

    with TestClient(create_app()) as client:
        assert client.get("/health").json()["ok"] is True
        assert events == ["start"]
    assert events == ["start", "stop"]


def test_health_keeps_deploy_contract_when_nothing_initialized(monkeypatch):
    """未装配任何服务时仍须 200 + ok（部署门禁契约），但必须暴露 degraded。

    deploy.sh/service.sh 用 `curl -fsS | grep -q '"ok"'` 做启动门禁：-f 会把
    503 当失败并死等到超时，所以降级绝不能用状态码表达。
    """

    _stub_lifespan(monkeypatch)

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert '"ok"' in response.text
    payload = response.json()
    assert payload["ok"] is True
    assert payload["degraded"] is True
    assert not any(payload["services"].values())


def test_health_reports_ready_services_without_degradation(monkeypatch):
    _stub_lifespan(monkeypatch, _healthy_state)

    with TestClient(create_app()) as client:
        payload = client.get("/health").json()

    assert payload == {
        "ok": True,
        "degraded": False,
        "services": {
            "edge_llm": True,
            "embedding_runtime": True,
            "personal_file_service": True,
            "expense_service": True,
            "metrics": True,
        },
        "cloud_enabled": False,
    }


def test_health_flags_single_silent_degradation(monkeypatch):
    """embedding_runtime 单独失败会让搜索静默退化为纯关键词匹配，必须可见。"""

    def _initialize(app):
        _healthy_state(app)
        app.state.embedding_runtime = None

    _stub_lifespan(monkeypatch, _initialize)

    with TestClient(create_app()) as client:
        payload = client.get("/health").json()

    assert payload["degraded"] is True
    assert payload["services"]["embedding_runtime"] is False
    assert payload["services"]["personal_file_service"] is True


def test_health_treats_cloud_disabled_as_config_not_degradation(monkeypatch):
    """CLOUD_ENABLED 默认 false，是配置选择；若计入降级会恒定告警。"""

    def _initialize(app):
        _healthy_state(app)
        app.state.orchestrator = SimpleNamespace(edge=None, cloud_cfg=SimpleNamespace(enabled=True))

    _stub_lifespan(monkeypatch, _initialize)

    with TestClient(create_app()) as client:
        payload = client.get("/health").json()

    assert payload["cloud_enabled"] is True
    assert payload["services"]["edge_llm"] is False
    assert payload["degraded"] is True


def test_http_errors_keep_trace_and_public_error_shape():
    client = TestClient(create_app())
    trace_id = "refactor-check-123"
    missing = client.get("/v1/not-a-route", headers={"x-trace-id": trace_id})
    assert missing.status_code == 404
    assert missing.headers["x-trace-id"] == trace_id
    assert missing.json()["error"] == {
        "code": "not_found",
        "message": "Not Found",
        "status": 404,
        "trace_id": trace_id,
        "path": "/v1/not-a-route",
        "detail": None,
    }

    invalid = client.post("/v1/search-agent/search", json={"query": ""})
    assert invalid.status_code == 422
    error = invalid.json()["error"]
    assert error["code"] == "validation_error"
    assert error["trace_id"] == invalid.headers["x-trace-id"]
    assert error["detail"]["errors"]


def test_web_static_assets_keep_cache_rule_and_trace_header():
    response = TestClient(create_app()).get("/web/app.js")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-cache, must-revalidate"
    assert response.headers["x-trace-id"]
