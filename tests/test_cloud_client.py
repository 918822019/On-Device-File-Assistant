"""云端客户端：显式配置校验与响应解析。

CLOUD_ENABLED 默认 false，所以这条路径平时不跑；一旦启用，配置缺失必须给出
可读的中文报错，而不是把空值发给远端换回不透明的 4xx —— 那种错误会被
orchestrator 吞成通用降级提示，排查时完全看不出是配置问题。
"""

import json

import pytest

from edge_cloud_agent.config import CloudConfig
from edge_cloud_agent.llm import cloud_client as cloud_client_module
from edge_cloud_agent.llm.cloud_client import CloudClient

MESSAGES = [{"role": "user", "content": "你好"}]


def _cfg(**overrides) -> CloudConfig:
    """构造确定性的云端配置，避免受本机 .env 影响。"""

    base = {
        "enabled": True,
        "api_base": "http://127.0.0.1:9999/v1",
        "api_key": "test-key",
        "model": "test/model",
        "timeout_seconds": 5,
        "max_new_tokens": 64,
        "temperature": 0.5,
    }
    base.update(overrides)
    return CloudConfig(**base)


class _Response:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _stub_post(monkeypatch, payload, status_code: int = 200):
    """替换 requests.post，回传捕获到的请求以便断言。"""

    captured = {}

    def _fake_post(url, headers=None, data=None, timeout=None):
        captured.update(url=url, headers=headers or {}, body=json.loads(data or "{}"), timeout=timeout)
        return _Response(payload, status_code)

    monkeypatch.setattr(cloud_client_module.requests, "post", _fake_post)
    return captured


def test_missing_api_base_raises_actionable_error():
    with pytest.raises(RuntimeError) as exc:
        CloudClient(_cfg(api_base="")).complete(MESSAGES)
    assert "CLOUD_API_BASE" in str(exc.value)


def test_whitespace_only_api_base_is_treated_as_missing():
    with pytest.raises(RuntimeError) as exc:
        CloudClient(_cfg(api_base="   ")).complete(MESSAGES)
    assert "CLOUD_API_BASE" in str(exc.value)


def test_missing_model_raises_actionable_error():
    """model 为空时必须本地报错；否则会发出 "model": "" 换回远端 4xx。"""

    with pytest.raises(RuntimeError) as exc:
        CloudClient(_cfg(model="")).complete(MESSAGES)
    assert "CLOUD_MODEL_ID" in str(exc.value)


def test_missing_model_does_not_reach_network(monkeypatch):
    called = []
    monkeypatch.setattr(
        cloud_client_module.requests, "post", lambda *a, **k: called.append((a, k)) or _Response({})
    )
    with pytest.raises(RuntimeError):
        CloudClient(_cfg(model="  ")).complete(MESSAGES)
    assert called == [], "model 校验必须在发请求之前完成"


def test_complete_returns_text_and_model(monkeypatch):
    captured = _stub_post(
        monkeypatch,
        {"model": "served/model", "choices": [{"message": {"content": "  你好呀  "}}]},
    )

    result = CloudClient(_cfg()).complete(MESSAGES)

    assert result.text == "你好呀", "返回文本应被 strip"
    assert result.model == "served/model", "优先采用响应里的 model"
    assert captured["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["body"]["model"] == "test/model"
    assert captured["body"]["messages"] == MESSAGES
    assert captured["timeout"] == 5


def test_complete_falls_back_to_configured_model_when_response_omits_it(monkeypatch):
    _stub_post(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})
    assert CloudClient(_cfg()).complete(MESSAGES).model == "test/model"


def test_api_key_omitted_when_empty(monkeypatch):
    captured = _stub_post(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})
    CloudClient(_cfg(api_key="")).complete(MESSAGES)
    assert "Authorization" not in captured["headers"]


def test_empty_choices_raises_with_payload(monkeypatch):
    _stub_post(monkeypatch, {"choices": []})
    with pytest.raises(RuntimeError) as exc:
        CloudClient(_cfg()).complete(MESSAGES)
    assert "missing choices" in str(exc.value)


def test_http_error_propagates_for_orchestrator_to_absorb(monkeypatch):
    """CloudClient 不负责降级；orchestrator._call_cloud 会把它吞成 None。"""

    _stub_post(monkeypatch, {"detail": "boom"}, status_code=500)
    with pytest.raises(RuntimeError):
        CloudClient(_cfg()).complete(MESSAGES)
