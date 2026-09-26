"""
Cloud side client.
For simplicity and portability this uses OpenAI-compatible chat/completions endpoint.
"""

import json
import logging
from dataclasses import dataclass

import requests

from ..config import CloudConfig


@dataclass
class CloudResult:
    text: str
    model: str


class CloudClient:
    def __init__(self, cfg: CloudConfig) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger("cloud_client")

    def complete(self, messages: list[dict]) -> CloudResult:
        api_base = (self.cfg.api_base or "").strip().rstrip("/")
        if not api_base:
            raise RuntimeError(
                "CLOUD_API_BASE 未配置。启用云端（CLOUD_ENABLED=true）时必须显式指定 "
                "OpenAI 兼容接口地址。"
            )

        headers = {
            "Content-Type": "application/json",
        }
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
        }

        resp = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self.cfg.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"cloud response missing choices: {data}")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        model = data.get("model") or self.cfg.model
        return CloudResult(text=text.strip(), model=model)

