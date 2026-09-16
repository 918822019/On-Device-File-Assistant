"""Routing policy for deciding edge vs cloud execution path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    """Decision object consumed by orchestrator."""

    use_edge: bool
    reason: str
    escalated: bool
    confidence_threshold_check: bool = False


class RoutingPolicy:
    """Pure policy engine independent of runtime implementations."""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _extract_latest_prompt(messages: list[dict]) -> str:
        for message in reversed(messages):
            content = message.get("content")
            if content:
                return str(content)
        return ""

    def _contains_force_cloud_keyword(self, prompt: str) -> bool:
        lowered = prompt.lower()
        return any(keyword.lower() in lowered for keyword in self.config.force_cloud_keywords)

    def next_decision(self, messages: list[dict], force_cloud: bool = False) -> RouteDecision:
        latest_input = self._extract_latest_prompt(messages)

        # Strategy A: user explicitly asks for cloud or prompt is judged sensitive/long.
        if force_cloud or self._needs_cloud_first(latest_input):
            return RouteDecision(use_edge=False, reason="policy_force_cloud_first", escalated=True)

        # Strategy B: global policy says do not use tiny-first.
        if not self.config.use_tiny_first:
            return RouteDecision(use_edge=False, reason="policy_tiny_disabled", escalated=True)

        return RouteDecision(use_edge=True, reason="edge_ok", escalated=False, confidence_threshold_check=True)

    def _needs_cloud_first(self, prompt: str) -> bool:
        if len(prompt) > self.config.max_input_chars:
            return True
        return self._contains_force_cloud_keyword(prompt)


class EdgeFallbackPolicy:
    """Policy used after edge attempt to decide whether to escalate."""

    def __init__(self, config):
        self.config = config

    def should_fallback(self, edge_text: str, edge_confidence: float) -> bool:
        is_short = len(edge_text) < 8
        low_conf = edge_confidence < self.config.min_edge_confidence
        return is_short or low_conf
