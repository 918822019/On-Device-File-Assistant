"""Edge-cloud orchestrator."""

import logging
from dataclasses import dataclass

from .cloud_client import CloudClient, CloudResult
from .config import CloudConfig, EdgeConfig, RouteConfig
from .edge_runtime import EdgeInferenceResult, EdgeRuntime
from .routing import EdgeFallbackPolicy, RoutingPolicy


@dataclass
class RoutingResult:
    final_text: str
    used_source: str
    escalated: bool
    reason: str
    edge_confidence: float | None = None
    model: str = ""


class EdgeCloudOrchestrator:
    def __init__(self, edge_cfg: EdgeConfig, cloud_cfg: CloudConfig, route_cfg: RouteConfig):
        self.logger = logging.getLogger("orchestrator")
        self.edge_cfg = edge_cfg
        self.cloud_cfg = cloud_cfg
        self.route_cfg = route_cfg

        self.edge = None
        self.cloud = CloudClient(cloud_cfg)
        self.route_policy = RoutingPolicy(route_cfg)
        self.fallback_policy = EdgeFallbackPolicy(route_cfg)

        self._init_edge_runtime()

    def _init_edge_runtime(self) -> None:
        try:
            self.edge = EdgeRuntime(self.edge_cfg)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Edge 初始化失败: %s", exc)
            self.edge = None

    def ask(self, messages: list[dict], force_cloud: bool = False) -> RoutingResult:
        decision = self.route_policy.next_decision(messages, force_cloud=force_cloud)

        # Route directly to cloud and keep behavior deterministic.
        if decision.use_edge is False and self.cloud_cfg.enabled:
            return self._ask_cloud(messages, decision.reason)

        # tiny-first is disabled and cloud not available.
        if not decision.use_edge and not self.cloud_cfg.enabled:
            return self._edge_only(messages, reason=decision.reason)

        # Edge-first path.
        edge_resp = self._try_edge(messages)
        if edge_resp is None:
            if self.cloud_cfg.enabled:
                return self._ask_cloud(messages, reason="edge_unavailable")
            return self._edge_unavailable_msg()

        if self.cloud_cfg.enabled and decision.confidence_threshold_check and self.fallback_policy.should_fallback(
            edge_resp.text, edge_resp.confidence
        ):
            return self._ask_cloud(
                messages,
                reason="edge_low_confidence",
                edge_resp=edge_resp,
            )

        return RoutingResult(
            final_text=edge_resp.text,
            used_source="edge",
            escalated=False,
            reason="edge_ok",
            edge_confidence=edge_resp.confidence,
            model=edge_resp.used_model,
        )

    def _ask_cloud(
        self,
        messages: list[dict],
        reason: str,
        edge_resp: EdgeInferenceResult | None = None,
    ) -> RoutingResult:
        cloud_resp = self._call_cloud(messages)
        return RoutingResult(
            final_text=cloud_resp.text,
            used_source="cloud",
            escalated=True,
            reason=reason,
            edge_confidence=edge_resp.confidence if edge_resp else None,
            model=cloud_resp.model,
        )

    def _edge_only(self, messages: list[dict], reason: str) -> RoutingResult:
        if self.edge is None or not self.edge.ready:
            return self._edge_unavailable_msg(reason=reason)

        try:
            edge_resp = self.edge.generate(messages)
            return RoutingResult(
                final_text=edge_resp.text,
                used_source="edge",
                escalated=False,
                reason="edge_ok",
                edge_confidence=edge_resp.confidence,
                model=edge_resp.used_model,
            )
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Edge 推理失败（云端关闭）: %s", exc)
            return self._edge_error_msg()

    def _try_edge(self, messages: list[dict]) -> EdgeInferenceResult | None:
        if self.edge is None or not self.edge.ready:
            return None

        try:
            return self.edge.generate(messages)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Edge 推理失败: %s", exc)
            return None

    def _edge_unavailable_msg(self, reason: str = "edge_unavailable_cloud_disabled") -> RoutingResult:
        return RoutingResult(
            final_text="端侧模型未就绪且未开启云端兜底，当前仅支持本地 tiny llm 推理。",
            used_source="edge",
            escalated=False,
            reason=reason,
            model=self.edge_cfg.model_id,
        )

    def _edge_error_msg(self, reason: str = "edge_error_cloud_disabled") -> RoutingResult:
        return RoutingResult(
            final_text="端侧推理异常，当前未开启云端兜底。",
            used_source="edge",
            escalated=False,
            reason=reason,
            model=self.edge_cfg.model_id,
        )

    def _call_cloud(self, messages: list[dict]) -> CloudResult:
        return self.cloud.complete(messages)
