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
        """走云端。云端失败时依次回退：已有端侧结果 -> 端侧推理 -> 明确降级提示。

        云端本身是兜底路径，因此它的失败绝不能向上抛成 HTTP 500——
        那会比不启用云端更糟。
        """
        cloud_resp = self._call_cloud(messages)
        if cloud_resp is not None:
            return RoutingResult(
                final_text=cloud_resp.text,
                used_source="cloud",
                escalated=True,
                reason=reason,
                edge_confidence=edge_resp.confidence if edge_resp else None,
                model=cloud_resp.model,
            )

        # 云端失败但端侧已经给出结果（低置信度升级场景）：保留端侧结果
        if edge_resp is not None:
            self.logger.warning("云端失败，保留端侧结果: reason=%s", reason)
            return RoutingResult(
                final_text=edge_resp.text,
                used_source="edge",
                escalated=False,
                reason=f"{reason}_cloud_failed_kept_edge",
                edge_confidence=edge_resp.confidence,
                model=edge_resp.used_model,
            )

        # 云端失败且没有端侧结果：再试一次端侧
        fallback = self._try_edge(messages)
        if fallback is not None:
            self.logger.warning("云端失败，回退端侧推理: reason=%s", reason)
            return RoutingResult(
                final_text=fallback.text,
                used_source="edge",
                escalated=False,
                reason="cloud_failed_edge_fallback",
                edge_confidence=fallback.confidence,
                model=fallback.used_model,
            )

        return self._cloud_unavailable_msg(reason=reason)

    def _edge_only(self, messages: list[dict], reason: str) -> RoutingResult:
        """路由判定应走云端、但云端未启用时的端侧兜底。

        reason 必须透传：此前这里硬编码返回 "edge_ok"，把真实的路由判定
        （policy_force_cloud_first / policy_tiny_disabled）丢掉了，
        导致响应无法反映「本来想去云端」这一事实，排查时极易误判。
        """
        if self.edge is None or not self.edge.ready:
            return self._edge_unavailable_msg(reason=reason)

        self.logger.warning(
            "路由判定应走云端（%s）但 CLOUD_ENABLED=false，改由端侧应答", reason
        )
        try:
            edge_resp = self.edge.generate(messages)
            return RoutingResult(
                final_text=edge_resp.text,
                used_source="edge",
                escalated=False,
                reason=f"{reason}_edge_only",
                edge_confidence=edge_resp.confidence,
                model=edge_resp.used_model,
            )
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Edge 推理失败（云端关闭）: %s", exc)
            return self._edge_error_msg(reason=f"{reason}_edge_error")

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

    def _cloud_unavailable_msg(self, reason: str) -> RoutingResult:
        """云端与端侧都不可用。

        used_source 用 none 而非 edge/cloud，避免调用方误判为某一方已作答；
        model 留空，不报未真正参与的模型。
        """
        return RoutingResult(
            final_text="云端调用失败且端侧不可用，请检查 CLOUD_API_BASE / CLOUD_API_KEY 与端侧模型配置。",
            used_source="none",
            escalated=False,
            reason=f"{reason}_cloud_unavailable",
            model="",
        )

    def _call_cloud(self, messages: list[dict]) -> CloudResult | None:
        """调用云端；任何失败都返回 None，由调用方决定如何降级。"""
        try:
            return self.cloud.complete(messages)
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                "云端调用失败 api_base=%r model=%r: %s",
                self.cloud_cfg.api_base,
                self.cloud_cfg.model,
                exc,
            )
            return None
