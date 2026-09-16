"""Chat endpoints for the edge-cloud agent."""

from fastapi import APIRouter, Request

from ..agent import EdgeCloudOrchestrator, RoutingResult
from ..schemas import ChatRequest, ChatResponse

router = APIRouter()


def _build_default_history(message: str) -> list[dict]:
    return [{"role": "user", "content": message}]


@router.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    orchestrator: EdgeCloudOrchestrator = request.app.state.orchestrator
    routing: RoutingResult = orchestrator.ask(
        messages=_build_default_history(req.message),
        force_cloud=req.force_cloud,
    )

    return ChatResponse(
        source=routing.used_source,
        escalated=routing.escalated,
        reason=routing.reason,
        used_model=routing.model,
        edge_confidence=routing.edge_confidence,
        text=routing.final_text,
    )
