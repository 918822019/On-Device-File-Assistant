"""HTTP adapter for the chat Agent."""

from fastapi import APIRouter, Request

from ..agents.chat import ChatAgent
from ..personal_search.search_flow import start_file_search
from . import deps
from .schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """把检索与 LLM 决策交给 ChatAgent；索引不可用时退化为纯模型问答。"""

    service = deps.optional_personal_file_service(request)

    def file_search(query: str):
        return start_file_search(
            service,
            query,
            trace_id=deps.trace_id(request),
            refresher=deps.personal_file_refresher(request),
            metrics=deps.metrics(request),
        )

    agent = ChatAgent(
        request.app.state.orchestrator,
        search_files=file_search if service is not None else None,
    )
    return ChatResponse(**vars(agent.answer(
        req.message,
        force_cloud=req.force_cloud,
        auto_search=req.auto_search,
    )))
