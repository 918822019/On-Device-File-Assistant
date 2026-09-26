"""Embedding endpoints for the edge-cloud agent."""

from fastapi import APIRouter, HTTPException, Request

from ..runtime.embedding_runtime import EdgeEmbeddingRuntime
from .schemas import EmbeddingRequest, EmbeddingResponse

router = APIRouter()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
def embedding(req: EmbeddingRequest, request: Request):
    runtime: EdgeEmbeddingRuntime | None = getattr(request.app.state, "embedding_runtime", None)

    if runtime is None or not runtime.ready:
        raise HTTPException(
            status_code=503,
            detail="端侧 embedding 模型未就绪，请检查 EDGE_EMBEDDING_* 配置后重启服务。",
        )

    try:
        result = runtime.embed(req.texts, normalize=req.normalize)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc

    return EmbeddingResponse(
        source="edge",
        model=result.used_model,
        embeddings=result.embeddings,
    )
