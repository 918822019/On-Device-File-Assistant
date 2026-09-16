"""Shared request/response schemas for API and error payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入")
    force_cloud: bool = Field(default=False, description="强制走云端")
    session_id: str | None = None


class ChatResponse(BaseModel):
    source: str
    escalated: bool
    reason: str
    used_model: str
    edge_confidence: float | None = None
    text: str


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="待生成向量的文本列表")
    normalize: bool = Field(default=True, description="是否做 L2 标准化")


class EmbeddingResponse(BaseModel):
    source: str
    model: str
    embeddings: list[list[float]]


class ErrorResponse(BaseModel):
    code: str
    message: str
    status: int
    trace_id: str | None = None
    path: str | None = None
    detail: object | None = None
