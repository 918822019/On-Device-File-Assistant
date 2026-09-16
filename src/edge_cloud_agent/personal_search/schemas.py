"""Schemas for the personal file memory search API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FileSearchRequest(BaseModel):
    """First-turn search request from the user."""

    query: str = Field(..., min_length=1, max_length=260, description="用户口述线索")
    top_k: int = Field(default=6, ge=1, le=20, description="返回候选上限")
    force_disambiguation: bool = Field(
        default=False,
        description="若为 true，则总是返回多个候选以便用户确认",
    )


class FileSearchSessionRequest(BaseModel):
    """Follow-up input after initial retrieval."""

    session_id: str = Field(..., min_length=8, description="上一次 search 返回的会话 ID")
    top_k: int = Field(default=6, ge=1, le=20, description="追问阶段返回候选上限")
    selected_file_id: str | None = Field(
        default=None,
        description="用户直接选中的文件 ID（若确认）",
    )
    reply: str | None = Field(
        default=None,
        description="用户补充线索，用于再次筛选，比如“5月的/微信群里的/蓝色背景”",
        min_length=1,
        max_length=160,
    )


class FileSearchCandidate(BaseModel):
    """一个可确认的候选文件。"""

    file_id: str
    title: str
    source_app: str
    doc_type: str
    mime_type: str | None
    file_uri: str | None
    captured_at: str | None
    score: float
    evidence: str
    preview: str
    visual_hints: list[str] = Field(default_factory=list)
    matched_clues: list[str] = Field(default_factory=list)


class FileSearchResponse(BaseModel):
    """Search round output for the first pass or clarification pass."""

    query: str
    session_id: str
    state: Literal["resolved", "needs_clarification", "not_found"]
    needs_disambiguation: bool
    question: str | None = None
    candidates: list[FileSearchCandidate] = Field(default_factory=list)
    selected_file_id: str | None = None
    next_action_suggestions: list[str] = Field(default_factory=list)


class FileActionRequest(BaseModel):
    """Action after a material is resolved."""

    session_id: str | None = Field(
        default=None,
        description="若来自当前会话可填，不填则直接按 file_id 执行",
    )
    file_id: str = Field(..., min_length=8, description="目标文件ID")
    action: Literal["open", "share", "compare", "annotate", "archive"] = Field(...)
    share_to: str | None = Field(default=None, description="分享目标昵称/账号")
    peer_file_id: str | None = Field(
        default=None,
        description="compare 动作下用的对比文件",
        min_length=8,
    )
    note: str | None = Field(default=None, description="用户备注")


class FileActionResponse(BaseModel):
    """Result of an executable action request."""

    action: str
    status: Literal["ok", "failed", "unsupported"]
    file_id: str
    file_title: str | None = None
    file_uri: str | None = None
    message: str
    share_payload: dict | None = None
    compare_payload: dict | None = None
    annotations: str | None = None
    archived: bool = False
    next_action_suggestions: list[str] = Field(default_factory=list)


class RebuildIndexResponse(BaseModel):
    scanned: int = 0
    imported: int = 0
    skipped: int = 0
    errors: int = 0
    material_ids: list[str] = Field(default_factory=list)
