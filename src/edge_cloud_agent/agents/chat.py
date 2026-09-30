"""聊天 Agent：按业务意图决定查本地文件还是交给 LLM。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..personal_search.schemas import FileSearchResponse

if TYPE_CHECKING:
    from ..llm.orchestrator import RoutingResult


class ChatModel(Protocol):
    """Agent 只依赖生成能力的接口，不负责加载具体模型。"""

    def ask(self, messages: list[dict], force_cloud: bool = False) -> RoutingResult: ...

_FILE_INTENT = re.compile(r"找(?:到|回|一下|出|个|份)?|搜索|检索|查找|文件|文档|照片|截图|下载|本地|在哪(?:里)?")
_CHAT_INTENT = re.compile(r"什么是|为什么|如何|怎么|讲解|介绍|解释|方案|步骤|流程|写一|生成|总结")


@dataclass(frozen=True)
class ChatAgentResult:
    source: str
    escalated: bool
    reason: str
    used_model: str
    text: str
    edge_confidence: float | None = None
    search: FileSearchResponse | None = None


def _should_search(message: str) -> tuple[bool, bool]:
    """明确找文件直接查；简短主题需由检索证据确认。"""
    explicit = bool(_FILE_INTENT.search(message))
    if not message.strip() or len(message) > 260:
        return False, explicit
    if not explicit and (
        _CHAT_INTENT.search(message) or len(message) > 40 or re.search(r"[？?。！!]", message)
    ):
        return False, explicit
    return True, explicit


class ChatAgent:
    def __init__(
        self,
        orchestrator: ChatModel,
        search_files: Callable[[str], FileSearchResponse] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.search_files = search_files

    def answer(self, message: str, *, force_cloud: bool = False, auto_search: bool = True) -> ChatAgentResult:
        if auto_search and not force_cloud:
            should_search, explicit = _should_search(message)
            if should_search and self.search_files is not None:
                result = self.search_files(message)
                best = result.candidates[0] if result.candidates else None
                if explicit or (
                    result.state != "not_found"
                    and best is not None
                    and best.score >= 0.32
                    and best.matched_clues
                ):
                    count = len(result.candidates)
                    return ChatAgentResult(
                        source="file_search",
                        escalated=False,
                        reason="local_file_match" if count else "local_file_not_found",
                        used_model="local-file-index",
                        text=(
                            f"找到 {count} 个可能的本地文件，请根据文件名和命中证据确认。"
                            if count else "当前索引中没有找到匹配的本地文件，请补充文件名、来源或时间。"
                        ),
                        search=result,
                    )
            elif should_search and explicit:
                return ChatAgentResult(
                    source="file_search",
                    escalated=False,
                    reason="local_file_index_unavailable",
                    used_model="local-file-index",
                    text="本地文件索引尚未就绪，请检查扫描目录和服务状态。",
                )

        routing = self.orchestrator.ask(
            messages=[{"role": "user", "content": message}],
            force_cloud=force_cloud,
        )
        return ChatAgentResult(
            source=routing.used_source,
            escalated=routing.escalated,
            reason=routing.reason,
            used_model=routing.model,
            edge_confidence=routing.edge_confidence,
            text=routing.final_text,
        )
