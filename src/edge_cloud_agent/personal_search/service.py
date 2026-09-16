"""Personal-file search service for disambiguation-first retrieval.

该模块定义了“搜索 -> 追问 -> 结果确认 -> 执行动作”的闭环。
核心职责包括候选检索、候选重排、会话状态管理和动作执行日志。
"""

from __future__ import annotations
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import md5
from pathlib import Path
from threading import RLock
from time import perf_counter
from uuid import uuid4

from ..config import PersonalFileConfig
from ..embedding_runtime import EdgeEmbeddingRuntime
from .storage import PersonalFileItem, PersonalFileStore
from .schemas import FileSearchCandidate
from .vector_index import PersonalFileVectorIndex


TIME_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"上周"), 7),
    (re.compile(r"上个?月"), 30),
    (re.compile(r"本周"), 7),
    (re.compile(r"最近(\d+)?天"), 3),
    (re.compile(r"今天|昨日|昨天|前天|明天"), 1),
]

_CLUE_COLOR_TOKENS = {
    "蓝色": "blue",
    "白色": "white",
    "黑色": "black",
    "红色": "red",
    "绿色": "green",
    "黄色": "yellow",
    "橙色": "orange",
    "紫色": "purple",
    "灰色": "gray",
}

_SOURCE_KEYWORDS = {
    "wechat": {"微信", "weixin", "wechat", "微信好友", "微信群", "群里", "群聊"},
    "email": {"邮箱", "email", "邮箱", "mail", "gmail", "outlook"},
    "image": {"图库", "相册", "screenshot", "截图", "screen"},
    "office": {"word", "excel", "ppt", "文件", "doc", "pdf"},
}

_LOGGER = logging.getLogger("agent_server.personal_search")


@dataclass
class _SearchCandidate:
    """内部候选结构体：原始 item + 打分 + 命中证据。"""

    item: PersonalFileItem
    score: float
    evidence: str
    matched_clues: list[str]


@dataclass
class _SearchSession:
    """单次搜索会话上下文，保存候选和对话轮次。"""

    session_id: str
    query: str
    candidates: list[_SearchCandidate]
    turn: int


def _now_iso() -> str:
    """返回标准 UTC 时间字符串，统一日志和存储中的时间展示。"""

    return datetime.utcnow().isoformat(timespec="seconds")


def _to_dt(value: str | None) -> datetime | None:
    """安全转换 ISO 字符串为 datetime，失败时返回 None。"""

    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _tokenize(query: str) -> list[str]:
    """将中文查询切词并提取中文短语，避免空 token 干扰匹配。"""

    parts = re.split(r"[\s,，。；;:：!！?？、/\\|()（）【】\-]+", query)
    tokens = [item.strip().lower() for item in parts if item.strip()]
    phrases = [seg for seg in re.findall(r"[\u4e00-\u9fff]{2,}", query)]
    for phrase in phrases:
        lower = phrase.lower()
        if lower not in tokens:
            tokens.append(lower)
    return sorted(set(tokens))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度，输入向量长度不一致时用最小维度对齐。"""

    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(size):
        dot += a[i] * b[i]
        norm_a += a[i] ** 2
        norm_b += b[i] ** 2
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _clamp01(value: float) -> float:
    """将分值限定在 0~1 区间，避免下游排序异常。"""

    return max(0.0, min(1.0, value))


def _has_time_match(captured_at: str | None, query: str) -> tuple[bool, str]:
    """从查询文本中判断时间约束是否命中，返回匹配说明用于 evidence。"""

    if not captured_at:
        return False, ""

    q = query.replace(" ", "")
    item_time = _to_dt(captured_at)
    if item_time is None:
        return False, ""

    now = datetime.utcnow()
    for pattern, days in TIME_PATTERNS:
        if pattern.search(q):
            if days == 1 and "今天" in q and abs((now.date() - item_time.date()).days) == 0:
                return True, "时间线索匹配: 今天"
            if "昨天" in q and (now.date() - item_time.date()).days == 1:
                return True, "时间线索匹配: 昨天"
            if "前天" in q and (now.date() - item_time.date()).days == 2:
                return True, "时间线索匹配: 前天"
            if "上周" in q and (now - item_time) <= timedelta(days=14):
                return True, "时间线索匹配: 上周"
            if "上月" in q and (now - item_time) <= timedelta(days=45):
                return True, "时间线索匹配: 上月"
            if "本周" in q and item_time.date().isocalendar().week == now.date().isocalendar().week:
                return True, "时间线索匹配: 本周"
            if q.startswith("最近") and "天" in q:
                try:
                    num = int(re.findall(r"最近(\d+)天", q)[0])
                except Exception:
                    num = days
                if (now - item_time) <= timedelta(days=num):
                    return True, f"时间线索匹配: 最近{num}天"
                return False, ""
            if item_time >= now - timedelta(days=days):
                return True, f"时间线索匹配: {days}天内"

    return False, ""


def _infer_source_from_path(file_path: Path) -> str:
    """基于文件路径做来源弱识别（wechat/微信/email/图片库等）。"""

    lowered = file_path.as_posix().lower()
    if "wechat" in lowered or "微信" in lowered or "weixin" in lowered:
        return "wechat"
    if "email" in lowered or "邮箱" in lowered or "mail" in lowered or "gmail" in lowered:
        return "email"
    if "image" in lowered or "images" in lowered or "图库" in lowered or "相册" in lowered:
        return "gallery"
    if "camera" in lowered or "截图" in lowered or "screenshot" in lowered:
        return "camera"
    return "local"


class PersonalFileSearchService:
    """Personal search core service.

提供三类能力：
1. 规则 + 向量混合检索
2. 澄清会话管理（用于继续追问）
3. 动作执行与结构化日志记录
"""

    def __init__(
        self,
        config: PersonalFileConfig,
        store: PersonalFileStore,
        embedding_runtime: EdgeEmbeddingRuntime | None,
    ) -> None:
        self.config = config
        self.store = store
        self.embedding_runtime = embedding_runtime
        self._sessions: dict[str, _SearchSession] = {}
        self._session_lock = RLock()
        self._annotations: dict[str, str] = {}
        self._archived: set[str] = set()
        self._vector_index = PersonalFileVectorIndex(config)
        # 初始化时记录向量索引可用性，便于启动期排障。
        _LOGGER.info(
            "personal-search-service-initialized",
            extra={
                "event": "service_init",
                "faiss_enabled": self.config.enable_faiss,
                "faiss_ready": self.vector_index_ready,
                "store_size": len(self.store.list_all()),
            },
        )

    @property
    def vector_index_ready(self) -> bool:
        return self._vector_index.is_ready()

    @property
    def vector_index_available(self) -> bool:
        return self._vector_index.is_available()

    def rebuild_vector_index(self) -> bool:
        if not self.vector_index_available:
            return False

        items = self.store.list_all()
        try:
            started = perf_counter()
            self._vector_index.rebuild(items)
            _LOGGER.info(
                "personal-search-faiss-rebuild",
                extra={
                    "event": "faiss.rebuild",
                    "count": len(items),
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "ready": self.vector_index_ready,
                },
            )
            return self.vector_index_ready
        except Exception:
            return False

    def next_file_id(self, file_path: Path) -> str:
        normalized = file_path.as_posix().encode("utf-8", errors="ignore")
        return md5(normalized).hexdigest()[:14]

    def detect_source_app(self, file_path: Path) -> str:
        return _infer_source_from_path(file_path)

    def remove_by_uri(self, file_uri: str) -> None:
        self.store.delete_by_file_uri(file_uri)

    def embed_text(self, text: str) -> list[float] | None:
        if not self.embedding_runtime or not text:
            return None
        try:
            return self.embedding_runtime.embed([text]).embeddings[0]
        except Exception:
            return None

    def make_summary(self, text: str) -> str:
        if not text:
            return ""
        stripped = text.strip()
        return stripped[:110] if len(stripped) > 110 else stripped

    def extract_tags(self, file_path: Path, text: str, source_app: str) -> list[str]:
        tags = [source_app]
        stem = file_path.stem.replace("_", " ").replace("-", " ").lower()
        tags.append(stem)
        if text:
            tags.extend(_tokenize(text)[:20])
        return sorted(set(t for t in tags if t))

    def extract_visual_hints(self, file_path: Path, text: str) -> list[str]:
        hints: set[str] = set()
        stem = file_path.stem.lower()
        for zh, en in _CLUE_COLOR_TOKENS.items():
            if zh in stem or en in stem:
                hints.add(zh)
        for token in ("会议", "讲", "设计", "柜子", "发票", "合同", "截图", "版本", "方案"):
            if token in text or token in stem:
                hints.add(token)
        return sorted(hints)

    def get_session(self, session_id: str) -> _SearchSession | None:
        return self._sessions.get(session_id)

    def get_file(self, file_id: str) -> PersonalFileItem | None:
        return self.store.get(file_id)

    def _log_file_action(
        self,
        action: str,
        file_id: str,
        status: str,
        **fields: object,
    ) -> None:
        """记录文件动作日志，统一成功/失败事件名与字段结构。"""

        payload = {
            "event": f"action.{action}_{status}",
            "action": action,
            "file_id": file_id,
            "status": status,
            "trace_id": fields.pop("trace_id", None),
            **fields,
        }
        if status == "ok":
            _LOGGER.info(
                "personal-search-file-action",
                extra=payload,
            )
            return
        _LOGGER.warning(
            "personal-search-file-action",
            extra=payload,
        )

    def add_annotation(self, file_id: str, note: str) -> bool:
        """给指定文件补充/覆盖备注。

        成功返回 True，目标文件不存在返回 False，并打 failed 日志。
        """

        if not self.get_file(file_id):
            self._log_file_action("annotate", file_id, "failed", note_provided=bool(note))
            return False
        self._annotations[file_id] = note.strip()
        self._log_file_action("annotate", file_id, "ok", note_size=len(note), note_preview=(note or "")[:40])
        return True

    def get_annotation(self, file_id: str) -> str | None:
        return self._annotations.get(file_id)

    def archive_file(self, file_id: str) -> bool:
        """标记文件为已归档，仅影响运行时展示与后续动作入口。"""

        if not self.get_file(file_id):
            self._log_file_action("archive", file_id, "failed")
            return False
        self._archived.add(file_id)
        self._log_file_action("archive", file_id, "ok")
        return True

    def is_archived(self, file_id: str) -> bool:
        return file_id in self._archived

    def compare_payload(self, left_file_id: str, right_file_id: str) -> dict:
        """返回两个文件的对比建议，含推荐保留项和原因。"""

        left = self.get_file(left_file_id)
        right = self.get_file(right_file_id)
        if left is None or right is None:
            self._log_file_action(
                "compare",
                left_file_id,
                "failed",
                right_file_id=right_file_id,
                reason="file_not_found",
            )
            raise ValueError("file_not_found")

        def _to_dt(value: str | None) -> datetime | None:
            if not value:
                return None
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return None

        left_time = _to_dt(left.captured_at)
        right_time = _to_dt(right.captured_at)
        if left_time and right_time:
            newer_id = left_file_id if left_time >= right_time else right_file_id
            reason = "按 captured_at 时间优先"
        else:
            newer_id = left_file_id if left.file_path >= right.file_path else right_file_id
            reason = "未命中时间戳，按文件路径字典序兜底"

        _LOGGER.info(
            "personal-search-compare-built",
            extra={
                "event": "action.compare_ok",
                "left_file_id": left_file_id,
                "right_file_id": right_file_id,
                "recommended_keep": newer_id,
                "reason": reason,
            },
        )

        return {
            "left": {
                "file_id": left.file_id,
                "title": left.title,
                "source_app": left.source_app,
                "captured_at": left.captured_at,
                "score_hint": self._match_hint_in_text(left.raw_text),
                "archived": self.is_archived(left.file_id),
                "note": self.get_annotation(left.file_id),
            },
            "right": {
                "file_id": right.file_id,
                "title": right.title,
                "source_app": right.source_app,
                "captured_at": right.captured_at,
                "score_hint": self._match_hint_in_text(right.raw_text),
                "archived": self.is_archived(right.file_id),
                "note": self.get_annotation(right.file_id),
            },
            "recommended_keep": newer_id,
            "recommended_reason": reason,
        }

    @staticmethod
    def _match_hint_in_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "无可检索文本"
        return text[:40]

    def start_search(
        self,
        query: str,
        top_k: int = 6,
        force_disambiguation: bool = False,
        trace_id: str | None = None,
    ) -> tuple[str, list[FileSearchCandidate], str, bool, str | None]:
        """发起一次搜索会话。

        返回 `(session_id, candidates, state, should_disambiguate, question)`。
        如果无需澄清直接返回 resolved。
        """

        parsed_query = query[: self.config.max_search_query_len]
        query_digest = md5(parsed_query.encode("utf-8", errors="ignore")).hexdigest()[:10]
        search_id = uuid4().hex
        started = perf_counter()

        _LOGGER.info(
            "personal-search-start",
            extra={
                "event": "search.started",
                "trace_id": trace_id,
                "search_id": search_id,
                "query_digest": query_digest,
                "query_len": len(parsed_query),
                "top_k": top_k,
                "force_disambiguation": force_disambiguation,
            },
        )

        candidates = self._search_items(parsed_query, top_k, search_id=search_id, trace_id=trace_id)
        state, question, should_disambiguate = self._decide_state(candidates, force_disambiguation)
        session = _SearchSession(
            session_id=search_id,
            query=parsed_query,
            candidates=candidates,
            turn=0,
        )
        with self._session_lock:
            self._sessions[session.session_id] = session

        response_candidates = [self._to_schema(candidate) for candidate in candidates]
        _LOGGER.info(
            "personal-search-completed",
            extra={
                "event": "search.completed",
                "trace_id": trace_id,
                "search_id": search_id,
                "state": state,
                "should_disambiguate": should_disambiguate,
                "candidate_count": len(response_candidates),
                "top_k_returned": [candidate.file_id for candidate in response_candidates[:3]],
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return session.session_id, response_candidates, state, should_disambiguate, question

    def continue_search(
        self,
        session_id: str,
        selected_file_id: str | None,
        reply: str | None,
        top_k: int = 6,
        trace_id: str | None = None,
    ) -> tuple[str, list[FileSearchCandidate], str, bool, str | None, str | None]:
        """对 clarify 会话进行下一轮处理。

        - selected_file_id 存在时优先直接确认；
- 否则用 reply 文本过滤/重排候选；
- 无结果时返回 not_found 并附带二次引导语。
        """

        started = perf_counter()
        session = self.get_session(session_id)
        if session is None:
            _LOGGER.warning(
                "personal-search-session-not-found",
                extra={
                    "event": "clarify.session_not_found",
                    "trace_id": trace_id,
                    "search_id": session_id,
                },
            )
            raise ValueError("session_not_found")

        candidates = list(session.candidates)
        selected = selected_file_id.strip() if selected_file_id else None
        normalized_reply = reply.strip() if reply else None
        question = None

        _LOGGER.info(
            "personal-search-clarify-started",
            extra={
                "event": "clarify.started",
                "trace_id": trace_id,
                "search_id": session_id,
                "session_turn": session.turn,
                "selected_file_id": selected,
                "reply_len": len(normalized_reply or ""),
            },
        )

        if selected is not None:
            selected_candidate = next((item for item in candidates if item.item.file_id == selected), None)
            if selected_candidate is not None:
                _LOGGER.info(
                    "personal-search-direct-select",
                    extra={
                        "event": "clarify.selected_direct",
                        "trace_id": trace_id,
                        "search_id": session_id,
                        "selected_file_id": selected,
                    },
                )
                return (
                    session_id,
                    [self._to_schema(selected_candidate)],
                    "resolved",
                    False,
                    None,
                    selected,
                )
            _LOGGER.warning(
                "personal-search-direct-select-miss",
                extra={
                    "event": "clarify.selected_direct_miss",
                    "trace_id": trace_id,
                    "search_id": session_id,
                    "selected_file_id": selected,
                },
            )

        if normalized_reply:
            before_filter = len(candidates)
            candidates = self._filter_candidates_by_reply(candidates, normalized_reply)
            _LOGGER.info(
                "personal-search-filter",
                extra={
                    "event": "clarify.filtered",
                    "trace_id": trace_id,
                    "search_id": session_id,
                    "before_filter": before_filter,
                    "after_filter": len(candidates),
                },
            )
            if not candidates:
                _LOGGER.warning(
                    "personal-search-no-match-after-filter",
                    extra={
                        "event": "clarify.no_match",
                        "trace_id": trace_id,
                        "search_id": session_id,
                        "reply_len": len(reply),
                    },
                )
                return session_id, [], "not_found", False, "没有找到可确认的文件，请再给 1~2 个线索，比如时间/群聊/颜色/场景。", None

        if not candidates:
            _LOGGER.info(
                "personal-search-clarify-empty",
                extra={
                    "event": "clarify.empty",
                    "trace_id": trace_id,
                    "search_id": session_id,
                    "reply_len": len(normalized_reply or ""),
                    "state_after_filter": "not_found",
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )
            return session_id, [], "not_found", False, "没有找到可确认的文件，请再给 1~2 个线索，比如时间/来源/场景。", None

        candidates = self._rerank_within(candidates, normalized_reply if normalized_reply else session.query, top_k)

        state, question, should_disambiguate = self._decide_state(candidates, False)
        session.turn += 1
        session.query = session.query
        session.candidates = candidates
        self._sessions[session_id] = session
        _LOGGER.info(
            "personal-search-clarify-completed",
            extra={
                "event": "clarify.completed",
                "trace_id": trace_id,
                "search_id": session_id,
                "state": state,
                "candidate_count": len(candidates),
                "should_disambiguate": should_disambiguate,
                "selected_file_id": selected,
                "duration_ms": round((perf_counter() - started) * 1000, 2),
            },
        )
        return (
            session_id,
            [self._to_schema(candidate) for candidate in candidates],
            state,
            should_disambiguate,
            question,
            selected,
        )

    def get_item(self, file_id: str) -> PersonalFileItem | None:
        return self.store.get(file_id)

    def _search_items(
        self,
        query: str,
        top_k: int = 6,
        search_id: str | None = None,
        trace_id: str | None = None,
    ) -> list[_SearchCandidate]:
        """内部主检索流程：候选池构建 + 分值打分 + 截断返回。"""

        started = perf_counter()
        all_items = self.store.list_all()
        if not all_items:
            if search_id:
                _LOGGER.warning(
                    "personal-search-no-items",
                    extra={
                        "event": "search.no_items",
                        "trace_id": trace_id,
                        "search_id": search_id,
                    },
                )
            return []

        query_embedding = None
        embedding_used = False
        if self.embedding_runtime is not None and query:
            try:
                query_embedding = self.embedding_runtime.embed([query]).embeddings[0]
                embedding_used = True
            except Exception as exc:
                _LOGGER.warning(
                    "personal-search-embedding-failed",
                    extra={
                        "event": "search.embedding_error",
                        "trace_id": trace_id,
                        "search_id": search_id,
                        "exception": type(exc).__name__,
                    },
                )
                query_embedding = None

        parsed_tokens = _tokenize(query)
        source_hints = self._extract_source_hints(query)
        visual_hints = self._extract_visual_hints(query)
        version_hint = self._extract_version_hint(query)
        candidate_pool = self._select_candidates_for_scoring(
            all_items,
            query_embedding,
            top_k,
            search_id=search_id,
            trace_id=trace_id,
        )
        if search_id is not None:
            _LOGGER.info(
                "personal-search-pool-ready",
                extra={
                    "event": "search.candidate_pool",
                    "trace_id": trace_id,
                    "search_id": search_id,
                    "total_items": len(all_items),
                    "candidate_pool": len(candidate_pool),
                    "embedding_used": embedding_used,
                    "faiss_used": bool(query_embedding and self.vector_index_ready and self.config.enable_faiss),
                },
            )

        scored: list[_SearchCandidate] = []
        for item in candidate_pool:
            score, evidence, matched = self._score_item(
                item,
                query,
                parsed_tokens,
                source_hints,
                visual_hints,
                version_hint,
                query_embedding,
            )
            if score <= 0.01:
                continue
            scored.append(_SearchCandidate(item=item, score=score, evidence=evidence, matched_clues=matched))

        scored.sort(key=lambda item: item.score, reverse=True)
        deduped = self._dedupe(scored)[:top_k]
        if search_id is not None:
            _LOGGER.info(
                "personal-search-scored",
                extra={
                    "event": "search.scored",
                    "trace_id": trace_id,
                    "search_id": search_id,
                    "scored_count": len(scored),
                    "returned_count": len(deduped),
                    "top_scores": [round(item.score, 4) for item in deduped[:3]],
                    "top_ids": [item.item.file_id for item in deduped[:3]],
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                },
            )
        return deduped

    def _select_candidates_for_scoring(
        self,
        all_items: list[PersonalFileItem],
        query_embedding: list[float] | None,
        top_k: int,
        search_id: str | None = None,
        trace_id: str | None = None,
    ) -> list[PersonalFileItem]:
        """候选池前置筛选。

策略：
- 无 embedding/向量索引时回到全量扫描；
- 有向量索引时按 ANN 命中补齐不足项，防止召回过窄。
        """

        if not query_embedding or not self.vector_index_ready:
            if search_id is not None:
                _LOGGER.info(
                    "personal-search-fallback-scan",
                    extra={
                        "event": "search.fallback",
                        "trace_id": trace_id,
                        "search_id": search_id,
                        "reason": "no_embedding_or_faiss_not_ready",
                        "faiss_enabled": self.config.enable_faiss,
                        "faiss_ready": self.vector_index_ready,
                    },
                )
            return all_items

        item_by_id = {item.file_id: item for item in all_items}
        try:
            candidate_multiplier = max(1, self.config.faiss_candidate_multiplier)
            limit = max(top_k, top_k * candidate_multiplier)
            hits = self._vector_index.search(
                query_embedding,
                top_k=limit,
            )
        except Exception as exc:
            if search_id is not None:
                _LOGGER.warning(
                    "personal-search-faiss-fallback-error",
                    extra={
                        "event": "search.fallback",
                        "trace_id": trace_id,
                        "search_id": search_id,
                        "reason": "faiss_search_exception",
                        "exception": type(exc).__name__,
                    },
                )
            return all_items
        if not hits:
            if search_id is not None:
                _LOGGER.warning(
                    "personal-search-faiss-no-hits",
                    extra={
                        "event": "search.fallback",
                        "trace_id": trace_id,
                        "search_id": search_id,
                        "reason": "faiss_no_hits",
                    },
                )
            return all_items

        selected: list[PersonalFileItem] = []
        seen: set[str] = set()
        for hit in hits:
            item = item_by_id.get(hit.file_id)
            if item is None:
                continue
            if item.file_id in seen:
                continue
            selected.append(item)
            seen.add(item.file_id)

        if len(selected) < top_k:
            for item in all_items:
                if item.file_id in seen:
                    continue
                selected.append(item)
                seen.add(item.file_id)
                if len(selected) >= top_k * 2:
                    break
        if search_id is not None:
            _LOGGER.info(
                "personal-search-faiss-selected",
                extra={
                    "event": "search.faiss_selected",
                    "trace_id": trace_id,
                    "search_id": search_id,
                    "requested_top_k": top_k,
                    "faiss_limit": limit,
                    "selected_count": len(selected),
                    "first_hit_ids": [hit.file_id for hit in hits[: min(5, len(hits))]],
                },
            )
        return selected

    def _rerank_within(self, candidates: list[_SearchCandidate], query: str, top_k: int) -> list[_SearchCandidate]:
        """基于用户回复再打一次分，推动候选向真实目标收敛。"""

        if not candidates:
            return []
        parsed_tokens = _tokenize(query)
        source_hints = self._extract_source_hints(query)
        visual_hints = self._extract_visual_hints(query)

        ranked: list[_SearchCandidate] = []
        for candidate in candidates:
            base_score, evidence, matched = self._score_item(
                candidate.item,
                query,
                parsed_tokens,
                source_hints,
                visual_hints,
                self._extract_version_hint(query),
                None,
            )
            boosted = base_score + (1.0 if candidate.item.file_id == candidates[0].item.file_id else 0.0)
            ranked.append(_SearchCandidate(item=candidate.item, score=boosted, evidence=f"{candidate.evidence}；{evidence}", matched_clues=sorted(set(candidate.matched_clues + matched))))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _extract_source_hints(query: str) -> list[str]:
        normalized = query.lower()
        source_hints: list[str] = []
        for source, items in _SOURCE_KEYWORDS.items():
            if any(key in normalized for key in items):
                source_hints.append(source)
        return source_hints

    @staticmethod
    def _extract_visual_hints(query: str) -> list[str]:
        normalized = query.lower()
        return [token for token in _CLUE_COLOR_TOKENS if token in normalized]

    @staticmethod
    def _extract_version_hint(query: str) -> bool:
        q = query.replace(" ", "")
        return "后来的" in q or "最新" in q or "版本" in q or "哪个" in q

    def _text_weight(self) -> float:
        return max(0.0, self.config.faiss_text_weight)

    def _clue_weight(self) -> float:
        return max(0.0, self.config.faiss_clue_weight)

    def _semantic_weight(self) -> float:
        return max(0.0, self.config.faiss_semantic_weight)

    def _version_bonus_weight(self) -> float:
        return max(0.0, self.config.faiss_version_bonus)

    @property
    def _score_weight_sum(self) -> float:
        return max(
            0.01,
            self._text_weight() + self._semantic_weight() + self._clue_weight(),
        )

    def _decide_state(self, candidates: list[_SearchCandidate], force_disambiguation: bool) -> tuple[str, str | None, bool]:
        """根据候选数量和分差判断是否已经 resolved。"""

        if not candidates:
            return "not_found", "没找到命中项，请再给一条时间/来源/场景线索。", False
        if len(candidates) == 1 and not force_disambiguation:
            return "resolved", None, False

        if len(candidates) >= 2:
            gap = candidates[0].score - candidates[1].score
            if gap >= 0.35 and not force_disambiguation:
                return "resolved", None, False

        question = self._build_question(candidates)
        return "needs_clarification", question, True

    def _build_question(self, candidates: list[_SearchCandidate]) -> str:
        if len(candidates) >= 3:
            return (
                "我找到几条可能的候选。你可以补 1-2 个线索继续确认："
                "比如说时间（上周/昨天）、来源（微信群/邮箱）或者视觉线索（蓝色背景、聚餐、会议）。"
            )
        return "我有两条很接近的结果，选一个更确定的描述：例如‘5月的那张’/‘群里的那张’。"

    def _filter_candidates_by_reply(self, candidates: list[_SearchCandidate], reply: str) -> list[_SearchCandidate]:
        reply_lower = reply.lower()
        filtered = []
        for candidate in candidates:
            text = " ".join(
                [
                    candidate.item.title,
                    candidate.item.summary,
                    candidate.item.doc_type,
                    candidate.item.source_app,
                    " ".join(candidate.item.tags),
                    " ".join(candidate.item.visual_hints),
                ]
            ).lower()
            if reply_lower in text or any(token in text for token in _tokenize(reply_lower)):
                filtered.append(candidate)
        return filtered

    def _score_item(
        self,
        item: PersonalFileItem,
        query: str,
        parsed_tokens: list[str],
        source_hints: list[str],
        visual_hints: list[str],
        version_hint: bool,
        query_embedding: list[float] | None,
    ) -> tuple[float, str, list[str]]:
        """计算单条文件与查询的综合分值与命中证据。"""

        haystack = " ".join(
            [
                item.title,
                item.summary,
                item.doc_type,
                item.source_app,
                item.raw_text,
                " ".join(item.tags),
                " ".join(item.visual_hints),
                item.file_path,
            ]
        ).lower()

        text_hits = 0.0
        matched: list[str] = []
        for token in parsed_tokens:
            if token and token in haystack:
                text_hits += 1.0
                matched.append(token)
        if query.lower() in haystack:
            text_hits += 1.5
            matched.append("全文命中")

        clue_score = 0.0
        for source in source_hints:
            source_key = source
            if source_key in haystack:
                clue_score += 0.8
                matched.append(f"来源:{source_key}")

        for visual in visual_hints:
            if visual in haystack:
                clue_score += 0.7
                matched.append(f"视觉:{visual}")

        time_hit, reason = _has_time_match(item.captured_at, query)
        if time_hit:
            clue_score += 0.7
            matched.append(reason)

        if version_hint and ("版本" in haystack or re.search(r"v\d+", item.title.lower())):
            clue_score += 0.5
            matched.append("版本关系")

        version_rank_bonus = 0.0
        if "v" in haystack:
            version_rank_bonus = 0.2

        embed_score = 0.0
        if query_embedding and item.embedding:
            embed_score = _cosine_similarity(query_embedding, item.embedding)

        normalized_tokens = max(1.0, len(parsed_tokens))
        score = (
            (text_hits / normalized_tokens) * self._text_weight()
            + embed_score * self._semantic_weight()
            + clue_score * self._clue_weight()
            + version_rank_bonus * self._version_bonus_weight()
        )
        score = score / self._score_weight_sum
        score = _clamp01(score)
        evidence = "; ".join(sorted(set(matched))) if matched else "仅按语义近似"
        return score, evidence, sorted(set(matched))

    @staticmethod
    def _dedupe(items: list[_SearchCandidate]) -> list[_SearchCandidate]:
        seen = set()
        deduped: list[_SearchCandidate] = []
        for item in items:
            if item.item.file_id in seen:
                continue
            seen.add(item.item.file_id)
            deduped.append(item)
        return deduped

    @staticmethod
    def _to_schema(item: _SearchCandidate) -> FileSearchCandidate:
        preview = item.item.summary or item.item.raw_text[:60]
        if not preview:
            preview = "无可检索文本，保留文件名与来源线索"
        return FileSearchCandidate(
            file_id=item.item.file_id,
            title=item.item.title,
            source_app=item.item.source_app,
            doc_type=item.item.doc_type,
            mime_type=item.item.mime_type,
            file_uri=item.item.file_uri,
            captured_at=item.item.captured_at,
            score=round(item.score, 4),
            evidence=item.evidence,
            preview=preview[:180],
            visual_hints=item.item.visual_hints,
            matched_clues=item.matched_clues,
        )
