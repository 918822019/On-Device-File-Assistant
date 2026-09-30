"""Personal-file search service：检索编排、会话与文件状态。

分层约定：
- 相关性规则（线索识别、打分、过滤、重排）在 `relevance.py`
- 澄清会话治理（TTL / 容量淘汰）在 `sessions.py`
- 对外响应组装在 `presentation.py`
- 索引写入与扫描在 `ingest.py`
本模块只负责把上述能力编排成「搜索 -> 追问 -> 确认 -> 动作」的闭环，
并管理向量索引、备注/归档等持久化状态。
"""

from __future__ import annotations

import logging
from hashlib import md5
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ..common.text_utils import tokenize
from ..common.time_utils import parse_iso
from ..config import PersonalFileConfig
from ..engines.embedding_runtime import EdgeEmbeddingRuntime
from .presentation import to_search_candidate
from .relevance import (
    CLUE_COLOR_TOKENS,
    ScoreWeights,
    SearchCandidate,
    decide_state,
    dedupe,
    extract_source_hints,
    extract_version_hint,
    extract_visual_hints,
    filter_candidates_by_reply,
    infer_source_from_path,
    rerank_within,
    score_item,
)
from .schemas import FileSearchCandidate
from .sessions import SearchSession, SessionStore
from .storage import FileStateStore, PersonalFileItem, PersonalFileStore
from .vector_index import PersonalFileVectorIndex

_LOGGER = logging.getLogger("agent_server.personal_search")

# 摘要长度：索引与展示共用的截断口径。
_SUMMARY_MAX_CHARS = 110

# 视觉线索里除颜色外的高频场景词。
_SCENE_HINT_TOKENS = ("会议", "讲", "设计", "柜子", "发票", "合同", "截图", "版本", "方案")


class PersonalFileSearchService:
    """Personal search core service.

    提供三类能力：
    1. 规则 + 向量混合检索
    2. 澄清会话管理（用于继续追问）
    3. 文件状态（备注/归档）与动作日志
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
        self.sessions = SessionStore()
        # 备注/归档标记持久化（旧实现为内存 dict/set，重启即丢）
        self._state_store = FileStateStore(config.state_path)
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

    # ------------------------------------------------------------------
    # 向量索引
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 索引写入侧（ingest 调用）
    # ------------------------------------------------------------------

    @property
    def score_weights(self) -> ScoreWeights:
        """按配置构造打分权重，负值一律归零。"""

        return ScoreWeights(
            text=max(0.0, self.config.faiss_text_weight),
            semantic=max(0.0, self.config.faiss_semantic_weight),
            clue=max(0.0, self.config.faiss_clue_weight),
            version_bonus=max(0.0, self.config.faiss_version_bonus),
        )

    def next_file_id(self, file_path: Path) -> str:
        normalized = file_path.as_posix().encode("utf-8", errors="ignore")
        return md5(normalized).hexdigest()[:14]

    def detect_source_app(self, file_path: Path) -> str:
        return infer_source_from_path(file_path)

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
        return stripped[:_SUMMARY_MAX_CHARS] if len(stripped) > _SUMMARY_MAX_CHARS else stripped

    def extract_tags(self, file_path: Path, text: str, source_app: str) -> list[str]:
        tags = [source_app]
        stem = file_path.stem.replace("_", " ").replace("-", " ").lower()
        tags.append(stem)
        if text:
            tags.extend(tokenize(text)[:20])
        return sorted({t for t in tags if t})

    def extract_visual_hints(self, file_path: Path, text: str) -> list[str]:
        hints: set[str] = set()
        stem = file_path.stem.lower()
        for zh, en in CLUE_COLOR_TOKENS.items():
            if zh in stem or en in stem:
                hints.add(zh)
        for token in _SCENE_HINT_TOKENS:
            if token in text or token in stem:
                hints.add(token)
        return sorted(hints)

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> SearchSession | None:
        return self.sessions.get(session_id)

    # ------------------------------------------------------------------
    # 文件状态与动作日志
    # ------------------------------------------------------------------

    def get_file(self, file_id: str) -> PersonalFileItem | None:
        return self.store.get(file_id)

    def get_item(self, file_id: str) -> PersonalFileItem | None:
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
        """给指定文件补充/覆盖备注（落盘持久化，重启不丢）。

        成功返回 True，目标文件不存在返回 False，并打 failed 日志。
        """

        if not self.get_file(file_id):
            self._log_file_action("annotate", file_id, "failed", note_provided=bool(note))
            return False
        self._state_store.set_annotation(file_id, note.strip())
        self._log_file_action("annotate", file_id, "ok", note_size=len(note), note_preview=(note or "")[:40])
        return True

    def get_annotation(self, file_id: str) -> str | None:
        return self._state_store.get_annotation(file_id)

    def archive_file(self, file_id: str) -> bool:
        """标记文件为已归档（落盘持久化）。仅影响展示与动作入口，不改检索结果。"""

        if not self.get_file(file_id):
            self._log_file_action("archive", file_id, "failed")
            return False
        self._state_store.set_archived(file_id)
        self._log_file_action("archive", file_id, "ok")
        return True

    def is_archived(self, file_id: str) -> bool:
        return self._state_store.is_archived(file_id)

    def prune_file_state(self) -> int:
        """剪掉已不在索引中的备注/归档标记，返回清理条数。

        幽灵清理只删 store 里的条目，不会动状态存储，所以必须单独剪枝。
        详见 FileStateStore.prune 的说明（file_id 由路径派生，孤儿备注会
        在同路径出现新文件时凭空贴上去）。
        """

        return self._state_store.prune(item.file_id for item in self.store.list_all())

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

        left_time = parse_iso(left.captured_at)
        right_time = parse_iso(right.captured_at)
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
            "left": self._compare_side(left),
            "right": self._compare_side(right),
            "recommended_keep": newer_id,
            "recommended_reason": reason,
        }

    def _compare_side(self, item: PersonalFileItem) -> dict:
        return {
            "file_id": item.file_id,
            "title": item.title,
            "source_app": item.source_app,
            "captured_at": item.captured_at,
            "score_hint": self._match_hint_in_text(item.raw_text),
            "archived": self.is_archived(item.file_id),
            "note": self.get_annotation(item.file_id),
        }

    @staticmethod
    def _match_hint_in_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "无可检索文本"
        return text[:40]

    # ------------------------------------------------------------------
    # 检索闭环
    # ------------------------------------------------------------------

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

        # 查询超长则截断。前端 maxlength 由 index-status 下发的 max_query_len 设置，
        # 正常路径不会走到这里；走到说明调用方绕过了前端（curl / Android / 聊天转搜索），
        # 故补一条 warning，让「尾部线索被丢弃」在日志里可见而不是静默发生 ——
        # 此前的静默截断在用户侧只表现为「搜不准」，且无任何线索可查。
        # 与本项目既有口径一致：只记长度与摘要，不记录查询原文。
        limit = self.config.max_search_query_len
        parsed_query = query[:limit]
        if len(query) > limit:
            _LOGGER.warning(
                "personal-search-query-truncated",
                extra={
                    "event": "search.query_truncated",
                    "trace_id": trace_id,
                    "original_query_len": len(query),
                    "limit": limit,
                    "dropped_chars": len(query) - limit,
                },
            )
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
        state, question, should_disambiguate = decide_state(candidates, force_disambiguation)
        self.sessions.put(
            SearchSession(session_id=search_id, query=parsed_query, candidates=candidates)
        )

        response_candidates = [to_search_candidate(candidate) for candidate in candidates]
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
        return search_id, response_candidates, state, should_disambiguate, question

    def continue_search(
        self,
        session_id: str,
        selected_file_id: str | None,
        reply: str | None,
        top_k: int = 6,
        trace_id: str | None = None,
    ) -> tuple[str, str, list[FileSearchCandidate], str, bool, str | None, str | None]:
        """对 clarify 会话进行下一轮处理。

        返回 (session_id, original_query, candidates, state, should_disambiguate, question, selected_file_id)。
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
                    session.query,
                    [to_search_candidate(selected_candidate)],
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
            candidates = filter_candidates_by_reply(candidates, normalized_reply)
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
                return (
                    session_id,
                    session.query,
                    [],
                    "not_found",
                    False,
                    "没有找到可确认的文件，请再给 1~2 个线索，比如时间/群聊/颜色/场景。",
                    None,
                )

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
            return (
                session_id,
                session.query,
                [],
                "not_found",
                False,
                "没有找到可确认的文件，请再给 1~2 个线索，比如时间/来源/场景。",
                None,
            )

        candidates = rerank_within(
            candidates,
            normalized_reply if normalized_reply else session.query,
            top_k,
            weights=self.score_weights,
            embed_text=self.embed_text,
        )

        state, question, should_disambiguate = decide_state(candidates, False)
        self.sessions.advance(session_id, candidates)
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
            session.query,
            [to_search_candidate(candidate) for candidate in candidates],
            state,
            should_disambiguate,
            question,
            selected,
        )

    def _search_items(
        self,
        query: str,
        top_k: int = 6,
        search_id: str | None = None,
        trace_id: str | None = None,
    ) -> list[SearchCandidate]:
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

        tokens = tokenize(query)
        source_hints = extract_source_hints(query)
        visual_hints = extract_visual_hints(query)
        version_hint = extract_version_hint(query)
        weights = self.score_weights
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

        scored: list[SearchCandidate] = []
        for item in candidate_pool:
            score, evidence, matched = score_item(
                item,
                query,
                tokens=tokens,
                source_hints=source_hints,
                visual_hints=visual_hints,
                version_hint=version_hint,
                query_embedding=query_embedding,
                weights=weights,
            )
            if score <= 0.01:
                continue
            scored.append(SearchCandidate(item=item, score=score, evidence=evidence, matched_clues=matched))

        scored.sort(key=lambda item: item.score, reverse=True)
        deduped = dedupe(scored)[:top_k]
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
