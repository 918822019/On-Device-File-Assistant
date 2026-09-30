"""澄清会话存储与治理：TTL 过期 + 容量上限的最久未用淘汰。"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from time import monotonic

from .relevance import SearchCandidate

# 会话治理：30 分钟未访问自动过期；容量上限 200，超出按最久未用淘汰。
# 此前会话字典只进不出，长期运行内存只增不减。
SESSION_TTL_SECONDS = 30 * 60.0
SESSION_MAX_COUNT = 200


@dataclass
class SearchSession:
    """单次搜索会话上下文，保存候选和对话轮次。"""

    session_id: str
    query: str
    candidates: list[SearchCandidate]
    turn: int = 0
    last_access: float = field(default_factory=monotonic)


class SessionStore:
    """线程安全的会话表；读命中即刷新 last_access。"""

    def __init__(self, ttl_seconds: float = SESSION_TTL_SECONDS, max_count: int = SESSION_MAX_COUNT) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_count = max_count
        self._sessions: dict[str, SearchSession] = {}
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def get(self, session_id: str) -> SearchSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_access = monotonic()
            return session

    def put(self, session: SearchSession) -> None:
        """写入新会话前先做惰性淘汰（过期 + 超容）。"""

        with self._lock:
            self._evict_locked()
            session.last_access = monotonic()
            self._sessions[session.session_id] = session

    def advance(self, session_id: str, candidates: list[SearchCandidate]) -> SearchSession | None:
        """推进一轮澄清：轮次 +1、替换候选并刷新访问时间。"""

        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.turn += 1
            session.candidates = candidates
            session.last_access = monotonic()
            return session

    def _evict_locked(self) -> None:
        """淘汰过期与超容会话（调用方须持有 _lock）。"""

        now = monotonic()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_access > self.ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]
        overflow = len(self._sessions) - self.max_count
        if overflow > 0:
            oldest = sorted(self._sessions.items(), key=lambda kv: kv[1].last_access)[:overflow]
            for session_id, _ in oldest:
                del self._sessions[session_id]
