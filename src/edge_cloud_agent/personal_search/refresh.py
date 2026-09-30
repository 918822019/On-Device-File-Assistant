"""搜索请求触发的单实例节流刷新，避免将全量扫描放在请求路径上。"""

import logging
import threading
import time

from ..config import PersonalFileConfig
from .ingest import run_once
from .service import PersonalFileSearchService

_LOGGER = logging.getLogger("agent_server.personal_search.router")


class IndexRefreshScheduler:
    """每个应用实例各自管理扫描中的状态和最近触发时间。"""

    def __init__(self, service: PersonalFileSearchService, cfg: PersonalFileConfig) -> None:
        self.service = service
        self.cfg = cfg
        self._lock = threading.Lock()
        self._last_attempt = 0.0
        self._running = False

    def trigger(self, trace_id: str | None = None) -> None:
        """窗口内最多触发一次后台扫描，失败也释放单飞锁。"""

        if not self.cfg.source_dir:
            return
        now = time.monotonic()
        window = max(5, self.cfg.scan_interval_seconds)
        with self._lock:
            if self._running or now - self._last_attempt < window:
                return
            self._last_attempt = now
            self._running = True

        def _worker() -> None:
            try:
                run_once(self.service, self.cfg, trace_id=trace_id)
            except Exception:
                _LOGGER.warning(
                    "personal-search-api-refresh-failed",
                    extra={
                        "event": "api.search.refresh_failed",
                        "trace_id": trace_id,
                        "path": "/v1/search-agent/search",
                    },
                )
            finally:
                with self._lock:
                    self._running = False

        try:
            threading.Thread(target=_worker, name="personal-file-refresh", daemon=True).start()
        except Exception:
            with self._lock:
                self._running = False
            raise
