"""指标事件存储：追加式 JSONL。

与业务存储（快照式 tmp+os.replace 原子写）不同，事件流天然 append-only：
最坏情况是崩溃留下半行，读取端按行容错跳过（与 ExpenseStore.from_dict
的容错口径一致）。内存态列表与文件同步维护，compute 只读内存。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock

_LOGGER = logging.getLogger("agent_server.analytics")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


@dataclass
class MetricsEvent:
    event: str
    ts: str
    payload: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "MetricsEvent | None":
        """容错构造：缺 event/ts 的历史行直接丢弃，不让整库加载失败。"""

        event = data.get("event")
        ts = data.get("ts")
        if not event or not ts:
            return None
        payload = data.get("payload")
        return cls(event=str(event), ts=str(ts), payload=payload if isinstance(payload, dict) else {})

    def to_dict(self) -> dict:
        return {"event": self.event, "ts": self.ts, "payload": self.payload}


class MetricsEventStore:
    """线程安全的事件存储（路由端点跑在 FastAPI 线程池里）。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = RLock()
        self._events: list[MetricsEvent] = []
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = MetricsEvent.from_dict(json.loads(line))
                    except Exception:
                        continue  # 半行/脏行跳过
                    if event is not None:
                        self._events.append(event)
        except Exception as exc:  # pragma: no cover
            _LOGGER.warning("metrics-event-store-load-failed: %s", exc)

    def append(self, event: str, payload: dict | None = None, ts: str | None = None) -> MetricsEvent:
        record = MetricsEvent(event=event, ts=ts or _now_iso(), payload=payload or {})
        with self._lock:
            self._events.append(record)
            if self.path:
                try:
                    parent = os.path.dirname(self.path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                except Exception as exc:  # pragma: no cover
                    # 落盘失败不阻断业务：内存态仍在，指标降级为本次进程可见
                    _LOGGER.warning("metrics-event-append-failed: %s", exc)
        return record

    def list_all(self) -> list[MetricsEvent]:
        with self._lock:
            return list(self._events)
