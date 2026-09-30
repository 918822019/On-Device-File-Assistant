"""服务日志格式和请求上下文缺省字段。"""

import logging
import os
from typing import ClassVar

# LogRecord 自带的属性 + 已在固定格式串里渲染过的字段。
# 不在此集合中的 record 属性一律视为业务 extra，追加输出。
_RESERVED_ATTRS: frozenset[str] = frozenset({
    # logging 内部
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
    # 固定格式串已渲染
    "event", "trace_id", "method", "path", "status_code", "duration_ms",
})

# 列表/元组类字段最多渲染几个元素（sample_file_ids 之类可能很长）
_MAX_SEQ_ITEMS = 5
# 单个字符串值最长渲染多少字符
_MAX_STR_LEN = 120


def _render(value: object) -> str:
    """把 extra 值压成单行、无空格的 key=value 形式，便于 grep 与肉眼扫读。"""

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        head = ",".join(_render(item) for item in items[:_MAX_SEQ_ITEMS])
        if len(items) > _MAX_SEQ_ITEMS:
            head += f",…(+{len(items) - _MAX_SEQ_ITEMS})"
        return f"[{head}]" if head else "[]"
    if isinstance(value, dict):
        inner = ",".join(f"{k}={_render(v)}" for k, v in sorted(value.items()))
        return "{" + inner + "}"
    text = str(value).replace("\n", "\\n").replace(" ", "_")
    if len(text) > _MAX_STR_LEN:
        text = text[:_MAX_STR_LEN] + "…"
    return text or "-"


class _ExtraFormatter(logging.Formatter):
    """在固定格式串之后追加 extra 里的其余字段。

    为什么需要它：全代码库给 ``_LOGGER.info(...)`` 传了丰富的 ``extra`` 载荷
    （scanned / imported / skipped / removed / state / cost_ms / rehashed /
    mtime_backfilled / candidate_pool / query_len …），但原先的固定格式串只渲染
    event / trace_id / method / path / status_code / duration_ms 六个字段，
    其余全部挂在 LogRecord 上却**从不输出**。后果是 docs/API.md 的「日志字典」
    形同虚设：日志里只能看到事件名，看不到任何一个业务数值，排障时无从下手
    （例如 ingest 每轮扫了多少、跳过多少、清理了多少幽灵记录，全都不可见）。

    这里统一在行尾追加，而不是逐个改约 50 个调用点。
    """

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras: list[tuple[str, str]] = []
        for key in sorted(k for k in record.__dict__ if k not in _RESERVED_ATTRS and not k.startswith("_")):
            rendered = _render(record.__dict__[key])
            # 跳过占位符：_DefaultLogContext 会给每条记录补上 query_digest /
            # search_id / session_id / file_id / action / top_k / state 的默认值
            # "-"，而这些字段并不在固定格式串里。不过滤的话每行日志都会多出一串
            # action=- file_id=- … 的噪音。真实值不受影响，照常渲染。
            if rendered == "-":
                continue
            extras.append((key, rendered))
        if not extras:
            return base
        return base + " " + " ".join(f"{key}={value}" for key, value in extras)


class _DefaultLogContext(logging.Filter):
    """为结构化日志补齐字段，避免 formatter 因缺字段抛异常。"""

    _defaults: ClassVar[dict[str, str]] = {
        "event": "-",
        "trace_id": "-",
        "method": "-",
        "path": "-",
        "status_code": "-",
        "duration_ms": "-",
        "query_digest": "-",
        "search_id": "-",
        "session_id": "-",
        "file_id": "-",
        "action": "-",
        "top_k": "-",
        "state": "-",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self._defaults.items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def configure_logging() -> None:
    """读取 APP_LOG_LEVEL，给根 logger 配置统一日志格式。"""

    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = _ExtraFormatter(
        "%(asctime)s %(levelname)s %(name)s event=%(event)s trace_id=%(trace_id)s "
        "method=%(method)s path=%(path)s status=%(status_code)s dur_ms=%(duration_ms)s %(message)s"
    )

    handlers = root_logger.handlers or [logging.StreamHandler()]
    if not root_logger.handlers:
        root_logger.addHandler(handlers[0])

    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        handler.addFilter(_DefaultLogContext())
