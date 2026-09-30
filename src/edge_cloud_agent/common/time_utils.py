"""统一的时间工具：naive UTC 时间戳与 ISO 解析。

datetime.utcnow() 在 3.12+ 已弃用；存储与日志中的时间格式（naive UTC，秒精度）
由本模块统一，避免各业务模块各写一份。
"""

from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    """naive UTC now。"""

    return datetime.now(timezone.utc).replace(tzinfo=None)


def now_iso() -> str:
    """标准 UTC 时间字符串，统一日志和存储中的时间展示。"""

    return utcnow_naive().isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    """安全转换 ISO 字符串为 datetime，失败时返回 None。"""

    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None
