"""后台摄取循环骨架与统一的摄取结果 DTO。

两条业务线各有一份 ``start_watch_loop``，都是同一段十行代码：``while not
stop_event.is_set(): try: run_once() except Exception: pass; stop_event.wait(delay)``。

那个 ``except Exception: pass`` 是这里要解决的真问题，不是重复本身：守护线程里
的异常被完全吞掉，**连一行日志都没有**。watch 循环是索引保持新鲜的唯一自动
机制，它一旦持续失败（配置目录权限变了、store 落盘失败、embedding 运行时挂了），
外部看到的现象只是「新文件搜不到」—— 与 R14「/health 恒绿」是同一类可观测性
幻觉，而且更隐蔽，因为连降级路径都没走到。

本模块只负责循环本身：调度、停止、失败可见。``run_once`` 里的业务串行锁留在
各业务模块（它要保护的是 store 与向量索引的写入，而 ``run_once`` 有多个入口：
watch 线程、``/search`` 的节流刷新、``/rebuild-index``）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event

#: interval <= 0 时的兜底秒数。``Event.wait(0)`` 会立即返回，配上一个永远不
#: set 的 stop_event 就是一个 100% CPU 的忙等循环 —— 而 ``*_INTERVAL_SECONDS``
#: 是环境变量，写错成 0 不会有任何报错。正的小数（测试用）不受影响。
_MIN_INTERVAL_SECONDS = 1.0


@dataclass
class IngestResult:
    """一轮摄取的统计。两条业务线共用同一份定义。

    ``imported_ids`` 是本轮新导入记录的主键（personal 侧是 ``file_id``，
    expense 侧是 ``material_id``）。此前两份副本都叫 ``material_ids`` ——
    personal 侧那份是照抄 expense 留下的名字，装的其实是 file_id。

    HTTP 响应里的字段名仍是 ``material_ids``：Android 客户端用
    ``@SerializedName("material_ids")`` 钉住了它（见
    ``android/.../model/ApiModels.kt``），改名会静默让客户端拿到空列表。
    映射发生在 routers 层，内部名字不再误导。
    """

    scanned: int
    imported: int
    skipped: int
    errors: int
    removed: int = 0
    imported_ids: list[str] = field(default_factory=list)


def run_watch_loop(
    tick: Callable[[], object],
    stop_event: Event,
    *,
    interval: float,
    logger: logging.Logger | None = None,
    loop_name: str = "watch-loop",
    event_prefix: str = "watch",
) -> None:
    """按 ``interval`` 反复调用 ``tick()``，直到 ``stop_event`` 被 set。

    - 进入即先跑一轮（不等第一个间隔），与原实现一致：启动时的首轮扫描负责
      把上次停机期间的新文件补进来
    - ``tick`` 抛出的异常**永不打断循环**，但会记一条 warning，含异常类型名
      与连续失败次数。只记类型名不记 message：异常消息里常带完整文件路径，
      与项目「日志不落用户内容」的口径一致（参见 ``ingest.item_failed``）
    - 连续失败次数在成功一轮后归零，便于区分「偶发一次」与「一直在坏」
    """

    log = logger or logging.getLogger("agent_server.watch")
    delay = interval if interval > 0 else _MIN_INTERVAL_SECONDS
    if delay != interval:
        log.warning(
            f"{loop_name}-interval-clamped",
            extra={
                "event": f"{event_prefix}.interval_clamped",
                "configured_interval": interval,
                "effective_interval": delay,
            },
        )

    consecutive_failures = 0
    total_failures = 0
    ticks = 0
    log.info(
        f"{loop_name}-started",
        extra={"event": f"{event_prefix}.loop_started", "interval_seconds": delay},
    )

    while not stop_event.is_set():
        try:
            tick()
            ticks += 1
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            total_failures += 1
            log.warning(
                f"{loop_name}-tick-failed",
                extra={
                    "event": f"{event_prefix}.tick_failed",
                    "exception": type(exc).__name__,
                    "consecutive_failures": consecutive_failures,
                    "total_failures": total_failures,
                },
            )
        stop_event.wait(delay)

    log.info(
        f"{loop_name}-stopped",
        extra={
            "event": f"{event_prefix}.loop_stopped",
            "ticks": ticks,
            "total_failures": total_failures,
        },
    )
