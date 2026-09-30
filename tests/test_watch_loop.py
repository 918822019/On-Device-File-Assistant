"""common/watch_loop：后台摄取循环骨架与 IngestResult。

这条循环是索引保持新鲜的**唯一自动机制**，而它此前的两个实现都是
``except Exception: pass`` —— 守护线程里的持续失败被完全吞掉，连一行日志都没有。
外部看到的现象只是「新文件搜不到」，与 R14「/health 恒绿」是同一类可观测性幻觉。

所以这里的测试重点是**失败必须可见**，而不是「循环能转」：

- tick 抛异常不得打断循环（否则一次偶发错误就让索引永久停更）
- 每次失败都要落一条 warning，含异常类型名与连续失败次数
- 连续失败计数在成功一轮后归零（区分「偶发一次」与「一直在坏」）
- interval <= 0 必须被夹到下限：``Event.wait(0)`` 立即返回，配上永不 set 的
  stop_event 就是一个 100% CPU 的忙等循环，而 ``*_INTERVAL_SECONDS`` 是环境变量，
  写错成 0 不会有任何报错
- 日志只记异常**类型名**不记 message：消息里常带完整文件路径
"""

from __future__ import annotations

import logging
from threading import Event

import pytest

from edge_cloud_agent.common.watch_loop import (
    _MIN_INTERVAL_SECONDS,
    IngestResult,
    run_watch_loop,
)


@pytest.fixture()
def logger(caplog: pytest.LogCaptureFixture) -> logging.Logger:
    log = logging.getLogger("test.watch_loop")
    log.handlers.clear()
    log.propagate = True
    caplog.set_level(logging.DEBUG, logger="test.watch_loop")
    return log


def _events(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.event for r in caplog.records if getattr(r, "event", None)]


# ------------------------------------------------------------------ IngestResult


def test_ingest_result_defaults():
    r = IngestResult(scanned=3, imported=1, skipped=2, errors=0)
    assert (r.removed, r.imported_ids) == (0, [])


def test_ingest_result_field_is_business_neutral():
    """internal 名字不得再叫 material_ids —— personal 侧装的是 file_id。

    wire 字段名仍是 material_ids（Android 用 @SerializedName 钉住了），
    映射发生在 routers 层。
    """

    fields = set(IngestResult.__dataclass_fields__)
    assert "imported_ids" in fields
    assert "material_ids" not in fields


def test_both_ingest_modules_share_one_result_type():
    from edge_cloud_agent.expense.ingest import IngestResult as ExpenseResult
    from edge_cloud_agent.personal_search.ingest import IngestResult as PersonalResult

    assert PersonalResult is IngestResult
    assert ExpenseResult is IngestResult


# ------------------------------------------------------------------ 循环调度


def test_tick_runs_immediately_without_waiting_one_interval(logger, caplog):
    """进入即跑第一轮：启动时的首轮扫描负责补上停机期间的新文件。"""

    calls: list[int] = []
    stop = Event()

    def tick():
        calls.append(1)
        stop.set()  # 第一轮之后就退出，测试不必真的等一个 interval

    run_watch_loop(tick, stop, interval=3600, logger=logger)
    assert len(calls) == 1


def test_loop_stops_on_stop_event(logger):
    stop = Event()
    calls: list[int] = []

    def tick():
        calls.append(1)
        if len(calls) >= 3:
            stop.set()

    run_watch_loop(tick, stop, interval=0.01, logger=logger)
    assert len(calls) == 3


def test_loop_does_nothing_when_stop_event_is_already_set(logger):
    stop = Event()
    stop.set()
    calls: list[int] = []
    run_watch_loop(lambda: calls.append(1), stop, interval=0.01, logger=logger)
    assert calls == []


def test_loop_uses_the_default_logger_when_none_given(caplog):
    """不传 logger 也不能静默：必须落到 root logger 上。"""

    stop = Event()
    caplog.set_level(logging.INFO)
    run_watch_loop(lambda: stop.set(), stop, interval=0.01)
    assert "watch.loop_started" in _events(caplog)


# ------------------------------------------------------------------ 失败可见


def test_tick_exception_does_not_break_the_loop(logger, caplog):
    stop = Event()
    calls: list[int] = []

    def tick():
        calls.append(1)
        if len(calls) >= 4:
            stop.set()
        raise RuntimeError("boom")

    run_watch_loop(tick, stop, interval=0.01, logger=logger)
    assert len(calls) == 4, "异常打断了循环 —— 索引会永久停更"


def test_every_failure_is_logged_with_type_and_consecutive_count(logger, caplog):
    stop = Event()
    n = {"i": 0}

    def tick():
        n["i"] += 1
        if n["i"] >= 3:
            stop.set()
        raise ValueError("nope")

    run_watch_loop(tick, stop, interval=0.01, logger=logger)

    failures = [r for r in caplog.records if getattr(r, "event", "") == "watch.tick_failed"]
    assert len(failures) == 3
    assert [r.consecutive_failures for r in failures] == [1, 2, 3]
    assert [r.total_failures for r in failures] == [1, 2, 3]
    assert all(r.exception == "ValueError" for r in failures)


def test_failure_log_does_not_leak_the_exception_message(logger, caplog):
    """异常消息里常带完整文件路径，与项目「日志不落用户内容」的口径冲突。

    必须断言 ``record.exception``（即 extra 里的那一项）而不是 ``getMessage()``：
    固定消息串是 ``watch-loop-tick-failed``，extra 载荷不在里面 —— 但它会被
    ``logging_config._ExtraFormatter`` 渲染进真实日志输出（那正是上一轮修的
    「全代码库 50 个调用点的 extra 全部被丢弃」）。只看 getMessage() 的话，
    把 ``type(exc).__name__`` 改成 ``str(exc)`` 这条测试照样绿。
    """

    secret = "/Users/somebody/private/报销单.pdf 读不出来"
    stop = Event()

    def tick():
        stop.set()
        raise ValueError(secret)

    run_watch_loop(tick, stop, interval=0.01, logger=logger)

    failures = [r for r in caplog.records if getattr(r, "event", "") == "watch.tick_failed"]
    assert len(failures) == 1
    assert failures[0].exception == "ValueError"
    rendered = "\n".join(r.getMessage() for r in caplog.records) + repr(
        [getattr(r, "exception", "") for r in caplog.records]
    )
    assert "somebody" not in rendered
    assert "报销单" not in rendered


def test_consecutive_counter_resets_after_a_successful_tick(logger, caplog):
    stop = Event()
    seq = iter([True, False, True])  # True = 抛异常
    n = {"i": 0}

    def tick():
        n["i"] += 1
        if n["i"] >= 3:
            stop.set()
        if next(seq):
            raise RuntimeError("x")

    run_watch_loop(tick, stop, interval=0.01, logger=logger)
    failures = [r for r in caplog.records if getattr(r, "event", "") == "watch.tick_failed"]
    # 失败、成功、失败 —— 连续计数必须回到 1，否则「一直在坏」与「偶发」无法区分
    assert [r.consecutive_failures for r in failures] == [1, 1]
    assert [r.total_failures for r in failures] == [1, 2]


def test_clean_run_logs_no_failure(logger, caplog):
    stop = Event()
    run_watch_loop(lambda: stop.set(), stop, interval=0.01, logger=logger)
    assert "watch.tick_failed" not in _events(caplog)
    assert _events(caplog) == ["watch.loop_started", "watch.loop_stopped"]


def test_loop_stopped_carries_the_run_summary(logger, caplog):
    stop = Event()
    n = {"i": 0}

    def tick():
        n["i"] += 1
        if n["i"] >= 2:
            stop.set()
        if n["i"] == 1:
            raise RuntimeError("x")

    run_watch_loop(tick, stop, interval=0.01, logger=logger)
    stopped = [r for r in caplog.records if getattr(r, "event", "") == "watch.loop_stopped"]
    assert len(stopped) == 1
    assert (stopped[0].ticks, stopped[0].total_failures) == (1, 1)


# ------------------------------------------------------------------ interval 夹取


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_non_positive_interval_is_clamped_and_warned(logger, caplog, bad):
    """interval<=0 会让 Event.wait 立即返回 → 100% CPU 忙等循环。"""

    stop = Event()
    run_watch_loop(lambda: stop.set(), stop, interval=bad, logger=logger)

    clamped = [r for r in caplog.records if getattr(r, "event", "") == "watch.interval_clamped"]
    assert len(clamped) == 1
    assert clamped[0].configured_interval == bad
    assert clamped[0].effective_interval == _MIN_INTERVAL_SECONDS

    started = [r for r in caplog.records if getattr(r, "event", "") == "watch.loop_started"]
    assert started[0].interval_seconds == _MIN_INTERVAL_SECONDS


def test_small_positive_interval_is_left_alone(logger, caplog):
    """正的小数（测试与快速轮询用）不得被夹取。"""

    stop = Event()
    run_watch_loop(lambda: stop.set(), stop, interval=0.01, logger=logger)
    assert "watch.interval_clamped" not in _events(caplog)


# ------------------------------------------------------------------ 命名


def test_event_prefix_and_loop_name_are_honoured(caplog):
    stop = Event()
    log = logging.getLogger("test.named")
    caplog.set_level(logging.INFO, logger="test.named")
    run_watch_loop(
        lambda: stop.set(), stop, interval=0.01, logger=log,
        loop_name="expense-watch", event_prefix="expense.watch",
    )
    events = [r.event for r in caplog.records if getattr(r, "event", None)]
    assert events == ["expense.watch.loop_started", "expense.watch.loop_stopped"]
    assert any(r.getMessage() == "expense-watch-started" for r in caplog.records)
