"""后台刷新按应用实例单飞、节流，避免搜索请求执行同步扫描。"""

import threading
from types import SimpleNamespace

from edge_cloud_agent.personal_search import refresh


def test_refresh_is_single_flight_per_instance_and_throttled(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    calls = []
    clock = [1000.0]

    def scan(service, cfg, trace_id=None):
        calls.append((service, trace_id))
        entered.set()
        assert release.wait(2)
        completed.set()

    monkeypatch.setattr(refresh, "run_once", scan)
    monkeypatch.setattr(refresh.time, "monotonic", lambda: clock[0])
    cfg = SimpleNamespace(source_dir="/tmp/docs", scan_interval_seconds=60)
    scheduler = refresh.IndexRefreshScheduler(object(), cfg)
    scheduler.trigger("first")
    assert entered.wait(2)
    scheduler.trigger("same-instance")
    assert len(calls) == 1

    release.set()
    assert completed.wait(2)
    # worker 最终释放锁后，即使窗口内再次请求也不能重复扫描。
    for _ in range(50):
        with scheduler._lock:
            if not scheduler._running:
                break
        threading.Event().wait(0.01)
    else:
        raise AssertionError("refresh worker did not release its lock")
    clock[0] += 30
    scheduler.trigger("within-window")
    assert len(calls) == 1
    clock[0] += 31
    scheduler.trigger("after-window")
    assert len(calls) == 2


def test_refresh_schedulers_do_not_share_state(monkeypatch):
    completed = threading.Event()
    calls = []

    def scan(service, cfg, trace_id=None):
        calls.append(service)
        if len(calls) == 2:
            completed.set()

    monkeypatch.setattr(refresh, "run_once", scan)
    monkeypatch.setattr(refresh.time, "monotonic", lambda: 1000.0)
    cfg = SimpleNamespace(source_dir="/tmp/docs", scan_interval_seconds=60)
    first, second = object(), object()
    refresh.IndexRefreshScheduler(first, cfg).trigger()
    refresh.IndexRefreshScheduler(second, cfg).trigger()
    assert completed.wait(2)
    assert set(calls) == {first, second}
