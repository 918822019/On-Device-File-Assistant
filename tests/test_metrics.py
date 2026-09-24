"""复盘指标：事件存储容错、报销回访/补齐/纠正口径、文件搜索追问收敛漏斗。

时间全部用显式 ts 注入（store.append 的 ts 参数），不依赖真实时钟。
"""

import json

import pytest

from edge_cloud_agent.analytics.service import MetricsService
from edge_cloud_agent.analytics.storage import MetricsEventStore


@pytest.fixture()
def svc(tmp_path):
    store = MetricsEventStore(str(tmp_path / "events.jsonl"))
    return MetricsService(store)


# ------------------------------------------------------------------ 存储


def test_store_roundtrip_and_bad_lines(tmp_path):
    path = tmp_path / "ev.jsonl"
    store = MetricsEventStore(str(path))
    store.append("expense.collect", {"claim_id": "c1"})
    # 人为写入脏行：半行 JSON、缺字段行
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"event": "broken"\n')
        fh.write(json.dumps({"ts": "2026-09-01T00:00:00"}) + "\n")  # 缺 event
        fh.write("\n")

    reloaded = MetricsEventStore(str(path))
    events = reloaded.list_all()
    assert len(events) == 1, "脏行/缺字段行应被跳过，不炸整库"
    assert events[0].event == "expense.collect"


# ------------------------------------------------------------------ 报销口径


def test_revisit_within_window_same_second(svc):
    """ts 秒级同秒时按事件顺序判定：collect 之后的 search 即算回访。"""

    svc.store.append("expense.collect", {"claim_id": "c1"}, ts="2026-09-01T10:00:00")
    svc.store.append("expense.search", {"claim_id": "c1"}, ts="2026-09-01T10:00:00")
    e = svc.compute()["expense"]
    assert (e["claims_total"], e["claims_revisited"], e["revisit_rate"]) == (1, 1, 1.0)


def test_revisit_outside_window_not_counted(svc):
    svc.store.append("expense.collect", {"claim_id": "c1"}, ts="2026-09-01T10:00:00")
    svc.store.append("expense.search", {"claim_id": "c1"}, ts="2026-09-16T10:00:01")  # 15 天后
    e = svc.compute()["expense"]
    assert e["claims_revisited"] == 0
    assert e["revisit_rate"] == 0.0


def test_search_before_collect_not_revisit(svc):
    """顺序语义：先 search 后 collect，search 不构成回访。"""

    svc.store.append("expense.search", {"claim_id": "c1"}, ts="2026-09-01T09:00:00")
    svc.store.append("expense.collect", {"claim_id": "c1"}, ts="2026-09-01T10:00:00")
    e = svc.compute()["expense"]
    assert e["claims_revisited"] == 0


def test_search_without_claim_id_ignored(svc):
    svc.store.append("expense.collect", {"claim_id": "c1"}, ts="2026-09-01T10:00:00")
    svc.store.append("expense.search", {"claim_id": None}, ts="2026-09-02T10:00:00")
    assert svc.compute()["expense"]["claims_revisited"] == 0


def test_followup_completed_within_hour(svc):
    svc.store.append(
        "expense.collect",
        {"claim_id": "c1", "needs_follow_up": True, "missing_required_types": ["invoice"]},
        ts="2026-09-01T10:00:00",
    )
    svc.store.append(
        "expense.collect",
        {"claim_id": "c1", "needs_follow_up": False, "missing_required_types": []},
        ts="2026-09-01T10:30:00",
    )
    e = svc.compute()["expense"]
    assert (e["followup_needed_claims"], e["followup_completed"], e["followup_completion_rate"]) == (1, 1, 1.0)


def test_followup_too_late_not_completed(svc):
    svc.store.append(
        "expense.collect",
        {"claim_id": "c1", "needs_follow_up": True, "missing_required_types": ["invoice"]},
        ts="2026-09-01T10:00:00",
    )
    svc.store.append(
        "expense.collect",
        {"claim_id": "c1", "needs_follow_up": False, "missing_required_types": []},
        ts="2026-09-01T12:00:00",  # 2 小时后
    )
    e = svc.compute()["expense"]
    assert (e["followup_needed_claims"], e["followup_completed"]) == (1, 0)


def test_corrections_counted_only_when_fields_changed(svc):
    svc.store.append(
        "expense.correct",
        {"claim_id": "c1", "material_id": "m1", "corrected_fields": ["extracted_amount"]},
        ts="2026-09-01T10:00:00",
    )
    svc.store.append(
        "expense.correct",
        {"claim_id": "c1", "material_id": "m1", "corrected_fields": ["merchant"]},
        ts="2026-09-01T10:05:00",
    )
    svc.store.append(
        "expense.correct",
        {"claim_id": "c1", "material_id": "m2", "corrected_fields": []},  # 无变化不计数
        ts="2026-09-01T10:06:00",
    )
    e = svc.compute()["expense"]
    assert (e["corrections_total"], e["corrected_materials"]) == (2, 1)


def test_rates_none_when_no_samples(svc):
    e = svc.compute()["expense"]
    assert e["revisit_rate"] is None
    assert e["followup_completion_rate"] is None


# ------------------------------------------------------------------ 文件搜索漏斗


def test_funnel_clarify_resolve_and_execute(svc):
    svc.store.append("filesearch.state", {"session_id": "s1", "state": "needs_clarification", "round": "search"})
    svc.store.append("filesearch.state", {"session_id": "s1", "state": "resolved", "round": "clarify"})
    svc.store.append("filesearch.action", {"session_id": "s1", "action": "open", "status": "ok"})
    f = svc.compute()["file_search"]
    assert f["sessions_total"] == 1
    assert f["needs_clarification"] == 1
    assert f["clarify_resolved"] == 1
    assert f["clarify_resolved_executed"] == 1
    assert f["clarify_resolve_rate"] == 1.0
    assert f["clarify_exec_rate"] == 1.0


def test_funnel_resolved_but_action_failed(svc):
    svc.store.append("filesearch.state", {"session_id": "s1", "state": "needs_clarification", "round": "search"})
    svc.store.append("filesearch.state", {"session_id": "s1", "state": "resolved", "round": "clarify"})
    svc.store.append("filesearch.action", {"session_id": "s1", "action": "open", "status": "failed"})
    f = svc.compute()["file_search"]
    assert f["clarify_resolved"] == 1
    assert f["clarify_resolved_executed"] == 0
    assert f["clarify_exec_rate"] == 0.0


def test_funnel_action_without_session_not_counted(svc):
    svc.store.append("filesearch.state", {"session_id": "s1", "state": "needs_clarification", "round": "search"})
    svc.store.append("filesearch.state", {"session_id": "s1", "state": "resolved", "round": "clarify"})
    svc.store.append("filesearch.action", {"session_id": None, "action": "open", "status": "ok"})
    assert svc.compute()["file_search"]["clarify_resolved_executed"] == 0


def test_funnel_direct_resolve_not_in_clarify_denominator(svc):
    svc.store.append("filesearch.state", {"session_id": "s1", "state": "resolved", "round": "search"})
    svc.store.append("filesearch.state", {"session_id": "s2", "state": "not_found", "round": "search"})
    f = svc.compute()["file_search"]
    assert f["resolved_direct"] == 1
    assert f["not_found"] == 1
    assert f["needs_clarification"] == 0
    assert f["clarify_resolve_rate"] is None, "分母 0 → None，不以 0.0 冒充"
