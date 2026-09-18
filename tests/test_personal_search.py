"""个人文件搜索：时间线索匹配、状态判定、clarify 重排回归。"""

from datetime import datetime, timedelta

import pytest

from edge_cloud_agent.config import PersonalFileConfig
from edge_cloud_agent.personal_search.service import (
    PersonalFileSearchService,
    _has_time_match,
    _SearchCandidate,
)
from edge_cloud_agent.personal_search.storage import PersonalFileItem, PersonalFileStore


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _make_item(
    file_id: str,
    title: str,
    raw_text: str = "",
    captured_at: str | None = None,
    source_app: str = "local",
    doc_type: str = "file",
) -> PersonalFileItem:
    return PersonalFileItem(
        file_id=file_id,
        title=title,
        file_uri=f"file:///tmp/{title}",
        source_app=source_app,
        doc_type=doc_type,
        mime_type="text/plain",
        raw_text=raw_text,
        summary=raw_text[:110],
        file_path=f"/tmp/{title}",
        captured_at=captured_at,
    )


@pytest.fixture()
def service(tmp_path) -> PersonalFileSearchService:
    cfg = PersonalFileConfig(
        store_path=str(tmp_path / "store.jsonl"),
        faiss_index_path=str(tmp_path / "idx.index"),
        enable_faiss=False,
    )
    store = PersonalFileStore(cfg.store_path)
    return PersonalFileSearchService(config=cfg, store=store, embedding_runtime=None)


# ---------------------------------------------------------------------------
# _has_time_match：captured_at 修复后时间线索才有意义，这里锁住匹配语义
# ---------------------------------------------------------------------------

def test_time_match_today():
    ok, reason = _has_time_match(_iso(datetime.utcnow()), "今天的会议记录")
    assert ok and "今天" in reason


def test_time_match_yesterday():
    ok, reason = _has_time_match(_iso(datetime.utcnow() - timedelta(days=1)), "昨天拍的照片")
    assert ok and "昨天" in reason


def test_time_match_last_week_within_window():
    ok, _ = _has_time_match(_iso(datetime.utcnow() - timedelta(days=5)), "上周的聚餐照片")
    assert ok


def test_time_no_match_last_week_for_old_file():
    # 60 天前的文件不应命中"上周"
    ok, _ = _has_time_match(_iso(datetime.utcnow() - timedelta(days=60)), "上周的聚餐照片")
    assert not ok


def test_time_match_recent_n_days():
    ok, _ = _has_time_match(_iso(datetime.utcnow() - timedelta(days=2)), "最近3天的文件")
    assert ok
    ok, _ = _has_time_match(_iso(datetime.utcnow() - timedelta(days=10)), "最近3天的文件")
    assert not ok


def test_time_no_captured_at_never_matches():
    ok, _ = _has_time_match(None, "昨天的照片")
    assert not ok


# ---------------------------------------------------------------------------
# _decide_state：resolved / needs_clarification / not_found 判定
# ---------------------------------------------------------------------------

def test_decide_state_empty_not_found(service):
    state, question, disambiguate = service._decide_state([], False)
    assert state == "not_found"
    assert question
    assert not disambiguate


def test_decide_state_single_resolved(service):
    cand = _SearchCandidate(item=_make_item("f1", "a"), score=0.8, evidence="", matched_clues=[])
    state, _, disambiguate = service._decide_state([cand], False)
    assert state == "resolved"
    assert not disambiguate


def test_decide_state_big_gap_resolved(service):
    c1 = _SearchCandidate(item=_make_item("f1", "a"), score=0.9, evidence="", matched_clues=[])
    c2 = _SearchCandidate(item=_make_item("f2", "b"), score=0.4, evidence="", matched_clues=[])
    state, _, _ = service._decide_state([c1, c2], False)
    assert state == "resolved"


def test_decide_state_close_scores_need_clarification(service):
    c1 = _SearchCandidate(item=_make_item("f1", "a"), score=0.60, evidence="", matched_clues=[])
    c2 = _SearchCandidate(item=_make_item("f2", "b"), score=0.55, evidence="", matched_clues=[])
    state, question, disambiguate = service._decide_state([c1, c2], False)
    assert state == "needs_clarification"
    assert question and disambiguate


def test_decide_state_force_disambiguation(service):
    cand = _SearchCandidate(item=_make_item("f1", "a"), score=0.8, evidence="", matched_clues=[])
    state, _, disambiguate = service._decide_state([cand], True)
    assert state == "needs_clarification"
    assert disambiguate


# ---------------------------------------------------------------------------
# _rerank_within 回归：reply 指向另一名候选时必须能翻盘，且分值不越界
# （旧实现对上一轮第一名硬加 +1.0，追问永远收敛不到别的候选）
# ---------------------------------------------------------------------------

def test_rerank_reply_can_flip_order(service):
    item1 = _make_item("fm_aaa1", "会议纪要", raw_text="会议 纪要 讨论 项目进度")
    item2 = _make_item("fm_bbb2", "聚餐照片", raw_text="部门 聚餐 烧烤")
    c1 = _SearchCandidate(item=item1, score=0.9, evidence="e1", matched_clues=["会议"])
    c2 = _SearchCandidate(item=item2, score=0.5, evidence="e2", matched_clues=[])

    ranked = service._rerank_within([c1, c2], "聚餐", top_k=2)

    assert ranked[0].item.file_id == "fm_bbb2", "reply 明确指向聚餐，旧第一名不应被硬保送"
    assert all(0.0 <= r.score <= 1.0 for r in ranked), "重排后分值必须仍在 [0,1]"


def test_rerank_keeps_prior_when_reply_has_no_signal(service):
    item1 = _make_item("fm_aaa1", "会议纪要", raw_text="会议 纪要")
    item2 = _make_item("fm_bbb2", "聚餐照片", raw_text="部门 聚餐")
    c1 = _SearchCandidate(item=item1, score=0.9, evidence="e1", matched_clues=[])
    c2 = _SearchCandidate(item=item2, score=0.5, evidence="e2", matched_clues=[])

    # reply 无任何可匹配线索时，先验权重应保住原排序，而不是随机塌缩
    ranked = service._rerank_within([c1, c2], "呃", top_k=2)
    assert ranked[0].item.file_id == "fm_aaa1"


def test_rerank_empty(service):
    assert service._rerank_within([], "任意", top_k=5) == []


# ---------------------------------------------------------------------------
# _filter_candidates_by_reply 回归：时间/来源类回复不应被字面过滤滤空
# （旧实现只做字面 token 匹配，"上周的那张"这类最常见的澄清回复直接 not_found）
# ---------------------------------------------------------------------------

def _days_ago_iso(days: int) -> str:
    return _iso(datetime.utcnow() - timedelta(days=days))


def test_filter_reply_by_time_clue(service):
    item_new = _make_item("fm_new1", "聚餐照片", captured_at=_days_ago_iso(5))
    item_old = _make_item("fm_old1", "团建聚餐", captured_at=_days_ago_iso(40))
    cands = [
        _SearchCandidate(item=item_new, score=0.5, evidence="", matched_clues=[]),
        _SearchCandidate(item=item_old, score=0.4, evidence="", matched_clues=[]),
    ]
    kept = service._filter_candidates_by_reply(cands, "上周的那张")
    assert [c.item.file_id for c in kept] == ["fm_new1"]


def test_filter_reply_by_source_clue(service):
    item_wx = _make_item("fm_wx01", "聚餐照片", source_app="wechat")
    item_local = _make_item("fm_lo01", "团建聚餐", source_app="local")
    cands = [
        _SearchCandidate(item=item_wx, score=0.5, evidence="", matched_clues=[]),
        _SearchCandidate(item=item_local, score=0.4, evidence="", matched_clues=[]),
    ]
    kept = service._filter_candidates_by_reply(cands, "微信群里那张照片")
    assert [c.item.file_id for c in kept] == ["fm_wx01"]


def test_filter_reply_literal_still_works(service):
    item1 = _make_item("fm_a1", "聚餐照片")
    item2 = _make_item("fm_b2", "会议纪要")
    cands = [
        _SearchCandidate(item=item1, score=0.5, evidence="", matched_clues=[]),
        _SearchCandidate(item=item2, score=0.4, evidence="", matched_clues=[]),
    ]
    kept = service._filter_candidates_by_reply(cands, "聚餐")
    assert [c.item.file_id for c in kept] == ["fm_a1"]
