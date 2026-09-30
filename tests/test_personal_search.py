"""个人文件搜索：时间线索匹配、状态判定、clarify 重排回归。"""

from datetime import datetime, timedelta

import pytest

from edge_cloud_agent.common.time_utils import utcnow_naive
from edge_cloud_agent.config import PersonalFileConfig
from edge_cloud_agent.personal_search.relevance import (
    SearchCandidate,
    decide_state,
    filter_candidates_by_reply,
    has_time_match,
    rerank_within,
)
from edge_cloud_agent.personal_search.service import PersonalFileSearchService
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
        state_path=str(tmp_path / "state.json"),
        faiss_index_path=str(tmp_path / "idx.index"),
        enable_faiss=False,
    )
    store = PersonalFileStore(cfg.store_path)
    return PersonalFileSearchService(config=cfg, store=store, embedding_runtime=None)


# ---------------------------------------------------------------------------
# has_time_match：captured_at 修复后时间线索才有意义，这里锁住匹配语义
# ---------------------------------------------------------------------------

def test_time_match_today():
    ok, reason = has_time_match(_iso(utcnow_naive()), "今天的会议记录")
    assert ok and "今天" in reason


def test_time_match_yesterday():
    ok, reason = has_time_match(_iso(utcnow_naive() - timedelta(days=1)), "昨天拍的照片")
    assert ok and "昨天" in reason


def test_time_match_last_week_within_window():
    ok, _ = has_time_match(_iso(utcnow_naive() - timedelta(days=5)), "上周的聚餐照片")
    assert ok


def test_time_no_match_last_week_for_old_file():
    # 60 天前的文件不应命中"上周"
    ok, _ = has_time_match(_iso(utcnow_naive() - timedelta(days=60)), "上周的聚餐照片")
    assert not ok


def test_time_match_recent_n_days():
    ok, _ = has_time_match(_iso(utcnow_naive() - timedelta(days=2)), "最近3天的文件")
    assert ok
    ok, _ = has_time_match(_iso(utcnow_naive() - timedelta(days=10)), "最近3天的文件")
    assert not ok


def test_time_no_captured_at_never_matches():
    ok, _ = has_time_match(None, "昨天的照片")
    assert not ok


# ---------------------------------------------------------------------------
# decide_state：resolved / needs_clarification / not_found 判定
# ---------------------------------------------------------------------------

def test_decide_state_empty_not_found(service):
    state, question, disambiguate = decide_state([], False)
    assert state == "not_found"
    assert question
    assert not disambiguate


def test_decide_state_single_resolved(service):
    cand = SearchCandidate(item=_make_item("f1", "a"), score=0.8, evidence="", matched_clues=[])
    state, _, disambiguate = decide_state([cand], False)
    assert state == "resolved"
    assert not disambiguate


def test_decide_state_big_gap_resolved(service):
    c1 = SearchCandidate(item=_make_item("f1", "a"), score=0.9, evidence="", matched_clues=[])
    c2 = SearchCandidate(item=_make_item("f2", "b"), score=0.4, evidence="", matched_clues=[])
    state, _, _ = decide_state([c1, c2], False)
    assert state == "resolved"


def test_decide_state_close_scores_need_clarification(service):
    c1 = SearchCandidate(item=_make_item("f1", "a"), score=0.60, evidence="", matched_clues=[])
    c2 = SearchCandidate(item=_make_item("f2", "b"), score=0.55, evidence="", matched_clues=[])
    state, question, disambiguate = decide_state([c1, c2], False)
    assert state == "needs_clarification"
    assert question and disambiguate


def test_decide_state_force_disambiguation(service):
    cand = SearchCandidate(item=_make_item("f1", "a"), score=0.8, evidence="", matched_clues=[])
    state, _, disambiguate = decide_state([cand], True)
    assert state == "needs_clarification"
    assert disambiguate


# ---------------------------------------------------------------------------
# rerank_within 回归：reply 指向另一名候选时必须能翻盘，且分值不越界
# （旧实现对上一轮第一名硬加 +1.0，追问永远收敛不到别的候选）
# ---------------------------------------------------------------------------

def test_rerank_reply_can_flip_order(service):
    item1 = _make_item("fm_aaa1", "会议纪要", raw_text="会议 纪要 讨论 项目进度")
    item2 = _make_item("fm_bbb2", "聚餐照片", raw_text="部门 聚餐 烧烤")
    c1 = SearchCandidate(item=item1, score=0.9, evidence="e1", matched_clues=["会议"])
    c2 = SearchCandidate(item=item2, score=0.5, evidence="e2", matched_clues=[])

    ranked = rerank_within(
        [c1, c2],
        "聚餐",
        top_k=2,
        weights=service.score_weights,
        embed_text=service.embed_text,
    )

    assert ranked[0].item.file_id == "fm_bbb2", "reply 明确指向聚餐，旧第一名不应被硬保送"
    assert all(0.0 <= r.score <= 1.0 for r in ranked), "重排后分值必须仍在 [0,1]"


def test_rerank_keeps_prior_when_reply_has_no_signal(service):
    item1 = _make_item("fm_aaa1", "会议纪要", raw_text="会议 纪要")
    item2 = _make_item("fm_bbb2", "聚餐照片", raw_text="部门 聚餐")
    c1 = SearchCandidate(item=item1, score=0.9, evidence="e1", matched_clues=[])
    c2 = SearchCandidate(item=item2, score=0.5, evidence="e2", matched_clues=[])

    # reply 无任何可匹配线索时，先验权重应保住原排序，而不是随机塌缩
    ranked = rerank_within(
        [c1, c2],
        "呃",
        top_k=2,
        weights=service.score_weights,
        embed_text=service.embed_text,
    )
    assert ranked[0].item.file_id == "fm_aaa1"


def test_rerank_empty(service):
    assert (
        rerank_within(
            [],
            "任意",
            top_k=5,
            weights=service.score_weights,
            embed_text=service.embed_text,
        )
        == []
    )


# ---------------------------------------------------------------------------
# filter_candidates_by_reply 回归：时间/来源类回复不应被字面过滤滤空
# （旧实现只做字面 token 匹配，"上周的那张"这类最常见的澄清回复直接 not_found）
# ---------------------------------------------------------------------------

def _days_ago_iso(days: int) -> str:
    return _iso(utcnow_naive() - timedelta(days=days))


def test_filter_reply_by_time_clue(service):
    item_new = _make_item("fm_new1", "聚餐照片", captured_at=_days_ago_iso(5))
    item_old = _make_item("fm_old1", "团建聚餐", captured_at=_days_ago_iso(40))
    cands = [
        SearchCandidate(item=item_new, score=0.5, evidence="", matched_clues=[]),
        SearchCandidate(item=item_old, score=0.4, evidence="", matched_clues=[]),
    ]
    kept = filter_candidates_by_reply(cands, "上周的那张")
    assert [c.item.file_id for c in kept] == ["fm_new1"]


def test_filter_reply_by_source_clue(service):
    item_wx = _make_item("fm_wx01", "聚餐照片", source_app="wechat")
    item_local = _make_item("fm_lo01", "团建聚餐", source_app="local")
    cands = [
        SearchCandidate(item=item_wx, score=0.5, evidence="", matched_clues=[]),
        SearchCandidate(item=item_local, score=0.4, evidence="", matched_clues=[]),
    ]
    kept = filter_candidates_by_reply(cands, "微信群里那张照片")
    assert [c.item.file_id for c in kept] == ["fm_wx01"]


def test_filter_reply_literal_still_works(service):
    item1 = _make_item("fm_a1", "聚餐照片")
    item2 = _make_item("fm_b2", "会议纪要")
    cands = [
        SearchCandidate(item=item1, score=0.5, evidence="", matched_clues=[]),
        SearchCandidate(item=item2, score=0.4, evidence="", matched_clues=[]),
    ]
    kept = filter_candidates_by_reply(cands, "聚餐")
    assert [c.item.file_id for c in kept] == ["fm_a1"]


# ---------------------------------------------------------------------------
# evidence 合并上限：澄清轮数没有上限，而 evidence 直接展示在候选卡片上
# ---------------------------------------------------------------------------

def test_merge_evidence_bounds_growth_over_many_rounds():
    from edge_cloud_agent.personal_search.relevance import (
        EVIDENCE_MAX_SEGMENTS,
        _merge_evidence,
    )

    evidence = ""
    for round_index in range(30):
        evidence = _merge_evidence(evidence, f"第{round_index}轮命中线索")

    segments = [seg for seg in evidence.split("；") if seg.strip()]
    # 省略提示「…」占一段，加上最近 N 段
    assert len(segments) <= EVIDENCE_MAX_SEGMENTS + 1, (
        f"30 轮澄清后 evidence 仍有 {len(segments)} 段，未受上限约束"
    )
    assert segments[0] == "…", "被省略时应以「…」提示用户证据不完整"
    assert segments[-1] == "第29轮命中线索", "必须保留最新的证据"
    assert "第0轮命中线索" not in evidence, "最早的证据应已被丢弃"


def test_merge_evidence_keeps_short_history_intact():
    """段数未超限时不应有任何删改或省略号。"""

    from edge_cloud_agent.personal_search.relevance import _merge_evidence

    assert _merge_evidence("", "文件名命中") == "文件名命中"
    assert _merge_evidence("文件名命中", "来源:wechat") == "文件名命中；来源:wechat"
    assert "…" not in _merge_evidence("a", "b")


def test_merge_evidence_tolerates_empty_and_messy_input():
    from edge_cloud_agent.personal_search.relevance import _merge_evidence

    assert _merge_evidence("", "") == ""
    assert _merge_evidence(None, "") == ""
    assert _merge_evidence("a；；b", "  ") == "a；b", "空段应被清理"


def test_rerank_evidence_does_not_grow_across_turns(service):
    """端到端：连续多轮 rerank 后，候选的 evidence 长度必须收敛而非线性增长。"""

    from edge_cloud_agent.personal_search.relevance import rerank_within

    item = _make_item("fm_loop1", "聚餐照片", raw_text="部门 聚餐 烧烤 去年")
    candidates = [SearchCandidate(item=item, score=0.5, evidence="文件名命中", matched_clues=[])]

    lengths = []
    for _ in range(12):
        candidates = rerank_within(
            candidates,
            "聚餐",
            top_k=1,
            weights=service.score_weights,
            embed_text=service.embed_text,
        )
        lengths.append(len(candidates[0].evidence))

    assert lengths[-1] <= lengths[5], (
        f"evidence 随轮数持续增长：{lengths}"
    )
