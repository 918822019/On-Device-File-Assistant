"""打磨批次回归：版本加分收紧、来源别名、会话治理、状态持久化、指标修真、raw_text 截断。"""

from time import monotonic

import pytest

from edge_cloud_agent.config import PersonalFileConfig
from edge_cloud_agent.personal_search.ingest import run_once
from edge_cloud_agent.personal_search.service import (
    _SearchCandidate,
    _SearchSession,
    _SESSION_MAX_COUNT,
    _source_hit,
    _tokenize,
    _VERSION_MARK_RE,
    PersonalFileSearchService,
)
from edge_cloud_agent.personal_search.storage import PersonalFileItem, PersonalFileStore


def _make_item(file_id, title, raw_text="", doc_type="file", source_app="local", captured_at=None):
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
    return PersonalFileSearchService(
        config=cfg,
        store=PersonalFileStore(cfg.store_path),
        embedding_runtime=None,
    )


def _score(service, item, query):
    score, _evidence, matched = service._score_item(
        item,
        query,
        _tokenize(query),
        service._extract_source_hints(query),
        service._extract_visual_hints(query),
        service._extract_version_hint(query),
        None,
    )
    return score, matched


# ---------------------------------------------------------------------------
# 版本加分收紧：需要「查询有版本意图」且「候选有版本标记」同时成立
# ---------------------------------------------------------------------------

def test_version_mark_regex():
    assert _VERSION_MARK_RE.search("方案v2")
    assert _VERSION_MARK_RE.search("doc v1.3 final")
    assert _VERSION_MARK_RE.search("最终版本")
    assert not _VERSION_MARK_RE.search("video save csv")
    assert not _VERSION_MARK_RE.search("myvideo2")  # v 前面是字母，不算版本标记


def test_no_version_bonus_without_intent(service):
    """旧实现 `if "v" in haystack` 会让含字母 v 的无关文件凭空拿加分。"""

    with_v = _make_item("fm_a1", "note", raw_text="save csv video")
    without_v = _make_item("fm_b2", "note", raw_text="plain text here")
    s_v, _ = _score(service, with_v, "聚餐")
    s_plain, _ = _score(service, without_v, "聚餐")
    assert s_v == s_plain == 0.0


def test_version_bonus_with_intent_and_mark(service):
    versioned = _make_item("fm_v2", "方案v2", raw_text="方案 v2")
    plain = _make_item("fm_v0", "方案", raw_text="方案")
    s_versioned, matched = _score(service, versioned, "哪个是最新版本")
    s_plain, _ = _score(service, plain, "哪个是最新版本")
    assert s_versioned > s_plain
    assert any("版本" in m for m in matched)


# ---------------------------------------------------------------------------
# 来源提示别名：gallery↔image、camera↔screenshot、document↔office/pdf
# ---------------------------------------------------------------------------

def test_source_hit_aliases():
    assert _source_hit(["gallery"], "doc_type: image mime: image/png") == ["gallery"]
    assert _source_hit(["camera"], "source_app: camera") == ["camera"]
    assert _source_hit(["document"], "mime: application/ms-office") == ["document"]
    assert _source_hit(["document"], "mime: application/pdf") == ["document"]
    assert _source_hit(["wechat"], "source_app: local doc_type: file") == []


def test_filter_reply_gallery_alias(service):
    photo = _make_item("fm_p1", "聚餐照片", doc_type="image")
    note = _make_item("fm_n1", "会议纪要", doc_type="file")
    cands = [
        _SearchCandidate(item=photo, score=0.5, evidence="", matched_clues=[]),
        _SearchCandidate(item=note, score=0.4, evidence="", matched_clues=[]),
    ]
    kept = service._filter_candidates_by_reply(cands, "相册里那张")
    assert [c.item.file_id for c in kept] == ["fm_p1"]


def test_score_gallery_clue_via_alias(service):
    photo = _make_item("fm_p1", "照片", doc_type="image")
    _, matched = _score(service, photo, "相册里的照片")
    assert any(m.startswith("来源:") for m in matched)


# ---------------------------------------------------------------------------
# 会话治理：TTL 过期 + 容量上限
# ---------------------------------------------------------------------------

def test_session_ttl_eviction(service):
    sid, *_ = service.start_search("任意查询", top_k=3)
    assert service.get_session(sid) is not None

    # 人为把会话拨到 TTL 之外，下一次 start_search 应将其淘汰
    with service._session_lock:
        for s in service._sessions.values():
            s.last_access = monotonic() - 31 * 60
    service.start_search("另一个查询", top_k=3)
    assert service.get_session(sid) is None


def test_session_capacity_eviction(service):
    total = _SESSION_MAX_COUNT + 10
    with service._session_lock:
        now = monotonic()
        for i in range(total):
            sid = f"fake_{i:04d}"
            service._sessions[sid] = _SearchSession(
                session_id=sid,
                query="q",
                candidates=[],
                turn=0,
                last_access=now - (total - i),  # i 越小越旧
            )
    service.start_search("触发淘汰", top_k=3)
    with service._session_lock:
        # 淘汰发生在新会话入表之前：210 -> 200，再 +1 个新会话
        assert len(service._sessions) <= _SESSION_MAX_COUNT + 1
        assert "fake_0000" not in service._sessions
        assert f"fake_{total - 1:04d}" in service._sessions


# ---------------------------------------------------------------------------
# 备注/归档持久化：重启（新 service 实例、同 state 文件）不丢
# ---------------------------------------------------------------------------

def test_annotation_and_archive_survive_restart(tmp_path):
    cfg_kwargs = dict(
        store_path=str(tmp_path / "store.jsonl"),
        state_path=str(tmp_path / "state.json"),
        faiss_index_path=str(tmp_path / "idx.index"),
        enable_faiss=False,
    )
    store = PersonalFileStore(cfg_kwargs["store_path"])
    store.add_or_update(_make_item("fm_keep1", "重要文件"))

    svc1 = PersonalFileSearchService(
        config=PersonalFileConfig(**cfg_kwargs), store=store, embedding_runtime=None
    )
    assert svc1.add_annotation("fm_keep1", "这个是最终版")
    assert svc1.archive_file("fm_keep1")

    svc2 = PersonalFileSearchService(
        config=PersonalFileConfig(**cfg_kwargs),
        store=PersonalFileStore(cfg_kwargs["store_path"]),
        embedding_runtime=None,
    )
    assert svc2.get_annotation("fm_keep1") == "这个是最终版"
    assert svc2.is_archived("fm_keep1")


def test_annotate_missing_file_returns_false(service):
    assert not service.add_annotation("fm_ghost", "note")
    assert not service.archive_file("fm_ghost")


# ---------------------------------------------------------------------------
# 报销指标修真：extraction_confidence 反映真实抽取质量
# ---------------------------------------------------------------------------

def test_extraction_confidence_real_signal():
    from edge_cloud_agent.expense.storage import ExpenseMaterial
    from edge_cloud_agent.routers.expense import _extraction_confidence

    def mat(**kw):
        base = dict(
            material_id="m", claim_id="c", title="t", doc_type="invoice",
            source_app="w", raw_text="r", summary="s",
        )
        base.update(kw)
        return ExpenseMaterial(**base)

    assert _extraction_confidence(mat()) == 0.0
    assert _extraction_confidence(mat(extracted_amount=1.0)) == pytest.approx(0.33)
    assert _extraction_confidence(
        mat(extracted_amount=1.0, extracted_date="2026-09-01", merchant="京东")
    ) == 1.0


# ---------------------------------------------------------------------------
# raw_text 截断用独立配置，不再复用查询截断长度
# ---------------------------------------------------------------------------

def test_raw_text_truncation_uses_own_config(tmp_path):
    src = tmp_path / "files"
    src.mkdir()
    (src / "long.txt").write_text("聚餐照片内容" * 100, encoding="utf-8")  # 600 字符

    cfg = PersonalFileConfig(
        store_path=str(tmp_path / "s.jsonl"),
        state_path=str(tmp_path / "st.json"),
        source_dir=str(src),
        faiss_index_path=str(tmp_path / "i.index"),
        enable_faiss=False,
        raw_text_max_chars=50,
        max_search_query_len=120,
    )
    service = PersonalFileSearchService(
        config=cfg, store=PersonalFileStore(cfg.store_path), embedding_runtime=None
    )
    result = run_once(service, cfg)
    assert result.imported == 1
    item = service.store.list_all()[0]
    assert len(item.raw_text) == 50
