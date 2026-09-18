"""报销业务线：字段抽取、中文关键词打分、from_dict 容错。"""

import pytest

from edge_cloud_agent.config import ExpenseConfig
from edge_cloud_agent.expense.service import ExpenseService
from edge_cloud_agent.expense.storage import ExpenseMaterial, ExpenseStore


@pytest.fixture()
def service(tmp_path) -> ExpenseService:
    cfg = ExpenseConfig(
        store_path=str(tmp_path / "expense.jsonl"),
        enable_embedding_search=False,
    )
    return ExpenseService(config=cfg, store=ExpenseStore(cfg.store_path), embedding_runtime=None)


def _make_material(**overrides) -> ExpenseMaterial:
    base = dict(
        material_id="m1",
        claim_id="c1",
        title="京东发票",
        doc_type="invoice",
        source_app="微信",
        raw_text="京东 发票 金额 128 元",
        summary="金额 128.0 元",
    )
    base.update(overrides)
    return ExpenseMaterial(**base)


# ---------------------------------------------------------------------------
# 字段抽取
# ---------------------------------------------------------------------------

def test_extract_amount_variants():
    assert ExpenseService._extract_amount("金额: 128.00 元 日期:2026-09-01") == 128.0
    assert ExpenseService._extract_amount("应付 ¥99.9") == 99.9
    assert ExpenseService._extract_amount("共计：200") == 200.0
    assert ExpenseService._extract_amount("没有金额信息") is None


def test_extract_date_iso_and_chinese():
    assert ExpenseService._extract_date("日期:2026-09-01") == "2026-09-01"
    d = ExpenseService._extract_date("9月1日开的票")
    assert d is not None and d.endswith("-09-01")
    assert ExpenseService._extract_date("无日期") is None


def test_extract_merchant():
    assert ExpenseService._extract_merchant("商户: 京东  金额: 128.00 元") == "京东"
    assert ExpenseService._extract_merchant("收款方：协和医院") == "协和医院"
    assert ExpenseService._extract_merchant("没有商户字段") is None


# ---------------------------------------------------------------------------
# 打分：中文关键词必须切词（旧实现按空格 split，标点连接的查询必失配）
# ---------------------------------------------------------------------------

def test_score_tokenizes_punctuated_keyword(service):
    material = _make_material()
    score = service._score(material, "京东，发票", None)
    assert score > 0, "带中文标点的关键词应被切分后命中"


def test_score_no_keyword_zero(service):
    material = _make_material()
    assert service._score(material, "", None) == 0.0


def test_score_unrelated_keyword_zero(service):
    material = _make_material()
    assert service._score(material, "火车票 打车", None) == 0.0


# ---------------------------------------------------------------------------
# 存储容错
# ---------------------------------------------------------------------------

def test_from_dict_tolerates_extra_and_missing_fields():
    payload = {"material_id": "m9", "raw_text": "x", "unknown_future_field": 42}
    m = ExpenseMaterial.from_dict(payload)
    assert m.material_id == "m9"
    assert m.doc_type == "receipt"      # 缺失字段回退默认
    assert m.keywords == []


def test_store_roundtrip(tmp_path):
    cfg = ExpenseConfig(store_path=str(tmp_path / "e.jsonl"))
    store = ExpenseStore(cfg.store_path)
    material = _make_material()
    store.add_or_update(material)

    reloaded = ExpenseStore(cfg.store_path)
    got = reloaded.get("m1")
    assert got is not None
    assert got.title == material.title
    assert got.raw_text == material.raw_text


def test_collect_fills_extracted_fields(service):
    from edge_cloud_agent.expense.schemas import ExpenseCollectRequest

    req = ExpenseCollectRequest(
        claim_id="claim-1",
        source_app="微信",
        doc_type="invoice",
        raw_text="商户: 京东  金额: 128.00 元 日期:2026-09-01",
    )
    material = service.collect(req, claim_id="claim-1")
    assert material.extracted_amount == 128.0
    assert material.extracted_date == "2026-09-01"
    assert material.merchant == "京东"
    assert service.store.get(material.material_id) is not None
