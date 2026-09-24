"""共享分词器行为测试（jieba 口径 + 无 jieba 降级口径）。"""

import edge_cloud_agent.text_utils as text_utils
from edge_cloud_agent.text_utils import tokenize


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_punctuated_chinese():
    # 中文标点应切开，避免"京东，发票"退化成单个整 token
    assert set(tokenize("京东，发票")) == {"京东", "发票"}


def test_tokenize_chinese_phrase_segmented():
    # M0 起 jieba 细分无标点短语；整句精确命中由打分端 +1.5 加成承接，
    # 不再依赖保留整段长 token
    assert set(tokenize("京东发票")) == {"京东", "发票"}


def test_tokenize_long_query_segmented():
    """回归核心动机：长查询不再退化成单个整 token。"""

    tokens = tokenize("上周群里发的聚餐照片")
    for word in ("上周", "聚餐", "照片"):
        assert word in tokens
    assert "的" not in tokens, "单字虚词应被过滤"
    assert "上周群里发的聚餐照片" not in tokens, "不应保留切不开的整段长 token"


def test_tokenize_single_cjk_char_preserved():
    # 切完只剩单字时保留原 token，避免整段丢失
    assert tokenize("票") == ["票"]


def test_tokenize_mixed_alnum_cjk():
    assert {"微信", "wx123", "账单"} <= set(tokenize("微信wx123账单"))


def test_tokenize_mixed_case_and_space():
    tokens = tokenize("WeChat 微信, 发票")
    assert "wechat" in tokens
    assert "微信" in tokens
    assert "发票" in tokens


def test_tokenize_dedup_and_sorted():
    tokens = tokenize("发票 发票 invoice")
    assert tokens == sorted(set(tokens))
    assert tokens.count("发票") == 1


def test_tokenize_fallback_without_jieba(monkeypatch):
    """jieba 不可用时退回旧口径：粗段 + ≥2 字中文短语补充。"""

    monkeypatch.setattr(text_utils, "_jieba", None)
    monkeypatch.setattr(text_utils, "_jieba_resolved", True)
    assert set(tokenize("京东发票")) == {"京东发票"}
    assert set(tokenize("京东，发票")) == {"京东", "发票"}
