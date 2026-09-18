"""共享分词器行为测试。"""

from edge_cloud_agent.text_utils import tokenize


def test_tokenize_empty():
    assert tokenize("") == []


def test_tokenize_punctuated_chinese():
    # 中文标点应切开，避免"京东，发票"退化成单个整 token
    assert set(tokenize("京东，发票")) == {"京东", "发票"}


def test_tokenize_chinese_phrase_kept_whole():
    assert set(tokenize("京东发票")) == {"京东发票"}


def test_tokenize_mixed_case_and_space():
    tokens = tokenize("WeChat 微信, 发票")
    assert "wechat" in tokens
    assert "微信" in tokens
    assert "发票" in tokens


def test_tokenize_dedup_and_sorted():
    tokens = tokenize("发票 发票 invoice")
    assert tokens == sorted(set(tokens))
    assert tokens.count("发票") == 1
