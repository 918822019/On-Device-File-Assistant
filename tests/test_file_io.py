"""file_io：编码降级读取（utf-8-sig → gb18030）与二进制启发式。"""

from edge_cloud_agent.common.file_io import (
    DEFAULT_ENCODINGS,
    looks_binary,
    parse_encodings,
    read_text_with_fallback,
)


def test_read_text_plain_utf8(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes("会议纪要 聚餐".encode("utf-8"))
    result = read_text_with_fallback(f)
    assert result is not None
    text, encoding = result
    assert text == "会议纪要 聚餐"
    assert encoding == "utf-8-sig"


def test_read_text_utf8_bom_stripped(tmp_path):
    """记事本另存的 BOM 不得混进正文首字符。"""

    f = tmp_path / "bom.txt"
    f.write_bytes("\ufeff会议纪要".encode("utf-8"))
    text, _ = read_text_with_fallback(f)
    assert text == "会议纪要"
    assert "\ufeff" not in text


def test_read_text_gb18030_fallback(tmp_path):
    """Windows 常见 GBK/GB2312 文件：utf-8 失败后按 gb18030 正确解码。"""

    f = tmp_path / "gbk.txt"
    f.write_bytes("会议纪要 报销单 金额128元".encode("gb18030"))
    result = read_text_with_fallback(f)
    assert result is not None
    text, encoding = result
    assert text == "会议纪要 报销单 金额128元"
    assert encoding == "gb18030"


def test_read_text_gbk_subset_fallback(tmp_path):
    f = tmp_path / "gbk2.txt"
    f.write_bytes("发票".encode("gbk"))
    text, encoding = read_text_with_fallback(f)
    assert text == "发票"
    assert encoding == "gb18030"


def test_read_text_undecodable_returns_none(tmp_path):
    """两种编码都解不出的字节序列：返回 None（调用方降级），不产生乱码。"""

    f = tmp_path / "bad.txt"
    f.write_bytes(b"\x80\x81\x82\x83")  # \x80 非 gb18030 合法首字节，也非 utf-8
    assert read_text_with_fallback(f) is None


def test_read_text_binary_nul_skipped(tmp_path):
    f = tmp_path / "bin.txt"
    f.write_bytes(b"abc\x00def\x00ghi")
    assert read_text_with_fallback(f) is None


def test_looks_binary_head_only():
    assert looks_binary(b"abc") is False
    assert looks_binary(b"a\x00b") is True
    assert looks_binary(b"") is False


def test_read_text_custom_encoding_order(tmp_path):
    f = tmp_path / "cjk.txt"
    f.write_bytes("会议".encode("utf-8"))
    # 仅 ascii：解码失败 → None
    assert read_text_with_fallback(f, encodings=("ascii",)) is None
    # 显式顺序生效
    text, encoding = read_text_with_fallback(f, encodings=("utf-8-sig", "gb18030"))
    assert (text, encoding) == ("会议", "utf-8-sig")


def test_read_text_missing_file_returns_none(tmp_path):
    assert read_text_with_fallback(tmp_path / "not_exists.txt") is None


def test_read_text_max_bytes_truncates(tmp_path):
    f = tmp_path / "long.txt"
    f.write_bytes("abcdefghij".encode("utf-8"))
    text, _ = read_text_with_fallback(f, max_bytes=5)
    assert text == "abcde"


def test_parse_encodings():
    assert parse_encodings("") == DEFAULT_ENCODINGS
    assert parse_encodings(None) == DEFAULT_ENCODINGS
    assert parse_encodings("  , ") == DEFAULT_ENCODINGS
    assert parse_encodings(" utf-8 , gb18030 ") == ("utf-8", "gb18030")
