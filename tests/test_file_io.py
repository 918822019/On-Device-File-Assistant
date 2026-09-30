"""file_io：编码降级读取（utf-8-sig → gb18030）、二进制启发式与流式判重 hash。"""

from hashlib import md5

from edge_cloud_agent.common.file_io import (
    DEFAULT_ENCODINGS,
    HASH_PREFIX_BYTES,
    content_hash,
    looks_binary,
    parse_encodings,
    read_text_with_fallback,
)


def test_read_text_plain_utf8(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes("会议纪要 聚餐".encode())
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
    f.write_bytes("会议".encode())
    # 仅 ascii：解码失败 → None
    assert read_text_with_fallback(f, encodings=("ascii",)) is None
    # 显式顺序生效
    text, encoding = read_text_with_fallback(f, encodings=("utf-8-sig", "gb18030"))
    assert (text, encoding) == ("会议", "utf-8-sig")


def test_read_text_missing_file_returns_none(tmp_path):
    assert read_text_with_fallback(tmp_path / "not_exists.txt") is None


def test_read_text_max_bytes_truncates(tmp_path):
    f = tmp_path / "long.txt"
    f.write_bytes(b"abcdefghij")
    text, _ = read_text_with_fallback(f, max_bytes=5)
    assert text == "abcde"


def test_parse_encodings():
    assert parse_encodings("") == DEFAULT_ENCODINGS
    assert parse_encodings(None) == DEFAULT_ENCODINGS
    assert parse_encodings("  , ") == DEFAULT_ENCODINGS
    assert parse_encodings(" utf-8 , gb18030 ") == ("utf-8", "gb18030")


# ------------------------------------------------------- content_hash（判重）


def _legacy_hash(data: bytes) -> str:
    """改动前的实现：md5(path.read_bytes()[:1MB])。作为等价性基准。"""

    return md5(data[: 1024 * 1024]).hexdigest()


def test_content_hash_matches_legacy_implementation(tmp_path):
    """必须与旧实现逐字节等价。

    这是本次改动最重要的安全性质：hash 是增量扫描的变更检测键，一旦口径变了，
    存量索引里每条记录的 file_hash 都会失配，下一次扫描就会把**整个语料**当成
    变更文件重导一遍 —— 包括用 300M embedding 模型重新编码全部内容。
    """

    cases = {
        "empty.bin": b"",
        "small.txt": b"hello",
        "cjk.txt": "会议纪要 聚餐照片".encode(),
        "exactly_1mb.bin": bytes(range(256)) * 4096,  # 恰好 1 MiB
        "over_1mb.bin": b"A" * (1024 * 1024) + b"B" * 512,
        "nul_binary.bin": b"\x00\x01\x02binary",
    }
    assert len(cases["exactly_1mb.bin"]) == 1024 * 1024

    for name, data in cases.items():
        path = tmp_path / name
        path.write_bytes(data)
        assert content_hash(path) == _legacy_hash(data), f"{name} 的 hash 与旧实现不一致"


def test_content_hash_reads_only_the_prefix(tmp_path):
    """超出 max_bytes 的部分不能被读取，否则大视频仍会产生全量 IO。"""

    head = b"H" * 2048
    prefix_only = tmp_path / "prefix.bin"
    prefix_only.write_bytes(head)

    with_huge_tail = tmp_path / "tailed.bin"
    with_huge_tail.write_bytes(head + b"T" * (8 * 1024 * 1024))

    assert content_hash(with_huge_tail, max_bytes=2048) == content_hash(prefix_only, max_bytes=2048)
    assert content_hash(with_huge_tail, max_bytes=4096) != content_hash(prefix_only, max_bytes=4096)


def test_content_hash_default_prefix_is_1mib(tmp_path):
    assert HASH_PREFIX_BYTES == 1024 * 1024
    big = tmp_path / "big.bin"
    big.write_bytes(b"X" * (1024 * 1024) + b"Y" * 4096)
    assert content_hash(big) == md5(b"X" * (1024 * 1024)).hexdigest()


def test_content_hash_distinguishes_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"same-length-1")
    b.write_bytes(b"same-length-2")
    assert content_hash(a) != content_hash(b)
    assert content_hash(a) == content_hash(a), "同一文件重复计算必须稳定"


def test_content_hash_returns_empty_string_on_failure(tmp_path):
    """读不出来时返回 ""，而不是抛异常打断整轮扫描。"""

    assert content_hash(tmp_path / "not_exists.bin") == ""
    assert content_hash(tmp_path) == "", "目录不可读，应返回空串"
