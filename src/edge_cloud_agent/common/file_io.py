"""跨平台文本读取工具：编码降级 + 二进制启发式。

Windows 上 GBK/ANSI 编码的文本文件很常见，此前 ingest 只按 utf-8 读取，
解码失败即静默降级为"文件名语义"，正文完全不可检索。默认降级链：

1. ``utf-8-sig``：带 BOM / 不带 BOM 都正确（记事本另存的 BOM 不再把
   \\ufeff 混进正文首 token）
2. ``gb18030``：GBK/GB2312/cp936 的超集，strict 解码，失败即整体放弃
   （返回 None，由调用方降级为文件名语义，不产生乱码入库）

gb18030 对任意字节序列过于宽容（几乎不会解码失败），故先做 NUL 二进制
启发式：头部含 \\x00 的文件直接判为二进制跳过，避免把图片/可执行文件
"成功解码"成乱码污染索引。

仅依赖 stdlib。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

# 默认编码降级链（config.FILE_MEMORY_TEXT_ENCODINGS 可覆盖）
DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "gb18030")

# 二进制启发式只看头部，避免大文件全量扫描
_BINARY_PROBE_BYTES = 4096


def looks_binary(head: bytes) -> bool:
    """头部含 NUL 字节即判为二进制（对文本/常见中文编码均安全）。"""

    return b"\x00" in head


def read_text_with_fallback(
    path: Path,
    encodings: Sequence[str] = DEFAULT_ENCODINGS,
    max_bytes: int | None = None,
) -> tuple[str, str] | None:
    """读取文本文件，按 encodings 顺序尝试解码。

    返回 ``(text, encoding_used)``；文件不可读 / 判为二进制 / 全部编码
    解码失败时返回 None（调用方自行降级）。max_bytes 非 None 时只读取
    头部该长度的字节（用于限制大文件 IO）。
    """

    try:
        if max_bytes is None:
            data = path.read_bytes()
        else:
            with path.open("rb") as fh:
                data = fh.read(max_bytes)
    except OSError:
        return None

    if looks_binary(data[:_BINARY_PROBE_BYTES]):
        return None

    for encoding in encodings:
        try:
            return data.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def parse_encodings(raw: str | None) -> tuple[str, ...]:
    """解析逗号分隔的编码配置串；空配置回落 DEFAULT_ENCODINGS。"""

    if not raw:
        return DEFAULT_ENCODINGS
    encodings = tuple(chunk.strip() for chunk in raw.split(",") if chunk.strip())
    return encodings or DEFAULT_ENCODINGS
