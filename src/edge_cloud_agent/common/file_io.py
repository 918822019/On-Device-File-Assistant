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

import contextlib
import os
from collections.abc import Iterable, Iterator, Sequence
from hashlib import md5
from pathlib import Path

# 默认编码降级链（config.FILE_MEMORY_TEXT_ENCODINGS 可覆盖）
DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "gb18030")

# 二进制启发式只看头部，避免大文件全量扫描
_BINARY_PROBE_BYTES = 4096

# 判重 hash 只取文件头部这么多字节
HASH_PREFIX_BYTES = 1024 * 1024
_HASH_CHUNK_BYTES = 256 * 1024


def content_hash(path: Path, max_bytes: int = HASH_PREFIX_BYTES) -> str:
    """流式计算文件头部 ``max_bytes`` 的 md5；读不出来时返回 ``""``。

    必须是流式的：此前两条业务线各自实现为 ``md5(path.read_bytes()[:1MB])``，
    而 ``read_bytes()`` 会把**整个文件**读进内存再切片 —— 扫描后缀白名单里含
    ``.mp4`` / ``.mov``，一个几 GB 的视频就是一次几 GB 的分配。

    输出与旧实现逐字节等价（同样的字节序列喂给同一个 md5），因此存量索引里的
    ``file_hash`` 不会因这次改动全部失配、进而触发整库重导 + 重新 embedding。

    异常口径也与旧实现一致：任何失败都返回 ``""`` 而不抛出，让调用方按
    「取不到 hash」处理（配合 size 变化仍会重导），而不是把该文件计成扫描错误。
    """

    hasher = md5()
    try:
        with path.open("rb") as handle:
            remaining = max_bytes
            while remaining > 0:
                chunk = handle.read(min(remaining, _HASH_CHUNK_BYTES))
                if not chunk:
                    break
                hasher.update(chunk)
                remaining -= len(chunk)
    except Exception:
        # 与旧实现同口径：hash 失败不抛出、返回空串，让调用方按「取不到 hash」
        # 处理（配合 size 变化仍会重导），而不是把该文件计成扫描错误。
        return ""
    return hasher.hexdigest()


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


# ---------------------------------------------------------------------------
# 原子写
# ---------------------------------------------------------------------------
# 项目里所有持久化产物（两份 JSONL 快照、备注/归档 JSON、FAISS ids 边车、
# .env）都必须是「要么旧内容、要么新内容」，不能出现半截文件：读取端虽然有
# 容错（坏行跳过、JSON 解析失败回落空状态），但那会把一次崩溃静默降级成
# 「索引丢了一半」，比直接报错难查得多。
#
# 此前这套 tmp + os.replace 在 5 个地方各写了一遍。代价不是行数，而是**新增
# 站点时会漏** —— vector_index 的 ids 边车就曾经直接写目标文件，直到上一轮
# 才补上。抽到这里之后，「原子」不再依赖每个作者记得。

_TMP_SUFFIX = ".tmp"


@contextlib.contextmanager
def _atomic_tmp(path: Path) -> Iterator[Path]:
    """产出同目录的 .tmp 路径；正常退出时 os.replace 到目标，异常时清掉残留。

    必须与目标**同目录**：os.replace 是 rename(2) 语义，跨文件系统会抛 EXDEV，
    而系统临时目录与 data/ 往往不在同一个卷上（macOS 上 /tmp 还是符号链接）。

    异常分支要 unlink：调用方下次成功写入虽然会覆盖同名 .tmp，但在那之前它是
    一个看起来像正常产物的残留文件，且占着目标文件的磁盘配额。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + _TMP_SUFFIX)
    try:
        yield tmp_path
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
    os.replace(tmp_path, path)


def atomic_write_text(path: Path, text: str, *, newline: str = "\n") -> None:
    """原子写入整段文本。

    ``newline="\\n"`` 是刻意的默认值：数据文件在所有平台上都用 LF，避免同一份
    索引在 macOS 写出、在 Windows 重写后产生纯换行符差异。所有读取端都是
    换行无关的（逐行 ``strip()`` 后 ``json.loads``、或 ``read_text`` 后解析），
    因此这不改变任何解析结果。
    """

    with _atomic_tmp(path) as tmp_path:
        tmp_path.write_text(text, encoding="utf-8", newline=newline)


def atomic_write_lines(path: Path, lines: Iterable[str], *, newline: str = "\n") -> None:
    """流式原子写入多行；``lines`` 可以是生成器。

    与 ``atomic_write_text("\\n".join(...))`` 的区别在于峰值内存：JSONL 快照里每
    行都带 embedding 向量（当前 1.2 MB / 59 条），先拼成一整个字符串等于把整份
    索引在内存里再复制一份。这里边生成边写，峰值只有一行。

    每行需自带行尾分隔符（调用方通常写 ``json.dumps(...) + "\\n"``）—— 不代加，
    是因为 .env 那类「最后一行也要有换行、行间不加空行」的格式要求由调用方决定
    更清楚。
    """

    with (
        _atomic_tmp(path) as tmp_path,
        tmp_path.open("w", encoding="utf-8", newline=newline) as handle,
    ):
        for line in lines:
            handle.write(line)
