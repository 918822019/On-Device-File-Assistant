"""计算个人文件索引覆盖范围，不暴露文件名或文档内容。"""

from pathlib import Path

from ..config import PersonalFileConfig
from .schemas import IndexedSource, IndexStatusResponse
from .service import PersonalFileSearchService


def build_index_status(service: PersonalFileSearchService, cfg: PersonalFileConfig) -> IndexStatusResponse:
    """将文件分配到最具体的配置根目录，嵌套目录只计数一次。"""

    roots = [Path(part.strip()).expanduser().resolve() for part in cfg.source_dir.split(",") if part.strip()]
    sources = [IndexedSource(path=str(root), indexed_files=0, available=root.is_dir()) for root in roots]
    items = service.store.list_all()
    last_indexed_at = max((item.last_scanned_at for item in items if item.last_scanned_at), default=None)
    for item in items:
        if not item.file_path:
            continue
        file_path = Path(item.file_path)
        # Nested roots belong to the most specific configured source, not both.
        # strict=True：roots 与 sources 由同一份列表推导而来，长度必须恒等；
        # 显式声明可防止将来重构把两者拆开后 zip 静默截断（少计嵌套根的文件数）。
        for root, source in sorted(
            zip(roots, sources, strict=True), key=lambda pair: len(pair[0].parts), reverse=True
        ):
            if file_path.is_relative_to(root):
                source.indexed_files += 1
                break
    return IndexStatusResponse(
        indexed_files=len(items),
        scan_recursive=cfg.scan_recursive,
        scan_interval_seconds=cfg.scan_interval_seconds,
        vector_ready=service.vector_index_ready,
        max_query_len=cfg.max_search_query_len,
        last_indexed_at=last_indexed_at,
        sources=sources,
    )
