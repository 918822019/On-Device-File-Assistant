"""FAISS 向量索引与 ids.json 的一致性。

index 文件与 ids.json 是两次独立写入，中间崩溃会留下错位组合。此前
`_load_index()` 只判断「index 存在且 ids 非空」就算就绪，而 search() 是
**按位置**把命中下标映射到 file_id 的（第 i 个向量 -> _ids[i]），所以
「新 index + 旧 ids」会静默返回**错误的文件**——比抛异常更糟。

这里用真实的 faiss（venv 里已安装）构造出错位状态，验证它被判为未就绪、
且 search() 返回空而不是错误结果。
"""

import json

import pytest

from edge_cloud_agent.config import PersonalFileConfig
from edge_cloud_agent.personal_search.storage import PersonalFileItem
from edge_cloud_agent.personal_search.vector_index import (
    PersonalFileVectorIndex,
    _FaissBackend,
)

faiss = pytest.importorskip("faiss", reason="faiss 未安装时跳过向量索引用例")


def _item(file_id: str, vector: list[float]) -> PersonalFileItem:
    return PersonalFileItem(
        file_id=file_id,
        title=file_id,
        file_uri=f"file:///tmp/{file_id}.txt",
        source_app="local",
        doc_type="file",
        mime_type="text/plain",
        raw_text="内容",
        summary="内容",
        file_path=f"/tmp/{file_id}.txt",
        embedding=vector,
    )


def _cfg(tmp_path, **overrides) -> PersonalFileConfig:
    base = {
        "store_path": str(tmp_path / "store.jsonl"),
        "state_path": str(tmp_path / "state.json"),
        "source_dir": str(tmp_path),
        "faiss_index_path": str(tmp_path / "idx.index"),
        "enable_faiss": True,
    }
    base.update(overrides)
    return PersonalFileConfig(**base)


# 三维单位向量，互相正交，便于断言命中顺序
_V1 = [1.0, 0.0, 0.0]
_V2 = [0.0, 1.0, 0.0]
_V3 = [0.0, 0.0, 1.0]


def test_build_then_load_roundtrip_is_ready(tmp_path):
    cfg = _cfg(tmp_path)
    index = PersonalFileVectorIndex(cfg)
    index.rebuild([_item("fm_a", _V1), _item("fm_b", _V2), _item("fm_c", _V3)])
    assert index.is_ready()

    reloaded = PersonalFileVectorIndex(cfg)
    assert reloaded.is_ready(), "落盘后重新加载必须仍是就绪态"

    hits = reloaded.search(_V2, top_k=3)
    assert hits[0].file_id == "fm_b", "位置映射必须正确：第 2 个向量对应 fm_b"


def test_stale_ids_shorter_than_index_is_not_ready(tmp_path, caplog):
    """核心回归：新 index + 旧（更短）ids 必须判为未就绪。"""

    cfg = _cfg(tmp_path)
    index = PersonalFileVectorIndex(cfg)
    index.rebuild([_item("fm_a", _V1), _item("fm_b", _V2), _item("fm_c", _V3)])

    # 模拟崩溃在「index 已写完、ids 还是上一轮」的窗口：把 ids 换成更短的陈旧版本
    ids_path = index._backend._ids_path
    ids_path.write_text(json.dumps({"ids": ["fm_old"]}), encoding="utf-8")

    with caplog.at_level("WARNING"):
        reloaded = PersonalFileVectorIndex(cfg)

    assert not reloaded.is_ready(), "ids 与向量数不一致时绝不能算就绪"
    assert reloaded.search(_V1, top_k=3) == [], "错位状态下必须返回空，而不是错误的 file_id"
    assert any("ids_mismatch" in (r.message or "") or "faiss" in (r.message or "").lower()
               for r in caplog.records), "错位应留下可排查的 warning"


def test_stale_ids_longer_than_index_is_not_ready(tmp_path):
    """反方向同样危险：ids 比向量多，会让下标落在陈旧条目上。"""

    cfg = _cfg(tmp_path)
    index = PersonalFileVectorIndex(cfg)
    index.rebuild([_item("fm_a", _V1)])

    index._backend._ids_path.write_text(
        json.dumps({"ids": ["fm_a", "fm_ghost1", "fm_ghost2"]}), encoding="utf-8"
    )

    reloaded = PersonalFileVectorIndex(cfg)
    assert not reloaded.is_ready()
    assert reloaded.search(_V1, top_k=3) == []


def test_corrupt_ids_json_falls_back_to_not_ready(tmp_path):
    """半截 JSON（崩溃留下的）应容错为未就绪，下一轮扫描会全量重建。"""

    cfg = _cfg(tmp_path)
    index = PersonalFileVectorIndex(cfg)
    index.rebuild([_item("fm_a", _V1), _item("fm_b", _V2)])

    index._backend._ids_path.write_text('{"ids": ["fm_a"', encoding="utf-8")  # 截断

    reloaded = PersonalFileVectorIndex(cfg)
    assert not reloaded.is_ready()
    assert reloaded.search(_V1, top_k=2) == []


def test_persist_ids_is_atomic_and_leaves_no_tmp(tmp_path):
    """ids 落盘必须走 tmp + os.replace，且正常路径不残留 .tmp 文件。"""

    cfg = _cfg(tmp_path)
    index = PersonalFileVectorIndex(cfg)
    index.rebuild([_item("fm_a", _V1), _item("fm_b", _V2)])

    ids_path = index._backend._ids_path
    assert ids_path.exists()
    assert not ids_path.with_name(ids_path.name + ".tmp").exists(), "落盘后不应残留临时文件"
    assert json.loads(ids_path.read_text(encoding="utf-8"))["ids"] == ["fm_a", "fm_b"]


def test_backend_is_ready_requires_matching_lengths(tmp_path):
    """直接就 _FaissBackend.is_ready 断言长度不变量。"""

    backend = _FaissBackend(tmp_path / "idx.index")
    assert not backend.is_ready(), "没有 index 时不该就绪"

    backend.build([_item("fm_a", _V1), _item("fm_b", _V2)])
    assert backend.is_ready()
    assert len(backend.ids) == backend._index.ntotal == 2

    # 手工破坏一致性：只改 ids，不重建 index
    backend._ids = ["fm_a"]
    assert not backend.is_ready(), "ids 与 ntotal 不等时必须立刻失去就绪态"


def test_rebuild_after_mismatch_restores_service(tmp_path):
    """错位是可自愈的：store 是唯一事实源，FAISS 只是可丢弃的缓存。"""

    cfg = _cfg(tmp_path)
    index = PersonalFileVectorIndex(cfg)
    items = [_item("fm_a", _V1), _item("fm_b", _V2)]
    index.rebuild(items)

    index._backend._ids_path.write_text(json.dumps({"ids": ["fm_stale"]}), encoding="utf-8")
    broken = PersonalFileVectorIndex(cfg)
    assert not broken.is_ready()

    broken.rebuild(items)
    assert broken.is_ready()
    assert broken.search(_V2, top_k=2)[0].file_id == "fm_b"
