"""共享存储原语的回归测试：atomic_write_* / cosine_similarity / JsonlSnapshotStore。

这三个东西现在承载了两条业务线的全部持久化与全部相似度打分，所以测试重点不在
「函数算得对」，而在**几件一旦悄悄退化就会造成数据损坏或口径分叉的事**：

- 原子写失败时必须清掉 .tmp，且**目标文件保持原样**（半截文件比旧文件危险得多）
- JSONL 的字节格式不能变：``ensure_ascii=False``、一行一个对象、末尾有换行。
  存量 data/personal_file_store.jsonl 是按这个格式写的，格式一变就等于全库失配
- 坏行必须被跳过而不是让整份快照读不进来
- 二级索引不能积累悬空键
- 两条业务线必须共用同一份 cosine 实现（去重的意义所在）
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

from edge_cloud_agent.common.file_io import atomic_write_lines, atomic_write_text
from edge_cloud_agent.common.jsonl_store import JsonlSnapshotStore
from edge_cloud_agent.common.vectors import cosine_similarity
from edge_cloud_agent.expense.storage import ExpenseMaterial, ExpenseStore
from edge_cloud_agent.personal_search import relevance
from edge_cloud_agent.personal_search.storage import (
    FileStateStore,
    PersonalFileItem,
    PersonalFileStore,
)

# ---------------------------------------------------------------------------
# atomic_write_*
# ---------------------------------------------------------------------------


def test_atomic_write_text_creates_parent_dirs(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c.json"
    atomic_write_text(target, '{"k": 1}')
    assert target.read_text(encoding="utf-8") == '{"k": 1}'


def test_atomic_write_text_leaves_no_tmp_residue(tmp_path: Path):
    target = tmp_path / "c.json"
    atomic_write_text(target, "x")
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]


def test_atomic_write_lines_matches_the_legacy_jsonl_byte_format(tmp_path: Path):
    """钉住字节格式：存量快照必须继续可读，且新写入与旧实现逐字节一致。"""

    target = tmp_path / "store.jsonl"
    rows = [{"file_id": "fm_1", "title": "会议纪要"}, {"file_id": "fm_2", "title": "b"}]
    atomic_write_lines(target, (json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    # ensure_ascii=False：中文原样落盘，不是 \uXXXX
    expected = (
        '{"file_id": "fm_1", "title": "会议纪要"}\n'
        '{"file_id": "fm_2", "title": "b"}\n'
    )
    assert target.read_bytes() == expected.encode()


def test_atomic_write_lines_accepts_a_generator_and_streams(tmp_path: Path):
    """生成器必须被逐条消费，不能先物化成 list（否则失去流式写的意义）。"""

    consumed: list[int] = []

    def gen():
        for i in range(5):
            consumed.append(i)
            yield f"line{i}\n"

    target = tmp_path / "s.txt"
    atomic_write_lines(target, gen())
    assert consumed == [0, 1, 2, 3, 4]
    assert target.read_text(encoding="utf-8") == "line0\nline1\nline2\nline3\nline4\n"


def test_failed_write_leaves_previous_content_intact(tmp_path: Path):
    """写入中途失败时，目标文件必须还是旧内容 —— 这是「原子」的全部意义。"""

    target = tmp_path / "s.jsonl"
    atomic_write_text(target, "GOOD")

    def boom():
        yield "partial\n"
        raise RuntimeError("disk on fire")

    with pytest.raises(RuntimeError, match="disk on fire"):
        atomic_write_lines(target, boom())

    assert target.read_text(encoding="utf-8") == "GOOD"


def test_failed_write_cleans_up_the_tmp_file(tmp_path: Path):
    """残留的 .tmp 会被误认成正常产物，必须清掉。"""

    target = tmp_path / "s.jsonl"
    atomic_write_text(target, "GOOD")

    def boom():
        yield "partial\n"
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        atomic_write_lines(target, boom())

    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.jsonl"]


def test_atomic_write_uses_same_directory_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """.tmp 必须与目标同目录：os.replace 是 rename(2) 语义，跨文件系统会 EXDEV。"""

    target = tmp_path / "sub" / "s.txt"
    target.parent.mkdir(parents=True)

    import os

    real_replace = os.replace
    calls: list[tuple[str, str]] = []

    def spy(src, dst, **kwargs):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", spy)
    atomic_write_text(target, "x")

    assert calls, "atomic_write_text 没有走 os.replace"
    src, dst = calls[-1]
    assert dst == str(target)
    assert Path(src).parent == Path(dst).parent, ".tmp 与目标不在同一目录"
    assert src == str(target) + ".tmp"


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([3.0, 4.0], [3.0, 4.0], 1.0),
        ([], [1.0], 0.0),
        ([1.0], [], 0.0),
        ([0.0, 0.0], [1.0, 1.0], 0.0),  # 零向量兜底
        ([1.0, 2.0, 3.0], [1.0, 2.0], 1.0),  # 长度不一致按最小维度对齐
    ],
)
def test_cosine_similarity(a: list[float], b: list[float], expected: float):
    assert cosine_similarity(a, b) == pytest.approx(expected, abs=1e-9)


def test_cosine_similarity_is_scale_invariant():
    assert cosine_similarity([1.0, 2.0], [100.0, 200.0]) == pytest.approx(1.0)


def test_both_business_lines_share_one_cosine_implementation():
    """去重的意义所在：两条业务线必须解析到同一个函数对象。

    只要有人图省事在业务模块里再抄一份，这条就会失败 —— 而那正是当初两份
    拷贝（personal_search/relevance.py 与 expense/service.py）的由来。
    """

    assert relevance.cosine_similarity is cosine_similarity
    source = Path(relevance.__file__).read_text(encoding="utf-8")
    assert "def cosine_similarity" not in source, "relevance.py 里又出现本地实现"
    expense_src = Path(__file__).resolve().parent.parent / "src/edge_cloud_agent/expense/service.py"
    assert "_cosine_similarity" not in expense_src.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# JsonlSnapshotStore
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    row_id: str
    name: str = ""
    uri: str | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict) -> _Row:
        return cls(
            row_id=payload.get("row_id", ""),
            name=payload.get("name", ""),
            uri=payload.get("uri"),
            tags=payload.get("tags", []) or [],
        )

    def to_dict(self) -> dict:
        return asdict(self)


class _Store(JsonlSnapshotStore[_Row]):
    row_type = _Row
    id_attr = "row_id"
    index_attr = "uri"


def test_store_roundtrip(tmp_path: Path):
    path = tmp_path / "s.jsonl"
    store = _Store(str(path))
    store.add_or_update(_Row("r1", "会议纪要", "file:///a"))
    store.add_or_update(_Row("r2", "发票", None))

    reloaded = _Store(str(path))
    assert {r.row_id for r in reloaded.list_all()} == {"r1", "r2"}
    assert reloaded.get("r1").name == "会议纪要"
    assert reloaded.get_by_index("file:///a").row_id == "r1"
    # uri 为假的行不建索引项
    assert reloaded.get_by_index("") is None


def test_store_skips_corrupt_lines_without_losing_the_rest(tmp_path: Path):
    """一行坏数据不能导致整份快照读不进来（那等于全库重新 embedding）。"""

    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"row_id": "r1", "name": "ok"}\n'
        "{ this is not json\n"
        "\n"
        '{"row_id": "r2", "name": "also ok"}\n',
        encoding="utf-8",
    )
    store = _Store(str(path))
    assert {r.row_id for r in store.list_all()} == {"r1", "r2"}


def test_store_drops_rows_with_empty_id(tmp_path: Path):
    """空主键的行必须丢弃，否则所有这类行会挤在同一个键上互相覆盖。"""

    path = tmp_path / "s.jsonl"
    path.write_text(
        '{"row_id": "", "name": "ghost"}\n{"row_id": "r1", "name": "real"}\n',
        encoding="utf-8",
    )
    store = _Store(str(path))
    assert [r.row_id for r in store.list_all()] == ["r1"]


def test_store_clears_stale_index_key_on_uri_change(tmp_path: Path):
    """URI 变化后旧键必须清掉，否则按旧 URI 查得到、删不干净。"""

    store = _Store(str(tmp_path / "s.jsonl"))
    store.add_or_update(_Row("r1", "v1", "file:///old"))
    store.add_or_update(_Row("r1", "v2", "file:///new"))

    assert store.get_by_index("file:///old") is None
    assert store.get_by_index("file:///new").name == "v2"
    assert len(store.list_all()) == 1


def test_store_does_not_steal_an_index_key_owned_by_another_id(tmp_path: Path):
    """旧键若已被别的 id 占用，不得误删。"""

    store = _Store(str(tmp_path / "s.jsonl"))
    store.add_or_update(_Row("r1", "v1", "file:///shared"))
    store.add_or_update(_Row("r2", "v2", "file:///shared"))  # 键改指 r2
    store.add_or_update(_Row("r1", "v3", "file:///elsewhere"))

    assert store.get_by_index("file:///shared").row_id == "r2"
    assert store.get_by_index("file:///elsewhere").row_id == "r1"


def test_store_delete_by_index_is_idempotent(tmp_path: Path):
    store = _Store(str(tmp_path / "s.jsonl"))
    store.add_or_update(_Row("r1", "v", "file:///a"))

    assert store.delete_by_index("file:///a") is True
    assert store.delete_by_index("file:///a") is False
    assert store.get("r1") is None
    assert store.delete_by_index("file:///never-existed") is False


def test_store_get_many_preserves_order_and_skips_missing(tmp_path: Path):
    store = _Store(str(tmp_path / "s.jsonl"))
    for rid in ("a", "b", "c"):
        store.add_or_update(_Row(rid, rid))
    assert [r.row_id for r in store.get_many(["c", "missing", "a"])] == ["c", "a"]


def test_store_persist_false_defers_writes_until_flush(tmp_path: Path):
    """批量导入路径：N 条记录只落盘一次，否则是 O(N²) 全量重写。"""

    path = tmp_path / "s.jsonl"
    store = _Store(str(path))
    for i in range(5):
        store.add_or_update(_Row(f"r{i}", str(i)), persist=False)

    assert not path.exists(), "persist=False 期间不应落盘"
    store.flush()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 5


def test_store_persists_utf8_not_ascii_escapes(tmp_path: Path):
    """钉住 store 自己的序列化口径（不是 atomic_write_lines 的）。

    ``ensure_ascii=True`` 不会让文件读不回来 —— ``json.loads`` 两种都吃 —— 所以
    这个退化是**静默**的：只会让每次落盘把中文标题从 3 字节 UTF-8 膨胀成 6 字节
    ``\\uXXXX``，1.2 MB 的快照平白涨一截，且再也没法用编辑器直接看。
    """

    path = tmp_path / "s.jsonl"
    _Store(str(path)).add_or_update(_Row("r1", "会议纪要"))
    raw = path.read_bytes()
    assert "会议纪要".encode() in raw, raw
    assert b"\\u" not in raw


def test_store_leaves_no_tmp_residue_after_many_writes(tmp_path: Path):
    store = _Store(str(tmp_path / "s.jsonl"))
    for i in range(20):
        store.add_or_update(_Row(f"r{i}", str(i), f"file:///{i}"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.jsonl"]


def test_store_missing_file_creates_parent_dir_and_starts_empty(tmp_path: Path):
    path = tmp_path / "deep" / "nested" / "s.jsonl"
    store = _Store(str(path))
    assert store.list_all() == []
    assert path.parent.is_dir()


def test_concurrent_writers_do_not_lose_rows_or_corrupt_the_file(tmp_path: Path):
    """写方法持锁、读方法不持锁 —— 这个口径必须真的成立。"""

    path = tmp_path / "s.jsonl"
    store = _Store(str(path))
    errors: list[BaseException] = []

    def writer(offset: int) -> None:
        try:
            for i in range(50):
                store.add_or_update(_Row(f"r{offset}_{i}", str(i), f"file:///{offset}/{i}"),
                                    persist=False)
                store.get(f"r{offset}_{i}")
                store.list_all()
        # 收集而不是就地抛：要断言的是「6 个线程都没出事」，第一个异常就中断
        # 会让后面 5 个线程的状态看不见。
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.flush()

    assert errors == []
    assert len(store.list_all()) == 300
    reloaded = _Store(str(path))
    assert len(reloaded.list_all()) == 300, "并发写入后落盘内容不完整"


# ---------------------------------------------------------------------------
# 两条真实业务线都建立在基类上（防止有人再 fork 一份实现）
# ---------------------------------------------------------------------------


def test_real_stores_subclass_the_shared_base():
    assert issubclass(PersonalFileStore, JsonlSnapshotStore)
    assert issubclass(ExpenseStore, JsonlSnapshotStore)
    assert (PersonalFileStore.id_attr, PersonalFileStore.index_attr) == ("file_id", "file_uri")
    assert (ExpenseStore.id_attr, ExpenseStore.index_attr) == ("material_id", "file_uri")


def test_personal_file_store_keeps_its_public_uri_api(tmp_path: Path):
    store = PersonalFileStore(str(tmp_path / "p.jsonl"))
    item = PersonalFileItem(
        file_id="fm_1", title="t", file_uri="file:///a", source_app="local",
        doc_type="document", mime_type="", raw_text="", summary="", file_path="/a",
    )
    store.add_or_update(item)
    assert store.get_by_file_uri("file:///a") is item
    assert store.delete_by_file_uri("file:///a") is True
    assert store.get_by_file_uri("file:///a") is None


def test_expense_store_business_queries_survive_the_refactor(tmp_path: Path):
    store = ExpenseStore(str(tmp_path / "e.jsonl"))
    store.add_or_update(ExpenseMaterial(
        material_id="m1", claim_id="c1", title="发票A", doc_type="receipt",
        source_app="manual", raw_text="", summary="", file_uri="file:///a",
    ))
    store.add_or_update(ExpenseMaterial(
        material_id="m2", claim_id="c1", title="发票B", doc_type="receipt",
        source_app="manual", raw_text="", summary="", file_uri=None,
    ))
    store.add_or_update(ExpenseMaterial(
        material_id="m3", claim_id="c2", title="发票C", doc_type="receipt",
        source_app="manual", raw_text="", summary="", file_uri="file:///c",
    ))

    assert store.all_claim_ids() == ["c1", "c2"]
    assert {m.material_id for m in store.list_by_claim("c1")} == {"m1", "m2"}
    # file_uri 为 None 的手工录入行只能按主键取
    assert store.get_by_file_uri("file:///c").material_id == "m3"
    assert store.get("m2").title == "发票B"


def test_file_state_store_uses_the_shared_atomic_write(tmp_path: Path):
    path = tmp_path / "state.json"
    store = FileStateStore(str(path))
    store.set_annotation("fm_1", "这是备注")
    store.set_archived("fm_2")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]

    reloaded = FileStateStore(str(path))
    assert reloaded.get_annotation("fm_1") == "这是备注"
    assert reloaded.is_archived("fm_2") is True
