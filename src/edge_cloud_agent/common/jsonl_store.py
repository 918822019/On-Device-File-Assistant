"""JSONL 快照存储基类：一份内存 dict + 原子落盘。

``PersonalFileStore`` 与 ``ExpenseStore`` 此前是两份**逐字符几乎相同**的实现：
同样的 ``_load``（逐行 ``json.loads``、坏行跳过、不阻塞启动）、同样的
``_persist``（tmp + ``os.replace``）、同样的 ``add_or_update`` / ``flush`` /
``get`` / ``get_many`` / ``list_all`` / ``delete_by_file_uri``。两者真正的差异
只有三点：行类型、主键字段名（``file_id`` vs ``material_id``）、以及 expense
多了两个业务查询。

抽成基类之后，两条业务线的持久化语义**只有一处定义**，这比省下的行数更重要：
上一轮给 personal 侧补的「同一 id 换了 URI 时清理旧二级索引键」这类修复，此前
必须记得在 expense 侧再写一遍才会同时生效（实际就漏了）。

存储模型是**快照**而非日志：每次落盘全量重写整个文件。这对当前规模（59 个
文件 / 1.2 MB）是正确的选择 —— 读取端不需要回放、崩溃后不需要截断修复。
代价是单次写入 O(N)，因此批量导入必须用 ``persist=False`` + ``flush()``，
否则 N 条记录 = N 次全量写 = O(N²)。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import ClassVar, Generic, TypeVar

from .file_io import atomic_write_lines

RowT = TypeVar("RowT")


class JsonlSnapshotStore(Generic[RowT]):
    """子类通过三个类属性声明结构，业务专属查询留在子类里。

    线程口径与抽出前完全一致：**读方法不加锁**，写方法持 ``self._lock``。
    不加锁的读是有意的 —— 快照语义下读到的是「稍旧但自洽」的视图，而给
    ``get`` / ``list_all`` 加锁会让检索热路径与后台扫描的落盘互相阻塞。
    """

    #: 行对象类型，需提供 ``from_dict(dict)`` classmethod 与 ``to_dict()``
    row_type: ClassVar[type]
    #: 主键属性名；该属性为假的行会被整行丢弃（避免所有空 id 挤在同一个键上）
    id_attr: ClassVar[str]
    #: 可选的二级索引属性名（如 ``file_uri``）；属性值为假时不建索引项
    index_attr: ClassVar[str | None] = None

    def __init__(self, path: str) -> None:
        self.path = Path(path).resolve()
        self._items: dict[str, RowT] = {}
        self._by_index: dict[str, str] = {}
        self._lock = Lock()
        self._load()

    # -- 读 ----------------------------------------------------------------

    def get(self, row_id: str) -> RowT | None:
        return self._items.get(row_id)

    def get_by_index(self, key: str) -> RowT | None:
        """按二级索引取行；索引项可能指向已被删除的 id，故再校验一次主表。"""

        row_id = self._by_index.get(key)
        if row_id is None:
            return None
        return self._items.get(row_id)

    def list_all(self) -> list[RowT]:
        return list(self._items.values())

    def get_many(self, row_ids: Iterable[str]) -> list[RowT]:
        """按给定顺序返回存在的行；缺失的 id 静默跳过（调用方拿到的是子集）。"""

        result: list[RowT] = []
        for row_id in row_ids:
            item = self._items.get(row_id)
            if item is not None:
                result.append(item)
        return result

    # -- 写 ----------------------------------------------------------------

    def add_or_update(self, item: RowT, persist: bool = True) -> None:
        """写入内存索引；``persist=False`` 时延迟落盘（批量导入配合 ``flush``）。"""

        with self._lock:
            self._index_item(item)
            if persist:
                self._persist()

    def delete_by_index(self, key: str, persist: bool = True) -> bool:
        """按二级索引删除；返回是否真的删掉了东西（幂等，重复删返回 False）。"""

        with self._lock:
            row_id = self._by_index.pop(key, None)
            if row_id is None:
                return False
            self._items.pop(row_id, None)
            if persist:
                self._persist()
            return True

    def flush(self) -> None:
        """把当前内存索引一次性落盘（配合 ``persist=False`` 的批量写入）。"""

        with self._lock:
            self._persist()

    # -- 内部 --------------------------------------------------------------

    def _index_item(self, item: RowT) -> None:
        """把一行写进主表与二级索引，并清理该 id 遗留的旧索引键。

        清理旧键这一步是必需的：``file_uri`` 会因为路径规范化而变化，若只写新
        键，``_by_index`` 会积累指向同一 id 的悬空键 —— 之后按旧 URI 查得到、
        按旧 URI 删却删不干净。``get(prev_key) == row_id`` 的校验防止误删已经
        被**别的** id 重新占用的键。
        """

        row_id = getattr(item, self.id_attr)
        if not row_id:
            return
        key = getattr(item, self.index_attr) if self.index_attr else None
        prev = self._items.get(row_id)
        if prev is not None and self.index_attr:
            prev_key = getattr(prev, self.index_attr, None)
            if prev_key and prev_key != key and self._by_index.get(prev_key) == row_id:
                del self._by_index[prev_key]
        self._items[row_id] = item
        if key:
            self._by_index[key] = row_id

    def _load(self) -> None:
        """读回快照。文件不存在时只建目录、保持空索引（首次启动的正常路径）。"""

        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return

        with self.path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                row = raw_line.strip()
                if not row:
                    continue
                try:
                    payload = json.loads(row)
                    item = self.row_type.from_dict(payload)
                except Exception:
                    # 向后兼容：一行坏数据直接跳过。整份快照因为一行损坏而读不
                    # 进来，代价是丢掉全部索引 + 触发整库重新 embedding，远大于
                    # 少一条记录。
                    continue
                self._index_item(item)

    def _persist(self) -> None:
        """全量重写 JSONL。调用方必须已持有 ``self._lock``。

        先 ``list(...)`` 拷一份再序列化：``atomic_write_lines`` 是边迭代边写的，
        直接传 ``self._items.values()`` 视图的话，任何并发写入都会让迭代抛
        ``RuntimeError: dictionary changed size during iteration``。所有写方法
        确实都持锁，但读方法不持锁、且 ``values()`` 视图的失效不取决于调用方
        是否守规矩 —— 拷贝一次就把这个隐患关掉了，代价相对逐行 json.dumps
        可以忽略。
        """

        items = list(self._items.values())
        atomic_write_lines(
            self.path,
            (json.dumps(item.to_dict(), ensure_ascii=False) + "\n" for item in items),
        )
