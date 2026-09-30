"""两个推理运行时的并发串行化。

所有端点都是同步 def，FastAPI 把它们丢进同一个 anyio 线程池（默认 40 个 token），
因此并发请求会同时对同一个 HF 模型对象调用前向。HF 模型没有内部同步，且
EdgeEmbeddingRuntime 是**单实例共享**的（bootstrap 只造一个，同时注入
expense_service 与 personal_file_service，后台 ingest 线程也会调它）。

运行时在 __init__ 里就加载真实模型（9.5 GiB + 300 MB），无法在测试中构造，
故用 object.__new__ 绕过 __init__ 并注入桩模型，只验证「锁确实把前向串行化」
这一个性质。

每个正向断言都配一个反证：绕过锁直接调 *_locked 时必须**能**并发。否则
peak == 1 可能只是桩模型本身不会并发，断言就是空的。
"""

import inspect
import logging
import re
import threading
import time
from types import SimpleNamespace

import torch

from edge_cloud_agent.config import EdgeConfig, EmbeddingConfig
from edge_cloud_agent.engines.edge_runtime import EdgeRuntime
from edge_cloud_agent.engines.embedding_runtime import EdgeEmbeddingRuntime

THREADS = 8
HOLD_SECONDS = 0.05


class _Tracker:
    """记录进入模型前向的并发峰值。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def __enter__(self):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(HOLD_SECONDS)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self.active -= 1
        return False


# ------------------------------------------------------------------ EdgeRuntime


class _FakeIds:
    """最小张量替身：只需支持 .shape[-1]、.to(device) 与切片。"""

    def __init__(self, ids):
        self._ids = ids

    @property
    def shape(self):
        return (1, len(self._ids))

    def to(self, device):
        return self

    def __getitem__(self, item):
        return self._ids[item]


class _FakeGenTokenizer:
    """刻意不提供 apply_chat_template，让 generate 走朴素拼接分支。"""

    eos_token_id = 2

    def __call__(self, prompt, **kwargs):
        return {"input_ids": _FakeIds([1, 2, 3]), "attention_mask": _FakeIds([1, 1, 1])}

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


class _FakeGenModel:
    device = "cpu"

    def __init__(self, tracker):
        self._tracker = tracker

    def generate(self, **kwargs):
        with self._tracker:
            return [_FakeIds([1, 2, 3, 4, 5])]


def _bare_edge_runtime(tracker) -> EdgeRuntime:
    rt = object.__new__(EdgeRuntime)
    rt.cfg = EdgeConfig()
    rt.logger = logging.getLogger("test.edge_runtime")
    rt._inference_lock = threading.RLock()
    rt._ready = True
    rt.model = _FakeGenModel(tracker)
    rt.tokenizer = _FakeGenTokenizer()
    rt.model_id = "fake/model"
    rt.display_model_id = "fake/model"
    return rt


_MESSAGES = [{"role": "user", "content": "你好"}]


def test_edge_generate_serializes_concurrent_calls():
    # 所有线程必须打同一个 runtime 实例，实例锁才有意义
    tracker = _Tracker()
    rt = _bare_edge_runtime(tracker)

    threads = [threading.Thread(target=lambda: rt.generate(_MESSAGES)) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.calls == THREADS
    assert tracker.peak == 1, f"generate 未被串行化，并发峰值 {tracker.peak}"


def test_edge_generate_lock_is_what_serializes():
    """反证：绕过锁直接调 _generate_locked 必须能并发，否则上一条断言是空的。"""

    tracker = _Tracker()
    rt = _bare_edge_runtime(tracker)

    threads = [
        threading.Thread(target=lambda: rt._generate_locked(_MESSAGES, 8)) for _ in range(THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.calls == THREADS
    assert tracker.peak > 1, "桩模型本身就能并发，说明 peak==1 的断言并非来自锁"


def test_edge_generate_raises_before_touching_lock_when_not_ready():
    """未就绪时应立刻报错，而不是排在一次长生成后面等锁。"""

    rt = _bare_edge_runtime(_Tracker())
    rt._ready = False
    held = threading.Event()

    # 占住锁，模拟一次正在进行的长生成
    def _hold():
        with rt._inference_lock:
            held.set()
            time.sleep(0.3)

    holder = threading.Thread(target=_hold)
    holder.start()
    assert held.wait(1.0)

    started = time.monotonic()
    try:
        rt.generate(_MESSAGES)
        raise AssertionError("未就绪却成功返回")
    except RuntimeError as exc:
        assert "not ready" in str(exc)
    elapsed = time.monotonic() - started
    holder.join()
    assert elapsed < 0.2, f"未就绪的调用被锁阻塞了 {elapsed:.3f}s，就绪校验应留在锁外"


# ------------------------------------------------------- EdgeEmbeddingRuntime


class _FakeEmbModel:
    device = torch.device("cpu")

    def __init__(self, tracker, hidden: int = 8):
        self._tracker = tracker
        self._hidden = hidden

    def __call__(self, **inputs):
        batch = inputs["input_ids"].shape[0]
        seq = inputs["input_ids"].shape[1]
        with self._tracker:
            return SimpleNamespace(
                last_hidden_state=torch.ones((batch, seq, self._hidden), dtype=torch.float32)
            )


class _FakeEmbTokenizer:
    def __call__(self, chunk, **kwargs):
        n = len(chunk)
        return {
            "input_ids": torch.ones((n, 4), dtype=torch.long),
            "attention_mask": torch.ones((n, 4), dtype=torch.float32),
        }


def _bare_embedding_runtime(tracker) -> EdgeEmbeddingRuntime:
    rt = object.__new__(EdgeEmbeddingRuntime)
    rt.cfg = EmbeddingConfig()
    rt.logger = logging.getLogger("test.embedding_runtime")
    rt._inference_lock = threading.RLock()
    rt._ready = True
    rt.model = _FakeEmbModel(tracker)
    rt.tokenizer = _FakeEmbTokenizer()
    rt.model_id = "fake/embedding"
    return rt


def test_embed_serializes_concurrent_calls():
    tracker = _Tracker()
    rt = _bare_embedding_runtime(tracker)

    def call():
        result = rt.embed(["一段文本", "另一段文本"])
        assert len(result.embeddings) == 2
        assert all(len(v) == 8 for v in result.embeddings)

    threads = [threading.Thread(target=call) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.calls == THREADS
    assert tracker.peak == 1, f"embed 未被串行化，并发峰值 {tracker.peak}"


def test_embed_lock_is_what_serializes():
    """反证：绕过锁直接调 _embed_locked 必须能并发。"""

    tracker = _Tracker()
    rt = _bare_embedding_runtime(tracker)

    threads = [
        threading.Thread(target=lambda: rt._embed_locked(["文本"], True)) for _ in range(THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert tracker.calls == THREADS
    assert tracker.peak > 1, "桩模型本身就能并发，说明 peak==1 的断言并非来自锁"


def test_embed_normalizes_and_keeps_values_in_range():
    """顺带锁定 mean-pooling + L2 归一化的输出性质（串行化改动不应影响数值）。"""

    rt = _bare_embedding_runtime(_Tracker())
    result = rt.embed(["a", "b", "c"], normalize=True)

    assert len(result.embeddings) == 3
    for vec in result.embeddings:
        assert abs(torch.norm(torch.tensor(vec)).item() - 1.0) < 1e-5, "归一化后模长应为 1"
    assert result.used_model == "fake/embedding"


def test_runtimes_create_their_inference_lock_in_init():
    """堵一个覆盖缺口：上面的用例都用 object.__new__ 绕过 __init__ 并自带一把锁，
    所以它们只能证明 generate/embed **使用** self._inference_lock，无法证明
    __init__ **创建** 了它。若那一行被删，生产环境会在首次推理时 AttributeError，
    而全部测试仍然通过 —— 故在此做源码级断言。
    """

    for cls in (EdgeRuntime, EdgeEmbeddingRuntime):
        source = inspect.getsource(cls.__init__)
        assert re.search(r"self\._inference_lock\s*=\s*threading\.RLock\(\)", source), (
            f"{cls.__name__}.__init__ 未创建 self._inference_lock = threading.RLock()"
        )


def test_embed_empty_input_returns_early_without_lock():
    """空输入不碰模型，也不应排队等锁。"""

    rt = _bare_embedding_runtime(_Tracker())
    assert rt.embed([]).embeddings == []
    assert rt._inference_lock.acquire(blocking=False), "空输入路径不应持有锁"
    rt._inference_lock.release()
