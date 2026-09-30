"""日志 formatter：extra 载荷必须真的输出，占位符不得污染每一行。

原先的固定格式串只渲染 event / trace_id / method / path / status_code /
duration_ms 六个字段，而全代码库给 _LOGGER.info(...) 传的丰富 extra 载荷
（scanned / imported / removed / state / cost_ms / rehashed / …）全部挂在
LogRecord 上却从不输出 —— docs/API.md 的「日志字典」因此形同虚设：日志里
只能看到事件名，看不到任何一个业务数值。
"""

import logging

import pytest

from edge_cloud_agent.logging_config import _DefaultLogContext, _ExtraFormatter, _render

FORMAT = (
    "%(levelname)s %(name)s event=%(event)s trace_id=%(trace_id)s "
    "method=%(method)s path=%(path)s status=%(status_code)s dur_ms=%(duration_ms)s %(message)s"
)


@pytest.fixture()
def logged():
    """返回一个 capture 函数，产出与生产完全一致的格式化结果。"""

    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    logger = logging.getLogger("test.logging_config")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    saved = list(logger.handlers)
    logger.handlers.clear()

    handler = _Collect()
    handler.setFormatter(_ExtraFormatter(FORMAT))
    handler.addFilter(_DefaultLogContext())
    logger.addHandler(handler)

    def capture(message="msg", **extra):
        records.clear()
        logger.info(message, extra=extra)
        assert len(records) == 1
        return records[0]

    yield capture

    logger.handlers = saved


def test_extra_fields_are_rendered(logged):
    out = logged("ingest-finished", event="ingest.finished", scanned=400, imported=0, removed=3)
    assert "scanned=400" in out
    assert "imported=0" in out
    assert "removed=3" in out


def test_fixed_part_still_rendered(logged):
    out = logged(
        "http-request",
        event="http.request",
        trace_id="abc123",
        method="GET",
        path="/health",
        status_code=200,
        duration_ms=0.77,
    )
    assert "event=http.request" in out
    assert "trace_id=abc123" in out
    assert "method=GET" in out
    assert "path=/health" in out
    assert "status=200" in out
    assert "dur_ms=0.77" in out
    # 已在固定串里渲染过的字段不应作为 extra 重复出现
    assert out.count("trace_id=") == 1
    assert out.count("method=") == 1


def test_placeholder_defaults_are_not_rendered(logged):
    """_DefaultLogContext 补的 "-" 不得污染每一行日志。"""

    out = logged("bare-message", event="some.event")
    for key in ("query_digest", "search_id", "session_id", "file_id", "action", "top_k", "state"):
        assert f"{key}=-" not in out, f"占位符 {key}=- 泄漏到日志里"


def test_real_values_of_defaulted_keys_are_rendered(logged):
    """同一个 key：占位符 "-" 不渲染，但真实值必须渲染。"""

    out = logged("search-completed", event="search.completed", state="needs_clarification", top_k=6)
    assert "state=needs_clarification" in out
    assert "top_k=6" in out


def test_explicit_none_is_skipped(logged):
    out = logged("m", event="e", trace_id=None, some_field=None)
    assert "some_field" not in out


def test_list_truncated_with_overflow_marker(logged):
    out = logged("m", event="e", sample_file_ids=["a", "b", "c", "d", "e", "f", "g"])
    assert "sample_file_ids=[a,b,c,d,e,…(+2)]" in out


def test_empty_and_short_lists(logged):
    assert "ids=[]" in logged("m", event="e", ids=[])
    assert "ids=[a,b]" in logged("m", event="e", ids=["a", "b"])


def test_spaces_and_newlines_are_sanitized(logged):
    """值里的空格会破坏 key=value 的可解析性，换行会破坏单行日志。"""

    out = logged("m", event="e", note="第一行\n第二行 带空格")
    assert "\n" not in out.split(" ", 1)[1], "extra 里不应出现真实换行"
    assert "note=第一行\\n第二行_带空格" in out


def test_long_string_truncated(logged):
    out = logged("m", event="e", blob="x" * 500)
    assert "…" in out
    assert "x" * 121 not in out


def test_dict_and_scalar_rendering(logged):
    out = logged("m", event="e", payload={"b": 2, "a": 1}, flag=True, ratio=0.123456)
    assert "payload={a=1,b=2}" in out, "dict 应按 key 排序渲染，保证输出稳定可 diff"
    assert "flag=true" in out
    assert "ratio=0.1235" in out


def test_extras_are_sorted_for_stable_output(logged):
    out = logged("m", event="e", zebra=1, alpha=2, middle=3)
    positions = {key: out.index(f"{key}=") for key in ("alpha", "middle", "zebra")}
    assert positions["alpha"] < positions["middle"] < positions["zebra"], (
        f"extra 未按 key 排序，输出无法稳定 diff：{positions}"
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "-"),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        ("", "-"),
        ([], "[]"),
        ({"k": None}, "{k=-}"),
    ],
)
def test_render_scalars(value, expected):
    assert _render(value) == expected


def test_log_record_internals_never_leak(logged):
    """logging 自带属性不能被当成业务 extra 输出。"""

    out = logged("m", event="e", scanned=1)
    for internal in ("levelname", "funcName", "lineno", "module", "processName", "threadName", "msg", "args"):
        assert f"{internal}=" not in out
