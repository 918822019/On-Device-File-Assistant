"""Shared lightweight text utilities.

个人文件搜索与报销两条业务线此前各自实现分词：
- personal_search 有中文短语切分；
- expense 只用 keyword.split()（按空格），"京东发票" 这类无空格中文查询
  会退化成单个整 token，命中率显著偏低。
统一收敛到本模块，避免同仓库两套分词口径。

2026-09（M0）：中文段接入 jieba 搜索引擎模式。此前无标点长查询（如
"上周群里发的聚餐照片"）整体是一个 token，字面路几乎必然 0 分，检索靠
时间/来源线索硬撑（BUSINESS_LAYER §四.1 的第一遗留项）。jieba 为可选
依赖：缺失时自动退回「标点切分 + 中文短语提取」的纯规则口径，业务不中断。
"""

from __future__ import annotations

import logging
import re

_SPLIT_RE = re.compile(r"[\s,，。；;:：!！?？、/\\|()（）【】\-]+")
_CJK_PHRASE_RE = re.compile(r"[一-鿿]{2,}")
_CJK_CHAR_RE = re.compile(r"[一-鿿]")

_jieba = None
_jieba_resolved = False


def _resolve_jieba():
    """懒加载 jieba 并压掉其启动日志；不可用时返回 None（降级纯规则）。"""

    global _jieba, _jieba_resolved
    if not _jieba_resolved:
        _jieba_resolved = True
        try:
            import jieba as _mod

            _mod.setLogLevel(logging.ERROR)
            # 预建词典（首切约 1s）：避免首个请求或 watch 线程首扫时的并发抖动
            _mod.initialize()
            _jieba = _mod
        except Exception:
            _jieba = None
    return _jieba


def _refine_cjk_token(token: str) -> list[str]:
    """对含中文的 token 做 jieba 细分；丢弃无区分度的单字段。"""

    mod = _resolve_jieba()
    if mod is None:
        return [token]
    words = [w.strip().lower() for w in mod.cut_for_search(token)]
    refined = [
        w for w in words
        if w and not (len(w) == 1 and _CJK_CHAR_RE.fullmatch(w))
    ]
    # 切完只剩单字时（如查询"票"），保留原 token，避免整段丢失
    return refined or [token]


def tokenize(query: str) -> list[str]:
    """将中英混合查询切词。

    - 先按空白/中英文标点切出粗段；
    - 含中文的段用 jieba 搜索引擎模式细分（jieba 不可用时保留粗段，并额外
      提取 ≥2 字中文短语作为补充线索，即旧口径）；
    - 不保留切不开的整段长 token：整句精确命中由打分端独立承接
      （_score_item 对 query 整句 substring 命中有 +1.5 加成），保留只会
      抬高计分分母、稀释词命中比例。

    返回去重、排序后的小写 token 列表；空输入返回空列表。
    """

    if not query:
        return []
    parts = _SPLIT_RE.split(query)
    coarse = [item.strip().lower() for item in parts if item.strip()]

    tokens: list[str] = []
    for token in coarse:
        if _CJK_CHAR_RE.search(token):
            tokens.extend(_refine_cjk_token(token))
        else:
            tokens.append(token)

    if _resolve_jieba() is None:
        # 降级口径：≥2 字中文短语补充为 token（与 jieba 可用时互斥）
        for phrase in _CJK_PHRASE_RE.findall(query):
            lower = phrase.lower()
            if lower not in tokens:
                tokens.append(lower)

    return sorted(set(tokens))
