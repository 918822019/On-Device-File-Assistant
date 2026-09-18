"""Shared lightweight text utilities.

个人文件搜索与报销两条业务线此前各自实现分词：
- personal_search 有中文短语切分；
- expense 只用 keyword.split()（按空格），"京东发票" 这类无空格中文查询
  会退化成单个整 token，命中率显著偏低。
统一收敛到本模块，避免同仓库两套分词口径。
"""

from __future__ import annotations

import re

_SPLIT_RE = re.compile(r"[\s,，。；;:：!！?？、/\\|()（）【】\-]+")
_CJK_PHRASE_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


def tokenize(query: str) -> list[str]:
    """将中英混合查询切词，并额外提取 2 字以上的中文短语。

    返回去重、排序后的小写 token 列表；空输入返回空列表。
    """

    if not query:
        return []
    parts = _SPLIT_RE.split(query)
    tokens = [item.strip().lower() for item in parts if item.strip()]
    for phrase in _CJK_PHRASE_RE.findall(query):
        lower = phrase.lower()
        if lower not in tokens:
            tokens.append(lower)
    return sorted(set(tokens))
