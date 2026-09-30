"""向量度量工具：与业务无关、只依赖 stdlib。

两条业务线（personal_search 的混合打分、expense 的材料相关性）此前各自实现了
一份 ``cosine_similarity``，逻辑逐行相同 —— 连「长度不一致时按最小维度对齐」
这个不那么显然的取舍都一样。两份拷贝的风险不在于行数，而在于**只会改一份**：
一旦有人调整了归一化或零向量兜底口径，两条业务线的相似度分数会静默分叉，而
它们的阈值（0.7/0.3 融合权重、命中判定）都是按同一个口径调出来的。

这里刻意用纯 Python 循环而不是 numpy：embedding 维度是 768，单次查询只算几十
条候选，numpy 的 import 与数组构造开销远大于计算本身；而且 numpy 在本项目是
可选依赖（见 vector_index 的局部导入约定），common/ 不能依赖它。
"""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度，输入向量长度不一致时用最小维度对齐。

    三种情况返回 0.0 而不是抛异常，因为调用方都在打分热路径上、且降级到
    「纯关键词得分」比整个请求失败更符合项目的全链路静默降级口径：
    空向量、长度对齐后为 0、任一向量范数为 0（全零 embedding）。
    """

    if not a or not b:
        return 0.0
    size = min(len(a), len(b))
    if size == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(size):
        dot += a[i] * b[i]
        norm_a += a[i] ** 2
        norm_b += b[i] ** 2
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)
