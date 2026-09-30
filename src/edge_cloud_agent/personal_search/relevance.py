"""相关性规则：线索识别、混合打分、候选过滤与重排。

本模块是与存储/会话无关的纯规则层：输入候选与查询线索，输出分值、命中证据和
排序结果。权重由调用方（service）按配置传入，便于测试固定口径。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from ..common.text_utils import tokenize
from ..common.time_utils import parse_iso, utcnow_naive
from ..common.vectors import cosine_similarity
from .storage import PersonalFileItem

TIME_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"上周"), 7),
    (re.compile(r"上个?月"), 30),
    (re.compile(r"本周"), 7),
    (re.compile(r"最近(\d+)?天"), 3),
    # "明天" 不纳入时间线索：文件 captured_at 是过去时刻，"明天"作为时间约束
    # 语义上无意义，纳入后会误匹配最近 24 小时的所有文件。
    (re.compile(r"今天|昨日|昨天|前天"), 1),
]

CLUE_COLOR_TOKENS = {
    "蓝色": "blue",
    "白色": "white",
    "黑色": "black",
    "红色": "red",
    "绿色": "green",
    "黄色": "yellow",
    "橙色": "orange",
    "紫色": "purple",
    "灰色": "gray",
}

# 来源提示的 key 与 infer_source_from_path 的返回值域（wechat/email/gallery/camera）
# 及 doc_type（document）对齐；此前用 image/office 作 key，与任何 source_app 值都
# 对不上，只能靠 mime/doc_type 里的子串巧合命中。
SOURCE_KEYWORDS = {
    "wechat": {"微信", "weixin", "wechat", "微信好友", "微信群", "群里", "群聊"},
    "email": {"邮箱", "email", "mail", "gmail", "outlook"},
    "gallery": {"图库", "相册", "gallery"},
    "camera": {"拍照", "相机", "camera", "截图", "截屏", "screenshot", "screen"},
    "document": {"word", "excel", "ppt", "文档", "文件", "doc", "pdf"},
}

# 来源提示 -> 候选文本中的字面别名。source_app / doc_type / mime 三处值域不同
# （如"相册"提示=gallery，而图片文件 doc_type="image"、mime="image/*"），
# 打分与过滤两侧共用本别名表，保证口径一致。
SOURCE_TEXT_ALIASES = {
    "wechat": ("wechat",),
    "email": ("email", "mail"),
    "gallery": ("gallery", "image"),
    "camera": ("camera", "screenshot"),
    "document": ("document", "office", "pdf"),
}

# 版本标记：前面不是字母数字的 v+数字（方案v2 / V4 / v1.3），或中文"版本"。
# 不用 \b：中文语境（如"方案v2"）里 CJK 与字母间没有词边界，\bv 命不中。
VERSION_MARK_RE = re.compile(r"(?<![a-z0-9])v\d+(?:\.\d+)*|版本")

# resolved 判定：首名与次名分差达到该值即认为无需澄清。
RESOLVED_SCORE_GAP = 0.35

# 重排时上一轮得分作为先验的权重；过低会让线索稀少的回复排序塌缩。
PRIOR_SCORE_WEIGHT = 0.3

# 候选卡片上展示的命中证据最多保留几段（澄清轮数无上限，不设界会无限增长）。
EVIDENCE_MAX_SEGMENTS = 4


@dataclass
class SearchCandidate:
    """内部候选结构体：原始 item + 打分 + 命中证据。"""

    item: PersonalFileItem
    score: float
    evidence: str
    matched_clues: list[str]


@dataclass(frozen=True)
class ScoreWeights:
    """打分权重（来自 PersonalFileConfig），负值一律按 0 处理。"""

    text: float
    semantic: float
    clue: float
    version_bonus: float

    @property
    def total(self) -> float:
        return max(0.01, self.text + self.semantic + self.clue)


def source_hit(source_hints: list[str], text_lower: str) -> list[str]:
    """返回在候选文本中命中的来源提示列表（经别名展开）。"""

    hits: list[str] = []
    for hint in source_hints:
        keys = SOURCE_TEXT_ALIASES.get(hint, (hint,))
        if any(key in text_lower for key in keys):
            hits.append(hint)
    return hits


def clamp01(value: float) -> float:
    """将分值限定在 0~1 区间，避免下游排序异常。"""

    return max(0.0, min(1.0, value))


def has_time_match(captured_at: str | None, query: str) -> tuple[bool, str]:
    """从查询文本中判断时间约束是否命中，返回匹配说明用于 evidence。"""

    if not captured_at:
        return False, ""

    q = query.replace(" ", "")
    item_time = parse_iso(captured_at)
    if item_time is None:
        return False, ""

    now = utcnow_naive()
    for pattern, days in TIME_PATTERNS:
        if pattern.search(q):
            if days == 1 and "今天" in q and abs((now.date() - item_time.date()).days) == 0:
                return True, "时间线索匹配: 今天"
            if "昨天" in q and (now.date() - item_time.date()).days == 1:
                return True, "时间线索匹配: 昨天"
            if "前天" in q and (now.date() - item_time.date()).days == 2:
                return True, "时间线索匹配: 前天"
            if "上周" in q and (now - item_time) <= timedelta(days=14):
                return True, "时间线索匹配: 上周"
            if ("上月" in q or "上个月" in q) and (now - item_time) <= timedelta(days=45):
                return True, "时间线索匹配: 上月"
            # isocalendar()[:2] = (iso_year, iso_week)；只比较 week 不比较 year 会跨年误匹配
            if "本周" in q and item_time.date().isocalendar()[:2] == now.date().isocalendar()[:2]:
                return True, "时间线索匹配: 本周"
            if q.startswith("最近") and "天" in q:
                try:
                    num = int(re.findall(r"最近(\d+)天", q)[0])
                except Exception:
                    num = days
                if (now - item_time) <= timedelta(days=num):
                    return True, f"时间线索匹配: 最近{num}天"
                return False, ""
            if item_time >= now - timedelta(days=days):
                return True, f"时间线索匹配: {days}天内"

    return False, ""


def infer_source_from_path(file_path: Path) -> str:
    """基于文件路径做来源弱识别（wechat/微信/email/图片库等）。"""

    lowered = file_path.as_posix().lower()
    if "wechat" in lowered or "微信" in lowered or "weixin" in lowered:
        return "wechat"
    if "email" in lowered or "邮箱" in lowered or "mail" in lowered or "gmail" in lowered:
        return "email"
    if "image" in lowered or "images" in lowered or "图库" in lowered or "相册" in lowered:
        return "gallery"
    # "截屏" 为 macOS 中文系统截图文件名前缀（截屏2026-09-24 ….png）；
    # "screen shot" 为旧版 macOS 英文命名（Screen Shot 2026-…），与 screenshot 并列。
    if (
        "camera" in lowered
        or "截图" in lowered
        or "截屏" in lowered
        or "screenshot" in lowered
        or "screen shot" in lowered
    ):
        return "camera"
    return "local"


def extract_source_hints(query: str) -> list[str]:
    normalized = query.lower()
    return [
        source
        for source, keywords in SOURCE_KEYWORDS.items()
        if any(key in normalized for key in keywords)
    ]


def extract_visual_hints(query: str) -> list[str]:
    normalized = query.lower()
    return [token for token in CLUE_COLOR_TOKENS if token in normalized]


def extract_version_hint(query: str) -> bool:
    q = query.replace(" ", "")
    return "后来的" in q or "最新" in q or "版本" in q or "哪个" in q


def score_item(
    item: PersonalFileItem,
    query: str,
    *,
    tokens: list[str],
    source_hints: list[str],
    visual_hints: list[str],
    version_hint: bool,
    query_embedding: list[float] | None,
    weights: ScoreWeights,
) -> tuple[float, str, list[str]]:
    """计算单条文件与查询的综合分值与命中证据。"""

    haystack = " ".join(
        [
            item.title,
            item.summary,
            item.doc_type,
            item.source_app,
            item.raw_text,
            " ".join(item.tags),
            " ".join(item.visual_hints),
            item.file_path,
        ]
    ).lower()

    text_hits = 0.0
    matched: list[str] = []
    for token in tokens:
        if token and token in haystack:
            text_hits += 1.0
            matched.append(token)
    if query.lower() in haystack:
        text_hits += 1.5
        matched.append("全文命中")

    clue_score = 0.0
    for source in source_hit(source_hints, haystack):
        clue_score += 0.8
        matched.append(f"来源:{source}")

    for visual in visual_hints:
        if visual in haystack:
            clue_score += 0.7
            matched.append(f"视觉:{visual}")

    time_hit, reason = has_time_match(item.captured_at, query)
    if time_hit:
        clue_score += 0.7
        matched.append(reason)

    if version_hint and ("版本" in haystack or re.search(r"v\d+", item.title.lower())):
        clue_score += 0.5
        matched.append("版本关系")

    # 仅当查询带版本意图（"最新/后来的/哪个版本"）且候选确实带版本标记
    # （v2、v1.3、"版本"字样）时才给排序加分。
    # 旧实现 `if "v" in haystack` 过松：任何含字母 v 的文本（video、save、
    # csv、临时目录名）都拿 0.2 加分，与查询意图完全无关。
    version_rank_bonus = 0.0
    if version_hint and VERSION_MARK_RE.search(haystack):
        version_rank_bonus = 0.2

    embed_score = 0.0
    if query_embedding and item.embedding:
        embed_score = cosine_similarity(query_embedding, item.embedding)

    normalized_tokens = max(1.0, len(tokens))
    score = (
        (text_hits / normalized_tokens) * weights.text
        + embed_score * weights.semantic
        + clue_score * weights.clue
        + version_rank_bonus * weights.version_bonus
    )
    score = clamp01(score / weights.total)
    evidence = "; ".join(sorted(set(matched))) if matched else "仅按语义近似"
    return score, evidence, sorted(set(matched))


def decide_state(candidates: list[SearchCandidate], force_disambiguation: bool) -> tuple[str, str | None, bool]:
    """根据候选数量和分差判断是否已经 resolved。"""

    if not candidates:
        return "not_found", "没找到命中项，请再给一条时间/来源/场景线索。", False
    if len(candidates) == 1 and not force_disambiguation:
        return "resolved", None, False

    if len(candidates) >= 2:
        gap = candidates[0].score - candidates[1].score
        if gap >= RESOLVED_SCORE_GAP and not force_disambiguation:
            return "resolved", None, False

    return "needs_clarification", build_question(candidates), True


def build_question(candidates: list[SearchCandidate]) -> str:
    if len(candidates) >= 3:
        return (
            "我找到几条可能的候选。你可以补 1-2 个线索继续确认："
            "比如说时间（上周/昨天）、来源（微信群/邮箱）或者视觉线索（蓝色背景、聚餐、会议）。"
        )
    return "我有两条很接近的结果，选一个更确定的描述：例如‘5月的那张’/‘群里的那张’。"


def filter_candidates_by_reply(candidates: list[SearchCandidate], reply: str) -> list[SearchCandidate]:
    """按用户回复过滤候选。

    除字面命中外，还识别两类最常见的自然回复线索：
    - 时间线索（"上周的/昨天的"）：按 captured_at 匹配，字面文本里没有"上周"字样；
    - 来源线索（"微信群里的"）：中文关键词映射到 source_app（wechat 等）再比对。
    此前只做字面 token 匹配，这两类回复会把全部候选滤空，clarify 直接死路。
    """

    reply_lower = reply.lower()
    reply_tokens = tokenize(reply_lower)
    source_hints = extract_source_hints(reply)
    filtered = []
    for candidate in candidates:
        item = candidate.item
        text = candidate_text(item)
        if reply_lower in text or any(token in text for token in reply_tokens):
            filtered.append(candidate)
            continue
        time_hit, _ = has_time_match(item.captured_at, reply)
        if time_hit:
            filtered.append(candidate)
            continue
        # 与 score_item 同口径：来源提示经别名展开命中即保留
        # （如"相册"提示=gallery，可命中 doc_type="image" 的图片文件）。
        if source_hit(source_hints, text):
            filtered.append(candidate)
    return filtered


def candidate_text(item: PersonalFileItem) -> str:
    """过滤/澄清阶段用于字面比对的候选文本（不含 raw_text 全文）。"""

    return " ".join(
        [
            item.title,
            item.summary,
            item.doc_type,
            item.source_app,
            " ".join(item.tags),
            " ".join(item.visual_hints),
        ]
    ).lower()


def _merge_evidence(prior: str, current: str) -> str:
    """合并上一轮与本轮的命中证据，并限制保留的段数。

    澄清轮数没有上限，而 evidence 是逐轮用「；」拼接后**直接展示给用户**的
    （web/js/search.js 渲染成「证据: …」）。原先无条件拼接会让这个字符串随
    对话轮数无限增长，候选卡片越拉越长。

    只保留最近 EVIDENCE_MAX_SEGMENTS 段：新线索比旧线索更能解释当前排序，
    被省略的以「…」开头提示。完整的命中类型仍保留在 matched_clues 里
    （它是去重集合，不会随轮数线性增长）。
    """

    segments = [seg.strip() for seg in (prior or "").split("；") if seg.strip()]
    if current and current.strip():
        segments.append(current.strip())
    if len(segments) > EVIDENCE_MAX_SEGMENTS:
        segments = ["…", *segments[-EVIDENCE_MAX_SEGMENTS:]]
    return "；".join(segments)


def rerank_within(
    candidates: list[SearchCandidate],
    query: str,
    top_k: int,
    *,
    weights: ScoreWeights,
    embed_text: Callable[[str], list[float] | None],
) -> list[SearchCandidate]:
    """基于用户回复再打一次分，推动候选向真实目标收敛。

    打分策略：
    - 回复文本同时以字面线索和语义向量参与打分（此前 embedding 传 None，
      导致 clarify 之后语义信号整体丢失）；
    - 上一轮得分只作为低权重先验保留，避免回复线索较少时排序塌缩；
    - 不再对旧第一名做 +1.0 硬性加分：那会突破 [0,1] 分值域，并让
      decide_state 的分差判定几乎必然命中，追问一轮后总是 resolved 成旧第一名，
      用户补充的线索被架空。
    """

    if not candidates:
        return []
    tokens = tokenize(query)
    source_hints = extract_source_hints(query)
    visual_hints = extract_visual_hints(query)
    version_hint = extract_version_hint(query)
    query_embedding = embed_text(query)

    prior_weight = PRIOR_SCORE_WEIGHT
    ranked: list[SearchCandidate] = []
    for candidate in candidates:
        base_score, evidence, matched = score_item(
            candidate.item,
            query,
            tokens=tokens,
            source_hints=source_hints,
            visual_hints=visual_hints,
            version_hint=version_hint,
            query_embedding=query_embedding,
            weights=weights,
        )
        combined = clamp01((1.0 - prior_weight) * base_score + prior_weight * candidate.score)
        ranked.append(
            SearchCandidate(
                item=candidate.item,
                score=combined,
                evidence=_merge_evidence(candidate.evidence, evidence),
                matched_clues=sorted(set(candidate.matched_clues + matched)),
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


def dedupe(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """按 file_id 去重，保留首次出现（即分值更高）的候选。"""

    seen: set[str] = set()
    deduped: list[SearchCandidate] = []
    for candidate in candidates:
        if candidate.item.file_id in seen:
            continue
        seen.add(candidate.item.file_id)
        deduped.append(candidate)
    return deduped
