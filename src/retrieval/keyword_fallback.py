"""查询关键词规则回退（LLM 抽取失败或返回空列表时使用）。

面向理工科科研文献（论文 / 专著 / 技术报告）的通用启发式，不绑定具体子领域。
"""

from __future__ import annotations

import re

from src.retrieval.relation_keywords import (
    SCHOLARLY_HIGH_HINTS,
    SCHOLARLY_LOW_PATTERNS,
    enhance_keywords_for_retrieval,
    extract_user_query_from_prompt,
)

_DEFAULT_HIGH_KEYWORDS = ("研究方法", "理论模型")


def scholarly_keyword_fallback(question: str) -> tuple[list[str], list[str]]:
    """从问题文本拆出 high/low 关键词，供 LightRAG 图检索使用。"""
    q = extract_user_query_from_prompt(question) or (question or "").strip()
    if not q:
        return [], []

    low: list[str] = []
    for pat in SCHOLARLY_LOW_PATTERNS:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            s = m.group(0).strip()
            if s and s not in low:
                low.append(s)

    try:
        import jieba

        for tok in jieba.lcut(q):
            t = tok.strip()
            if len(t) >= 2 and t not in low and not t.isspace():
                if re.search(r"[\u4e00-\u9fffA-Za-z0-9_]", t):
                    low.append(t)
    except Exception:
        pass

    low = [x for x in low if len(x) <= 48][:12]

    high: list[str] = []
    ql = q.lower()
    for hint in SCHOLARLY_HIGH_HINTS:
        if hint in q or (hint.isascii() and hint in ql):
            if hint not in high:
                high.append(hint)
    if not high:
        high = list(_DEFAULT_HIGH_KEYWORDS)
    if len(q) <= 80 and q not in low:
        low.insert(0, q)

    return enhance_keywords_for_retrieval(q, high[:6], low[:10])
