"""FEC 领域查询关键词启发式回退（LLM 抽取失败或返回空列表时使用）。"""

from __future__ import annotations

import re

from src.retrieval.relation_keywords import enhance_keywords_for_retrieval, extract_user_query_from_prompt

_FEC_LOW_PATTERNS = (
    r"\bRM\b",
    r"Reed[\s-]?Muller",
    r"Polar",
    r"极化码",
    r"RPA",
    r"BEC",
    r"BSC",
    r"AGNC",
    r"BER",
    r"SCL",
    r"BP\s*译码",
    r"Plotkin",
    r"Arıkan",
    r"Arikan",
    r"G_RM",
    r"N_max",
    r"Proj\s*\(",
    r"\(\d+\s*,\s*\d+\)",
    r"RM\s*\(\s*m\s*,\s*r\s*\)",
    r"Table\s+[IVX\d]+",
    r"Algorithm\s+\d+",
)

_FEC_HIGH_HINTS = (
    "性能",
    "对比",
    "比较",
    "译码",
    "编码",
    "构造",
    "复杂度",
    "等价",
    "包含关系",
    "步骤",
    "算法",
    "矩阵",
    "投影",
    "信道",
    "误码",
    "作者",
    "发表",
)


def fec_keyword_fallback(question: str) -> tuple[list[str], list[str]]:
    """从问题文本拆出 high/low 关键词，供 LightRAG 图检索使用。"""
    q = extract_user_query_from_prompt(question) or (question or "").strip()
    if not q:
        return [], []

    low: list[str] = []
    for pat in _FEC_LOW_PATTERNS:
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
    for hint in _FEC_HIGH_HINTS:
        if hint in q and hint not in high:
            high.append(hint)
    if not high:
        high = ["信道编码", "差错控制"]
    if len(q) <= 80 and q not in low:
        low.insert(0, q)

    return enhance_keywords_for_retrieval(q, high[:6], low[:10])
