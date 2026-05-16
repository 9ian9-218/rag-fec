"""关系检索用高层关键词增强（供 global / hybrid / mix 的 edge 查询）。"""

from __future__ import annotations

import re

_FEC_LOW_PATTERNS = (
    r"\bRM\b",
    r"Reed[\s-]?Muller",
    r"Polar",
    r"极化码",
    r"RPA",
    r"BEC",
    r"BSC",
    r"AGNC",
)

_FEC_HIGH_HINTS = (
    "性能",
    "对比",
    "译码",
    "编码",
    "构造",
    "复杂度",
    "等价",
    "算法",
    "信道",
    "误码",
)

_JUNK_KEYWORDS = frozenset(
    {
        "role",
        "you",
        "are",
        "an",
        "expert",
        "keyword",
        "extractor",
        "specializing",
        "in",
        "analyzing",
        "user",
        "query",
        "given",
        "task",
        "json",
    }
)

_RELATION_THEME_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"性能|BER|误码|对比|比较", "信道编码性能"),
    (r"译码|解码|RPA|SCL|BP", "译码方法与算法"),
    (r"编码|构造|生成矩阵|Plotkin", "码构造与代数结构"),
    (r"复杂度", "编解码复杂度"),
    (r"等价|包含|子码|投影|Proj", "码族关系与投影"),
    (r"作者|发表|文献", "论文元信息"),
    (r"BEC|BSC|AGNC|信道", "信道模型与仿真"),
)


def extract_user_query_from_prompt(text: str) -> str:
    """从 LightRAG 关键词抽取 prompt 中还原用户问题。"""
    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(r"User Query:\s*(.+?)(?:\n-{3,}|\Z)", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "User Query:" not in t and len(t) < 500:
        return t
    return ""


def _is_junk_token(tok: str) -> bool:
    s = tok.strip().lower()
    if not s or len(s) < 2:
        return True
    if s in _JUNK_KEYWORDS:
        return True
    if re.fullmatch(r"[a-z]{1,2}", s):
        return True
    return False


def fec_relation_high_keywords(question: str) -> list[str]:
    """面向关系向量检索的高层主题短语。"""
    q = (question or "").strip()
    if not q:
        return []
    out: list[str] = []
    for pat, phrase in _RELATION_THEME_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE) and phrase not in out:
            out.append(phrase)
    for hint in _FEC_HIGH_HINTS:
        if hint in q and hint not in out:
            out.append(hint)
    codes: list[str] = []
    for pat in _FEC_LOW_PATTERNS:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            c = m.group(0).strip()
            if c and c not in codes:
                codes.append(c)
    if len(codes) >= 2:
        pair = f"{codes[0]}与{codes[1]}"
        if pair not in out:
            out.append(pair)
    return out[:8]


def enhance_keywords_for_retrieval(
    question: str,
    hl: list[str],
    ll: list[str],
) -> tuple[list[str], list[str]]:
    """清洗低层噪声并补强关系向 edge 查询用的高层词。"""
    q = extract_user_query_from_prompt(question) or (question or "").strip()
    hl_out: list[str] = []
    for x in hl or []:
        s = str(x).strip()
        if s and not _is_junk_token(s) and s not in hl_out:
            hl_out.append(s)
    for phrase in fec_relation_high_keywords(q):
        if phrase not in hl_out:
            hl_out.append(phrase)

    ll_out: list[str] = []
    for x in ll or []:
        s = str(x).strip()
        if s and not _is_junk_token(s) and s not in ll_out:
            ll_out.append(s)
    if not ll_out and q:
        ll_out = [q[:120]]
    return hl_out[:8], ll_out[:12]
