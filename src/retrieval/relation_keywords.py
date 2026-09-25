"""关系检索用高层关键词增强（供 global / hybrid / mix 的 edge 查询）。

词表面向理工科科研文献（论文 / 专著 / 技术报告）通用，不绑定具体子领域，
与 ``config/schema_defaults`` 的实体 schema 配套使用。
"""

from __future__ import annotations

import re

# 值得作为 low-level 关键词的技术记号：编号化的定理、算法、公式、图表等
SCHOLARLY_LOW_PATTERNS: tuple[str, ...] = (
    r"\b(?:Theorem|Lemma|Corollary|Proposition|Definition|Axiom|Remark|Claim|Proof)"
    r"\s*\.?\s*\d+(?:\.\d+)*",
    r"\bAlgorithm\s*\d+(?:\.\d+)*",
    r"\bEq(?:uation)?\.?\s*\(?\d+(?:\.\d+)*\)?",
    r"\b(?:Table|Fig(?:ure)?|Section|Chapter|Appendix)\s*\.?\s*[IVXLCivxlc\d]+",
    r"\bO\([^)]{1,40}\)",
    r"\b[A-Za-z]+_\{?[A-Za-z0-9]+\}?",
)

# 通用科研高层主题词（命中即作为 high-level 关键词参与 edge 检索）
SCHOLARLY_HIGH_HINTS: tuple[str, ...] = (
    "定理", "引理", "命题", "推论", "证明", "推导", "定义", "公式", "方程",
    "模型", "机制", "理论", "假设", "框架", "定律",
    "算法", "方法", "流程", "步骤", "实现", "优化",
    "实验", "仿真", "测试", "评测", "数据集", "样本", "数据",
    "指标", "性能", "误差", "精度", "复杂度", "效率", "收敛",
    "对比", "比较", "基线", "综述", "局限", "优势", "创新", "贡献", "结论",
    "参数", "超参数", "阈值", "变量", "结构", "性质", "应用", "场景",
    "作者", "机构", "发表", "期刊", "会议", "基金", "资助",
    "theorem", "proof", "model", "algorithm", "method", "experiment",
    "dataset", "benchmark", "metric", "performance", "complexity",
    "comparison", "survey", "limitation", "contribution",
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
    (r"性能|效率|误差|精度|对比|比较|benchmark|performance", "性能对比"),
    (r"定理|证明|推导|引理|命题|公理", "理论与证明"),
    (r"模型|机制|理论|假设|框架|定律", "模型与机制"),
    (r"算法|方法|流程|步骤|实现|优化|协议", "方法与实现"),
    (r"复杂度|开销|加速|收敛|效率", "复杂度与效率"),
    (r"实验|仿真|测试|评测|数据集|样本", "实验与数据"),
    (r"参数|超参数|阈值|配置|设置", "参数与设置"),
    (r"结构|性质|等价|包含|关系", "结构与性质"),
    (r"应用|场景|任务|领域", "应用与场景"),
    (r"作者|机构|发表|期刊|会议|基金|资助", "文献元信息"),
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


def scholarly_relation_high_keywords(question: str) -> list[str]:
    """面向关系向量检索的高层主题短语。"""
    q = (question or "").strip()
    if not q:
        return []
    out: list[str] = []
    for pat, phrase in _RELATION_THEME_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE) and phrase not in out:
            out.append(phrase)
    ql = q.lower()
    for hint in SCHOLARLY_HIGH_HINTS:
        if hint in q or (hint.isascii() and hint in ql):
            if hint not in out:
                out.append(hint)
    tokens: list[str] = []
    for pat in SCHOLARLY_LOW_PATTERNS:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            c = m.group(0).strip()
            if c and c not in tokens:
                tokens.append(c)
    if len(tokens) >= 2:
        pair = f"{tokens[0]}与{tokens[1]}"
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
    for phrase in scholarly_relation_high_keywords(q):
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
