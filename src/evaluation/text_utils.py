"""文本標準化（參考 graph-rag-agent/evaluation/utils/text_utils）。"""

from __future__ import annotations

import re
import string


_SIMP_CONVERTER: object | None = None


def to_simplified_chinese(text: str) -> str:
    """繁轉簡（評測匹配用）；無 opencc 時原樣返回。"""
    global _SIMP_CONVERTER
    t = text or ""
    if not t.strip():
        return t
    try:
        if _SIMP_CONVERTER is None:
            from opencc import OpenCC

            _SIMP_CONVERTER = OpenCC("t2s")
        return str(_SIMP_CONVERTER.convert(t))
    except Exception:
        return t


def normalize_answer(s: str) -> str:
    """移除冠詞、標點，小寫並壓縮空白，便於 EM / 集合匹配。"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|一个|一种|这个|那个)\b", " ", text, flags=re.I)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation + "，。！？《》【】\"'：；（）、")
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
