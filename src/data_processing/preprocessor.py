"""文字清洗與輕量正規化。"""

from __future__ import annotations

import re

_WS_LINES = re.compile(r"[ \t]+\n", re.MULTILINE)
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_PAGE_HEADER = re.compile(r"^\s*(第\s*\d+\s*頁|Page\s*\d+)\s*$", re.MULTILINE | re.IGNORECASE)


def clean_whitespace(text: str) -> str:
    """合併多餘空白與換行。"""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = _MULTI_SPACE.sub(" ", t)
    t = _WS_LINES.sub("\n", t)
    t = _MULTI_NL.sub("\n\n", t)
    return t.strip()


def strip_boilerplate(text: str) -> str:
    """移除常見頁碼列等雜訊。"""
    return _PAGE_HEADER.sub("", text)


def normalize_units(text: str) -> str:
    """極簡單位與空白正規化（可依專案擴充）。"""
    t = text
    t = re.sub(r"(\d)\s+%", r"\1%", t)
    return t


def preprocess(text: str, *, aggressive: bool = False) -> str:
    """完整預處理管線。"""
    t = clean_whitespace(text)
    t = strip_boilerplate(t)
    t = normalize_units(t)
    if aggressive:
        t = re.sub(r"[^\S\n]+", " ", t)
    return t.strip()
