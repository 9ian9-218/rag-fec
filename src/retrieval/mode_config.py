"""各檢索模式的預設參數與簡單路由。"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from lightrag.base import QueryParam

RetrievalMode = Literal["naive", "local", "global", "hybrid", "mix", "bypass"]


MODE_DEFAULTS: dict[str, dict[str, object]] = {
    "naive": {"chunk_top_k": 20},
    "local": {"top_k": 40},
    "global": {"top_k": 40},
    "hybrid": {"top_k": 30},
    "mix": {"top_k": 10, "chunk_top_k": 12},
    "bypass": {},
}


def build_query_param(
    mode: RetrievalMode,
    *,
    top_k: int | None = None,
    stream: bool = False,
    only_need_context: bool = False,
    conversation_history: list[dict[str, str]] | None = None,
) -> QueryParam:
    """建立 ``QueryParam``，並套用模式預設。"""
    defaults = MODE_DEFAULTS.get(mode, {})
    param = QueryParam(
        mode=mode,
        stream=stream,
        only_need_context=only_need_context,
        conversation_history=conversation_history or [],
    )
    if top_k is not None:
        param = replace(param, top_k=top_k)
    elif "top_k" in defaults:
        param = replace(param, top_k=int(defaults["top_k"]))
    if "chunk_top_k" in defaults:
        param = replace(param, chunk_top_k=int(defaults["chunk_top_k"]))
    return param


def suggest_mode_from_question(question: str) -> RetrievalMode:
    """極簡啟發式：含「總結、整體、全書」偏 global，否則 mix。"""
    q = question.strip().lower()
    keys = ("總結", "整體", "全書", "概述", "global summary", "overview")
    if any(k.lower() in q for k in keys):
        return "global"
    return "mix"
