"""各檢索模式的預設參數與簡單路由。"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from lightrag.base import QueryParam

from config.settings import Settings, get_settings

RetrievalMode = Literal["naive", "local", "global", "hybrid", "mix", "bypass"]


def _lightrag_query_overrides(settings: Settings) -> dict[str, object]:
    lr = settings.lightrag
    chunk_top_k = lr.chunk_top_k if lr.chunk_top_k is not None else settings.retrieval.top_k
    return {
        "max_entity_tokens": lr.max_entity_tokens,
        "max_relation_tokens": lr.max_relation_tokens,
        "max_total_tokens": lr.max_total_tokens,
    }


MODE_DEFAULTS: dict[str, dict[str, object]] = {
    "naive": {"chunk_top_k": 12},
    "local": {},
    "global": {},
    "hybrid": {},
    "mix": {"chunk_top_k": 9},
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
    settings = get_settings()
    defaults = MODE_DEFAULTS.get(mode, {})
    lr_over = _lightrag_query_overrides(settings)
    param = QueryParam(
        mode=mode,
        stream=stream,
        only_need_context=only_need_context,
        conversation_history=conversation_history or [],
        **lr_over,  # type: ignore[arg-type]
    )
    base_top_k = top_k if top_k is not None else settings.retrieval.top_k
    param = replace(param, top_k=base_top_k)
    if "chunk_top_k" in defaults:
        param = replace(param, chunk_top_k=int(defaults["chunk_top_k"]))
    else:
        ct = settings.lightrag.chunk_top_k
        if ct is not None:
            param = replace(param, chunk_top_k=int(ct))
    if not settings.rerank_runtime_available():
        param = replace(param, enable_rerank=False)
    return param


def suggest_mode_from_question(question: str) -> RetrievalMode:
    """極簡啟發式：含「總結、整體、全書」偏 global，否則 mix。"""
    q = question.strip().lower()
    keys = ("總結", "整體", "全書", "概述", "global summary", "overview")
    if any(k.lower() in q for k in keys):
        return "global"
    return "mix"
