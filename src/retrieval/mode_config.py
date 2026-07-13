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
    user_prompt: str | None = None,
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
        user_prompt=user_prompt,
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
    """改進啟發式：根據關鍵詞選擇模式，不再默認 mix。"""
    q = question.strip().lower()

    # Global：總結、概述、宏觀、對比趨勢
    global_keys = (
        "總結", "整體", "全書", "概述", "全局", "趨勢", "對比", "比較", "宏觀",
        "global summary", "overview", "compare", "comparison", "contrast",
    )
    if any(k in q for k in global_keys):
        return "global"

    # Naive：簡單事實、定義查詢、直接提問
    naive_keys = (
        "什麼是", "什麼叫", "定義", "概念", "簡述", "簡單說", "一句話",
        "what is", "define", "meaning of", "什麼意思",
    )
    if any(k in q for k in naive_keys):
        return "naive"

    # Local：具體實體、算法細節、性質、參數
    local_keys = (
        "算法", "步驟", "流程", "性質", "參數", "複雜度", "時間", "空間",
        "algorithm", "step", "procedure", "property", "complexity", "parameter",
        "如何", "怎麼", "方法", "具體",
    )
    if any(k in q for k in local_keys):
        return "local"

    # 默認 hybrid（比 mix 更輕量，但仍能利用圖譜結構）
    return "hybrid"
