"""测试规则路由：不调用 LLM，基于关键词决策。"""

from __future__ import annotations

import json

from src.retrieval.mode_router import resolve_retrieval_mode


async def test_explicit_mode_wins() -> None:
    route = await resolve_retrieval_mode("随便问", "mix")
    assert route.mode == "mix"
    assert route.source == "explicit"


async def test_rule_based_routes_without_llm() -> None:
    # 定义类问题 -> naive
    route = await resolve_retrieval_mode("什么是 Reed-Muller 码？", None)
    assert route.mode == "naive"
    assert route.source == "heuristic"

    # 对比/宏观类问题 -> global
    route = await resolve_retrieval_mode("比较 RPA_RM 和 Chase 列表译码的性能差异", None)
    assert route.mode == "global"


async def test_keyword_extraction_uses_rule_fallback_no_llm() -> None:
    from config.settings import get_settings
    from src.storage.lightrag_init import _build_llm_func

    func = _build_llm_func(get_settings())
    result = await func("User Query: 什么是循环码？", keyword_extraction=True)
    parsed = json.loads(result)
    assert "high_level_keywords" in parsed
    assert "low_level_keywords" in parsed
