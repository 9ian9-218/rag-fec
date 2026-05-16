from __future__ import annotations

import pytest

from src.retrieval.mode_router import (
    ModeRouteResult,
    normalize_mode,
    parse_mode_router_response,
    resolve_retrieval_mode,
)


def test_parse_mode_router_json() -> None:
    raw = '{"difficulty":"medium","complexity":"high","context_richness":"high","mode":"hybrid","reason":"多跳"}'
    d = parse_mode_router_response(raw)
    assert d["mode"] == "hybrid"
    assert normalize_mode(d["mode"]) == "hybrid"


def test_normalize_native_typo() -> None:
    assert normalize_mode("native") == "naive"


@pytest.mark.asyncio
async def test_resolve_explicit_mode() -> None:
    from config.settings import get_settings

    r = await resolve_retrieval_mode("任意问题", "local", settings=get_settings(), use_llm_router=False)
    assert r.mode == "local"
    assert r.source == "explicit"


@pytest.mark.asyncio
async def test_resolve_llm_router_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    from config.settings import get_settings

    async def _fake(_q: str, **_: object) -> ModeRouteResult:
        return ModeRouteResult(
            mode="global",
            difficulty="hard",
            complexity="high",
            context_richness="high",
            reason="需要全局关系",
            source="llm",
        )

    monkeypatch.setattr("src.retrieval.mode_router.route_mode_with_llm", _fake)
    s = get_settings().model_copy(
        update={"retrieval": get_settings().retrieval.model_copy(update={"llm_mode_router_enabled": True})}
    )
    r = await resolve_retrieval_mode("请总结全书", None, settings=s, use_llm_router=True)
    assert r.mode == "global"
    assert r.source == "llm"
