"""检索引擎异常降级：外部 embedding/LLM API 故障时返回空上下文而非 500。"""

from __future__ import annotations

import pytest

from src.retrieval import retriever as r_mod
from src.retrieval.mode_config import build_query_param


class _FakeRagBroken:
    async def aquery_data(self, question, param):
        raise RuntimeError("embedding 402: balance insufficient")


@pytest.mark.asyncio
async def test_retrieval_exception_degrades_to_empty_bundle(monkeypatch) -> None:
    """aquery_data 抛异常（如 embedding 402）时必须降级为空 bundle，不得向上抛 500。"""
    calls = {"set": 0}

    async def fake_get(key):
        return None

    async def fake_set(key, bundle, ttl=300):
        calls["set"] += 1
        return True

    async def fake_lightrag():
        return _FakeRagBroken()

    async def fake_refine(q, d, settings=None):
        return d

    monkeypatch.setattr(r_mod, "get_retrieval_cache", fake_get)
    monkeypatch.setattr(r_mod, "set_retrieval_cache", fake_set)
    monkeypatch.setattr(r_mod, "get_lightrag", fake_lightrag)
    monkeypatch.setattr(r_mod, "refine_retrieval_bundle", fake_refine)

    rv = r_mod.GraphRAGRetriever()
    param = build_query_param("mix", top_k=8)
    bundle = await rv._get_retrieval_bundle("外部 API 故障时的问题", param, "mix", 8)

    assert isinstance(bundle, dict)
    assert bundle.get("status") == "failure"
    # 空结果不得写缓存（T04 语义）
    assert calls["set"] == 0