"""T04 空检索结果不写入缓存：空 bundle 必须重新走真实检索。"""

from __future__ import annotations

import pytest

from src.retrieval import retriever as r_mod
from src.retrieval.mode_config import build_query_param


class _FakeRagEmpty:
    async def aquery_data(self, question, param):
        return {"status": "success", "data": {"entities": [], "relationships": [], "chunks": []}}


class _FakeRagWithChunks:
    async def aquery_data(self, question, param):
        return {
            "status": "success",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [{"chunk_id": "c1", "content": "有内容的 chunk"}],
            },
        }


@pytest.mark.asyncio
async def test_empty_bundle_is_not_cached(monkeypatch) -> None:
    """空检索结果（实体/关系/引文全空）不得写入检索缓存。"""
    calls = {"get": 0, "set": 0}

    async def fake_get(key):
        calls["get"] += 1
        return None

    async def fake_set(key, bundle, ttl=300):
        calls["set"] += 1
        return True

    async def fake_lightrag():
        return _FakeRagEmpty()

    monkeypatch.setattr(r_mod, "get_retrieval_cache", fake_get)
    monkeypatch.setattr(r_mod, "set_retrieval_cache", fake_set)
    monkeypatch.setattr(r_mod, "get_lightrag", fake_lightrag)

    rv = r_mod.GraphRAGRetriever()
    param = build_query_param("mix", top_k=8)
    bundle = await rv._get_retrieval_bundle("一个查不到的问题", param, "mix", 8)

    assert bundle is not None
    assert calls["set"] == 0
    assert calls["get"] == 1


@pytest.mark.asyncio
async def test_nonempty_bundle_is_cached(monkeypatch) -> None:
    """有引文内容的检索结果必须照常缓存。"""
    calls = {"set": 0}

    async def fake_get(key):
        return None

    async def fake_set(key, bundle, ttl=300):
        calls["set"] += 1
        return True

    async def fake_lightrag():
        return _FakeRagWithChunks()

    monkeypatch.setattr(r_mod, "get_retrieval_cache", fake_get)
    monkeypatch.setattr(r_mod, "set_retrieval_cache", fake_set)
    monkeypatch.setattr(r_mod, "get_lightrag", fake_lightrag)

    rv = r_mod.GraphRAGRetriever()
    param = build_query_param("mix", top_k=8)
    bundle = await rv._get_retrieval_bundle("什么是 RS 码", param, "mix", 8)

    assert calls["set"] == 1