"""测试检索上下文缓存：命中后不再执行 LightRAG aquery_data。"""

from __future__ import annotations

from types import SimpleNamespace

import fakeredis.aioredis

from src.retrieval import retriever as retriever_module
from src.retrieval.retriever import GraphRAGRetriever
from src.storage import redis_cache


async def test_retrieval_cache_skips_second_lightrag_call(monkeypatch) -> None:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_cache, "get_redis_client", lambda *a, **k: fake)

    calls = 0
    bundle = {"status": "success", "data": {"chunks": []}}

    class FakeRag:
        async def aquery_data(self, question, param):
            nonlocal calls
            calls += 1
            return bundle

    async def fake_get_lightrag():
        return FakeRag()

    monkeypatch.setattr(retriever_module, "get_lightrag", fake_get_lightrag)
    async def fake_refine(question, data, settings=None):
        return data

    monkeypatch.setattr(retriever_module, "refine_retrieval_bundle", fake_refine)

    r = GraphRAGRetriever()
    param = SimpleNamespace(chunk_top_k=8)

    first = await r._get_retrieval_bundle("同一个问题", param, "mix", 8)
    second = await r._get_retrieval_bundle("同一个问题", param, "mix", 8)

    assert first == bundle
    assert second == bundle
    assert calls == 1
