"""测试 Rerank 结果缓存：相同 query+documents 第二次不调用远程 API。"""

from __future__ import annotations

from types import SimpleNamespace

import fakeredis.aioredis

from config.settings import Settings
from src.storage import redis_cache
from src.storage.remote_rerank import build_remote_rerank_model_func


async def test_rerank_cache_skips_second_remote_call(monkeypatch) -> None:
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_cache, "get_redis_client", lambda *a, **k: fake)

    calls = 0

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.1},
                ]
            }

    class FakeHttp:
        async def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return FakeResp()

    monkeypatch.setattr(
        "src.storage.remote_rerank.get_httpx_client",
        lambda *a, **k: FakeHttp(),
    )

    # 显式给出 rerank 配置，避免依赖开发者本机 .env 的开关
    settings = Settings()
    settings.models.rerank_api_enabled = True
    settings.models.rerank_api_key = "test-key"
    settings.models.rerank_api_base_url = "https://example.invalid"
    settings.models.rerank_api_model_name = "test-reranker"

    rerank = build_remote_rerank_model_func(settings)
    assert rerank is not None

    docs = ["doc1", "doc2"]
    r1 = await rerank(query="q", documents=docs, top_n=2)
    r2 = await rerank(query="q", documents=docs, top_n=2)

    assert len(r1) == 2
    assert r1 == r2
    assert calls == 1
