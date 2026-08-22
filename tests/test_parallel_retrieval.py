"""测试并行检索 patch：图查询与向量查询并行，多路召回合并。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from src.storage.lightrag_patches import _parallel_perform_kg_search


async def _fake_embeddings(texts, **kwargs):
    return [[1.0] for _ in texts]


async def test_parallel_search_merges_local_global_vector(monkeypatch) -> None:
    import lightrag.operate as op

    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def track_start():
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        async with lock:
            active -= 1

    async def fake_node(*args, **kwargs):
        await track_start()
        return ([{"entity_name": "e1"}], [])

    async def fake_edge(*args, **kwargs):
        await track_start()
        return ([{"src_id": "a", "tgt_id": "b"}], [])

    async def fake_vector(*args, **kwargs):
        await track_start()
        return [{"chunk_id": "c1"}]

    monkeypatch.setattr(op, "_get_node_data", fake_node)
    monkeypatch.setattr(op, "_get_edge_data", fake_edge)
    monkeypatch.setattr(op, "_get_vector_context", fake_vector)

    text_chunks_db = SimpleNamespace(
        global_config={"kg_chunk_pick_method": "VECTOR"},
        embedding_func=lambda texts, **kwargs: _fake_embeddings(texts),
    )
    query_param = SimpleNamespace(mode="mix")
    result = await _parallel_perform_kg_search(
        query="q",
        ll_keywords="ll",
        hl_keywords="hl",
        knowledge_graph_inst=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks_db=text_chunks_db,
        query_param=query_param,
        chunks_vdb=object(),
    )

    assert result["final_entities"] == [{"entity_name": "e1"}]
    assert result["final_relations"] == [{"src_id": "a", "tgt_id": "b"}]
    assert result["vector_chunks"] == [{"chunk_id": "c1"}]
    assert max_active >= 2


def test_parallel_patch_is_installed() -> None:
    import lightrag.operate as op

    from src.storage.lightrag_patches import (
        _PATCHED,
        _parallel_perform_kg_search,
        apply_lightrag_relation_patches,
    )

    if not _PATCHED:
        apply_lightrag_relation_patches()
    assert op._perform_kg_search is _parallel_perform_kg_search
