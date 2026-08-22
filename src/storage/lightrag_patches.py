"""LightRAG 关系检索运行期补丁（relation_top_k / related_relation_chunk_number）。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("storage.lightrag_patches")

_PATCHED = False


def _gc_val(global_config: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in global_config and global_config[key] is not None:
        return global_config[key]
    addon = global_config.get("addon_params")
    if isinstance(addon, dict) and key in addon:
        return addon[key]
    return default


async def _parallel_perform_kg_search(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    text_chunks_db,
    query_param,
    chunks_vdb=None,
):
    """并行版 _perform_kg_search：local/global/vector 三路可并行时并发执行。"""
    import lightrag.operate as op

    local_entities: list = []
    local_relations: list = []
    global_entities: list = []
    global_relations: list = []
    vector_chunks: list = []
    chunk_tracking: dict = {}

    kg_chunk_pick_method = text_chunks_db.global_config.get(
        "kg_chunk_pick_method", op.DEFAULT_KG_CHUNK_PICK_METHOD
    )
    actual_embedding_func = text_chunks_db.embedding_func
    query_embedding = None
    ll_embedding = None
    hl_embedding = None

    mode = query_param.mode
    need_ll = mode in ("local", "hybrid", "mix") and bool(ll_keywords)
    need_hl = mode in ("global", "hybrid", "mix") and bool(hl_keywords)

    if actual_embedding_func:
        texts_to_embed: list[str] = []
        text_purposes: list[str] = []

        if query and (kg_chunk_pick_method == "VECTOR" or chunks_vdb):
            texts_to_embed.append(query)
            text_purposes.append("query")
        if need_ll:
            texts_to_embed.append(ll_keywords)
            text_purposes.append("ll")
        if need_hl:
            texts_to_embed.append(hl_keywords)
            text_purposes.append("hl")

        if texts_to_embed:
            try:
                all_embeddings = await actual_embedding_func(
                    texts_to_embed, context="query", _priority=op.DEFAULT_QUERY_PRIORITY
                )
                for i, purpose in enumerate(text_purposes):
                    if purpose == "query":
                        query_embedding = all_embeddings[i]
                    elif purpose == "ll":
                        ll_embedding = all_embeddings[i]
                    elif purpose == "hl":
                        hl_embedding = all_embeddings[i]
            except Exception as e:
                import logging
                logging.getLogger("storage.lightrag_patches").warning(
                    f"Failed to batch pre-compute embeddings: {e}"
                )

    if mode == "local" and len(ll_keywords) > 0:
        local_entities, local_relations = await op._get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
            query_embedding=ll_embedding,
        )
    elif mode == "global" and len(hl_keywords) > 0:
        global_relations, global_entities = await op._get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
            query_embedding=hl_embedding,
        )
    else:
        tasks = []
        if len(ll_keywords) > 0:
            tasks.append(
                (
                    "local",
                    op._get_node_data(
                        ll_keywords,
                        knowledge_graph_inst,
                        entities_vdb,
                        query_param,
                        query_embedding=ll_embedding,
                    ),
                )
            )
        if len(hl_keywords) > 0:
            tasks.append(
                (
                    "global",
                    op._get_edge_data(
                        hl_keywords,
                        knowledge_graph_inst,
                        relationships_vdb,
                        query_param,
                        query_embedding=hl_embedding,
                    ),
                )
            )
        if mode == "mix" and chunks_vdb:
            tasks.append(
                (
                    "vector",
                    op._get_vector_context(
                        query,
                        chunks_vdb,
                        query_param,
                        query_embedding,
                    ),
                )
            )

        if tasks:
            results = await asyncio.gather(*[t for _, t in tasks])
            for (kind, _), result in zip(tasks, results):
                if kind == "local":
                    local_entities, local_relations = result
                elif kind == "global":
                    global_relations, global_entities = result
                else:
                    vector_chunks = result
                    for i, chunk in enumerate(vector_chunks):
                        chunk_id = chunk.get("chunk_id") or chunk.get("id")
                        if chunk_id:
                            chunk_tracking[chunk_id] = {
                                "source": "C",
                                "frequency": 1,
                                "order": i + 1,
                            }

    final_entities = []
    seen_entities = set()
    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        if i < len(local_entities):
            entity = local_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)
        if i < len(global_entities):
            entity = global_entities[i]
            entity_name = entity.get("entity_name")
            if entity_name and entity_name not in seen_entities:
                final_entities.append(entity)
                seen_entities.add(entity_name)

    final_relations = []
    seen_relations = set()
    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        if i < len(local_relations):
            relation = local_relations[i]
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )
            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)
        if i < len(global_relations):
            relation = global_relations[i]
            if "src_tgt" in relation:
                rel_key = tuple(sorted(relation["src_tgt"]))
            else:
                rel_key = tuple(
                    sorted([relation.get("src_id"), relation.get("tgt_id")])
                )
            if rel_key not in seen_relations:
                final_relations.append(relation)
                seen_relations.add(rel_key)

    return {
        "final_entities": final_entities,
        "final_relations": final_relations,
        "vector_chunks": vector_chunks,
        "chunk_tracking": chunk_tracking,
        "query_embedding": query_embedding,
    }


def apply_lightrag_relation_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import lightrag.operate as op

    _orig_edge = op._get_edge_data

    # 兼容不同 LightRAG 版本：1.4.x 使用 _find_related_text_unit_from_relationships，1.5.x 使用 _find_related_text_unit_from_relations
    _rel_chunks_attr = None
    for attr in ("_find_related_text_unit_from_relations", "_find_related_text_unit_from_relationships"):
        if hasattr(op, attr):
            _rel_chunks_attr = attr
            break

    if _rel_chunks_attr is None:
        logger.warning("LightRAG operate 模組中找不到關係 chunk 檢索函數，跳過關係檢索補丁")
        _orig_rel_chunks = None
    else:
        _orig_rel_chunks = getattr(op, _rel_chunks_attr)

    import inspect
    _edge_sig = inspect.signature(_orig_edge)
    _edge_has_query_embedding = "query_embedding" in _edge_sig.parameters

    async def _get_edge_data_patched(
        keywords,
        knowledge_graph_inst,
        relationships_vdb,
        query_param,
        query_embedding=None,
    ):
        gc = relationships_vdb.global_config
        rtk = int(_gc_val(gc, "relation_top_k", 0) or 0)
        if rtk > 0:
            query_param = replace(query_param, top_k=rtk)
        if _edge_has_query_embedding:
            return await _orig_edge(
                keywords,
                knowledge_graph_inst,
                relationships_vdb,
                query_param,
                query_embedding=query_embedding,
            )
        return await _orig_edge(
            keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

    async def _find_related_text_unit_from_relations_patched(*args, **kwargs):
        text_chunks_db = args[2] if len(args) > 2 else kwargs.get("text_chunks_db")
        gc = getattr(text_chunks_db, "global_config", {}) or {}
        rel_n = int(_gc_val(gc, "related_relation_chunk_number", 0) or 0)
        if rel_n > 0 and isinstance(gc, dict):
            patched_gc = dict(gc)
            patched_gc["related_chunk_number"] = rel_n
            text_chunks_db.global_config = patched_gc
        try:
            return await _orig_rel_chunks(*args, **kwargs)
        finally:
            if rel_n > 0 and isinstance(gc, dict):
                text_chunks_db.global_config = gc

    op._get_edge_data = _get_edge_data_patched
    if _orig_rel_chunks is not None:
        setattr(op, _rel_chunks_attr, _find_related_text_unit_from_relations_patched)

    # Patch LightRAG 1.4.0 bug: pipeline_status missing 'history_messages' key
    import lightrag.kg.shared_storage as _shared_storage
    import lightrag.lightrag as _lr_module

    _orig_get_namespace_data = _shared_storage.get_namespace_data

    async def _patched_get_namespace_data(namespace: str, *args, **kwargs):
        result = await _orig_get_namespace_data(namespace, *args, **kwargs)
        if namespace == "pipeline_status":
            result.setdefault("history_messages", [])
        return result

    _shared_storage.get_namespace_data = _patched_get_namespace_data
    _lr_module.get_namespace_data = _patched_get_namespace_data

    if hasattr(op, "_perform_kg_search"):
        op._perform_kg_search = _parallel_perform_kg_search
        logger.info("LightRAG 检索补丁：图/向量并行检索已启用")

    _PATCHED = True
    logger.info("LightRAG 关系检索补丁已启用（relation_top_k / related_relation_chunk_number）")
