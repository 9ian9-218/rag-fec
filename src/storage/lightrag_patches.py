"""LightRAG 关系检索运行期补丁（relation_top_k / related_relation_chunk_number）。"""

from __future__ import annotations

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

    _PATCHED = True
    logger.info("LightRAG 关系检索补丁已启用（relation_top_k / related_relation_chunk_number）")
