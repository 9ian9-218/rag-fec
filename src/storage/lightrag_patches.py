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
    _orig_rel_chunks = op._find_related_text_unit_from_relations

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
        return await _orig_edge(
            keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
            query_embedding=query_embedding,
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
    op._find_related_text_unit_from_relations = _find_related_text_unit_from_relations_patched
    _PATCHED = True
    logger.info("LightRAG 关系检索补丁已启用（relation_top_k / related_relation_chunk_number）")
