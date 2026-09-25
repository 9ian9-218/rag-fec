"""LightRAG 實例建立、環境注入與生命週期管理。"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.settings import Settings, apply_settings_to_environ, get_settings
from src.utils.concurrency import get_semaphore
from src.utils.logger import get_logger
from src.utils.retry import call_with_retry

logger = get_logger("storage.lightrag_init")

def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


if TYPE_CHECKING:
    from lightrag import LightRAG

_lightrag_instance: Any = None
_init_lock = asyncio.Lock()



def _project_root() -> Path:
    return Path(get_settings().paths.project_root).resolve()





def _keyword_json_from_lists(prompt: str, hl: list, ll: list, settings: Settings) -> str:
    import json

    from src.retrieval.relation_keywords import enhance_keywords_for_retrieval

    hl2, ll2 = enhance_keywords_for_retrieval(prompt, list(hl or []), list(ll or []))
    return json.dumps(
        {"high_level_keywords": hl2, "low_level_keywords": ll2},
        ensure_ascii=False,
    )


def _keyword_fallback_json(prompt: str, settings: Settings) -> str:
    from src.retrieval.keyword_fallback import scholarly_keyword_fallback

    hl, ll = scholarly_keyword_fallback(prompt)
    return _keyword_json_from_lists(prompt, hl, ll, settings)


def _is_keyword_extraction_prompt(prompt: str) -> bool:
    """判斷是否為 LightRAG 的關鍵詞抽取提示詞。"""
    p = prompt or ""
    return "high_level_keywords" in p and "low_level_keywords" in p

def _build_llm_func(settings: Settings):
    from lightrag.llm.openai import openai_complete

    from src.utils.concurrency import get_phase, get_semaphore

    async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        # 查询关键词不再调用 LLM，直接使用规则启发式生成（科研文献通用词表），避免 query 改写/意图识别。
        # 關鍵詞抽取不調 LLM：LightRAG 1.5.x 經 keyword 角色調用，不帶 keyword_extraction 參數，
        # 因此再按提示詞特徵（同時要求 high/low level keywords）判斷。
        if kwargs.get("keyword_extraction") or _is_keyword_extraction_prompt(prompt):
            return _keyword_fallback_json(prompt, settings)

        # 查询期与插入期使用独立配额，避免增量更新挤占在线查询。
        phase = get_phase()
        limit = (
            int(settings.llm.max_concurrent_calls)
            if phase == "query"
            else int(settings.llm.max_concurrent_insert_calls)
        )

        async with get_semaphore(f"llm:{phase}", limit):
            return await openai_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                timeout=int(settings.llm.timeout),
                **kwargs,
            )

    return _llm


def build_lightrag(settings: Settings | None = None) -> "LightRAG":
    """依設定建立 ``LightRAG``（尚未 ``initialize_storages``）。"""
    s = settings or get_settings()
    apply_settings_to_environ(s)
    from src.storage.lightrag_patches import (
        apply_lightrag_relation_patches,
        apply_openai_session_header_patch,
    )
    from src.storage.pymilvus_timeout_patch import ensure_pymilvus_connection_timeout

    ensure_pymilvus_connection_timeout()
    apply_lightrag_relation_patches()
    apply_openai_session_header_patch()
    from lightrag import LightRAG

    root = _project_root()
    working_dir = str(root / s.paths.lightrag_working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # Embedding：本地模型（EMBEDDING_BACKEND=local）或第三方线上 API
    if s.embedding.backend == "local":
        from src.storage.local_embedding import build_local_embedding_func

        embedding_func = build_local_embedding_func(s)
        logger.info(
            "Embedding backend: local, path=%s device=%s dtype=%s dim=%d",
            s.embedding.local_model_path,
            s.embedding.local_device,
            s.embedding.local_dtype,
            s.embedding.dimension,
        )
    else:
        from src.storage.remote_embedding import build_remote_embedding_func

        if not s.embedding.api_enabled:
            raise RuntimeError(
                "Embedding 未配置：请设置 EMBEDDING_BACKEND=local 使用本地模型，"
                "或设置 EMBEDDING_API_ENABLED=true 并补齐 EMBEDDING_API_KEY / BASE_URL / MODEL_NAME。"
            )

        embedding_func = build_remote_embedding_func(s)
        logger.info(
            "Embedding backend: remote API, model=%s, base_url=%s, dim=%d",
            s.embedding.api_model_name,
            s.embedding.api_base_url,
            s.embedding.dimension,
        )

    llm_model_func = _build_llm_func(s)

    # Rerank：线上 API（显式启用优先）→ 本地 CrossEncoder 权重 → 显式关闭
    rerank_model_func = None

    if s.models.rerank_api_enabled:
        from src.storage.remote_rerank import build_remote_rerank_model_func

        rerank_model_func = build_remote_rerank_model_func(s)
        if rerank_model_func is not None:
            logger.info(
                "Rerank enabled: remote API, model=%s base_url=%s min_score=%s",
                s.models.rerank_api_model_name,
                s.models.rerank_api_base_url,
                s.retrieval.rerank_min_score,
            )
        else:
            logger.warning(
                "MODELS_RERANK_API_ENABLED=true 但线上 rerank 配置不完整"
                "（需 KEY / BASE_URL / MODEL_NAME），将尝试本地 CrossEncoder"
            )

    if rerank_model_func is None:
        from src.storage.bge_rerank import build_local_rerank_model_func

        rerank_model_func = build_local_rerank_model_func(s)
        if rerank_model_func is not None:
            logger.info(
                "Rerank enabled: local CrossEncoder, batch_size=%d min_score=%s",
                s.models.rerank_batch_size,
                s.retrieval.rerank_min_score,
            )

    if rerank_model_func is None:
        logger.warning(
            "Chunk 精排未启用：线上 rerank 未配置，且本地未找到 CrossEncoder 权重，"
            "查询将显式 enable_rerank=False（不再出现『看似在重排』的状态）。\n"
            "启用本地免费重排：python scripts/download_reranker.py"
        )

    lr = s.lightrag
    max_graph_nodes = min(1000, max(64, int(lr.max_graph_nodes)))
    chunk_top_k = lr.chunk_top_k if lr.chunk_top_k is not None else s.retrieval.top_k

    entity_types = s.domain.resolve_entity_types()
    entity_types_guidance = s.domain.resolve_entity_types_guidance()
    addon_params = {
        "language": s.domain.summary_language,
        # LightRAG 1.5+ 以 entity_types_guidance（字串）決定實體類型；
        # 舊版（<=1.4）讀 entity_types 列表，一併傳入以保持相容。
        "entity_types_guidance": entity_types_guidance,
        "entity_types": [name for name, _ in entity_types],
        "relation_top_k": lr.relation_top_k,
        "related_relation_chunk_number": lr.related_relation_chunk_number,
    }

    import inspect

    _lightrag_sig = inspect.signature(LightRAG.__init__)
    _lightrag_params = set(_lightrag_sig.parameters.keys())

    _rag_kwargs = {
        "working_dir": working_dir,
        "workspace": s.lightrag_workspace or "",
        "llm_model_func": llm_model_func,
        "llm_model_name": s.resolved_llm_model_name(),
        "llm_model_kwargs": {"temperature": s.resolved_llm_temperature()},
        "embedding_func": embedding_func,
        "kv_storage": "JsonKVStorage",
        "vector_storage": "MilvusVectorDBStorage",
        "graph_storage": "Neo4JStorage",
        "doc_status_storage": "JsonDocStatusStorage",
        "chunk_token_size": s.chunk.chunk_size,
        "chunk_overlap_token_size": s.chunk.chunk_overlap,
        "top_k": s.retrieval.top_k,
        "chunk_top_k": chunk_top_k,
        "max_entity_tokens": lr.max_entity_tokens,
        "max_relation_tokens": lr.max_relation_tokens,
        "max_total_tokens": lr.max_total_tokens,
        "related_chunk_number": lr.related_entity_chunk_number,
        "max_graph_nodes": max_graph_nodes,
        "cosine_better_than_threshold": float(lr.cosine_better_than_threshold),
        "entity_extract_max_gleaning": lr.entity_extract_max_gleaning,
        "embedding_batch_num": s.embedding.batch_size,
        "embedding_func_max_async": s.embedding.max_async,
        "addon_params": addon_params,
        "rerank_model_func": rerank_model_func,
    }

    # 可选参数：依 LightRAG 版本動態加入
    if "kg_chunk_pick_method" in _lightrag_params:
        _rag_kwargs["kg_chunk_pick_method"] = lr.kg_chunk_pick_method.strip().upper()
    if "default_embedding_timeout" in _lightrag_params:
        _rag_kwargs["default_embedding_timeout"] = s.embedding.lightrag_embedding_timeout
    if "min_rerank_score" in _lightrag_params:
        _rag_kwargs["min_rerank_score"] = s.retrieval.rerank_min_score

    rag = LightRAG(**_rag_kwargs)
    logger.info(
        "LightRAG runtime: top_k=%s chunk_top_k=%s max_total_tokens=%s "
        "max_entity_tokens=%s max_relation_tokens=%s related_chunk_number=%s "
        "kg_chunk_pick=%s max_graph_nodes=%s min_rerank=%s ref_ctx_chars=%s",
        s.retrieval.top_k,
        chunk_top_k,
        lr.max_total_tokens,
        lr.max_entity_tokens,
        lr.max_relation_tokens,
        lr.related_entity_chunk_number,
        lr.kg_chunk_pick_method,
        max_graph_nodes,
        s.retrieval.rerank_min_score,
        s.multimodal.reference_context_max_chars,
    )
    logger.info(
        "LightRAG schema addon_params: language=%s entity_types=%d kinds guidance_chars=%d gleaning=%s",
        addon_params["language"],
        len(entity_types),
        len(entity_types_guidance),
        lr.entity_extract_max_gleaning,
    )
    return rag


async def get_lightrag() -> Any:
    """取得已初始化儲存後端的單例 ``LightRAG``。"""
    global _lightrag_instance
    async with _init_lock:
        if _lightrag_instance is None:
            _lightrag_instance = build_lightrag()
            await _lightrag_instance.initialize_storages()
            logger.info("LightRAG storages initialized")
        return _lightrag_instance


def get_lightrag_blocking() -> Any:
    """在無事件迴圈環境取得單例（例如腳本）。"""
    from lightrag.utils import always_get_an_event_loop

    loop = always_get_an_event_loop()
    return loop.run_until_complete(get_lightrag())


def reset_lightrag_singleton() -> None:
    """測試或重建索引時清除單例參考。"""
    global _lightrag_instance
    _lightrag_instance = None
    try:
        from src.storage.bge_rerank import reset_rerank_singleton

        reset_rerank_singleton()
    except ImportError:
        pass
