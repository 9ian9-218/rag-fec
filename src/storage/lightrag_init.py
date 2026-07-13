"""LightRAG 實例建立、環境注入與生命週期管理。"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.settings import Settings, apply_settings_to_environ, get_settings
from src.utils.logger import get_logger

logger = get_logger("storage.lightrag_init")

_CREATE_EXTRA_KEYS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "logit_bias",
    }
)


def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _keyword_extraction_via_json_object() -> bool:
    """DeepSeek 等不支援 ``completions.parse`` + ``GPTKeywordExtractionFormat``，改走 ``json_object`` 或純文本 JSON。"""
    if _env_truthy("LIGHTRAG_KEYWORD_USE_OPENAI_PARSE"):
        return False
    if _env_truthy("LIGHTRAG_KEYWORD_JSON_OBJECT"):
        return True
    base = (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").lower()
    return "deepseek" in base


def _effective_openai_http_timeout_seconds(**kwargs: Any) -> int:
    t = kwargs.get("timeout")
    if t is not None:
        try:
            return max(1, int(t))
        except (TypeError, ValueError):
            pass
    for env in ("OPENAI_TIMEOUT", "LLM_TIMEOUT"):
        v = (os.getenv(env) or "").strip()
        if v:
            try:
                return max(1, int(v))
            except ValueError:
                pass
    return 180


async def _openai_keyword_extraction_compat(
    *,
    model: str,
    prompt: str,
    system_prompt: str | None,
    history_messages: list[dict[str, Any]],
    base_url: str,
    api_key: str | None,
    http_timeout: int,
    client_configs: dict[str, Any] | None,
    safe_kw: dict[str, Any],
) -> str:
    from lightrag.llm.openai import create_openai_async_client

    bu = (base_url or "").strip().rstrip("/") or "https://api.openai.com/v1"
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for rf in ({"type": "json_object"}, None):
        try:
            client = create_openai_async_client(
                api_key=api_key,
                base_url=bu,
                timeout=http_timeout,
                client_configs=client_configs,
            )
            create_kw: dict[str, Any] = {**safe_kw, "timeout": http_timeout}
            if rf is not None:
                create_kw["response_format"] = rf
            async with client:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **create_kw,
                )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if rf is None:
                break
            logger.warning(
                "關鍵詞抽取：response_format=json_object 不可用或失敗，改為不帶 response_format 重試: %s",
                e,
            )
    assert last_err is not None
    raise last_err


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
    from src.retrieval.keyword_fallback import fec_keyword_fallback

    hl, ll = fec_keyword_fallback(prompt)
    return _keyword_json_from_lists(prompt, hl, ll, settings)


def _build_llm_func(settings: Settings):
    from lightrag.llm.openai import openai_complete

    async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        if kwargs.get("keyword_extraction"):
            hashing_kv = kwargs.get("hashing_kv")
            if hashing_kv is None:
                raise RuntimeError("LightRAG LLM 呼叫缺少 hashing_kv，無法取得 llm_model_name")
            model = hashing_kv.global_config["llm_model_name"]
            base_raw = (
                os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or (settings.openai_base_url or settings.llm.base_url or "")
            ).strip()
            api_key = (os.getenv("OPENAI_API_KEY") or settings.openai_api_key or settings.llm.api_key or "").strip()
            api_key = api_key or None
            http_timeout = _effective_openai_http_timeout_seconds(**kwargs)
            client_cfgs = kwargs.get("openai_client_configs")
            safe_kw = {k: v for k, v in kwargs.items() if k in _CREATE_EXTRA_KEYS}

            async def _call_keyword_llm() -> str:
                if _keyword_extraction_via_json_object():
                    return await _openai_keyword_extraction_compat(
                        model=model,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        history_messages=list(history_messages or []),
                        base_url=base_raw,
                        api_key=api_key,
                        http_timeout=http_timeout,
                        client_configs=client_cfgs if isinstance(client_cfgs, dict) else None,
                        safe_kw=safe_kw,
                    )
                return await openai_complete(
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    **kwargs,
                )

            if settings.lightrag.keyword_fallback_enabled:
                try:
                    from lightrag.utils import remove_think_tags
                    import json_repair

                    raw = await _call_keyword_llm()
                    parsed = json_repair.loads(remove_think_tags(raw))
                    hl = (parsed or {}).get("high_level_keywords") or []
                    ll = (parsed or {}).get("low_level_keywords") or []
                    if hl or ll:
                        return _keyword_json_from_lists(prompt, hl, ll, settings)
                    logger.warning("關鍵詞抽取為空，使用 FEC 啟發式回退")
                except Exception as e:
                    logger.warning("關鍵詞抽取失敗，使用 FEC 啟發式回退: %s", e)
                return _keyword_fallback_json(prompt, settings)

            if _keyword_extraction_via_json_object():
                return await _openai_keyword_extraction_compat(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=list(history_messages or []),
                    base_url=base_raw,
                    api_key=api_key,
                    http_timeout=http_timeout,
                    client_configs=client_cfgs if isinstance(client_cfgs, dict) else None,
                    safe_kw=safe_kw,
                )

        return await openai_complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            **kwargs,
        )

    return _llm


def build_lightrag(settings: Settings | None = None) -> "LightRAG":
    """依設定建立 ``LightRAG``（尚未 ``initialize_storages``）。"""
    s = settings or get_settings()
    apply_settings_to_environ(s)
    from src.storage.lightrag_patches import apply_lightrag_relation_patches
    from src.storage.pymilvus_timeout_patch import ensure_pymilvus_connection_timeout

    ensure_pymilvus_connection_timeout()
    apply_lightrag_relation_patches()
    from lightrag import LightRAG

    root = _project_root()
    working_dir = str(root / s.paths.lightrag_working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # Embedding: 仅支持线上 API
    from src.storage.remote_embedding import build_remote_embedding_func

    if not s.embedding.api_enabled:
        raise RuntimeError(
            "Embedding API 未启用。本项目现已移除本地模型支持，必须使用线上 API。\n"
            "请在 .env 中设置：\n"
            "  EMBEDDING_API_ENABLED=true\n"
            "  EMBEDDING_API_KEY=your-api-key\n"
            "  EMBEDDING_API_BASE_URL=https://api.siliconflow.cn\n"
            "  EMBEDDING_API_MODEL_NAME=BAAI/bge-m3\n"
            "  EMBEDDING_DIMENSION=1024"
        )

    embedding_func = build_remote_embedding_func(s)
    logger.info(
        "Embedding backend: remote API, model=%s, base_url=%s, dim=%d",
        s.embedding.api_model_name,
        s.embedding.api_base_url,
        s.embedding.dimension,
    )
    llm_model_func = _build_llm_func(s)

    # Rerank: 仅支持线上 API
    rerank_model_func = None

    if not s.models.rerank_api_enabled:
        logger.warning(
            "Rerank API 未启用。本项目现已移除本地模型支持。\n"
            "如需启用 rerank，请在 .env 中设置：\n"
            "  MODELS_RERANK_API_ENABLED=true\n"
            "  MODELS_RERANK_API_KEY=your-api-key\n"
            "  MODELS_RERANK_API_BASE_URL=https://api.siliconflow.cn\n"
            "  MODELS_RERANK_API_MODEL_NAME=BAAI/bge-reranker-v2-m3"
        )
    else:
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
            logger.warning("Remote rerank API 初始化失敗，将跳过 rerank")

    lr = s.lightrag
    max_graph_nodes = min(1000, max(64, int(lr.max_graph_nodes)))
    chunk_top_k = lr.chunk_top_k if lr.chunk_top_k is not None else s.retrieval.top_k

    entity_types = s.fec.resolve_entity_types()
    addon_params = {
        "language": s.fec.summary_language,
        "entity_types": entity_types,
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
        "LightRAG FEC addon_params: language=%s entity_types=%d kinds gleaning=%s",
        addon_params["language"],
        len(entity_types),
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
