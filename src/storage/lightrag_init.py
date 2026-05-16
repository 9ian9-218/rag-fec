"""LightRAG 實例建立、環境注入與生命週期管理。"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.model_paths import resolve_embedding_model_load_path, resolve_reranker_model_load_path
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
_st_model_lock = threading.Lock()
_st_models: dict[str, Any] = {}


def _get_sentence_transformer(model_key: str) -> Any:
    """程序內單例載入；``model_key`` 為實際載入路徑或 Hub id。"""
    with _st_model_lock:
        if model_key not in _st_models:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer model: %s", model_key)
            _st_models[model_key] = SentenceTransformer(model_key, trust_remote_code=True)
        return _st_models[model_key]


def _warm_sentence_transformer(settings: Settings) -> None:
    load_path = resolve_embedding_model_load_path(settings)
    _get_sentence_transformer(load_path)


def _project_root() -> Path:
    return Path(get_settings().paths.project_root).resolve()


def _build_embedding_func(settings: Settings):
    from lightrag.utils import wrap_embedding_func_with_attrs

    import numpy as np

    s = settings
    dim = s.embedding.dimension
    model_id = s.embedding.model_name
    load_path = resolve_embedding_model_load_path(s)

    def _encode_sync(texts: list[str]) -> np.ndarray:
        m = _get_sentence_transformer(load_path)
        arr = m.encode(
            texts,
            normalize_embeddings=True,
            batch_size=s.embedding.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)

    @wrap_embedding_func_with_attrs(
        embedding_dim=dim,
        max_token_size=8192,
        model_name=model_id,
        send_dimensions=False,
    )
    async def _embed(texts: list[str], **_kwargs: Any) -> Any:
        return await asyncio.to_thread(_encode_sync, texts)

    return _embed


def _build_llm_func(settings: Settings):
    from lightrag.llm.openai import openai_complete

    async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        if kwargs.get("keyword_extraction") and _keyword_extraction_via_json_object():
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
    from src.storage.pymilvus_timeout_patch import ensure_pymilvus_connection_timeout

    ensure_pymilvus_connection_timeout()
    from lightrag import LightRAG

    root = _project_root()
    working_dir = str(root / s.paths.lightrag_working_dir)
    os.makedirs(working_dir, exist_ok=True)

    _warm_sentence_transformer(s)
    embedding_func = _build_embedding_func(s)
    logger.info(
        "Embedding backend: sentence-transformers, model=%s, load_path=%s, dim=%d",
        s.embedding.model_name,
        resolve_embedding_model_load_path(s),
        s.embedding.dimension,
    )
    llm_model_func = _build_llm_func(s)

    rerank_model_func = None
    if s.models.rerank_enabled and s.rerank_runtime_available():
        from src.storage.bge_rerank import build_bge_rerank_model_func

        rerank_model_func = build_bge_rerank_model_func(s)
        logger.info(
            "Rerank enabled: model=%s load_path=%s min_score=%s",
            s.models.rerank_model_name,
            resolve_reranker_model_load_path(s),
            s.retrieval.rerank_min_score,
        )
    else:
        logger.info(
            "Rerank skipped (rerank_enabled=%s runtime_available=%s)",
            s.models.rerank_enabled,
            s.rerank_runtime_available(),
        )

    max_graph_nodes = min(1000, max(64, s.retrieval.max_hop * 48))

    entity_types = s.fec.resolve_entity_types()
    addon_params = {
        "language": s.fec.summary_language,
        "entity_types": entity_types,
    }

    rag = LightRAG(
        working_dir=working_dir,
        workspace=s.lightrag_workspace or "",
        llm_model_func=llm_model_func,
        llm_model_name=s.resolved_llm_model_name(),
        llm_model_kwargs={"temperature": s.resolved_llm_temperature()},
        embedding_func=embedding_func,
        kv_storage="JsonKVStorage",
        vector_storage="MilvusVectorDBStorage",
        graph_storage="Neo4JStorage",
        doc_status_storage="JsonDocStatusStorage",
        chunk_token_size=s.chunk.chunk_size,
        chunk_overlap_token_size=s.chunk.chunk_overlap,
        top_k=s.retrieval.top_k,
        chunk_top_k=s.retrieval.top_k,
        max_graph_nodes=max_graph_nodes,
        cosine_better_than_threshold=0.2,
        embedding_batch_num=s.embedding.batch_size,
        default_embedding_timeout=s.embedding.lightrag_embedding_timeout,
        embedding_func_max_async=s.embedding.max_async,
        addon_params=addon_params,
        rerank_model_func=rerank_model_func,
        min_rerank_score=s.retrieval.rerank_min_score,
    )
    logger.info(
        "LightRAG FEC addon_params: language=%s entity_types=%d kinds",
        addon_params["language"],
        len(entity_types),
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
