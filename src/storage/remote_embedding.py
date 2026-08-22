"""第三方線上 Embedding API 封裝（OpenAI 相容介面）。

支援 SiliconFlow、智譜、OpenAI 等提供 OpenAI 相容 embedding API 的服務商。
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import numpy as np

from config.settings import Settings, get_settings
from src.storage.redis_cache import get_json_cache, set_json_cache
from src.utils.client_cache import get_openai_client
from src.utils.concurrency import get_semaphore
from src.utils.logger import get_logger
from src.utils.retry import call_with_retry

logger = get_logger("storage.remote_embedding")


def build_remote_embedding_func(settings: Settings | None = None):
    """建立基於第三方線上 API 的 embedding 函數，供 LightRAG 使用。

    返回的函數簽名：``async def _embed(texts: list[str]) -> np.ndarray``
    """
    s = settings or get_settings()

    if not s.embedding.api_enabled:
        raise RuntimeError("Embedding API 未啟用，請設定 EMBEDDING_API_ENABLED=true")

    api_key = (s.embedding.api_key or "").strip()
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY 未設定")

    api_base_url = (s.embedding.api_base_url or "").strip().rstrip("/")
    if not api_base_url:
        raise ValueError("EMBEDDING_API_BASE_URL 未設定")

    api_model_name = (s.embedding.api_model_name or "").strip()
    if not api_model_name:
        raise ValueError("EMBEDDING_API_MODEL_NAME 未設定")

    api_timeout = int(s.embedding.api_timeout)
    dimension = int(s.embedding.dimension)

    def _embedding_cache_key(text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"rag:embedding:{digest}"

    logger.info(
        "Remote Embedding API 已啟用: model=%s, base_url=%s, timeout=%ds",
        api_model_name,
        api_base_url,
        api_timeout,
    )

    async def _call_embedding_api(texts: list[str]) -> list[list[float]]:
        """調用 OpenAI 相容 embedding API。"""
        client = get_openai_client(
            api_key=api_key,
            base_url=f"{api_base_url}/v1" if "/v1" not in api_base_url else api_base_url,
            timeout=api_timeout,
        )

        # OpenAI embedding API 通常單次請求最多支援 100 個 texts
        batch_size = min(100, int(s.embedding.batch_size) if hasattr(s.embedding, "batch_size") else 16)
        all_embeddings: list[list[float]] = []
        embedding_sem = get_semaphore("embedding", int(s.embedding.api_max_concurrency))

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                async def _do_embed() -> Any:
                    async with embedding_sem:
                        return await client.embeddings.create(
                            model=api_model_name,
                            input=batch,
                            encoding_format="float",
                        )

                response = await call_with_retry(
                    _do_embed,
                    max_retries=int(s.embedding.max_retries),
                )
                # 按照返回順序組裝（API 保證順序與輸入一致）
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error("Embedding API 調用失敗 (batch %d-%d): %s", i, i + len(batch), e)
                raise

        return all_embeddings

    async def _embed(texts: list[str], **_kwargs: Any) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, dimension)

        try:
            # 先查 Redis 缓存，降低外部 API 调用量。
            keys = [_embedding_cache_key(t) for t in texts]
            cached_vectors: list[list[float] | None] = [None] * len(texts)
            missing_indices: list[int] = []
            for i, key in enumerate(keys):
                cached = await get_json_cache(key)
                if isinstance(cached, list):
                    cached_vectors[i] = [float(x) for x in cached]
                else:
                    missing_indices.append(i)

            if missing_indices:
                missing_texts = [texts[i] for i in missing_indices]
                embeddings = await _call_embedding_api(missing_texts)
                arr_missing = np.array(embeddings, dtype=np.float32)
                if arr_missing.ndim == 1:
                    arr_missing = arr_missing.reshape(1, -1)
                norms = np.linalg.norm(arr_missing, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                arr_missing = arr_missing / norms
                for idx, vec in zip(missing_indices, arr_missing.tolist()):
                    cached_vectors[idx] = vec
                    await set_json_cache(
                        keys[idx],
                        vec,
                        ttl=int(s.redis.cache_ttl_seconds),
                    )

            arr = np.array(cached_vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception as e:
            logger.error("Remote embedding 失敗: %s", e)
            raise

    try:
        from lightrag.utils import EmbeddingFunc
        return EmbeddingFunc(
            embedding_dim=dimension,
            max_token_size=8192,
            func=_embed,
            model_name=api_model_name,
        )
    except Exception:
        return _embed
