"""第三方線上 Embedding API 封裝（OpenAI 相容介面）。

支援 SiliconFlow、智譜、OpenAI 等提供 OpenAI 相容 embedding API 的服務商。
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from config.settings import Settings, get_settings
from src.utils.logger import get_logger

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

    logger.info(
        "Remote Embedding API 已啟用: model=%s, base_url=%s, timeout=%ds",
        api_model_name,
        api_base_url,
        api_timeout,
    )

    async def _call_embedding_api(texts: list[str]) -> list[list[float]]:
        """調用 OpenAI 相容 embedding API。"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("請安裝 openai: pip install openai")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=f"{api_base_url}/v1" if "/v1" not in api_base_url else api_base_url,
            timeout=api_timeout,
        )

        # OpenAI embedding API 通常單次請求最多支援 100 個 texts
        batch_size = min(100, int(s.embedding.batch_size) if hasattr(s.embedding, "batch_size") else 16)
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                response = await client.embeddings.create(
                    model=api_model_name,
                    input=batch,
                    encoding_format="float",
                )
                # 按照返回順序組裝（API 保證順序與輸入一致）
                batch_embeddings = sorted(
                    [item.embedding for item in response.data],
                    key=lambda x: getattr(x, "index", 0),
                )
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error("Embedding API 調用失敗 (batch %d-%d): %s", i, i + len(batch), e)
                raise

        return all_embeddings

    async def _embed(texts: list[str], **_kwargs: Any) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, dimension)

        try:
            embeddings = await _call_embedding_api(texts)
            arr = np.array(embeddings, dtype=np.float32)

            # 確保維度正確
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            # 正規化（可選，依模型需求）
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # 避免除零
            arr = arr / norms

            return arr
        except Exception as e:
            logger.error("Remote embedding 失敗: %s", e)
            raise

    return _embed
