"""第三方線上 Rerank API 封裝（OpenAI 相容介面）。

支援提供 reranking/score API 的第三方服務（如 Cohere、SiliconFlow rerank 等）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import numpy as np

from config.settings import Settings, get_settings
from src.evaluation.online_monitor import set_rerank_stats
from src.utils.logger import get_logger

logger = get_logger("storage.remote_rerank")


def minmax_normalize_scores(scores: list[float]) -> list[float]:
    """將分數壓到 [0,1]，便於與 ``MIN_RERANK_SCORE`` / ``rerank_min_score`` 配合。"""
    if not scores:
        return []
    a = np.asarray(scores, dtype=np.float64)
    if a.size == 1:
        return [1.0]
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return [1.0] * int(a.size)
    return ((a - lo) / (hi - lo)).tolist()


def build_remote_rerank_model_func(settings: Settings | None = None) -> Callable[..., Any] | None:
    """建立基於第三方線上 API 的 rerank 函數，供 LightRAG 使用。

    返回的函數簽名：``async def rerank_model_func(query, documents, top_n) -> list[dict]``
    """
    s = settings or get_settings()

    if not s.models.rerank_api_enabled:
        return None

    api_key = (s.models.rerank_api_key or "").strip()
    if not api_key:
        logger.warning("Rerank API 已啟用但 RERANK_API_KEY 未設定，降級為本地 rerank")
        return None

    api_base_url = (s.models.rerank_api_base_url or "").strip().rstrip("/")
    if not api_base_url:
        logger.warning("Rerank API 已啟用但 RERANK_API_BASE_URL 未設定，降級為本地 rerank")
        return None

    api_model_name = (s.models.rerank_api_model_name or "").strip()
    if not api_model_name:
        logger.warning("Rerank API 已啟用但 RERANK_API_MODEL_NAME 未設定，降級為本地 rerank")
        return None

    api_timeout = int(s.models.rerank_api_timeout)

    logger.info(
        "Remote Rerank API 已啟用: model=%s, base_url=%s, timeout=%ds",
        api_model_name,
        api_base_url,
        api_timeout,
    )

    async def rerank_model_func(
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        q = (query or "").strip()
        if not q:
            return [{"index": i, "relevance_score": 1.0} for i in range(len(documents))]

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("請安裝 openai: pip install openai")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=f"{api_base_url}/v1" if "/v1" not in api_base_url else api_base_url,
            timeout=api_timeout,
        )

        # 注意：不同服务商的 rerank API 格式可能不同
        # 这里使用通用的 OpenAI 相容格式（如 SiliconFlow rerank API）
        # 如果服务商使用不同格式，需要在此适配
        try:
            response = await client.rerank.create(
                model=api_model_name,
                query=q,
                documents=documents,
                top_n=top_n or len(documents),
            )

            # 解析返回结果（不同服务商可能结构不同）
            # OpenAI/SiliconFlow 格式：response.data 包含 list of {index, relevance_score}
            results = []
            for item in response.data:
                results.append({
                    "index": item.index,
                    "relevance_score": float(item.relevance_score),
                })

            # 按分数排序
            results.sort(key=lambda x: x["relevance_score"], reverse=True)

            # 统计
            scores = [r["relevance_score"] for r in results]
            norm = minmax_normalize_scores(scores)
            min_score = float(get_settings().retrieval.rerank_min_score)
            below_min = sum(1 for s in norm if s < min_score)

            set_rerank_stats(
                candidates=len(documents),
                returned=len(results),
                below_min_score=below_min,
            )

            return [{"index": r["index"], "relevance_score": float(norm[i])} for i, r in enumerate(results)]

        except AttributeError:
            # 如果 API 不支持 rerank.create，尝试使用 embeddings 相似度作为替代
            logger.warning(
                "Rerank API 不支持 rerank.create，尝试使用 embedding 相似度作为替代方案"
            )
            return await _fallback_with_embedding_similarity(
                client=client,
                model=api_model_name,
                query=q,
                documents=documents,
                top_n=top_n,
            )
        except Exception as e:
            logger.error("Rerank API 調用失敗: %s", e)
            return []

    return rerank_model_func


async def _fallback_with_embedding_similarity(
    client,
    model: str,
    query: str,
    documents: list[str],
    top_n: int | None,
) -> list[dict[str, Any]]:
    """当 rerank API 不可用时，使用 embedding 余弦相似度作为降级方案。"""
    try:
        # 获取 query 和所有 documents 的 embeddings
        all_texts = [query] + documents
        response = await client.embeddings.create(
            model=model,
            input=all_texts,
            encoding_format="float",
        )

        embeddings = sorted(
            [item.embedding for item in response.data],
            key=lambda x: getattr(x, "index", 0),
        )

        query_emb = np.array(embeddings[0], dtype=np.float32)
        doc_embs = np.array(embeddings[1:], dtype=np.float32)

        # 计算余弦相似度
        query_norm = np.linalg.norm(query_emb)
        doc_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)

        similarities = (doc_embs @ query_emb) / (doc_norms.flatten() * query_norm + 1e-8)
        scores = similarities.tolist()

        # 归一化
        norm = minmax_normalize_scores(scores)

        # 排序
        order = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
        if top_n is not None and int(top_n) > 0:
            order = order[: int(top_n)]

        min_score = float(get_settings().retrieval.rerank_min_score)
        below_min = sum(1 for s in norm if s < min_score)

        set_rerank_stats(
            candidates=len(documents),
            returned=len(order),
            below_min_score=below_min,
        )

        return [{"index": i, "relevance_score": float(norm[i])} for i in order]

    except Exception as e:
        logger.error("Embedding 相似度降级方案失敗: %s", e)
        return []
