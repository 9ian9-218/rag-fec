"""本地 BGE CrossEncoder 封裝（支持 sentence-transformers）。

提供關係重排所需的 CrossEncoder 單例管理、min-max 分數歸一化，以及可直接交給
LightRAG 作為 ``rerank_model_func`` 的本地重排函數。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from config.model_paths import resolve_local_reranker_dir
from config.settings import Settings, get_settings
from src.evaluation.online_monitor import set_rerank_stats
from src.storage.remote_rerank import minmax_normalize_scores
from src.utils.logger import get_logger

logger = get_logger("storage.bge_rerank")

_cross_encoder_instance: Any | None = None


def _get_cross_encoder(load_path: str | Path | None = None) -> Any:
    """獲取或創建 CrossEncoder 實例（單例緩存）。"""
    global _cross_encoder_instance

    if _cross_encoder_instance is not None:
        return _cross_encoder_instance

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise ImportError(
            "本地 CrossEncoder 需要 sentence-transformers，請安裝："
            "pip install sentence-transformers>=2.5.0"
        ) from e

    load_path = (load_path or "").strip() if load_path else ""
    if not load_path:
        load_path = "BAAI/bge-reranker-v2-m3"

    logger.info("加載 CrossEncoder: %s", load_path)
    _cross_encoder_instance = CrossEncoder(load_path)
    return _cross_encoder_instance


def reset_rerank_singleton() -> None:
    """清除 CrossEncoder 單例緩存。"""
    global _cross_encoder_instance
    _cross_encoder_instance = None


def _score_pairs(
    question: str,
    documents: list[str],
    *,
    load_path: str,
    batch_size: int,
) -> list[float]:
    """同步計算 query-document 相關性分數（供 ``asyncio.to_thread`` 調用）。"""
    model = _get_cross_encoder(load_path)
    pairs = [[question, d] for d in documents]
    raw = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(x) for x in raw]


def build_local_rerank_model_func(
    settings: Settings | None = None,
) -> Callable[..., Any] | None:
    """建立本地 CrossEncoder 的 rerank 函數，供 LightRAG 使用。

    簽名與線上版一致：``async def rerank_model_func(*, query, documents, top_n) -> list[dict]``。
    本地無權重時返回 ``None`` —— 不在請求路徑中觸發 Hub 下載。
    """
    s = settings or get_settings()
    local_dir = resolve_local_reranker_dir(s)
    if local_dir is None:
        return None
    load_path = str(local_dir)
    batch_size = int(s.models.rerank_batch_size)
    min_score = float(s.retrieval.rerank_min_score)

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
            raw = await asyncio.to_thread(
                _score_pairs, q, documents, load_path=load_path, batch_size=batch_size
            )
        except Exception as e:
            # 重排失敗不得影響檢索結果：保持原序返回，並保持「候選數」可見
            logger.warning("本地 CrossEncoder 重排失敗，保持原始順序: %s", e)
            set_rerank_stats(candidates=len(documents), returned=len(documents))
            return [{"index": i, "relevance_score": 1.0} for i in range(len(documents))]

        norm = minmax_normalize_scores(raw)
        order = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
        if top_n is not None and int(top_n) > 0:
            order = order[: int(top_n)]

        below_min = sum(1 for i in order if norm[i] < min_score)
        set_rerank_stats(
            candidates=len(documents),
            returned=len(order),
            below_min_score=below_min,
        )
        logger.debug(
            "本地 CrossEncoder 重排: candidates=%d returned=%d below_min=%d batch=%d",
            len(documents),
            len(order),
            below_min,
            batch_size,
        )
        return [{"index": i, "relevance_score": float(norm[i])} for i in order]

    return rerank_model_func
