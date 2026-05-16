"""本機 BGE CrossEncoder rerank，供 LightRAG ``rerank_model_func`` 使用。"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable

import numpy as np

from config.model_paths import resolve_reranker_model_load_path
from config.settings import Settings
from src.utils.logger import get_logger

logger = get_logger("storage.bge_rerank")

_rerank_lock = threading.Lock()
_rerank_models: dict[str, Any] = {}


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


def _get_cross_encoder(model_key: str) -> Any:
    from sentence_transformers import CrossEncoder

    with _rerank_lock:
        if model_key not in _rerank_models:
            logger.info("Loading CrossEncoder reranker: %s", model_key)
            _rerank_models[model_key] = CrossEncoder(
                model_key,
                trust_remote_code=True,
            )
        return _rerank_models[model_key]


def build_bge_rerank_model_func(settings: Settings | None = None) -> Callable[..., Any] | None:
    """
    建立 LightRAG 所需的異步 rerank 函數：``(query, documents, top_n) -> [{index, relevance_score}]``。
    """
    s = settings or get_settings()
    if not s.models.rerank_enabled or not s.rerank_runtime_available():
        return None

    load_path = resolve_reranker_model_load_path(s)
    batch_size = int(s.models.rerank_batch_size)

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

        def _predict() -> list[float]:
            model = _get_cross_encoder(load_path)
            pairs = [[q, d or ""] for d in documents]
            raw = model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            return [float(x) for x in arr.tolist()]

        try:
            scores = await asyncio.to_thread(_predict)
        except Exception as e:
            logger.error("Rerank predict failed: %s", e)
            return []

        if len(scores) != len(documents):
            logger.warning(
                "Rerank score length mismatch: scores=%d docs=%d",
                len(scores),
                len(documents),
            )
            return []

        norm = minmax_normalize_scores(scores)
        order = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
        if top_n is not None and int(top_n) > 0:
            order = order[: int(top_n)]

        return [{"index": i, "relevance_score": float(norm[i])} for i in order]

    return rerank_model_func


def reset_rerank_singleton() -> None:
    """測試或切換模型時清除 CrossEncoder 快取。"""
    with _rerank_lock:
        _rerank_models.clear()
