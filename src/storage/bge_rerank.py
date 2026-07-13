"""本地 BGE CrossEncoder 封裝（支持 sentence-transformers）。

提供關係重排所需的 CrossEncoder 單例管理與 min-max 分數歸一化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
