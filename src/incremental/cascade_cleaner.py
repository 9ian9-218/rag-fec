"""級聯清理：以 LightRAG 官方刪除為主，並同步應用層 SQLite 註冊。"""

from __future__ import annotations

from typing import Any

from src.storage.kv_client import KVClient
from src.utils.logger import get_logger

logger = get_logger("incremental.cascade_cleaner")


async def cascade_delete_document(rag: Any, doc_id: str, kv: KVClient | None = None) -> dict[str, Any]:
    """刪除單一文件在圖、向量、KV（LightRAG 內部）之資料，並可選清除 SQLite 註冊列。

    LightRAG 的 ``adelete_by_doc_id`` 已協調 Neo4j、Milvus 與 JSON KV 等後端。
    """
    result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=False)
    if kv is not None:
        try:
            kv.delete_document_row(doc_id)
        except Exception as e:
            logger.warning("SQLite 註冊列刪除失敗 doc_id=%s: %s", doc_id, e)
    logger.info("cascade_delete_document doc_id=%s status=%s", doc_id, getattr(result, "status", result))
    if isinstance(result, dict):
        return result
    return {
        "status": getattr(result, "status", "unknown"),
        "message": getattr(result, "message", ""),
        "doc_id": doc_id,
    }
