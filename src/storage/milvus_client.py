"""Milvus 向量庫輔助操作（LightRAG 仍自行管理 collection；此處提供維運級刪查）。"""

from __future__ import annotations

from typing import Any

from pymilvus import MilvusClient

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("storage.milvus")


class MilvusAdminClient:
    """以 MilvusClient 執行維運操作（依 collection 名稱刪除符合條件的向量）。"""

    def __init__(self, uri: str | None = None, db_name: str | None = None) -> None:
        s = get_settings()
        self._uri = uri or s.milvus.uri
        self._db = db_name or s.milvus.db_name
        self._client: MilvusClient | None = None

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri, db_name=self._db)
        return self._client

    def delete_by_doc_id(self, collection_name: str, doc_id: str) -> None:
        """刪除 ``full_doc_id`` 等於指定 doc_id 的資料列（若欄位不存在則由呼叫端確認 schema）。"""
        expr = f'full_doc_id == "{doc_id}"'
        try:
            self.client.delete(collection_name=collection_name, filter=expr)
            logger.info("Milvus delete: collection=%s filter=%s", collection_name, expr)
        except Exception as e:
            logger.warning("Milvus delete failed (collection may not exist): %s", e)

    def drop_collection_if_exists(self, collection_name: str) -> None:
        """刪除集合（若存在）。"""
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                logger.info("Dropped Milvus collection %s", collection_name)
        except Exception as e:
            logger.error("Failed to drop collection %s: %s", collection_name, e)
