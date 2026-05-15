"""RAG 業務服務：問答、文件 CRUD、增量更新、對話歷史。"""

from __future__ import annotations

import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator

from config.settings import get_settings
from src.incremental.update_manager import UpdateManager
from src.retrieval.retriever import GraphRAGRetriever
from src.storage.kv_client import KVClient
from src.utils.logger import get_logger

logger = get_logger("service.rag_service")


class RAGService:
    """應用層 Facade。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._retriever = GraphRAGRetriever()
        self._kv = KVClient()
        self._history: dict[str, list[dict[str, str]]] = defaultdict(list)

    def _hist(self, session_id: str | None) -> list[dict[str, str]]:
        sid = session_id or "default"
        return self._history[sid]

    async def query(
        self,
        question: str,
        *,
        session_id: str | None = None,
        mode: str | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        hist = self._hist(session_id)
        hist.append({"role": "user", "content": question})
        out = await self._retriever.query(
            question,
            mode=mode,  # type: ignore[arg-type]
            history=hist[:-1],
            stream=stream,
        )
        if stream:
            return out  # type: ignore[return-value]

        hist.append({"role": "assistant", "content": str(out)})
        return str(out)

    async def query_with_context(self, question: str, **kw: Any) -> dict[str, Any]:
        return await self._retriever.retrieve_data(question, **kw)

    async def incremental_update(self) -> dict[str, Any]:
        mgr = UpdateManager(kv=self._kv)
        return await mgr.run_incremental()

    async def add_document(self, path: Path) -> dict[str, Any]:
        mgr = UpdateManager(kv=self._kv)
        return await mgr.ingest_path(path, replace=False)

    async def update_document(self, path: Path) -> dict[str, Any]:
        mgr = UpdateManager(kv=self._kv)
        return await mgr.ingest_path(path, replace=True)

    async def delete_document_by_id(self, doc_id: str) -> dict[str, Any]:
        from src.incremental.cascade_cleaner import cascade_delete_document
        from src.storage.lightrag_init import get_lightrag

        rag = await get_lightrag()
        return await cascade_delete_document(rag, doc_id, self._kv)

    def list_documents(self) -> list[dict[str, Any]]:
        return self._kv.list_documents()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self._kv.get_doc_by_id(doc_id)

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self._history[sid] = []
        return sid
