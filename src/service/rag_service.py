"""RAG 業務服務：問答、文件 CRUD、增量更新、對話歷史。"""

from __future__ import annotations

import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, AsyncIterator

from config.settings import get_settings
from src.incremental.conversion_manager import ConversionManager
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

    def set_llm_mode_router(self, enabled: bool) -> None:
        """開關檢索前 LLM 智能路由（與 ``RETRIEVAL_LLM_MODE_ROUTER_ENABLED`` 共同生效）。"""
        self._retriever._use_llm_mode_router = enabled

    @property
    def last_mode_selection(self) -> dict[str, object] | None:
        route = self._retriever._last_mode_route
        return route.to_dict() if route is not None else None

    async def query(
        self,
        question: str,
        *,
        session_id: str | None = None,
        mode: str | None = None,
        stream: bool = False,
        multimodal: bool = False,
        use_llm_router: bool | None = None,
    ) -> str | AsyncIterator[str]:
        hist = self._hist(session_id)
        hist.append({"role": "user", "content": question})
        out = await self._retriever.query(
            question,
            mode=mode,  # type: ignore[arg-type]
            history=hist[:-1],
            stream=stream,
            multimodal=multimodal,
            use_llm_router=use_llm_router,
        )
        if stream:
            return out  # type: ignore[return-value]

        hist.append({"role": "assistant", "content": str(out)})
        return str(out)

    async def query_with_context(self, question: str, **kw: Any) -> dict[str, Any]:
        return await self._retriever.retrieve_data(question, **kw)  # 含 mode_selection

    async def incremental_update(self, *, convert_first: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if convert_first and self._settings.document.is_two_stage():
            out["conversion"] = ConversionManager().run_incremental()
        out["index"] = await UpdateManager(kv=self._kv).run_incremental()
        return out

    async def add_document(self, path: Path) -> dict[str, Any]:
        path = path.expanduser().resolve()
        if path.suffix.lower() == ".pdf" and self._settings.document.is_two_stage():
            ConversionManager().convert_path(path)
        mgr = UpdateManager(kv=self._kv)
        return await mgr.ingest_path(path, replace=False)

    async def update_document(self, path: Path) -> dict[str, Any]:
        mgr = UpdateManager(kv=self._kv)
        return await mgr.ingest_path(path, replace=True)

    async def delete_document_by_id(self, doc_id: str) -> dict[str, Any]:
        from src.incremental.cascade_cleaner import cascade_delete_document
        from src.incremental.document_manifest import legacy_cleanup_markdown_sidecars, purge_for_doc_id
        from src.storage.lightrag_init import get_lightrag

        row = self._kv.get_doc_by_id(doc_id)
        manifest_out = purge_for_doc_id(doc_id)
        rag = await get_lightrag()
        result = await cascade_delete_document(rag, doc_id, self._kv)
        if manifest_out.get("skipped") and row and isinstance(row.get("source_path"), str):
            legacy_cleanup_markdown_sidecars(row["source_path"])
        return result

    def list_documents(self) -> list[dict[str, Any]]:
        return self._kv.list_documents()

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self._kv.get_doc_by_id(doc_id)

    def new_session(self) -> str:
        sid = str(uuid.uuid4())
        self._history[sid] = []
        return sid
