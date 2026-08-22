"""RAG 業務服務：問答、文件 CRUD、增量更新、對話歷史。"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from config.settings import get_settings
from src.incremental.conversion_manager import ConversionManager
from src.incremental.update_manager import UpdateManager
from src.retrieval.retriever import GraphRAGRetriever
from src.storage.kv_client import KVClient
from src.storage.redis_cache import build_query_cache_key, get_json_cache, set_json_cache
from src.storage.redis_lock import acquire_lock, release_lock
from src.utils.logger import get_logger

logger = get_logger("service.rag_service")


class RAGService:
    """應用層 Facade。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._retriever = GraphRAGRetriever()
        self._kv = KVClient()
        self._query_sem = asyncio.Semaphore(self._settings.service.max_concurrent_queries)

    @property
    def last_mode_selection(self) -> dict[str, object] | None:
        route = self._retriever.last_mode_route
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
        # 每次問答僅使用本次檢索結果，不再將歷史對話作為上下文傳給 LLM
        custom_instructions = (
            "你是專業助手，請基於提供的檢索材料作答，尽量使用简体中文来进行回答。"
            "【嚴格規則】你只能使用下方提供的檢索材料中的信息來回答用戶問題；"
            "如果檢索材料中沒有足夠信息，你必須明確說明『根據提供的檢索材料，無法找到足夠的信息來回答該問題』，"
            "絕對不要使用你自身的知識來補充、推測或編造答案。"
        )

        if stream:
            async def _bounded_stream() -> AsyncIterator[str]:
                async with self._query_sem:
                    out = await self._retriever.query(
                        question,
                        mode=mode,  # type: ignore[arg-type]
                        history=[],
                        stream=True,
                        multimodal=multimodal,
                        use_llm_router=use_llm_router,
                        custom_instructions=custom_instructions,
                    )
                    if hasattr(out, "__aiter__"):
                        async for chunk in out:  # type: ignore[union-attr]
                            if chunk:
                                yield str(chunk)
                    else:
                        yield str(out)

            return _bounded_stream()

        cache_key = build_query_cache_key(
            question=question,
            mode=mode,
            top_k=self._settings.retrieval.top_k,
            multimodal=multimodal,
            use_llm_router=bool(
                use_llm_router if use_llm_router is not None else self._settings.retrieval.llm_mode_router_enabled
            ),
        )
        cached = await get_json_cache(cache_key)
        if isinstance(cached, dict) and "answer" in cached:
            return str(cached["answer"])

        async with self._query_sem:
            out = await self._retriever.query(
                question,
                mode=mode,  # type: ignore[arg-type]
                history=[],
                stream=False,
                multimodal=multimodal,
                use_llm_router=use_llm_router,
                custom_instructions=custom_instructions,
            )
            answer = str(out)

        await set_json_cache(
            cache_key,
            {"answer": answer},
            ttl=int(self._settings.redis.cache_ttl_seconds),
        )
        return answer

    async def query_with_context(self, question: str, **kw: Any) -> dict[str, Any]:
        return await self._retriever.retrieve_data(question, **kw)  # 含 mode_selection

    async def incremental_update(self, *, convert_first: bool = False) -> dict[str, Any]:
        lock_key = "rag:lock:incremental"
        token = await acquire_lock(lock_key, timeout=60)
        if token is None:
            raise RuntimeError("另一个增量更新正在进行，请稍后重试")
        try:
            out: dict[str, Any] = {}
            if convert_first and self._settings.document.is_two_stage():
                out["conversion"] = ConversionManager().run_incremental()
            out["index"] = await UpdateManager(kv=self._kv).run_incremental()
            return out
        finally:
            await release_lock(lock_key, token)

    async def add_document(self, path: Path) -> dict[str, Any]:
        path = path.expanduser().resolve()
        lock_key = f"rag:lock:doc:{path.name}"
        token = await acquire_lock(lock_key, timeout=60)
        if token is None:
            raise RuntimeError(f"文档正在处理中: {path.name}")
        try:
            if path.suffix.lower() == ".pdf" and self._settings.document.is_two_stage():
                ConversionManager().convert_path(path)
            mgr = UpdateManager(kv=self._kv)
            return await mgr.ingest_path(path, replace=False)
        finally:
            await release_lock(lock_key, token)

    async def update_document(self, path: Path) -> dict[str, Any]:
        path = path.expanduser().resolve()
        lock_key = f"rag:lock:doc:{path.name}"
        token = await acquire_lock(lock_key, timeout=60)
        if token is None:
            raise RuntimeError(f"文档正在处理中: {path.name}")
        try:
            mgr = UpdateManager(kv=self._kv)
            return await mgr.ingest_path(path, replace=True)
        finally:
            await release_lock(lock_key, token)

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
        return str(uuid.uuid4())
