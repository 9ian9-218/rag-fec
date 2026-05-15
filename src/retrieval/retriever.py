"""統一檢索介面：包裝 LightRAG，並以 LangChain Runnable 組合「檢索資料 → 後處理」。"""

from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config.settings import get_settings
from src.retrieval.mode_config import RetrievalMode, build_query_param, suggest_mode_from_question
from src.retrieval.result_processor import compact_retrieval_payload, extract_sources, kg_dict_to_bullets
from src.storage.lightrag_init import get_lightrag


class GraphRAGRetriever:
    """非同步檢索封裝。"""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def _rag(self):
        return await get_lightrag()

    async def retrieve_data(
        self,
        question: str,
        *,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        stream: bool = False,
        only_need_context: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """呼叫 ``aquery_data``，回傳結構化檢索結果。"""
        rag = await self._rag()
        m = mode or self._settings.retrieval.default_mode  # type: ignore[assignment]
        if not isinstance(m, str) or m not in (
            "naive",
            "local",
            "global",
            "hybrid",
            "mix",
            "bypass",
        ):
            m = suggest_mode_from_question(question)
        param = build_query_param(
            m,  # type: ignore[arg-type]
            top_k=top_k or self._settings.retrieval.top_k,
            stream=stream,
            only_need_context=only_need_context,
            conversation_history=history,
        )
        data = await rag.aquery_data(question, param)
        sources = extract_sources(data if isinstance(data, dict) else {})
        kg_text = ""
        if isinstance(data, dict):
            ent = data.get("entities") or []
            rel = data.get("relationships") or []
            if isinstance(ent, list) and isinstance(rel, list):
                kg_text = kg_dict_to_bullets(ent, rel)
        return {
            "data": data,
            "sources": sources,
            "kg_text": kg_text,
            "mode": m,
        }

    async def query(
        self,
        question: str,
        *,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """端到端問答（含 LLM）。``stream=True`` 時回傳 async iterator。"""
        rag = await self._rag()
        m = mode or self._settings.retrieval.default_mode
        if not isinstance(m, str) or m not in (
            "naive",
            "local",
            "global",
            "hybrid",
            "mix",
            "bypass",
        ):
            m = suggest_mode_from_question(question)
        param = build_query_param(
            m,  # type: ignore[arg-type]
            top_k=top_k or self._settings.retrieval.top_k,
            stream=stream,
            conversation_history=history or [],
        )
        return await rag.aquery(question, param, system_prompt=system_prompt)

    def build_postprocess_chain(self):
        """LangChain：檢索結果 → 壓縮 payload（方便鏈式組合）。"""
        return RunnablePassthrough.assign(
            compact=lambda d: compact_retrieval_payload(d.get("data", {}))
            if isinstance(d, dict)
            else {},
        )

    def build_retrieve_then_compact_chain(self):
        """Runnable：問句 → ``retrieve_data`` → 精簡。"""

        async def _retrieve(inputs: dict[str, Any]) -> dict[str, Any]:
            q = str(inputs["question"])
            return await self.retrieve_data(
                q,
                mode=inputs.get("mode"),
                top_k=inputs.get("top_k"),
            )

        return RunnableLambda(_retrieve) | RunnableLambda(
            lambda d: {**d, "compact": compact_retrieval_payload(d.get("data", {})) if isinstance(d, dict) else {}}
        )
