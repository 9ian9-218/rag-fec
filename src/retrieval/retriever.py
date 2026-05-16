"""統一檢索介面：包裝 LightRAG，並以 LangChain Runnable 組合「檢索資料 → 後處理」。"""

from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config.settings import get_settings
from src.retrieval.mode_config import RetrievalMode, build_query_param, suggest_mode_from_question
from src.retrieval.result_processor import compact_retrieval_payload, extract_sources, kg_dict_to_bullets
from src.storage.lightrag_init import get_lightrag
from src.utils.logger import get_logger

logger = get_logger("retrieval.retriever")


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
        multimodal: bool = False,
    ) -> str | AsyncIterator[str]:
        """端到端問答（含 LLM）。``stream=True`` 時回傳 async iterator。

        ``multimodal=True``：先 ``aquery_data`` 取 chunks，解析 ``![](images/...)``，
        將圖片一併送入支援 vision 的 OpenAI 相容 API（``stream`` 與多模態同開時暫不支援，將降級為非串流）。
        """
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

        if multimodal and stream:
            logger.warning("多模態與 stream 同時請求時，改為非串流多模態回答")
            stream = False
            param = build_query_param(
                m,  # type: ignore[arg-type]
                top_k=top_k or self._settings.retrieval.top_k,
                stream=False,
                conversation_history=history or [],
            )

        if multimodal and m != "bypass":
            bundle = await rag.aquery_data(question, param)
            if isinstance(bundle, dict) and bundle.get("status") == "success":
                from src.retrieval.multimodal_answer import (
                    answer_with_retrieved_images,
                    answer_with_retrieved_text_only,
                )

                try:
                    return await answer_with_retrieved_images(
                        settings=self._settings,
                        question=question,
                        bundle=bundle,
                        history_messages=history,
                    )
                except Exception as e:
                    logger.warning("多模態管線異常，嘗試僅以檢索文字作答: %s", e)
                    try:
                        return await answer_with_retrieved_text_only(
                            settings=self._settings,
                            question=question,
                            bundle=bundle,
                            history_messages=history,
                        )
                    except Exception as e2:
                        logger.warning("僅文字檢索作答仍失敗，降級為 LightRAG aquery: %s", e2)

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
