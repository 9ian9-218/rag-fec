"""統一檢索介面：包裝 LightRAG，並以 LangChain Runnable 組合「檢索資料 → 後處理」。"""

from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config.settings import get_settings
from src.evaluation.online_monitor import (
    QueryTimer,
    append_telemetry,
    build_telemetry,
    clear_rerank_stats,
)
from src.retrieval.mode_config import RetrievalMode, build_query_param
from src.retrieval.mode_router import ModeRouteResult, resolve_retrieval_mode
from src.retrieval.relation_optimizer import refine_retrieval_bundle
from src.retrieval.result_processor import compact_retrieval_payload, extract_sources, kg_dict_to_bullets
from src.storage.lightrag_init import get_lightrag
from src.utils.logger import get_logger

logger = get_logger("retrieval.retriever")


class GraphRAGRetriever:
    """非同步檢索封裝。"""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._use_llm_mode_router = True
        self._last_mode_route: ModeRouteResult | None = None

    async def _rag(self):
        return await get_lightrag()

    async def _resolve_mode(
        self,
        question: str,
        mode: RetrievalMode | None,
        *,
        use_llm_router: bool | None = None,
    ) -> ModeRouteResult:
        explicit = mode if isinstance(mode, str) else None
        router_on = self._use_llm_mode_router if use_llm_router is None else use_llm_router
        route = await resolve_retrieval_mode(
            question,
            explicit,
            settings=self._settings,
            use_llm_router=router_on,
        )
        self._last_mode_route = route
        return route

    async def retrieve_data(
        self,
        question: str,
        *,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        stream: bool = False,
        only_need_context: bool = False,
        history: list[dict[str, str]] | None = None,
        use_llm_router: bool | None = None,
    ) -> dict[str, Any]:
        """呼叫 ``aquery_data``，回傳結構化檢索結果。"""
        rag = await self._rag()
        route = await self._resolve_mode(question, mode, use_llm_router=use_llm_router)
        m = route.mode
        param = build_query_param(
            m,
            top_k=top_k or self._settings.retrieval.top_k,
            stream=stream,
            only_need_context=only_need_context,
            conversation_history=history,
        )
        clear_rerank_stats()
        timer = QueryTimer()
        data = await rag.aquery_data(question, param)
        if isinstance(data, dict):
            data = await refine_retrieval_bundle(question, data, settings=self._settings)
        sources = extract_sources(data if isinstance(data, dict) else {})
        kg_text = ""
        if isinstance(data, dict):
            ent = data.get("entities") or []
            rel = data.get("relationships") or []
            if isinstance(ent, list) and isinstance(rel, list):
                kg_text = kg_dict_to_bullets(ent, rel)
        bundle = data if isinstance(data, dict) else {}
        append_telemetry(
            build_telemetry(
                question=question,
                mode=str(m),
                bundle=bundle,
                latency_ms=timer.elapsed_ms(),
                kg_text=kg_text,
                min_rerank_score=float(self._settings.retrieval.rerank_min_score),
            )
        )
        clear_rerank_stats()
        return {
            "data": data,
            "sources": sources,
            "kg_text": kg_text,
            "mode": m,
            "mode_selection": route.to_dict(),
        }

    async def query(
        self,
        question: str,
        *,
        mode: RetrievalMode | None = None,
        top_k: int | None = None,
        custom_instructions: str | None = None,
        history: list[dict[str, str]] | None = None,
        stream: bool = False,
        multimodal: bool = False,
        use_llm_router: bool | None = None,
    ) -> str | AsyncIterator[str]:
        """端到端問答（含 LLM）。``stream=True`` 時回傳 async iterator。

        ``multimodal=True``：先 ``aquery_data`` 取 chunks，解析 ``![](images/...)``，
        將圖片一併送入支援 vision 的 OpenAI 相容 API（``stream`` 與多模態同開時暫不支援，將降級為非串流）。
        """
        rag = await self._rag()
        route = await self._resolve_mode(question, mode, use_llm_router=use_llm_router)
        m = route.mode
        if m == "bypass" and route.source != "explicit":
            logger.warning("自動路由選中 bypass，改為 mix 以保留檢索")
            m = "mix"
            route = ModeRouteResult(mode="mix", reason=route.reason or "bypass 已改為 mix", source=route.source)
        param = build_query_param(
            m,
            top_k=top_k or self._settings.retrieval.top_k,
            stream=stream,
            conversation_history=history or [],
            user_prompt=custom_instructions,
        )

        if multimodal and stream:
            logger.warning("多模態與 stream 同時請求時，改為非串流多模態回答")
            stream = False
            param = build_query_param(
                m,
                top_k=top_k or self._settings.retrieval.top_k,
                stream=False,
                conversation_history=history or [],
                user_prompt=custom_instructions,
            )

        clear_rerank_stats()
        timer = QueryTimer()
        if multimodal and m != "bypass":
            bundle = await rag.aquery_data(question, param)
            if isinstance(bundle, dict):
                bundle = await refine_retrieval_bundle(question, bundle, settings=self._settings)
            if isinstance(bundle, dict) and bundle.get("status") == "success":
                from src.retrieval.multimodal_answer import (
                    _extract_chunks_and_kg,
                    _reference_trim_params,
                    _trim_chunks_for_reference,
                    answer_with_retrieved_images,
                    answer_with_retrieved_text_only,
                )

                chunks_raw, kg_text = _extract_chunks_and_kg(bundle)
                mc, mch = _reference_trim_params(self._settings)
                trimmed = _trim_chunks_for_reference(chunks_raw, max_chunk_count=mc, max_chars=mch)
                try:
                    answer = await answer_with_retrieved_images(
                        settings=self._settings,
                        question=question,
                        bundle=bundle,
                        history_messages=history,
                    )
                    append_telemetry(
                        build_telemetry(
                            question=question,
                            mode=str(m),
                            bundle=bundle,
                            latency_ms=timer.elapsed_ms(),
                            kg_text=kg_text,
                            reference_chunks_before=len(chunks_raw),
                            reference_chunks_after=len(trimmed),
                            min_rerank_score=float(self._settings.retrieval.rerank_min_score),
                        )
                    )
                    clear_rerank_stats()
                    return answer
                except Exception as e:
                    logger.warning("多模態管線異常，嘗試僅以檢索文字作答: %s", e)
                    try:
                        answer = await answer_with_retrieved_text_only(
                            settings=self._settings,
                            question=question,
                            bundle=bundle,
                            history_messages=history,
                        )
                        append_telemetry(
                            build_telemetry(
                                question=question,
                                mode=str(m),
                                bundle=bundle,
                                latency_ms=timer.elapsed_ms(),
                                kg_text=kg_text,
                                reference_chunks_before=len(chunks_raw),
                                reference_chunks_after=len(trimmed),
                                min_rerank_score=float(self._settings.retrieval.rerank_min_score),
                            )
                        )
                        clear_rerank_stats()
                        return answer
                    except Exception as e2:
                        logger.warning("僅文字檢索作答仍失敗，降級為 LightRAG aquery: %s", e2)

        bundle = await rag.aquery_data(question, param)
        if isinstance(bundle, dict):
            bundle = await refine_retrieval_bundle(question, bundle, settings=self._settings)
        sources = extract_sources(bundle if isinstance(bundle, dict) else {})
        kg_text = ""
        if isinstance(bundle, dict):
            ent = bundle.get("entities") or []
            rel = bundle.get("relationships") or []
            if isinstance(ent, list) and isinstance(rel, list):
                kg_text = kg_dict_to_bullets(ent, rel)

        out: str | AsyncIterator[str]
        if not stream:
            from src.retrieval.multimodal_answer import answer_with_retrieved_text_only
            try:
                out = await answer_with_retrieved_text_only(
                    settings=self._settings,
                    question=question,
                    bundle=bundle,
                    history_messages=history,
                )
            except Exception as e:
                logger.warning("自定义文字回答失败，降级为 LightRAG aquery: %s", e)
                out = await rag.aquery(question, param)
        else:
            out = await rag.aquery(question, param)
        append_telemetry(
            build_telemetry(
                question=question,
                mode=str(m),
                bundle=bundle if isinstance(bundle, dict) else {},
                latency_ms=timer.elapsed_ms(),
                kg_text=kg_text,
                min_rerank_score=float(self._settings.retrieval.rerank_min_score),
            )
        )
        clear_rerank_stats()
        return out

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
