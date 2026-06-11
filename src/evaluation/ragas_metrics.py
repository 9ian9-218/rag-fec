"""RAGAS 端到端指標：基於 ragas Python 包。"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

_RAGAS_METRICS: list[Any] | None = None


def _get_ragas_metrics() -> list[Any]:
    """取得與 ragas.evaluate 相容的預建指標單例。"""
    global _RAGAS_METRICS
    if _RAGAS_METRICS is None:
        # collections 指標與 evaluate 的 isinstance 檢查不兼容，使用 v1 單例
        from ragas.metrics import context_precision, context_recall, faithfulness

        _RAGAS_METRICS = [context_recall, context_precision, faithfulness]
    return _RAGAS_METRICS


def build_ragas_llm(settings: Any = None) -> Any:
    """從專案設定建立 ragas LLM（OpenAI 相容端點）。"""
    from openai import OpenAI
    from ragas.llms import llm_factory

    from config.settings import get_settings

    s = settings or get_settings()
    key = (s.openai_api_key or s.llm.api_key or s.multimodal.api_key or "").strip()
    base = (
        s.openai_base_url or s.llm.base_url or s.multimodal.base_url or "https://api.openai.com/v1"
    ).rstrip("/")
    model = s.resolved_llm_model_name()
    if not model:
        raise RuntimeError("RAGAS 需要配置 LLM 模型（OPENAI_MODEL / LLM_MODEL_NAME）")
    if not base:
        raise RuntimeError(
            "RAGAS 需要配置 LLM base_url（OPENAI_API_BASE / LLM_BASE_URL / MULTIMODAL_BASE_URL）"
        )
    # 本機 OpenAI 相容端點允許 api_key 為 none
    client = OpenAI(api_key=key or "none", base_url=base)
    # 評測裁判輸出 JSON，需足夠 token 避免截斷
    return llm_factory(model, client=client, max_tokens=4096)


def build_ragas_reference(
    reference: str,
    *,
    reference_bullets: list[str] | None = None,
    gold_evidence_texts: list[str] | None = None,
) -> str:
    """合併參考答案、要點與金標 evidence，作為 ragas reference。"""
    parts: list[str] = []
    if reference.strip():
        parts.append(reference.strip())
    if reference_bullets:
        parts.extend(str(x).strip() for x in reference_bullets if str(x).strip())
    if gold_evidence_texts:
        parts.extend(str(x).strip() for x in gold_evidence_texts if str(x).strip())
    return "\n".join(parts)


def build_ragas_contexts(
    retrieved_context: str,
    *,
    kg_text: str = "",
    chunks_only: bool = True,
) -> list[str]:
    """構建 ragas retrieved_contexts；chunk 正文與 KG 分開以便 precision 評估。"""
    from src.evaluation.context_utils import extract_chunk_sections

    ctx = (retrieved_context or "").strip()
    if not ctx and not (kg_text or "").strip():
        return []
    contexts: list[str] = []
    if chunks_only and "【片段" in ctx:
        body = extract_chunk_sections(ctx)
        if body.strip():
            contexts.append(body.strip())
        elif ctx:
            contexts.append(ctx)
    elif ctx:
        contexts.append(ctx)
    kg = (kg_text or "").strip()
    if kg:
        contexts.append(kg)
    return contexts or ([ctx] if ctx else [])


def row_to_ragas_sample(
    *,
    question: str,
    reference: str,
    prediction: str,
    retrieved_context: str,
    gold_evidence_texts: list[str] | None = None,
    kg_text: str = "",
    reference_bullets: list[str] | None = None,
    chunks_only_precision: bool = True,
) -> dict[str, Any]:
    """將評估 JSONL 單行映射為 ragas Dataset 樣本。"""
    return {
        "user_input": question or "",
        "response": prediction or "",
        "retrieved_contexts": build_ragas_contexts(
            retrieved_context,
            kg_text=kg_text,
            chunks_only=chunks_only_precision,
        ),
        "reference": build_ragas_reference(
            reference,
            reference_bullets=reference_bullets,
            gold_evidence_texts=gold_evidence_texts,
        ),
    }


def _safe_float(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(num) else num


def compute_ragas_batch(
    samples: list[dict[str, Any]],
    *,
    llm: Any = None,
    settings: Any = None,
) -> list[dict[str, float]]:
    """批量計算 RAGAS 三項指標，返回與 samples 同序的分數。"""
    if not samples:
        return []
    from datasets import Dataset
    from ragas import evaluate

    ragas_llm = llm or build_ragas_llm(settings)
    dataset = Dataset.from_list(samples)
    result = evaluate(
        dataset,
        metrics=_get_ragas_metrics(),
        llm=ragas_llm,
        show_progress=False,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    return [
        {
            "context_recall": _safe_float(row.get("context_recall")),
            "context_precision": _safe_float(row.get("context_precision")),
            "faithfulness": _safe_float(row.get("faithfulness")),
        }
        for _, row in df.iterrows()
    ]


def compute_ragas_row(
    *,
    reference: str,
    prediction: str,
    retrieved_context: str,
    question: str = "",
    gold_evidence_texts: list[str] | None = None,
    kg_text: str = "",
    reference_bullets: list[str] | None = None,
    chunks_only_precision: bool = True,
    use_embedding_faithfulness: bool = True,
    llm: Any = None,
    settings: Any = None,
) -> dict[str, float]:
    """單條樣本 RAGAS 評分（內部走 batch 以复用 ragas.evaluate）。"""
    del use_embedding_faithfulness  # ragas 包下由 LLM 裁判，保留參數僅為兼容舊調用
    sample = row_to_ragas_sample(
        question=question,
        reference=reference,
        prediction=prediction,
        retrieved_context=retrieved_context,
        gold_evidence_texts=gold_evidence_texts,
        kg_text=kg_text,
        reference_bullets=reference_bullets,
        chunks_only_precision=chunks_only_precision,
    )
    scores = compute_ragas_batch([sample], llm=llm, settings=settings)
    if scores:
        return scores[0]
    return {"context_recall": 0.0, "context_precision": 0.0, "faithfulness": 0.0}
