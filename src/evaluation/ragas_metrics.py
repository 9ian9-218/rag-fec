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
    # 本機 OpenAI 相容端點允許 api_key 為 none；OpenCode 網關強制要求會話頭，
    # 否則裁判請求一律 400（MissingSessionID），所有指標會變成 0 分
    from src.utils.openai_session import session_headers

    client = OpenAI(api_key=key or "none", base_url=base, default_headers=session_headers())
    # 评测裁判输出 JSON，需足够 token 避免截断；模型/答案越长越容易撞上限。
    # 注意：本网关模型即使 temperature=0 也非完全确定（实测同问两次结果不同），
    # 因此判官分数存在固有波动，小差异不可当作信号——详见 docs/jev-laya-step1-results。
    max_tokens = int(getattr(s, "evaluation", None).ragas_max_tokens)
    return llm_factory(model, client=client, max_tokens=max_tokens, temperature=0)


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


def _is_missing(value: Any) -> bool:
    """ragas 裁判失敗時該行指標為 NaN；與「真的是 0 分」必須區分開。"""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def compute_ragas_batch(
    samples: list[dict[str, Any]],
    *,
    llm: Any = None,
    settings: Any = None,
) -> list[dict[str, float]]:
    """批量計算 RAGAS 三項指標，返回與 samples 同序的分數。

    每行附帶 ``ragas_ok``：為 False 表示裁判 LLM 調用失敗、分數不可用
    （而不是「答案確實得 0 分」），上層據此把這些行排除在均值之外。
    """
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

    out: list[dict[str, Any]] = []
    failed_rows = 0
    failed_metrics: dict[str, int] = {}
    for _, row in df.iterrows():
        vals = {
            "context_recall": row.get("context_recall"),
            "context_precision": row.get("context_precision"),
            "faithfulness": row.get("faithfulness"),
        }
        missing = {k: _is_missing(v) for k, v in vals.items()}
        if all(missing.values()):
            failed_rows += 1
        for k, is_missing in missing.items():
            if is_missing:
                failed_metrics[k] = failed_metrics.get(k, 0) + 1
        rec: dict[str, Any] = {k: _safe_float(v) for k, v in vals.items()}
        # 逐指标标记是否真的算出来了；上層均值只統計 <metric>_scored=True 的行，
        # 避免「某指标裁判失败」被当成「該指標得 0 分」拉低均值
        rec |= {f"{k}_scored": not is_missing for k, is_missing in missing.items()}
        rec["ragas_ok"] = not all(missing.values())
        out.append(rec)

    if failed_rows or failed_metrics:
        logger.warning(
            "RAGAS 裁判部分/全部失败：整行失败 %d/%d；逐指标失败 %s。"
            "失败指标已标 <metric>_scored=False 并排除在均值外。"
            "常见原因：裁判 LLM 调用失败（连接/鉴权/缺 x-opencode-session 头）"
            "或输出被 max_tokens 截斷（IncompleteOutputException）。",
            failed_rows,
            len(out),
            failed_metrics or "无",
        )
    return out


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
) -> dict[str, Any]:
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
    return {
        "context_recall": 0.0,
        "context_precision": 0.0,
        "faithfulness": 0.0,
        "ragas_ok": False,
    }
