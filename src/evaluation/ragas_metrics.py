"""RAGAS 風格端到端指標：v2 改進啟發式 + 可選 LLM 裁判。"""

from __future__ import annotations

import re
from typing import Any

from src.evaluation.answer_metrics import char_f1, token_f1
from src.evaluation.context_utils import extract_chunk_sections, split_claim_sentences
from src.evaluation.text_utils import normalize_answer

_EMBED_MODEL: Any = None


def _jieba_tokens(text: str) -> list[str]:
    try:
        import jieba  # type: ignore[import-untyped]

        return [t.strip() for t in jieba.cut(text) if t.strip()]
    except ImportError:
        return [t for t in text.split() if t]


def _token_set(text: str) -> set[str]:
    return set(_jieba_tokens(normalize_answer(text)))


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _claim_supported(claim: str, support_text: str, *, char_threshold: float = 0.42) -> bool:
    c = normalize_answer(claim)
    s = normalize_answer(support_text)
    if not c or not s:
        return False
    if c in s:
        return True
    if char_f1(claim, support_text)["f1"] >= char_threshold:
        return True
    if token_f1(claim, support_text, use_jieba=True)["f1"] >= 0.38:
        return True
    return _overlap_ratio(claim, support_text) >= 0.45


def context_recall(
    reference: str,
    retrieved_context: str,
    *,
    gold_evidence_texts: list[str] | None = None,
    reference_bullets: list[str] | None = None,
) -> float:
    """v2：以中文參考句 + 可選要點為 claim，在上下文中做字符/詞面支撐判斷。"""
    ctx = retrieved_context or ""
    if not ctx.strip():
        return 0.0
    claims: list[str] = []
    if reference_bullets:
        claims.extend(str(x) for x in reference_bullets if str(x).strip())
    claims.extend(split_claim_sentences(reference))
    if gold_evidence_texts:
        claims.extend(str(e) for e in gold_evidence_texts if str(e).strip())
    # 去重（按 normalize）
    seen: set[str] = set()
    uniq: list[str] = []
    for c in claims:
        k = normalize_answer(c)
        if k and k not in seen:
            seen.add(k)
            uniq.append(c)
    if not uniq:
        return 1.0
    hits = sum(1 for c in uniq if _claim_supported(c, ctx))
    return hits / len(uniq)


def context_precision(
    retrieved_context: str,
    *,
    gold_evidence_texts: list[str] | None = None,
    reference: str = "",
    reference_bullets: list[str] | None = None,
    chunks_only: bool = True,
) -> float:
    """v2：預設僅對 chunk 正文分句，避免 KG 摘要拉低 precision。"""
    ctx = extract_chunk_sections(retrieved_context) if chunks_only else (retrieved_context or "")
    ctx = ctx.strip()
    if not ctx:
        return 0.0
    sents = [s.strip() for s in re.split(r"[\n。！？!?]+", ctx) if s.strip()]
    if not sents:
        return 0.0
    gold_parts: list[str] = list(reference_bullets or []) if reference_bullets else []
    gold_parts.extend(split_claim_sentences(reference))
    if gold_evidence_texts:
        gold_parts.extend(str(x) for x in gold_evidence_texts if str(x).strip())
    if not gold_parts:
        return 1.0
    rel = sum(
        1
        for s in sents
        if any(_claim_supported(g, s, char_threshold=0.35) for g in gold_parts)
    )
    return rel / len(sents)


def faithfulness_ngram(prediction: str, retrieved_context: str, *, kg_text: str = "") -> float:
    """保留舊版 n-gram 覆蓋（偏嚴，僅作對照）。"""
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    support = normalize_answer((retrieved_context or "") + "\n" + (kg_text or ""))
    if not support.strip():
        return 0.0
    pred_toks = _jieba_tokens(pred)
    if not pred_toks:
        return 1.0
    sup_toks = _jieba_tokens(support)
    tri = _ngrams(pred_toks, 3) or _ngrams(pred_toks, 2) or {(pred_toks[0],)}
    sup_tri = _ngrams(sup_toks, 3) | _ngrams(sup_toks, 2)
    if not tri:
        return 0.0
    return len(tri & sup_tri) / len(tri)


def faithfulness_claim(prediction: str, retrieved_context: str, *, kg_text: str = "") -> float:
    """v2：按預測答案拆句，每句是否可由上下文+圖譜支撐。"""
    pred = (prediction or "").strip()
    if not pred:
        return 0.0
    support = ((retrieved_context or "") + "\n" + (kg_text or "")).strip()
    if not support:
        return 0.0
    sents = split_claim_sentences(pred) or [pred]
    if len(pred) > 80 and len(sents) == 1:
        sents = re.split(r"\n{2,}|\n(?=#)", pred)
        sents = [x.strip() for x in sents if len(x.strip()) >= 10] or sents
    hits = sum(1 for s in sents if _claim_supported(s, support, char_threshold=0.38))
    return hits / len(sents)


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from config.model_paths import resolve_embedding_model_load_path
        from config.settings import get_settings
        from sentence_transformers import SentenceTransformer

        path = resolve_embedding_model_load_path(get_settings())
        _EMBED_MODEL = SentenceTransformer(path, trust_remote_code=True)
        return _EMBED_MODEL
    except Exception:
        return None


def faithfulness_embedding(
    prediction: str,
    retrieved_context: str,
    *,
    kg_text: str = "",
    threshold: float = 0.55,
) -> float:
    """可選：句級嵌入相似度（更貼近語義忠實）。"""
    model = _get_embed_model()
    if model is None:
        return faithfulness_claim(prediction, retrieved_context, kg_text=kg_text)
    import numpy as np

    pred = (prediction or "").strip()
    support = ((retrieved_context or "") + "\n" + (kg_text or ""))[:12000]
    if not pred or not support.strip():
        return 0.0
    pred_sents = split_claim_sentences(pred) or [pred[:500]]
    ctx_sents = split_claim_sentences(support)
    if not ctx_sents:
        ctx_sents = [support[:2000]]
    pe = model.encode(pred_sents, normalize_embeddings=True, show_progress_bar=False)
    ce = model.encode(ctx_sents, normalize_embeddings=True, show_progress_bar=False)
    pe = np.asarray(pe)
    ce = np.asarray(ce)
    if pe.ndim == 1:
        pe = pe.reshape(1, -1)
    if ce.ndim == 1:
        ce = ce.reshape(1, -1)
    sims = pe @ ce.T
    hits = sum(1 for i in range(len(pred_sents)) if float(sims[i].max()) >= threshold)
    return hits / len(pred_sents)


def _overlap_ratio(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta)


def compute_ragas_row(
    *,
    reference: str,
    prediction: str,
    retrieved_context: str,
    gold_evidence_texts: list[str] | None = None,
    kg_text: str = "",
    reference_bullets: list[str] | None = None,
    chunks_only_precision: bool = True,
    use_embedding_faithfulness: bool = True,
) -> dict[str, float]:
    faith = (
        faithfulness_embedding(prediction, retrieved_context, kg_text=kg_text)
        if use_embedding_faithfulness
        else faithfulness_claim(prediction, retrieved_context, kg_text=kg_text)
    )
    return {
        "context_recall": context_recall(
            reference,
            retrieved_context,
            gold_evidence_texts=gold_evidence_texts,
            reference_bullets=reference_bullets,
        ),
        "context_precision": context_precision(
            retrieved_context,
            gold_evidence_texts=gold_evidence_texts,
            reference=reference,
            chunks_only=chunks_only_precision,
        ),
        "faithfulness": faith,
        "faithfulness_ngram_legacy": faithfulness_ngram(prediction, retrieved_context, kg_text=kg_text),
        "faithfulness_claim": faithfulness_claim(prediction, retrieved_context, kg_text=kg_text),
    }


async def compute_ragas_row_llm(
    *,
    reference: str,
    prediction: str,
    retrieved_context: str,
    question: str,
    client: Any,
    model: str,
) -> dict[str, float]:
    prompt = (
        "你是 RAG 評估裁判。僅根據檢索上下文判斷，輸出 JSON："
        '{"context_recall":0-1,"context_precision":0-1,"faithfulness":0-1}。\n'
        "context_recall：參考答案中的關鍵結論是否都能從上下文推出。\n"
        "context_precision：上下文中與問題相關且必要的內容占比。\n"
        "faithfulness：模型答案是否無幻覺、可由上下文支持。\n"
        f"問題：{question}\n參考答案：{reference}\n模型答案：{prediction}\n"
        f"檢索上下文：{retrieved_context[:12000]}"
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    import json

    text = (resp.choices[0].message.content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM judge did not return JSON")
    data = json.loads(text[start : end + 1])
    return {
        "context_recall": float(data.get("context_recall", 0)),
        "context_precision": float(data.get("context_precision", 0)),
        "faithfulness": float(data.get("faithfulness", 0)),
    }
