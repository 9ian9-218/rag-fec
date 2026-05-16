"""文檔 / 塊 ID 級檢索指標（參考 graph-rag-agent 檢索評估中的集合重疊思想）。"""

from __future__ import annotations


def _strip_ids(ids: list[str]) -> set[str]:
    return {str(x).strip() for x in ids if str(x).strip()}


def retrieval_precision_recall_f1(gold_ids: list[str], retrieved_ids: list[str]) -> dict[str, float]:
    """將 gold / retrieved 視為集合（去重），計算標準 P/R/F1。"""
    g = _strip_ids(gold_ids)
    r = _strip_ids(retrieved_ids)
    if not g and not r:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not g or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    inter = g & r
    p = len(inter) / len(r) if r else 0.0
    rec = len(inter) / len(g) if g else 0.0
    f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
    return {"precision": p, "recall": rec, "f1": f1}


def hit_at_k(gold_ids: list[str], ranked_retrieved: list[str], k: int) -> float:
    """任一 gold 出現於前 k 個檢索結果中則為 1，否則 0。"""
    g = _strip_ids(gold_ids)
    if not g:
        return 1.0
    top = [str(x).strip() for x in ranked_retrieved[:k] if str(x).strip()]
    return 1.0 if g.intersection(top) else 0.0


def reciprocal_rank(gold_ids: list[str], ranked_retrieved: list[str]) -> float:
    """首個命中 gold 的排名倒數；無命中為 0。"""
    g = _strip_ids(gold_ids)
    if not g:
        return 1.0
    for i, rid in enumerate(ranked_retrieved):
        s = str(rid).strip()
        if s in g:
            return 1.0 / float(i + 1)
    return 0.0
