"""排序列表上的 Recall@K / Precision@K / NDCG@K（實體、關係、Chunk ID）。"""

from __future__ import annotations

import math
from typing import Callable


def _norm(s: str, normalizer: Callable[[str], str] | None) -> str:
    fn = normalizer or (lambda z: z.strip().lower())
    return fn(str(s).strip())


def _gold_set(gold: list[str], normalizer: Callable[[str], str] | None) -> set[str]:
    return {_norm(x, normalizer) for x in gold if _norm(x, normalizer)}


def _ranked_list(ranked: list[str], normalizer: Callable[[str], str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in ranked:
        n = _norm(x, normalizer)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def recall_at_k(gold: list[str], ranked: list[str], k: int, *, normalizer: Callable[[str], str] | None = None) -> float:
    g = _gold_set(gold, normalizer)
    if not g:
        return 1.0
    top = set(_ranked_list(ranked, normalizer)[: max(0, k)])
    return len(g & top) / len(g)


def precision_at_k(
    gold: list[str], ranked: list[str], k: int, *, normalizer: Callable[[str], str] | None = None
) -> float:
    g = _gold_set(gold, normalizer)
    top = _ranked_list(ranked, normalizer)[: max(0, k)]
    if not top:
        return 0.0 if g else 1.0
    hits = sum(1 for x in top if x in g)
    return hits / len(top)


def ndcg_at_k(gold: list[str], ranked: list[str], k: int, *, normalizer: Callable[[str], str] | None = None) -> float:
    g = _gold_set(gold, normalizer)
    if not g:
        return 1.0
    top = _ranked_list(ranked, normalizer)[: max(0, k)]
    if not top:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(top):
        rel = 1.0 if item in g else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    ideal_hits = min(len(g), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def ranked_metrics_bundle(
    gold: list[str],
    ranked: list[str],
    ks: tuple[int, ...],
    *,
    normalizer: Callable[[str], str] | None = None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in ks:
        key = f"@{k}"
        out[key] = {
            "recall": recall_at_k(gold, ranked, k, normalizer=normalizer),
            "precision": precision_at_k(gold, ranked, k, normalizer=normalizer),
            "ndcg": ndcg_at_k(gold, ranked, k, normalizer=normalizer),
        }
    return out
