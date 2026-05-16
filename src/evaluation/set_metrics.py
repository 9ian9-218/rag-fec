"""實體 / 關係 / 文檔 ID 等集合上的 P/R/F1（ multiset 版本）。"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def _norm_list(items: Iterable[str], norm: Callable[[str], str]) -> Counter[str]:
    c: Counter[str] = Counter()
    for x in items:
        s = norm(str(x).strip())
        if s:
            c[s] += 1
    return c


def multiset_precision_recall_f1(
    predicted: list[str],
    gold: list[str],
    *,
    normalizer: Callable[[str], str] | None = None,
) -> dict[str, float]:
    """
    基於多重集合的 micro 風格 P/R/F1（每個標籤可出現多次）。

    - precision = sum_x min(pred(x), gold(x)) / sum(pred)
    - recall    = sum_x min(pred(x), gold(x)) / sum(gold)
    """
    norm = normalizer or (lambda z: z.strip().lower())
    p = _norm_list(predicted, norm)
    g = _norm_list(gold, norm)
    if not p and not g:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not p or not g:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = 0
    for k in p:
        overlap += min(p[k], g.get(k, 0))
    sp = sum(p.values())
    sg = sum(g.values())
    precision = overlap / sp if sp else 0.0
    recall = overlap / sg if sg else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def set_jaccard(predicted: list[str], gold: list[str], *, normalizer: Callable[[str], str] | None = None) -> float:
    norm = normalizer or (lambda z: z.strip().lower())
    ps = {norm(x) for x in predicted if norm(x)}
    gs = {norm(x) for x in gold if norm(x)}
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    return len(ps & gs) / len(ps | gs)
