"""答案層指標：ROUGE、詞袋 F1、標準化後精確匹配等。"""

from __future__ import annotations

from typing import Any

from rouge_score import tokenizers as rouge_tokenizers

from src.evaluation.text_utils import normalize_answer


class _WhitespaceTokenizer(rouge_tokenizers.Tokenizer):
    """不依賴 [a-z0-9] 過濾，供中文等 Unicode 文本計算 ROUGE。"""

    def tokenize(self, text: str) -> list[str]:
        t = text.strip().lower()
        if not t:
            return []
        return [x for x in t.split() if x]


def _needs_unicode_rouge(reference: str, prediction: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in reference + prediction)


def _jieba_cut(text: str) -> list[str]:
    try:
        import jieba  # type: ignore[import-untyped]

        return [t.strip() for t in jieba.cut(text) if t.strip()]
    except ImportError:
        return [t for t in text.split() if t]


def token_f1(reference: str, prediction: str, *, use_jieba: bool = True) -> dict[str, float]:
    """基於分詞（可選 jieba）的詞集合 F1。"""
    ref_t = _jieba_cut(reference) if use_jieba else [t for t in reference.split() if t]
    pred_t = _jieba_cut(prediction) if use_jieba else [t for t in prediction.split() if t]
    if not ref_t and not pred_t:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ref_t or not pred_t:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    rs, ps = set(ref_t), set(pred_t)
    inter = rs & ps
    p = len(inter) / len(ps) if ps else 0.0
    r = len(inter) / len(rs) if rs else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def char_f1(reference: str, prediction: str) -> dict[str, float]:
    """字元級（去空白）多重集合 F1，對中文無分詞庫時更穩。"""
    ref_c = [c for c in normalize_answer(reference) if not c.isspace()]
    pred_c = [c for c in normalize_answer(prediction) if not c.isspace()]
    if not ref_c and not pred_c:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ref_c or not pred_c:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    from collections import Counter

    rc, pc = Counter(ref_c), Counter(pred_c)
    overlap = sum(min(pc[k], rc.get(k, 0)) for k in pc)
    sp, sr = sum(pc.values()), sum(rc.values())
    p = overlap / sp if sp else 0.0
    r = overlap / sr if sr else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def exact_match(reference: str, prediction: str) -> float:
    return 1.0 if normalize_answer(reference) == normalize_answer(prediction) else 0.0


def _rouge_friendly_text(s: str) -> str:
    """ROUGE 預設按空白分詞；純中文無空格時改為字間空格，避免指標全零。"""
    t = s.strip()
    if not t:
        return t
    if " " not in t and any("\u4e00" <= c <= "\u9fff" for c in t):
        return " ".join(t)
    return t


def rouge_bundle(
    reference: str,
    prediction: str,
    *,
    types: tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
    use_stemmer: bool = False,
) -> dict[str, dict[str, float]]:
    from rouge_score import rouge_scorer

    ref = _rouge_friendly_text(reference)
    pred = _rouge_friendly_text(prediction)
    if _needs_unicode_rouge(reference, prediction):
        scorer = rouge_scorer.RougeScorer(
            list(types),
            use_stemmer=False,
            tokenizer=_WhitespaceTokenizer(),
        )
    else:
        scorer = rouge_scorer.RougeScorer(list(types), use_stemmer=use_stemmer)
    s = scorer.score(ref, pred)
    out: dict[str, dict[str, float]] = {}
    for t in types:
        if t not in s:
            continue
        o = s[t]
        out[t] = {"f": float(o.fmeasure), "p": float(o.precision), "r": float(o.recall)}
    return out


def compute_answer_row(
    reference: str,
    prediction: str,
    *,
    rouge_types: tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
    use_stemmer: bool = False,
) -> dict[str, Any]:
    rb = rouge_bundle(reference, prediction, types=rouge_types, use_stemmer=use_stemmer)
    tf = token_f1(reference, prediction, use_jieba=True)
    cf = char_f1(reference, prediction)
    return {
        "exact_match": exact_match(reference, prediction),
        "token_f1": tf,
        "char_f1": cf,
        "rouge": rb,
    }
