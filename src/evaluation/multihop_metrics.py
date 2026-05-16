"""多跳問答準確率（要點/別名/字符級匹配）。"""

from __future__ import annotations

from src.evaluation.answer_metrics import char_f1, exact_match, token_f1
from src.evaluation.context_utils import split_claim_sentences
from src.evaluation.text_utils import normalize_answer


def _match_one_ref(ref: str, prediction: str, *, char_threshold: float, token_threshold: float) -> bool:
    if exact_match(ref, prediction) >= 1.0:
        return True
    if char_f1(ref, prediction)["f1"] >= char_threshold:
        return True
    if token_f1(ref, prediction, use_jieba=True)["f1"] >= token_threshold:
        return True
    pred_n = normalize_answer(prediction)
    ref_n = normalize_answer(ref)
    if ref_n and pred_n and ref_n in pred_n:
        return True
    # 要點詞覆蓋：參考中較長詞在預測中出現
    ref_toks = [t for t in ref_n.split() if len(t) >= 2]
    if len(ref_toks) >= 2:
        hit = sum(1 for t in ref_toks if t in pred_n)
        if hit / len(ref_toks) >= 0.6:
            return True
    return False


def multihop_correct(
    reference: str,
    prediction: str,
    *,
    aliases: list[str] | None = None,
    reference_bullets: list[str] | None = None,
    char_f1_threshold: float = 0.42,
    token_f1_threshold: float = 0.42,
) -> float:
    """
    多跳題：參考/別名/要點任一匹配即 1。
    默認閾值較 v1 寬鬆，適合中文長答案概括。
    """
    refs: list[str] = []
    if reference_bullets:
        refs.extend(str(x) for x in reference_bullets if str(x).strip())
    refs.append(reference)
    refs.extend(str(x) for x in (aliases or []) if str(x).strip())
    # 參考答案拆句作為軟要點（綜合題）
    refs.extend(split_claim_sentences(reference))
    seen: set[str] = set()
    uniq: list[str] = []
    for r in refs:
        k = normalize_answer(r)
        if k and k not in seen and len(k) >= 4:
            seen.add(k)
            uniq.append(r)
    if not uniq:
        return 0.0
    # 綜合題：至少一條核心要點命中即可（reference 首句 + bullets 優先）
    primary = []
    if reference_bullets:
        primary.extend(reference_bullets)
    else:
        primary.append(reference)
    if any(
        _match_one_ref(r, prediction, char_threshold=char_f1_threshold, token_threshold=token_f1_threshold)
        for r in primary
    ):
        return 1.0
    return 0.0
