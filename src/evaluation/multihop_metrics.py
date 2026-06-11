"""多跳問答準確率（要點/別名/字符級匹配）。"""

from __future__ import annotations

from src.evaluation.answer_metrics import char_f1, exact_match, token_f1
from src.evaluation.context_utils import split_claim_sentences
from src.evaluation.text_utils import normalize_answer, to_simplified_chinese


def _match_one_ref(ref: str, prediction: str, *, char_threshold: float, token_threshold: float) -> bool:
    ref = to_simplified_chinese(ref)
    prediction = to_simplified_chinese(prediction)
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
    # 短要点：关键词覆盖（预测可更长，非对称匹配）
    if len(ref_n) <= 80:
        try:
            import jieba  # type: ignore[import-untyped]

            toks = [normalize_answer(t) for t in jieba.cut(ref) if len(t.strip()) >= 2]
        except ImportError:
            toks = [ref_n]
        toks = [t for t in toks if t]
        if toks:
            hit = sum(1 for t in toks if t in pred_n)
            need = max(1, int(len(toks) * 0.45 + 0.5))
            if hit >= need:
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
    # 綜合題：要點/別名/首句任一命中即可（避免整段 reference 拉低匹配率）
    primary: list[str] = []
    if reference_bullets:
        primary.extend(reference_bullets)
    else:
        sents = split_claim_sentences(reference)
        if sents:
            primary.append(sents[0])
        primary.append(reference)
    if aliases:
        primary.extend(str(x) for x in aliases if str(x).strip())
    if any(
        _match_one_ref(r, prediction, char_threshold=char_f1_threshold, token_threshold=token_f1_threshold)
        for r in primary
    ):
        return 1.0
    return 0.0
