from __future__ import annotations

import pytest


def test_rouge_score_smoke() -> None:
    pytest.importorskip("rouge_score")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    s = scorer.score("參考答案", "參考答案近似")
    assert s["rougeL"].fmeasure >= 0.0
