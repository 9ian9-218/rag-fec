from __future__ import annotations

import pytest

from src.evaluation.ranked_metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_recall_precision_ndcg_at_k() -> None:
    gold = ["a", "b"]
    ranked = ["x", "a", "b", "y"]
    assert recall_at_k(gold, ranked, 3) == 1.0
    assert precision_at_k(gold, ranked, 3) == pytest.approx(2 / 3)
    assert ndcg_at_k(gold, ranked, 3) > 0.5
