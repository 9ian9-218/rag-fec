from __future__ import annotations

from src.storage.bge_rerank import minmax_normalize_scores


def test_minmax_normalize_scores_empty() -> None:
    assert minmax_normalize_scores([]) == []


def test_minmax_normalize_scores_single() -> None:
    assert minmax_normalize_scores([0.3]) == [1.0]


def test_minmax_normalize_scores_spread() -> None:
    out = minmax_normalize_scores([0.0, 0.5, 1.0])
    assert len(out) == 3
    assert min(out) == 0.0
    assert max(out) == 1.0


def test_minmax_normalize_scores_constant() -> None:
    out = minmax_normalize_scores([2.0, 2.0, 2.0])
    assert out == [1.0, 1.0, 1.0]
