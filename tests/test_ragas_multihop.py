from __future__ import annotations

from unittest.mock import patch

from src.evaluation.multihop_metrics import multihop_correct
from src.evaluation.ragas_metrics import compute_ragas_row


def test_ragas_row_delegates_to_batch() -> None:
    fake_scores = [{"context_recall": 0.9, "context_precision": 0.8, "faithfulness": 0.7}]
    with patch("src.evaluation.ragas_metrics.compute_ragas_batch", return_value=fake_scores) as mocked:
        row = compute_ragas_row(
            question="Polar 码和 RM 码哪个好?",
            reference="Polar 码优于 RM 码",
            prediction="Polar 码性能更好",
            retrieved_context="实验表明 Polar 码优于 RM 码",
            gold_evidence_texts=["Polar 码优于 RM 码"],
            llm=object(),
        )
    mocked.assert_called_once()
    assert row["context_recall"] == 0.9
    assert row["faithfulness"] == 0.7


def test_multihop_correct() -> None:
    assert multihop_correct("Polar 优于 RM", "Polar 优于 RM 码") == 1.0
    assert multihop_correct("答案A", "完全不同") == 0.0
    assert (
        multihop_correct(
            "取决于译码，不能一概而论",
            "性能取决于信道与译码算法，无法一概而论",
            aliases=["不能一概而论"],
        )
        == 1.0
    )
