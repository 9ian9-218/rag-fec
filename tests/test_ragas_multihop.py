from __future__ import annotations

from src.evaluation.multihop_metrics import multihop_correct
from src.evaluation.ragas_metrics import compute_ragas_row


def test_ragas_heuristic() -> None:
    row = compute_ragas_row(
        reference="Polar 码优于 RM 码",
        prediction="Polar 码性能更好",
        retrieved_context="实验表明 Polar 码优于 RM 码",
        gold_evidence_texts=["Polar 码优于 RM 码"],
    )
    assert row["context_recall"] >= 0.5
    assert row["faithfulness"] >= 0.0


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
