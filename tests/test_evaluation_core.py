from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_build_core_metrics_report(tmp_path: Path) -> None:
    from src.evaluation.runner import build_report_from_path

    row = {
        "id": "t1",
        "question": "q",
        "reference": "Polar 优于 RM",
        "prediction": "Polar 码更好",
        "k": 5,
        "multihop": True,
        "gold_entities": ["Polar", "RM"],
        "retrieved_entities_ranked": ["噪声", "Polar", "RM"],
        "gold_relations": ["Polar|优于|RM"],
        "retrieved_relations_ranked": ["Polar|优于|RM"],
        "gold_chunk_ids": ["c1"],
        "retrieved_chunk_ids_ranked": ["c2", "c1"],
        "gold_evidence_texts": ["Polar 优于 RM"],
        "retrieved_context": "表格顯示 Polar 优于 RM",
    }
    f = tmp_path / "eval.jsonl"
    f.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    report = build_report_from_path(f, eval_ks=(5,))
    assert "retrieval_at_k" in report
    assert report["retrieval_at_k"]["entities"]["recall_mean"] == 1.0
    assert "ragas" in report
    assert report["ragas"]["faithfulness_mean"] >= 0.0
    assert report["multihop"]["accuracy"] >= 0.0
