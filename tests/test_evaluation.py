from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_rouge_score_smoke() -> None:
    pytest.importorskip("rouge_score")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    s = scorer.score("參考答案", "參考答案近似")
    assert s["rougeL"].fmeasure >= 0.0


def test_normalize_answer() -> None:
    from src.evaluation.text_utils import normalize_answer

    assert "test" in normalize_answer("The Test.")


def test_retrieval_prf_and_mrr() -> None:
    from src.evaluation.retrieval_metrics import hit_at_k, reciprocal_rank, retrieval_precision_recall_f1

    prf = retrieval_precision_recall_f1(["a", "b"], ["b", "c"])
    assert prf["precision"] == 0.5
    assert prf["recall"] == 0.5
    assert reciprocal_rank(["a"], ["x", "a", "y"]) == 0.5
    assert hit_at_k(["a"], ["x", "a"], 2) == 1.0


def test_build_report_from_sample_file() -> None:
    pytest.importorskip("rouge_score")
    from src.evaluation.runner import build_report_from_path

    p = Path(__file__).resolve().parent / "fixtures" / "eval_sample_full.jsonl"
    report = build_report_from_path(p, max_detail_rows=5, include_answer=True)
    assert report["counts"]["rows_total"] >= 1
    assert "answer" in report
    assert "rouge_avg" in report["answer"]
    assert "retrieval" in report
    assert "graph" in report


def test_build_report_answer_only(tmp_path: Path) -> None:
    pytest.importorskip("rouge_score")
    from src.evaluation.runner import build_report_from_path

    f = tmp_path / "p.jsonl"
    f.write_text(
        json.dumps(
            {"question": "q", "reference": "hello world", "prediction": "hello there world"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    r = build_report_from_path(
        f, include_answer=True, include_retrieval=False, include_graph=False
    )
    assert r["counts"]["answer_evaluated"] == 1
    assert "retrieval" not in r
    assert "graph" not in r
