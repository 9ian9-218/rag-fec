"""拒答评分口径回归（负样本误答率 / 正样本误拒率）。"""

from __future__ import annotations

from src.evaluation.refusal_metrics import has_refusal_labels, looks_refused, score_rows

REFUSE_TEXT = "根据提供的检索材料，无法找到足够的信息来回答该问题。"
ANSWER_TEXT = "RPA 译码通过把码投影到陪集上、递归译码低阶 RM 码并聚合重建来完成译码。"


def test_looks_refused_detects_builtin_wording_and_variants() -> None:
    assert looks_refused(REFUSE_TEXT) is True
    assert looks_refused("材料中没有提及该算法的复杂度。") is True
    assert looks_refused("insufficient information to answer") is True
    assert looks_refused(ANSWER_TEXT) is False
    # 空答案视为未能回答
    assert looks_refused("") is True


def test_score_rows_separates_negative_and_positive_rates() -> None:
    rows = [
        {"id": "a", "label": "out_of_corpus", "expect": "refuse", "prediction": REFUSE_TEXT},
        {"id": "b", "label": "out_of_corpus", "expect": "refuse", "prediction": "LDPC 的密度进化阈值约为 0.43。"},
        {"id": "c", "label": "false_premise", "expect": "refuse_or_correct", "prediction": REFUSE_TEXT},
        {"id": "d", "label": "in_corpus", "expect": "answer", "prediction": ANSWER_TEXT},
        {"id": "e", "label": "in_corpus", "expect": "answer", "prediction": REFUSE_TEXT},
        {"id": "f", "label": "partial", "expect": "either", "prediction": REFUSE_TEXT},
    ]
    summary, details = score_rows(rows)

    # 负样本 3 题中有 1 题未拒答 -> 误答率 1/3
    assert abs(summary["negative_misanswer_rate"] - round(1 / 3, 4)) < 1e-9
    # 正样本 2 题中有 1 题误拒 -> 1/2
    assert abs(summary["positive_false_refusal_rate"] - 0.5) < 1e-9
    # partial 不计入误答率
    assert summary["counts"]["partial"] == 1
    assert summary["negative_misanswer_rate"] != 1.0
    assert len(details) == 6


def test_score_rows_flags_long_refusal_as_suspect() -> None:
    long_refusal = REFUSE_TEXT + "另外，" + ("补充内容" * 120)
    summary, _ = score_rows(
        [{"id": "x", "label": "out_of_corpus", "expect": "refuse", "prediction": long_refusal}]
    )
    assert summary["suspect_rows"] == ["x"]


def test_has_refusal_labels_gates_report_block() -> None:
    assert has_refusal_labels([{"label": "out_of_corpus"}]) is True
    assert has_refusal_labels([{"question": "q"}]) is False
    assert has_refusal_labels([{"label": "  "}]) is False


def test_evaluate_report_includes_refusal_block_for_labeled_rows() -> None:
    """整份评测报告与 score_refusal.py 共用同一口径：带 label 的行自动出 refusal 区块。"""
    from src.evaluation import runner

    rows = [
        {"id": "a", "label": "out_of_corpus", "expect": "refuse", "question": "q1", "prediction": REFUSE_TEXT},
        {"id": "b", "label": "in_corpus", "expect": "answer", "question": "q2", "prediction": ANSWER_TEXT},
    ]
    report = runner.build_report(rows, ragas_llm=False, max_detail_rows=10)

    assert report["refusal"]["negative_misanswer_rate"] == 0.0
    assert report["refusal"]["positive_false_refusal_rate"] == 0.0
    assert report["refusal"]["counts"] == {"in_corpus": 1, "out_of_corpus": 1}


def test_evaluate_report_omits_refusal_block_without_labels() -> None:
    from src.evaluation import runner

    report = runner.build_report(
        [{"id": "a", "question": "q1", "prediction": ANSWER_TEXT}], ragas_llm=False
    )
    assert "refusal" not in report


def test_score_rows_returns_none_when_a_label_class_is_absent() -> None:
    """历史缺陷：只有 in_corpus 的评估集会报 negative_misanswer_rate=1.0（伪值）。"""
    only_positive, _ = score_rows(
        [{"id": "a", "label": "in_corpus", "expect": "answer", "prediction": ANSWER_TEXT}]
    )
    assert only_positive["negative_misanswer_rate"] is None
    assert only_positive["positive_false_refusal_rate"] == 0.0

    only_negative, _ = score_rows(
        [{"id": "b", "label": "out_of_corpus", "expect": "refuse", "prediction": REFUSE_TEXT}]
    )
    assert only_negative["negative_misanswer_rate"] == 0.0
    assert only_negative["positive_false_refusal_rate"] is None
