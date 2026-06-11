from __future__ import annotations

from src.evaluation.context_utils import extract_chunk_sections
from src.evaluation.multihop_metrics import multihop_correct
from src.evaluation.ragas_metrics import build_ragas_contexts, build_ragas_reference, row_to_ragas_sample


def test_extract_chunk_sections() -> None:
    ctx = "【知識圖譜摘要】\n- 實體: X\n\n【片段 1】 來源: /a.md\n正文A\n\n【片段 2】 來源: /b.md\n正文B"
    body = extract_chunk_sections(ctx)
    assert "正文A" in body and "知識圖譜" not in body


def test_build_ragas_reference_merges_evidence() -> None:
    ref = build_ragas_reference(
        "Polar 优于 RM",
        reference_bullets=["取决于译码"],
        gold_evidence_texts=["实验表明 Polar 优于 RM"],
    )
    assert "Polar 优于 RM" in ref
    assert "取决于译码" in ref
    assert "实验表明 Polar 优于 RM" in ref


def test_build_ragas_contexts_splits_kg() -> None:
    ctx = "【知識圖譜摘要】\n- 實體: X\n\n【片段 1】 來源: /a.md\n正文A"
    contexts = build_ragas_contexts(ctx, kg_text="KG 摘要", chunks_only=True)
    assert len(contexts) == 2
    assert "正文A" in contexts[0]
    assert contexts[1] == "KG 摘要"


def test_row_to_ragas_sample_columns() -> None:
    sample = row_to_ragas_sample(
        question="q",
        reference="ref",
        prediction="pred",
        retrieved_context="ctx",
    )
    assert sample["user_input"] == "q"
    assert sample["response"] == "pred"
    assert sample["reference"] == "ref"
    assert isinstance(sample["retrieved_contexts"], list)


def test_multihop_q014_style() -> None:
    ref = "BP下Polar优于RM；RPA下RM可优于Polar；取决于译码方式，不能一概而论。"
    pred = "在相同参数下，Polar与RM的优劣取决于所选译码算法。BP译码时Polar通常优于RM；RPA译码时RM可优于Polar SCL。"
    assert (
        multihop_correct(
            ref,
            pred,
            aliases=["取决于译码器", "不能一概而论"],
        )
        == 1.0
    )
