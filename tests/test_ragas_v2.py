from __future__ import annotations

from src.evaluation.context_utils import extract_chunk_sections
from src.evaluation.multihop_metrics import multihop_correct
from src.evaluation.ragas_metrics import compute_ragas_row, context_recall


def test_context_recall_not_always_one() -> None:
    r = context_recall(
        "短码长接近，长码 Polar 优势明显",
        "only unrelated english text about cats",
        gold_evidence_texts=["polar codes win at long block length"],
    )
    assert r < 1.0


def test_extract_chunk_sections() -> None:
    ctx = "【知識圖譜摘要】\n- 實體: X\n\n【片段 1】 來源: /a.md\n正文A\n\n【片段 2】 來源: /b.md\n正文B"
    body = extract_chunk_sections(ctx)
    assert "正文A" in body and "知識圖譜" not in body


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


def test_faithfulness_claim_improved() -> None:
    rg = compute_ragas_row(
        reference="N_max 设为 ceil(m/2)",
        prediction="we set N_max = ceil(m/2) in practice",
        retrieved_context="we set N_max = ceil(m/2) for Reed-Muller decoding",
        use_embedding_faithfulness=False,
    )
    assert rg["faithfulness_claim"] >= 0.5
