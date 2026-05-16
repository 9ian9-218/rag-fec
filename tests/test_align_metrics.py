from __future__ import annotations

from src.evaluation.align_metrics import (
    chunk_match,
    doc_match,
    entity_match,
    ordered_sources_from_context,
    recall_at_k_chunks,
    relation_match,
)


def test_entity_alias_rm() -> None:
    assert entity_match("RM Code", "Reed-Muller Codes")
    assert entity_match("Polar Code", "Polar码")


def test_relation_pair() -> None:
    assert relation_match(
        "BEC|RM Code",
        "BEC Channel|Reed-Muller Codes|Reed-Muller码在BEC信道上的性能被研究。",
    )


def test_doc_stem_match() -> None:
    gold = "doc-8c265c0e55db1ee795344f3f5df9c346"
    ret = "/home/u/data/raw/A_performance_comparison_of_polar_codes_and_Reed-Muller_codes.md"
    assert doc_match(gold, ret) or doc_match(
        "A_performance_comparison_of_polar_codes_and_Reed-Muller_codes.md", ret
    )


def test_chunk_match_by_source() -> None:
    gold = "A_performance_comparison_of_polar_codes_and_Reed-Muller_codes.md"
    src = "/data/raw/A_performance_comparison_of_polar_codes_and_Reed-Muller_codes.md"
    assert chunk_match(gold, chunk_id="chunk-abc", source_path=src)


def test_chunk_recall_from_context() -> None:
    ctx = "【片段 1】 來源: /data/raw/A_performance_comparison_of_polar_codes_and_Reed-Muller_codes.md\nbody"
    paths = ordered_sources_from_context(ctx)
    assert paths
    r = recall_at_k_chunks(
        [gold := "A_performance_comparison_of_polar_codes_and_Reed-Muller_codes.md"],
        ["chunk-x"],
        paths,
        1,
    )
    assert r == 1.0
