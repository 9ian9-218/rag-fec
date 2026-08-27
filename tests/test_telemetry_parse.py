"""T07 遥测 bundle 解析：与 LightRAG 1.5.4 convert_to_user_format 输出对齐。"""

from __future__ import annotations

from src.evaluation.online_monitor import build_telemetry


_ONE_ENTITY = {"entity_name": "RM 码", "entity_type": "Code", "description": "Reed-Muller 纠错码"}
_ONE_REL = {"src_id": "RM 码", "tgt_id": "译码", "description": "描述", "keywords": "译码, 流程"}
_ONE_CHUNK = {"chunk_id": "c1", "content": "Reed-Muller 译码流程：第一步计算伴随式，第二步迭代译码。"}


def _bundle(entities=None, relationships=None, chunks=None) -> dict:
    return {
        "status": "success",
        "message": "Query processed successfully",
        "data": {
            "entities": entities or [],
            "relationships": relationships or [],
            "chunks": chunks or [],
        },
        "metadata": {"query_mode": "local"},
    }


def test_telemetry_counts_real_bundle_contents() -> None:
    """有结果时实体/关系/引文计数与 tokens 必须反映真实检索内容（非 0）。"""
    t = build_telemetry(
        question="RM 码译码流程",
        mode="local",
        bundle=_bundle([_ONE_ENTITY], [_ONE_REL], [_ONE_CHUNK]),
        latency_ms=123.4,
    )
    assert t.entities_found == 1
    assert t.relations_found == 1
    assert t.merged_chunks_count == 1
    assert t.final_chunks_count == 1
    assert t.tokens_chunks > 0
    assert t.tokens_total_estimated > 0
    assert t.graph_empty is False


def test_telemetry_graph_empty_only_when_nothing_found() -> None:
    """图模式空召回标记只在实体与关系均为 0 时置位。"""
    t = build_telemetry(
        question="不存在的内容",
        mode="local",
        bundle=_bundle([], [], []),
        latency_ms=9.0,
    )
    assert t.entities_found == 0
    assert t.relations_found == 0
    assert t.graph_empty is True


def test_telemetry_chunk_count_fallback_without_processing_info() -> None:
    """LightRAG 1.5.4 无 processing_info 时，merged 计数用实际 chunks 长度兜底。"""
    t = build_telemetry(
        question="有引文",
        mode="naive",
        bundle=_bundle(chunks=[_ONE_CHUNK]),
        latency_ms=5.0,
    )
    assert t.merged_chunks_count == 1
    assert t.final_chunks_count == 1