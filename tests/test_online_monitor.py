from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.online_monitor import (
    aggregate_metrics_jsonl,
    build_telemetry,
    set_rerank_stats,
)


def test_build_telemetry_graph_empty() -> None:
    bundle = {
        "status": "success",
        "metadata": {
            "processing_info": {
                "total_entities_found": 0,
                "total_relations_found": 0,
                "merged_chunks_count": 10,
                "final_chunks_count": 4,
            }
        },
        "data": {"chunks": [{"content": "x" * 100}], "entities": [], "relationships": []},
    }
    set_rerank_stats(candidates=10, returned=6, below_min_score=2)
    t = build_telemetry(
        question="q",
        mode="mix",
        bundle=bundle,
        latency_ms=120.5,
        min_rerank_score=0.2,
    )
    assert t.graph_empty is True
    assert t.chunk_truncation_rate == 0.6
    assert t.rerank_filter_rate >= 0.4


def test_aggregate_metrics(tmp_path: Path) -> None:
    log = tmp_path / "m.jsonl"
    log.write_text(
        json.dumps({"latency_ms": 100, "graph_empty_rate_component": 1, "chunk_truncation_rate": 0.5, "rerank_filter_rate": 0.1, "tokens_total_estimated": 1000, "mode": "mix"})
        + "\n"
        + json.dumps({"latency_ms": 200, "graph_empty_rate_component": 0, "chunk_truncation_rate": 0.0, "rerank_filter_rate": 0.0, "tokens_total_estimated": 500, "mode": "mix"})
        + "\n",
        encoding="utf-8",
    )
    s = aggregate_metrics_jsonl(log)
    assert s["count"] == 2
    assert s["latency_ms_mean"] == 150.0
