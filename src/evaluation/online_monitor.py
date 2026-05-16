"""線上查詢遙測：圖譜空召回、Chunk 截斷、重排過濾、延遲、Token 估算。"""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("evaluation.online_monitor")

GRAPH_MODES = frozenset({"local", "global", "hybrid", "mix"})
DEFAULT_METRICS_LOG = Path("data/logs/query_metrics.jsonl")

rerank_stats_ctx: ContextVar[dict[str, int] | None] = ContextVar("rerank_stats_ctx", default=None)


@dataclass
class QueryTelemetry:
    question: str
    mode: str
    latency_ms: float
    graph_empty: bool = False
    graph_empty_rate_component: float = 0.0
    chunk_truncation_rate: float = 0.0
    rerank_filter_rate: float = 0.0
    entities_found: int = 0
    relations_found: int = 0
    entities_after_truncation: int = 0
    relations_after_truncation: int = 0
    merged_chunks_count: int = 0
    final_chunks_count: int = 0
    reference_chunks_before_trim: int = 0
    reference_chunks_after_trim: int = 0
    tokens_entities: int = 0
    tokens_relations: int = 0
    tokens_chunks: int = 0
    tokens_kg_text: int = 0
    tokens_total_estimated: int = 0
    rerank_candidates: int = 0
    rerank_returned: int = 0
    rerank_below_min_score: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_rerank_stats(
    *,
    candidates: int,
    returned: int,
    below_min_score: int = 0,
) -> None:
    rerank_stats_ctx.set(
        {
            "candidates": int(candidates),
            "returned": int(returned),
            "below_min_score": int(below_min_score),
        }
    )


def clear_rerank_stats() -> None:
    rerank_stats_ctx.set(None)


def _estimate_tokens(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(t))
    except Exception:
        return max(1, len(t) // 4)


def _payload_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    inner = bundle.get("data") or {}
    if isinstance(inner, dict) and isinstance(inner.get("data"), dict):
        inner = inner["data"]
    if not isinstance(inner, dict):
        inner = {}
    return inner


def _processing_info(bundle: dict[str, Any]) -> dict[str, Any]:
    meta = bundle.get("metadata") or {}
    if not isinstance(meta, dict):
        return {}
    pi = meta.get("processing_info") or {}
    return pi if isinstance(pi, dict) else {}


def _entity_relation_names(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    ents: list[str] = []
    rels: list[str] = []
    for e in payload.get("entities") or []:
        if isinstance(e, dict):
            n = e.get("entity_name") or e.get("name") or e.get("id")
            if n:
                ents.append(str(n))
        elif e:
            ents.append(str(e))
    for r in payload.get("relationships") or []:
        if isinstance(r, dict):
            s = r.get("src_id") or r.get("source") or ""
            t = r.get("tgt_id") or r.get("target") or ""
            rels.append(f"{s}|{t}")
        elif r:
            rels.append(str(r))
    return ents, rels


def build_telemetry(
    *,
    question: str,
    mode: str,
    bundle: dict[str, Any] | None,
    latency_ms: float,
    kg_text: str = "",
    reference_chunks_before: int = 0,
    reference_chunks_after: int = 0,
    min_rerank_score: float = 0.0,
) -> QueryTelemetry:
    payload = _payload_from_bundle(bundle or {})
    pi = _processing_info(bundle or {})
    ents, rels = _entity_relation_names(payload)

    te = int(pi.get("total_entities_found") or len(ents))
    tr = int(pi.get("total_relations_found") or len(rels))
    ea = int(pi.get("entities_after_truncation") or len(ents))
    ra = int(pi.get("relations_after_truncation") or len(rels))
    merged = int(pi.get("merged_chunks_count") or pi.get("total_chunks_found") or 0)
    final = int(pi.get("final_chunks_count") or len(payload.get("chunks") or []))

    graph_empty = False
    if mode in GRAPH_MODES:
        graph_empty = te == 0 and tr == 0

    chunk_truncation_rate = 0.0
    if merged > 0 and final < merged:
        chunk_truncation_rate = (merged - final) / merged
    elif reference_chunks_before > 0 and reference_chunks_after < reference_chunks_before:
        chunk_truncation_rate = (reference_chunks_before - reference_chunks_after) / reference_chunks_before

    rs = rerank_stats_ctx.get() or {}
    rerank_candidates = int(rs.get("candidates") or 0)
    rerank_returned = int(rs.get("returned") or 0)
    below_min = int(rs.get("below_min_score") or 0)
    rerank_filter_rate = 0.0
    if rerank_candidates > 0:
        rerank_filter_rate = max(0.0, (rerank_candidates - rerank_returned) / rerank_candidates)
        if below_min > 0:
            rerank_filter_rate = max(rerank_filter_rate, below_min / rerank_candidates)
    elif merged > 0 and final < merged and min_rerank_score > 0:
        rerank_filter_rate = (merged - final) / merged

    ent_text = "\n".join(ents)
    rel_text = "\n".join(rels)
    chunk_text = "\n".join(
        str(c.get("content") or "") for c in (payload.get("chunks") or []) if isinstance(c, dict)
    )
    tok_e = _estimate_tokens(ent_text)
    tok_r = _estimate_tokens(rel_text)
    tok_c = _estimate_tokens(chunk_text)
    tok_kg = _estimate_tokens(kg_text)

    return QueryTelemetry(
        question=question[:500],
        mode=mode,
        latency_ms=round(latency_ms, 2),
        graph_empty=graph_empty,
        graph_empty_rate_component=1.0 if graph_empty else 0.0,
        chunk_truncation_rate=round(chunk_truncation_rate, 4),
        rerank_filter_rate=round(rerank_filter_rate, 4),
        entities_found=te,
        relations_found=tr,
        entities_after_truncation=ea,
        relations_after_truncation=ra,
        merged_chunks_count=merged,
        final_chunks_count=final,
        reference_chunks_before_trim=reference_chunks_before,
        reference_chunks_after_trim=reference_chunks_after,
        tokens_entities=tok_e,
        tokens_relations=tok_r,
        tokens_chunks=tok_c,
        tokens_kg_text=tok_kg,
        tokens_total_estimated=tok_e + tok_r + tok_c + tok_kg,
        rerank_candidates=rerank_candidates,
        rerank_returned=rerank_returned,
        rerank_below_min_score=below_min,
    )


def append_telemetry(
    telemetry: QueryTelemetry,
    *,
    log_path: Path | None = None,
) -> None:
    path = log_path or DEFAULT_METRICS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(telemetry.to_dict(), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.info(
        "eval_telemetry mode=%s latency_ms=%.1f graph_empty=%s chunk_trunc=%.3f rerank_filter=%.3f tokens=%d",
        telemetry.mode,
        telemetry.latency_ms,
        telemetry.graph_empty,
        telemetry.chunk_truncation_rate,
        telemetry.rerank_filter_rate,
        telemetry.tokens_total_estimated,
    )


def aggregate_metrics_jsonl(path: Path) -> dict[str, Any]:
    """匯總 ``query_metrics.jsonl`` 為監控面板用摘要。"""
    if not path.is_file():
        return {"count": 0}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return {"count": 0}

    def mean(key: str) -> float:
        vals = [float(r[key]) for r in rows if key in r and isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    graph_modes = [r for r in rows if r.get("mode") in GRAPH_MODES]
    return {
        "count": len(rows),
        "latency_ms_mean": mean("latency_ms"),
        "latency_ms_p95": _percentile([float(r.get("latency_ms", 0)) for r in rows], 0.95),
        "graph_empty_rate": mean("graph_empty_rate_component") if graph_modes else 0.0,
        "chunk_truncation_rate_mean": mean("chunk_truncation_rate"),
        "rerank_filter_rate_mean": mean("rerank_filter_rate"),
        "tokens_total_mean": mean("tokens_total_estimated"),
    }


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(p * len(s)))
    return s[idx]


class QueryTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
