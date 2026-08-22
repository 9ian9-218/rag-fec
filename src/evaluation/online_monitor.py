"""線上查詢遙測：圖譜空召回、Chunk 截斷、重排過濾、延遲、Token 估算。

存储策略（T5）：
- 不再写 JSONL。
- 请求路径只把遥测放入内存队列。
- 后台线程批量写入 SQLite。
- SQLite 开启 WAL + 批量事务。
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import get_settings
from src.storage.redis_client import get_sync_redis_client
from src.utils.logger import get_logger

logger = get_logger("evaluation.online_monitor")

GRAPH_MODES = frozenset({"local", "global", "hybrid", "mix"})
DEFAULT_METRICS_LOG = Path("data/logs/query_metrics.jsonl")
DEFAULT_FEEDBACK_LOG = Path("data/logs/feedback_metrics.jsonl")

# 内存队列：请求路径只入队，不写库。
_TELEMETRY_QUEUE: "queue.Queue[tuple[str, dict[str, Any]]]" = queue.Queue(maxsize=10000)
_WRITER_THREAD: threading.Thread | None = None
_WRITER_THREAD_LOCK = threading.Lock()
_DROPPED_COUNT = 0
_DROPPED_LOCK = threading.Lock()

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


@dataclass
class FeedbackTelemetry:
    """用戶回答反饋遙測。"""

    question: str
    answer: str
    feedback: str  # "correct" or "wrong"
    session_id: str | None = None
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


# ---------- SQLite 存储 ----------

def _sqlite_path() -> Path:
    s = get_settings()
    root = Path(s.paths.project_root).resolve()
    return root / s.paths.sqlite_path


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS query_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            mode TEXT,
            latency_ms REAL,
            graph_empty INTEGER,
            graph_empty_rate_component REAL,
            chunk_truncation_rate REAL,
            rerank_filter_rate REAL,
            entities_found INTEGER,
            relations_found INTEGER,
            entities_after_truncation INTEGER,
            relations_after_truncation INTEGER,
            merged_chunks_count INTEGER,
            final_chunks_count INTEGER,
            reference_chunks_before_trim INTEGER,
            reference_chunks_after_trim INTEGER,
            tokens_entities INTEGER,
            tokens_relations INTEGER,
            tokens_chunks INTEGER,
            tokens_kg_text INTEGER,
            tokens_total_estimated INTEGER,
            rerank_candidates INTEGER,
            rerank_returned INTEGER,
            rerank_below_min_score INTEGER,
            timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            feedback TEXT,
            session_id TEXT,
            timestamp TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_query_telemetry_timestamp ON query_telemetry(timestamp);
        CREATE INDEX IF NOT EXISTS idx_feedback_telemetry_timestamp ON feedback_telemetry(timestamp);
        """
    )


def _publish_redis_stream(rows: list[tuple[str, dict[str, Any]]]) -> None:
    """将遥测事件发布到 Redis Stream（可选，供多实例汇聚）。"""
    client = get_sync_redis_client()
    if client is None:
        return
    try:
        for kind, data in rows:
            client.xadd(
                "rag:telemetry",
                {
                    "kind": kind,
                    "payload_json": json.dumps(data, ensure_ascii=False),
                },
            )
    except Exception:
        logger.warning("发布遥测到 Redis Stream 失败", exc_info=True)


def _write_batch_to_sqlite(rows: list[tuple[str, dict[str, Any]]]) -> None:
    if not rows:
        return
    db_path = _sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        _init_schema(conn)
        query_rows = [d for kind, d in rows if kind == "query"]
        feedback_rows = [d for kind, d in rows if kind == "feedback"]
        with conn:
            if query_rows:
                conn.executemany(
                    """
                    INSERT INTO query_telemetry (
                        question, mode, latency_ms, graph_empty, graph_empty_rate_component,
                        chunk_truncation_rate, rerank_filter_rate, entities_found, relations_found,
                        entities_after_truncation, relations_after_truncation, merged_chunks_count,
                        final_chunks_count, reference_chunks_before_trim, reference_chunks_after_trim,
                        tokens_entities, tokens_relations, tokens_chunks, tokens_kg_text,
                        tokens_total_estimated, rerank_candidates, rerank_returned,
                        rerank_below_min_score, timestamp
                    ) VALUES (
                        :question, :mode, :latency_ms, :graph_empty, :graph_empty_rate_component,
                        :chunk_truncation_rate, :rerank_filter_rate, :entities_found, :relations_found,
                        :entities_after_truncation, :relations_after_truncation, :merged_chunks_count,
                        :final_chunks_count, :reference_chunks_before_trim, :reference_chunks_after_trim,
                        :tokens_entities, :tokens_relations, :tokens_chunks, :tokens_kg_text,
                        :tokens_total_estimated, :rerank_candidates, :rerank_returned,
                        :rerank_below_min_score, :timestamp
                    )
                    """,
                    query_rows,
                )
            if feedback_rows:
                conn.executemany(
                    """
                    INSERT INTO feedback_telemetry (question, answer, feedback, session_id, timestamp)
                    VALUES (:question, :answer, :feedback, :session_id, :timestamp)
                    """,
                    feedback_rows,
                )
    finally:
        conn.close()

    _publish_redis_stream(rows)


def _writer_loop() -> None:
    while True:
        item = _TELEMETRY_QUEUE.get()
        if item is None:
            break
        batch = [item]
        while len(batch) < 100:
            try:
                batch.append(_TELEMETRY_QUEUE.get_nowait())
            except queue.Empty:
                break
        try:
            _write_batch_to_sqlite(batch)
        except Exception:
            logger.exception("批量写入遥测 SQLite 失败，已丢弃 %d 条", len(batch))


def _ensure_writer() -> None:
    global _WRITER_THREAD
    with _WRITER_THREAD_LOCK:
        if _WRITER_THREAD is not None and _WRITER_THREAD.is_alive():
            return
        _WRITER_THREAD = threading.Thread(target=_writer_loop, name="telemetry-sqlite-writer", daemon=True)
        _WRITER_THREAD.start()


def _incr_dropped() -> None:
    global _DROPPED_COUNT
    with _DROPPED_LOCK:
        _DROPPED_COUNT += 1


def append_telemetry(
    telemetry: QueryTelemetry,
    *,
    log_path: Path | None = None,
) -> None:
    _ensure_writer()
    try:
        _TELEMETRY_QUEUE.put_nowait(("query", telemetry.to_dict()))
    except queue.Full:
        _incr_dropped()
        logger.warning("遥测队列已满，丢弃 query telemetry")
        return
    logger.info(
        "eval_telemetry mode=%s latency_ms=%.1f graph_empty=%s chunk_trunc=%.3f rerank_filter=%.3f tokens=%d",
        telemetry.mode,
        telemetry.latency_ms,
        telemetry.graph_empty,
        telemetry.chunk_truncation_rate,
        telemetry.rerank_filter_rate,
        telemetry.tokens_total_estimated,
    )


def append_feedback(
    telemetry: FeedbackTelemetry,
    *,
    log_path: Path | None = None,
) -> None:
    _ensure_writer()
    try:
        _TELEMETRY_QUEUE.put_nowait(("feedback", telemetry.to_dict()))
    except queue.Full:
        _incr_dropped()
        logger.warning("遥测队列已满，丢弃 feedback telemetry")
        return
    logger.info("feedback received: %s", telemetry.feedback)


def flush_telemetry() -> None:
    """将当前内存队列中的遥测立即写入 SQLite（测试/停机时使用）。"""
    rows: list[tuple[str, dict[str, Any]]] = []
    while True:
        try:
            rows.append(_TELEMETRY_QUEUE.get_nowait())
        except queue.Empty:
            break
    if rows:
        _write_batch_to_sqlite(rows)


def stop_telemetry_writer() -> None:
    """停止后台 writer 并 flush 剩余数据。"""
    global _WRITER_THREAD
    flush_telemetry()
    with _WRITER_THREAD_LOCK:
        thread = _WRITER_THREAD
        _WRITER_THREAD = None
    if thread is not None and thread.is_alive():
        _TELEMETRY_QUEUE.put(None)
        thread.join(timeout=5)


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(p * len(s)))
    return s[idx]


def aggregate_metrics_sqlite() -> dict[str, Any]:
    db_path = _sqlite_path()
    if not db_path.is_file():
        return {"count": 0}
    conn = sqlite3.connect(str(db_path))
    try:
        _init_schema(conn)
        rows = conn.execute(
            "SELECT latency_ms, graph_empty_rate_component, chunk_truncation_rate, rerank_filter_rate, tokens_total_estimated, mode FROM query_telemetry"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return {"count": 0}

    latencies = [float(r[0]) for r in rows]
    graph_modes = [r for r in rows if r[5] in GRAPH_MODES]
    return {
        "count": len(rows),
        "latency_ms_mean": sum(latencies) / len(latencies),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "graph_empty_rate": (
            sum(float(r[1]) for r in graph_modes) / len(graph_modes) if graph_modes else 0.0
        ),
        "chunk_truncation_rate_mean": sum(float(r[2]) for r in rows) / len(rows),
        "rerank_filter_rate_mean": sum(float(r[3]) for r in rows) / len(rows),
        "tokens_total_mean": sum(float(r[4]) for r in rows) / len(rows),
    }


def aggregate_feedback_sqlite() -> dict[str, Any]:
    db_path = _sqlite_path()
    if not db_path.is_file():
        return {"count": 0, "correct": 0, "wrong": 0}
    conn = sqlite3.connect(str(db_path))
    try:
        _init_schema(conn)
        rows = conn.execute("SELECT feedback FROM feedback_telemetry").fetchall()
    finally:
        conn.close()
    if not rows:
        return {"count": 0, "correct": 0, "wrong": 0}
    correct = sum(1 for r in rows if r[0] == "correct")
    wrong = sum(1 for r in rows if r[0] == "wrong")
    return {
        "count": len(rows),
        "correct": correct,
        "wrong": wrong,
        "correct_rate": round(correct / len(rows), 4) if rows else 0.0,
    }


# 保留旧 JSONL 聚合函数，避免外部调用方直接报错；但新数据不再写 JSONL。
def aggregate_metrics_jsonl(path: Path) -> dict[str, Any]:
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


def aggregate_feedback_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"count": 0, "correct": 0, "wrong": 0}
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
        return {"count": 0, "correct": 0, "wrong": 0}

    correct = sum(1 for r in rows if r.get("feedback") == "correct")
    wrong = sum(1 for r in rows if r.get("feedback") == "wrong")
    return {
        "count": len(rows),
        "correct": correct,
        "wrong": wrong,
        "correct_rate": round(correct / len(rows), 4) if rows else 0.0,
    }


class QueryTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0
