"""金標與系統輸出的對齊匹配（文檔 stem、實體別名、關係端點），避免格式不一致導致評分偏低。"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable

# 常見別名組（FEC / 兩篇論文語料）
_ENTITY_ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"polar code", "polar codes", "polar码", "极化码", "polar"}),
    frozenset(
        {
            "rm code",
            "rm codes",
            "reed-muller code",
            "reed-muller codes",
            "reed muller code",
            "reed muller codes",
            "reed-muller码",
            "reed-muller",
        }
    ),
    frozenset({"bec", "bec channel", "binary erasure channel", "二进制擦除信道"}),
    frozenset({"rpa", "rpa decoder", "rpa algorithm", "recursive projection-aggregation decoding"}),
    frozenset({"scl", "successive cancellation list", "scl decoder"}),
    frozenset({"fht", "fast hadamard transform"}),
)


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def normalize_entity(name: str) -> str:
    s = _collapse_ws(str(name).lower())
    s = re.sub(r"[（(].*?[）)]", "", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s, flags=re.UNICODE)
    return _collapse_ws(s)


def entity_match(gold: str, retrieved: str) -> bool:
    a, b = normalize_entity(gold), normalize_entity(retrieved)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    for group in _ENTITY_ALIAS_GROUPS:
        if a in group and b in group:
            return True
    return False


def parse_relation_pair(rel: str) -> tuple[str, str] | None:
    parts = [p.strip() for p in str(rel).split("|") if p.strip()]
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def relation_match(gold_rel: str, retrieved_rel: str) -> bool:
    g = parse_relation_pair(gold_rel)
    r = parse_relation_pair(retrieved_rel)
    if not g or not r:
        return False
    return (entity_match(g[0], r[0]) and entity_match(g[1], r[1])) or (
        entity_match(g[0], r[1]) and entity_match(g[1], r[0])
    )


def doc_stem(doc_id: str) -> str:
    s = str(doc_id).strip().replace("\\", "/")
    if s.startswith("doc-"):
        return s.lower()
    name = Path(s).name.lower()
    if name.endswith(".md"):
        return name[: -3]
    return name


def doc_keys(doc_id: str) -> set[str]:
    s = str(doc_id).strip()
    keys = {doc_stem(s)}
    if s.startswith("doc-"):
        keys.add(s.lower())
    keys.add(Path(s.replace("\\", "/")).name.lower())
    return {k for k in keys if k}


def doc_match(gold_doc: str, retrieved_doc: str) -> bool:
    g_keys = doc_keys(gold_doc)
    r_keys = doc_keys(retrieved_doc)
    if g_keys & r_keys:
        return True
    for g in g_keys:
        for r in r_keys:
            if g and r and (g in r or r in g):
                return True
    return False


def ordered_sources_from_context(context: str) -> list[str]:
    if not context:
        return []
    return [m.group(1).strip() for m in re.finditer(r"【片段\s+\d+】\s*來源:\s*(.+?)(?=\n)", context)]


def chunk_match(gold_chunk: str, *, chunk_id: str = "", source_path: str = "") -> bool:
    g_stem = doc_stem(gold_chunk)
    if not g_stem:
        return False
    if chunk_id and (chunk_id == gold_chunk or g_stem in chunk_id.lower()):
        return True
    if source_path:
        p_stem = doc_stem(source_path)
        if g_stem == p_stem or g_stem in p_stem or p_stem in g_stem:
            return True
        if Path(source_path).name.lower() == Path(gold_chunk).name.lower():
            return True
    if gold_chunk.startswith("chunk-") and chunk_id == gold_chunk:
        return True
    return False


def _ranked_slice(ranked: list[str], k: int) -> list[str]:
    return [str(x).strip() for x in ranked[: max(0, k)] if str(x).strip()]


def recall_at_k_aligned(
    gold: list[str],
    ranked: list[str],
    k: int,
    match_fn: Callable[[str, str], bool],
) -> float:
    g = [str(x) for x in gold if str(x).strip()]
    if not g:
        return 1.0
    top = _ranked_slice(ranked, k)
    hits = sum(1 for item in g if any(match_fn(item, r) for r in top))
    return hits / len(g)


def precision_at_k_aligned(
    gold: list[str],
    ranked: list[str],
    k: int,
    match_fn: Callable[[str, str], bool],
) -> float:
    g = [str(x) for x in gold if str(x).strip()]
    top = _ranked_slice(ranked, k)
    if not top:
        return 0.0 if g else 1.0
    hits = sum(1 for r in top if any(match_fn(item, r) for item in g))
    return hits / len(top)


def ndcg_at_k_aligned(
    gold: list[str],
    ranked: list[str],
    k: int,
    match_fn: Callable[[str, str], bool],
) -> float:
    g = [str(x) for x in gold if str(x).strip()]
    if not g:
        return 1.0
    top = _ranked_slice(ranked, k)
    if not top:
        return 0.0
    dcg = 0.0
    for i, r in enumerate(top):
        rel = 1.0 if any(match_fn(item, r) for item in g) else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    ideal = min(len(g), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k_chunks(
    gold_chunks: list[str],
    ranked_chunk_ids: list[str],
    source_paths: list[str],
    k: int,
) -> float:
    g = [str(x) for x in gold_chunks if str(x).strip()]
    if not g:
        return 1.0
    n = min(k, max(len(ranked_chunk_ids), len(source_paths), 0))
    if n == 0:
        return 0.0
    hits = 0
    for item in g:
        for i in range(n):
            cid = ranked_chunk_ids[i] if i < len(ranked_chunk_ids) else ""
            src = source_paths[i] if i < len(source_paths) else ""
            if chunk_match(item, chunk_id=cid, source_path=src):
                hits += 1
                break
    return hits / len(g)


def precision_at_k_chunks(
    gold_chunks: list[str],
    ranked_chunk_ids: list[str],
    source_paths: list[str],
    k: int,
) -> float:
    g = [str(x) for x in gold_chunks if str(x).strip()]
    n = min(k, max(len(ranked_chunk_ids), len(source_paths), 0))
    if n == 0:
        return 0.0 if g else 1.0
    hits = 0
    for i in range(n):
        cid = ranked_chunk_ids[i] if i < len(ranked_chunk_ids) else ""
        src = source_paths[i] if i < len(source_paths) else ""
        if any(chunk_match(item, chunk_id=cid, source_path=src) for item in g):
            hits += 1
    return hits / n


def _chunk_covers_gold(gold_item: str, *, chunk_id: str, source_path: str) -> bool:
    return chunk_match(gold_item, chunk_id=chunk_id, source_path=source_path)


def ndcg_at_k_chunks(
    gold_chunks: list[str],
    ranked_chunk_ids: list[str],
    source_paths: list[str],
    k: int,
) -> float:
    g = [str(x) for x in gold_chunks if str(x).strip()]
    if not g:
        return 1.0
    n = min(k, max(len(ranked_chunk_ids), len(source_paths), 0))
    if n == 0:
        return 0.0
    covered: set[int] = set()
    dcg = 0.0
    for i in range(n):
        cid = ranked_chunk_ids[i] if i < len(ranked_chunk_ids) else ""
        src = source_paths[i] if i < len(source_paths) else ""
        new_hit = False
        for gi, item in enumerate(g):
            if gi in covered:
                continue
            if _chunk_covers_gold(item, chunk_id=cid, source_path=src):
                covered.add(gi)
                new_hit = True
        if new_hit:
            dcg += 1.0 / math.log2(i + 2)
    ideal = min(len(g), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return min(1.0, dcg / idcg) if idcg > 0 else 0.0


def ranked_metrics_bundle_aligned(
    gold: list[str],
    ranked: list[str],
    ks: tuple[int, ...],
    *,
    kind: str,
    source_paths: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in ks:
        key = f"@{k}"
        if kind == "entity":
            out[key] = {
                "recall": recall_at_k_aligned(gold, ranked, k, entity_match),
                "precision": precision_at_k_aligned(gold, ranked, k, entity_match),
                "ndcg": ndcg_at_k_aligned(gold, ranked, k, entity_match),
            }
        elif kind == "relation":
            out[key] = {
                "recall": recall_at_k_aligned(gold, ranked, k, relation_match),
                "precision": precision_at_k_aligned(gold, ranked, k, relation_match),
                "ndcg": ndcg_at_k_aligned(gold, ranked, k, relation_match),
            }
        elif kind == "chunk":
            paths = source_paths or []
            out[key] = {
                "recall": recall_at_k_chunks(gold, ranked, paths, k),
                "precision": precision_at_k_chunks(gold, ranked, paths, k),
                "ndcg": ndcg_at_k_chunks(gold, ranked, paths, k),
            }
        else:
            raise ValueError(kind)
    return out


def doc_retrieval_prf_aligned(gold_docs: list[str], retrieved_docs: list[str]) -> dict[str, float]:
    g = [str(x) for x in gold_docs if str(x).strip()]
    r = [str(x) for x in retrieved_docs if str(x).strip()]
    if not g and not r:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not g or not r:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hit_g = sum(1 for item in g if any(doc_match(item, rd) for rd in r))
    hit_r = sum(1 for rd in r if any(doc_match(item, rd) for item in g))
    rec = hit_g / len(g)
    p = hit_r / len(r)
    f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
    return {"precision": p, "recall": rec, "f1": f1}


def hit_at_k_docs_aligned(gold_docs: list[str], ranked_docs: list[str], k: int) -> float:
    g = [str(x) for x in gold_docs if str(x).strip()]
    if not g:
        return 1.0
    top = _ranked_slice(ranked_docs, k)
    return 1.0 if any(doc_match(item, rd) for item in g for rd in top) else 0.0


def reciprocal_rank_docs_aligned(gold_docs: list[str], ranked_docs: list[str]) -> float:
    g = [str(x) for x in gold_docs if str(x).strip()]
    if not g:
        return 1.0
    for i, rd in enumerate(ranked_docs):
        if any(doc_match(item, rd) for item in g):
            return 1.0 / float(i + 1)
    return 0.0
