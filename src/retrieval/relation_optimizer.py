"""检索结果中的关系去冗、重排与截断。"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from config.settings import Settings, get_settings
from src.retrieval.relation_keywords import enhance_keywords_for_retrieval, extract_user_query_from_prompt
from src.storage.bge_rerank import minmax_normalize_scores
from src.utils.logger import get_logger

logger = get_logger("retrieval.relation_optimizer")

_REL_STOP = frozenset(
    "的 了 与 和 及 在 是 有 对 为 从 到 被 把 将 这 那 其 该 一种 一个".split()
)


def _payload_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    inner = bundle.get("data") or {}
    if isinstance(inner, dict) and isinstance(inner.get("data"), dict):
        return inner["data"]
    return inner if isinstance(inner, dict) else {}


def _norm_edge_key(src: str, tgt: str) -> str:
    a, b = src.strip(), tgt.strip()
    if not a or not b:
        return ""
    return "|".join(sorted([a, b]))


def _relation_text(r: dict[str, Any]) -> str:
    parts = [
        str(r.get("src_id") or r.get("source") or ""),
        str(r.get("tgt_id") or r.get("target") or ""),
        str(r.get("description") or ""),
        str(r.get("keywords") or r.get("relationship_keywords") or ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _keyword_overlap_score(text: str, keywords: list[str]) -> float:
    if not text or not keywords:
        return 0.0
    t = text.lower()
    hits = 0
    for kw in keywords:
        k = kw.strip().lower()
        if not k:
            continue
        if k in t:
            hits += 1
            continue
        for part in re.split(r"[\s,，、/|]+", k):
            if len(part) >= 2 and part.lower() in t:
                hits += 1
                break
    return hits / max(1, len(keywords))


def _online_rerank_available(settings: Settings) -> bool:
    m = settings.models
    return bool(
        m.rerank_api_enabled
        and m.rerank_api_key
        and m.rerank_api_base_url
        and m.rerank_api_model_name
    )


async def _rerank_relations_online(
    question: str,
    relations: list[dict[str, Any]],
    *,
    settings: Settings,
) -> list[tuple[dict[str, Any], float]]:
    """优先使用线上 Rerank API 对关系重排；不可用则回退到本地 CrossEncoder。"""
    from src.storage.remote_rerank import build_remote_rerank_model_func

    rerank_fn = build_remote_rerank_model_func(settings)
    if rerank_fn is None:
        return await asyncio.to_thread(
            _rerank_relations_sync, question, relations, settings=settings
        )

    docs = [_relation_text(r) or " " for r in relations]
    if not docs:
        return []

    results = await rerank_fn(query=question, documents=docs, top_n=len(docs))
    scored: list[tuple[dict[str, Any], float]] = []
    for item in results:
        idx = int(item.get("index", 0))
        score = float(item.get("relevance_score", 0.0))
        if 0 <= idx < len(relations):
            scored.append((relations[idx], score))
    return scored


def _rerank_relations_sync(
    question: str,
    relations: list[dict[str, Any]],
    *,
    settings: Settings,
) -> list[tuple[dict[str, Any], float]]:
    from config.model_paths import resolve_reranker_model_load_path
    from src.storage.bge_rerank import _get_cross_encoder

    load_path = resolve_reranker_model_load_path(settings)
    model = _get_cross_encoder(load_path)
    docs = [_relation_text(r) or " " for r in relations]
    pairs = [[question, d] for d in docs]
    raw = model.predict(pairs, batch_size=int(settings.models.rerank_batch_size), show_progress_bar=False)
    scores = minmax_normalize_scores([float(x) for x in raw])
    return list(zip(relations, scores))


async def filter_relationships(
    question: str,
    relations: list[dict[str, Any]],
    *,
    hl_keywords: list[str] | None = None,
    ll_keywords: list[str] | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """去重、关键词相关性过滤、CrossEncoder 重排、按 top_k 截断。"""
    s = settings or get_settings()
    lr = s.lightrag
    if not relations:
        return []

    q = extract_user_query_from_prompt(question) or (question or "").strip()
    hl, ll = enhance_keywords_for_retrieval(q, list(hl_keywords or []), list(ll_keywords or []))
    kw_all = hl + ll

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in relations:
        if not isinstance(r, dict):
            continue
        src = str(r.get("src_id") or r.get("source") or "").strip()
        tgt = str(r.get("tgt_id") or r.get("target") or "").strip()
        key = _norm_edge_key(src, tgt)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    min_kw = float(lr.relation_keyword_min_score)
    if min_kw > 0 and kw_all:
        scored_kw: list[tuple[dict[str, Any], float]] = []
        for r in deduped:
            scored_kw.append((r, _keyword_overlap_score(_relation_text(r), kw_all)))
        deduped = [r for r, sc in scored_kw if sc >= min_kw]
        if not deduped and scored_kw:
            scored_kw.sort(key=lambda x: x[1], reverse=True)
            deduped = [r for r, _ in scored_kw[: max(1, lr.relation_top_k)]]

    if lr.relation_rerank_enabled and (_online_rerank_available(s) or s.rerank_runtime_available()) and deduped:
        try:
            ranked = await _rerank_relations_online(q, deduped, settings=s)
            min_rr = float(lr.relation_min_rerank_score)
            ranked = [(r, sc) for r, sc in ranked if sc >= min_rr]
            ranked.sort(key=lambda x: x[1], reverse=True)
            deduped = [r for r, _ in ranked]
            logger.debug(
                "relation rerank: in=%d kept=%d min_score=%.2f",
                len(relations),
                len(deduped),
                min_rr,
            )
        except Exception as e:
            logger.warning("关系重排失败，保留关键词过滤结果: %s", e)

    top_k = max(1, int(lr.relation_top_k))
    return deduped[:top_k]


async def refine_retrieval_bundle(
    question: str,
    bundle: dict[str, Any],
    *,
    hl_keywords: list[str] | None = None,
    ll_keywords: list[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """就地优化 ``aquery_data`` 返回 bundle 中的 relationships 列表。"""
    if not isinstance(bundle, dict):
        return bundle
    payload = _payload_from_bundle(bundle)
    rels = payload.get("relationships")
    if not isinstance(rels, list) or not rels:
        return bundle

    before = len(rels)
    filtered = await filter_relationships(
        question,
        [r for r in rels if isinstance(r, dict)],
        hl_keywords=hl_keywords,
        ll_keywords=ll_keywords,
        settings=settings,
    )
    payload["relationships"] = filtered
    meta = bundle.get("metadata")
    if isinstance(meta, dict):
        pi = meta.get("processing_info")
        if isinstance(pi, dict):
            pi["relations_after_truncation"] = len(filtered)
            pi["total_relations_found"] = before
    logger.info("关系检索优化: %d → %d", before, len(filtered))
    return bundle
