"""社区摘要构建/刷新编排：读图 → 划分 → 复用判定 → 补摘要 → 向量索引 → 落盘。"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from config.settings import Settings, get_settings
from src.community import partition as P
from src.community import store
from src.community.summarizer import content_hash, sample_text, summarize_community
from src.storage.redis_lock import acquire_lock, release_lock
from src.utils.logger import get_logger

logger = get_logger("community.builder")

LOCK_KEY = "rag:lock:community:rebuild"


async def load_graph(settings: Settings | None = None) -> tuple[list[str], list[tuple[str, str, float]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """只读导出 Neo4j 图谱：``(节点ID, 边, 节点元数据, 关系明细)``。"""
    from src.storage.neo4j_client import Neo4jClient

    s = settings or get_settings()
    driver = Neo4jClient.async_driver()
    node_ids: list[str] = []
    node_meta: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, float]] = []
    relations: list[dict[str, Any]] = []
    try:
        async with driver.session(database=s.neo4j.database) as session:
            result = await session.run(
                "MATCH (n:base) RETURN n.entity_id AS id, n.entity_type AS type, "
                "n.file_path AS file_path, n.description AS description"
            )
            async for record in result:
                node_id = record["id"]
                if not node_id:
                    continue
                node_ids.append(node_id)
                node_meta[node_id] = {
                    "type": record["type"],
                    "file_path": record["file_path"],
                    "description": record["description"],
                }
            result = await session.run(
                "MATCH (a:base)-[r:DIRECTED]->(b:base) "
                "RETURN a.entity_id AS a, b.entity_id AS b, r.weight AS w, r.description AS description"
            )
            async for record in result:
                a, b = record["a"], record["b"]
                if not a or not b:
                    continue
                weight = 1.0 if record["w"] is None else float(record["w"])
                edges.append((a, b, weight))
                relations.append({"src": a, "tgt": b, "weight": weight, "description": record["description"]})
    finally:
        await driver.close()
    return node_ids, edges, node_meta, relations


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _community_payload(
    community: P.Community,
    node_meta: dict[str, dict[str, Any]],
    degrees: dict[str, int],
    relations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """组装单个社区的结构化材料（供取样与摘要使用）。"""
    member_set = set(community.members)
    inside = [
        r for r in relations
        if r["src"] in member_set and r["tgt"] in member_set
    ]
    if len(inside) < 5:  # 社区内部边过少时补充跨社区关系，避免摘要无据可依
        extra = [
            r for r in relations
            if (r["src"] in member_set) != (r["tgt"] in member_set)
        ]
        inside = inside + extra[: max(0, 20 - len(inside))]
    entities = [
        {
            "name": m,
            "type": (node_meta.get(m) or {}).get("type"),
            "degree": int(degrees.get(m, 0)),
            "description": (node_meta.get(m) or {}).get("description"),
        }
        for m in community.members
    ]
    return {
        "community_id": community.community_id,
        "size": community.size,
        "docs": list(community.docs),
        "entity_types": dict(community.entity_types),
        "entities": entities,
        "relations": list(inside),
    }


async def _embed_texts(texts: Sequence[str], settings: Settings) -> np.ndarray:
    """用本地/远程 embedding 编码文本（归一化，float32）。"""
    if settings.embedding.backend == "local":
        from src.storage.local_embedding import build_local_embedding_func

        func = build_local_embedding_func(settings)
    else:
        from src.storage.remote_embedding import build_remote_embedding_func

        func = build_remote_embedding_func(settings)
    vectors = await func(list(texts))
    arr = np.asarray(vectors, dtype=np.float32)
    return arr.reshape(len(texts), -1) if arr.ndim == 1 else arr


def _report_text(rep: dict[str, Any]) -> str:
    """摘要参与向量化的文本（标题 + 概述 + 要点）。"""
    parts = [str(rep.get("title") or "").strip(), str(rep.get("summary") or "").strip()]
    points = rep.get("key_points") or []
    if points:
        parts.append("；".join(str(p) for p in points))
    return "\n".join(p for p in parts if p).strip()


async def build_communities(
    *,
    mode: str = "auto",
    force: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    settings: Settings | None = None,
    reason: str = "manual",
) -> dict[str, Any]:
    """构建/刷新社区摘要。

    - ``mode="auto"``：算法与图谱指纹未变则直接跳过（0 次 LLM 调用）；
    - ``mode="full"`` 或 ``force=True``：忽略缓存，全部重新生成；
    - ``dry_run=True``：只做划分与预算估算，不调 LLM、不落盘；
    - ``limit``：本次最多新增多少次 LLM 摘要调用（超出的社区标记 pending）。
    """
    s = settings or get_settings()
    c = s.community
    started = time.perf_counter()

    if not dry_run and not c.enabled and not force:
        return {"skipped": True, "reason": "disabled_by_server"}

    try:
        node_ids, edges, node_meta, relations = await load_graph(s)
    except Exception as e:  # 图谱不可用不应影响调用方
        logger.error("社区构建：读取图谱失败: %s", e)
        return {"skipped": True, "reason": "graph_unavailable", "error": str(e)}

    if not node_ids:
        return {"skipped": True, "reason": "empty_graph"}

    fingerprint = P.graph_fingerprint(node_ids, edges)
    target = P.compute_target_count(
        len(node_ids),
        per_nodes=int(c.target_per_nodes),
        target_min=int(c.target_min),
        target_max=int(c.target_max),
    )
    previous = store.load_reports(s)
    prev_meta = (previous or {}).get("meta") or {}
    prev_reports = (previous or {}).get("reports") or {}
    if (
        not force
        and mode == "auto"
        and previous is not None
        and prev_meta.get("algorithm") == P.ALGORITHM
        and prev_meta.get("graph_fingerprint") == fingerprint
        and not bool(prev_meta.get("stale"))
    ):
        return {
            "skipped": True,
            "reason": "up_to_date",
            "communities": len(prev_reports),
            "fingerprint": fingerprint,
        }

    result = P.detect_communities(
        node_ids,
        edges,
        target=target,
        min_size=int(c.min_size),
        resolution=c.resolution,
        seed=int(c.seed),
        resolution_min=float(c.resolution_min),
        resolution_max=float(c.resolution_max),
    )
    graph = P.build_graph(node_ids, edges)
    degrees = dict(graph.degree())
    communities = P.enrich_communities(result.communities, node_meta, degrees)

    summary = {
        "algorithm": P.ALGORITHM,
        "reason": reason,
        "mode": mode,
        "fingerprint": fingerprint,
        "target": result.target,
        "resolution": round(float(result.resolution), 4),
        "modularity": round(float(result.modularity), 4),
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "unclustered": len(result.unclustered),
        "communities": len(communities),
    }

    if dry_run:
        return {
            "dry_run": True,
            **summary,
            "estimated_llm_calls": len(communities),
            "sizes": [cm.size for cm in communities],
            "docs": {cm.community_id: len(cm.docs) for cm in communities},
        }

    # ---- 复用判定：成员 Jaccard ≥ 阈值则沿用旧摘要，不调 LLM ----
    reuse_map: dict[str, tuple[str, float, dict[str, Any]]] = {}
    if prev_reports and mode == "auto":
        taken: set[str] = set()
        for community in communities:
            best: tuple[str, float, dict[str, Any]] | None = None
            members = set(community.members)
            for prev_id, rep in prev_reports.items():
                if prev_id in taken or not isinstance(rep, dict):
                    continue
                score = _jaccard(members, set(rep.get("members") or []))
                if score >= float(c.reuse_jaccard) and (best is None or score > best[1]):
                    best = (prev_id, score, rep)
            if best is not None:
                taken.add(best[0])
                reuse_map[community.community_id] = best

    # ---- 生成摘要（复用优先；limit 约束新增 LLM 调用数） ----
    model = (c.summary_model or "").strip() or s.resolved_llm_model_name()
    reports: dict[str, Any] = {}
    to_generate: list[tuple[P.Community, dict[str, Any], str]] = []
    llm_budget = None if limit is None else max(0, int(limit))
    reused = 0
    for community in communities:
        payload = _community_payload(community, node_meta, degrees, relations)
        sample = sample_text(payload, max_tokens=int(c.summary_max_input_tokens))
        payload["sample"] = sample
        rep_hash = content_hash(
            community.members,
            sample,
            prompt_version=str(c.prompt_version),
            model=model,
        )
        reused_entry = reuse_map.get(community.community_id)
        if reused_entry is not None:
            prev_rep = reused_entry[2]
            reports[community.community_id] = {
                **{k: prev_rep.get(k) for k in ("title", "summary", "key_points", "entities")},
                "community_id": community.community_id,
                "members": list(community.members),
                "size": community.size,
                "docs": list(community.docs),
                "entity_types": dict(community.entity_types),
                "content_hash": rep_hash,
                "model": prev_rep.get("model") or model,
                "generated_at": prev_rep.get("generated_at"),
                "reused_from": reused_entry[0],
                "reuse_jaccard": round(reused_entry[1], 4),
                "stale": bool(prev_rep.get("stale")),
            }
            reused += 1
            continue
        if llm_budget is not None and llm_budget <= 0:
            reports[community.community_id] = _pending_report(community, rep_hash, model)
            continue
        if llm_budget is not None:
            llm_budget -= 1
        to_generate.append((community, payload, rep_hash))

    llm_calls = 0
    failed = 0
    if to_generate:
        outcomes = await asyncio.gather(
            *(
                summarize_community(payload, payload["sample"], settings=s)
                for _, payload, _ in to_generate
            ),
            return_exceptions=True,
        )
        for (community, payload, rep_hash), outcome in zip(to_generate, outcomes):
            if isinstance(outcome, BaseException):
                failed += 1
                logger.warning("社区摘要生成失败 %s: %s", community.community_id, outcome)
                prev_rep = (reuse_map.get(community.community_id) or (None, 0.0, {}))[2]
                reports[community.community_id] = {
                    **{k: (prev_rep or {}).get(k) for k in ("title", "summary", "key_points", "entities")},
                    "community_id": community.community_id,
                    "members": list(community.members),
                    "size": community.size,
                    "docs": list(community.docs),
                    "entity_types": dict(community.entity_types),
                    "content_hash": rep_hash,
                    "model": model,
                    "generated_at": (prev_rep or {}).get("generated_at"),
                    "reused_from": None,
                    "reuse_jaccard": 0.0,
                    "stale": True,
                }
                continue
            llm_calls += 1
            reports[community.community_id] = {
                "community_id": community.community_id,
                "title": outcome.get("title") or "",
                "summary": outcome.get("summary") or "",
                "key_points": outcome.get("key_points") or [],
                "entities": outcome.get("entities") or [],
                "members": list(community.members),
                "size": community.size,
                "docs": list(community.docs),
                "entity_types": dict(community.entity_types),
                "content_hash": rep_hash,
                "model": outcome.get("model") or model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "reused_from": None,
                "reuse_jaccard": 0.0,
                "stale": False,
            }

    # ---- 向量索引（标题 + 概述 + 要点） ----
    index_ids: list[str] = []
    index_vectors: list[np.ndarray] = []
    indexed: list[str] = []
    usable = [
        cid for cid, rep in reports.items()
        if str(rep.get("summary") or "").strip()
    ]
    if failed and not usable:
        # Preserve the current reports and index when the LLM is unavailable.
        return {
            **summary,
            "skipped": True,
            "reason": "summaries_unavailable",
            "built": llm_calls,
            "reused": reused,
            "pending": sum(1 for rep in reports.values() if rep.get("pending")),
            "failed": failed,
            "indexed": 0,
        }
    if usable:
        try:
            vectors = await _embed_texts([_report_text(reports[cid]) for cid in usable], s)
            for cid, vec in zip(usable, vectors):
                index_ids.append(cid)
                index_vectors.append(np.asarray(vec, dtype=np.float32))
                indexed.append(cid)
        except Exception as e:
            logger.error("社区摘要向量化失败，本次不更新向量索引: %s", e)

    pending = sum(1 for rep in reports.values() if rep.get("pending"))
    meta = {
        "graph_fingerprint": fingerprint,
        "algorithm": P.ALGORITHM,
        "seed": int(c.seed),
        "min_size": int(c.min_size),
        "target": result.target,
        "resolution": round(float(result.resolution), 4),
        "modularity": round(float(result.modularity), 4),
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "unclustered": len(result.unclustered),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "mode": mode,
        "model": model,
        "prompt_version": str(c.prompt_version),
        "summarized": len(usable),
        "reused": reused,
        "pending": pending,
        "failed": failed,
        "llm_calls": llm_calls,
        "duration_s": round(time.perf_counter() - started, 2),
        "stale": False,
    }
    store.save_reports(meta, reports, s)
    if index_ids:
        store.save_index(index_ids, np.vstack(index_vectors), s)

    return {
        **summary,
        "built": llm_calls,
        "reused": reused,
        "pending": pending,
        "failed": failed,
        "indexed": len(index_ids),
        "duration_s": meta["duration_s"],
    }


def _pending_report(community: P.Community, rep_hash: str, model: str) -> dict[str, Any]:
    return {
        "community_id": community.community_id,
        "title": "",
        "summary": "",
        "key_points": [],
        "entities": [],
        "members": list(community.members),
        "size": community.size,
        "docs": list(community.docs),
        "entity_types": dict(community.entity_types),
        "content_hash": rep_hash,
        "model": model,
        "generated_at": None,
        "reused_from": None,
        "reuse_jaccard": 0.0,
        "pending": True,
        "stale": True,
    }


async def refresh_communities(
    *,
    reason: str = "incremental",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """带锁的增量刷新入口（入库后调用；失败不得影响入库结果）。"""
    s = settings or get_settings()
    if not s.community.enabled or s.community.refresh_mode != "auto":
        return {"skipped": True, "reason": "disabled_by_server"}
    token = await acquire_lock(LOCK_KEY, timeout=int(s.community.rebuild_timeout_seconds))
    if token is None:
        return {"skipped": True, "reason": "locked"}
    try:
        return await build_communities(mode="auto", reason=reason, settings=s)
    finally:
        await release_lock(LOCK_KEY, token)
