"""社区划分：实体关系图 → Leiden 社区（目标数自适应 + 小社区归并）。

算法说明：
- 用 leidenalg 的 Leiden（模块度优化）在**加权无向图**上划分，边权取关系 ``weight``；
- 不固定分辨率，而是二分搜索 ``resolution`` 命中目标社区数
  ``clamp(round(节点数 / target_per_nodes), target_min, target_max)``，
  这样语料增长时摘要数量仍被硬性封顶；
- 规模 < ``min_size`` 的社区按"到各大社区的总边权最大"并入相邻大社区，
  与任何大社区都不连通的孤立点记为 ``unclustered``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import igraph as ig
import leidenalg
import networkx as nx

from src.utils.logger import get_logger

logger = get_logger("community.partition")

ALGORITHM = "leiden"
MEMBER_SEP = "<SEP>"


@dataclass(slots=True)
class Community:
    """一个社区及其成员实体。"""

    community_id: str
    members: list[str]
    size: int
    entity_types: dict[str, int] = field(default_factory=dict)
    docs: list[str] = field(default_factory=list)
    top_entities: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PartitionResult:
    communities: list[Community]
    unclustered: list[str]
    resolution: float
    target: int
    modularity: float
    fingerprint: str
    node_count: int
    edge_count: int


def compute_target_count(
    node_count: int,
    *,
    per_nodes: int = 200,
    target_min: int = 6,
    target_max: int = 20,
) -> int:
    """目标社区数 = clamp(round(节点数 / per_nodes), target_min, target_max)。"""
    lo = max(1, int(target_min))
    hi = max(lo, int(target_max))
    if node_count <= 0:
        return lo
    raw = int(round(node_count / max(1, int(per_nodes))))
    return max(lo, min(hi, raw))


def build_graph(
    node_ids: Iterable[str],
    edges: Iterable[tuple[str, str, float | None]],
) -> nx.Graph:
    """构建加权无向图（同一对实体的多条关系累加权重）。"""
    g = nx.Graph()
    for node in node_ids:
        g.add_node(node)
    for a, b, w in edges:
        if not a or not b:
            continue
        weight = 1.0 if w is None else float(w)
        if g.has_edge(a, b):
            g[a][b]["weight"] = float(g[a][b].get("weight", 1.0)) + weight
        else:
            g.add_edge(a, b, weight=weight)
    return g


def graph_fingerprint(
    node_ids: Iterable[str],
    edges: Iterable[tuple[str, str, float | None]],
) -> str:
    """图谱指纹：节点集合 + 归一化边集合 + 权重。"""
    h = hashlib.sha1()
    for node in sorted(str(n) for n in node_ids):
        h.update(node.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"|")
    normalized = sorted(
        (str(a), str(b), round(1.0 if w is None else float(w), 6))
        for a, b, w in edges
        if a and b
    )
    for a, b, w in normalized:
        h.update(f"{a}\x01{b}\x01{w:.6f}".encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def leiden_communities(g: nx.Graph, resolution: float, seed: int = 42) -> list[set[str]]:
    """对图执行一次 Leiden，返回社区成员集合列表。"""
    nodes = list(g.nodes)
    if g.number_of_edges() == 0:
        return [{node} for node in nodes]
    node_index = {node: i for i, node in enumerate(nodes)}
    graph = ig.Graph(
        n=len(nodes),
        edges=[(node_index[a], node_index[b]) for a, b in g.edges],
        directed=False,
    )
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=[float(data.get("weight", 1.0)) for _, _, data in g.edges(data=True)],
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    return [{nodes[i] for i in community} for community in partition]


def count_summarizable(communities: Sequence[set[str]], min_size: int) -> int:
    """统计达到最小规模门槛（即需要写摘要）的社区数。"""
    return sum(1 for c in communities if len(c) >= int(min_size))


def search_resolution(
    g: nx.Graph,
    *,
    target: int,
    min_size: int,
    seed: int = 42,
    resolution_min: float = 0.05,
    resolution_max: float = 3.0,
    max_iter: int = 24,
) -> tuple[float, list[set[str]], int]:
    """二分搜索分辨率，使"≥min_size 的社区数"逼近 target。

    分辨率越大社区越碎（计数单调不减），因此可以二分；命中目标立即停止。
    返回 ``(resolution, communities, count)``，其中为迭代过程中最接近目标的一组。
    """
    lo, hi = float(resolution_min), float(resolution_max)
    best: tuple[float, list[set[str]], int] | None = None
    for _ in range(max(1, int(max_iter))):
        mid = (lo + hi) / 2.0
        communities = leiden_communities(g, mid, seed)
        k = count_summarizable(communities, min_size)
        if best is None or abs(k - target) < abs(best[2] - target):
            best = (mid, communities, k)
        if k == target:
            break
        if k < target:
            lo = mid
        else:
            hi = mid
    assert best is not None  # max_iter>=1 保证至少跑一次
    return best


def merge_small_communities(
    g: nx.Graph,
    communities: Sequence[set[str]],
    min_size: int,
) -> tuple[list[set[str]], list[str]]:
    """把 <min_size 的社区并入跨边权重最大的大社区；无连通的大社区则记为 unclustered。"""
    big = [set(c) for c in communities if len(c) >= int(min_size)]
    small = [set(c) for c in communities if len(c) < int(min_size)]
    if not big:
        unclustered = sorted(node for c in communities for node in c)
        return [], unclustered

    node_to_big = {node: idx for idx, c in enumerate(big) for node in c}
    unclustered: list[str] = []
    for group in small:
        votes: dict[int, float] = {}
        for node in group:
            for neighbor in g.neighbors(node):
                idx = node_to_big.get(neighbor)
                if idx is None:
                    continue
                votes[idx] = votes.get(idx, 0.0) + float(g[node][neighbor].get("weight", 1.0))
        if not votes:
            unclustered.extend(sorted(group))
            continue
        # 权重最大者优先，权重相同时取索引更小者，保证结果可复现
        best_idx = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        big[best_idx] |= group
        for node in group:
            node_to_big[node] = best_idx
    return big, unclustered


def assign_community_ids(
    fingerprint: str,
    communities: Sequence[set[str]],
) -> list[tuple[str, set[str]]]:
    """按「规模降序 + 成员最小 entity_id 破平」分配稳定 ID。"""
    short = fingerprint[:8]
    ordered = sorted(communities, key=lambda c: (-len(c), min(c) if c else ""))
    return [(f"com-{short}-{i:03d}", c) for i, c in enumerate(ordered, start=1)]


def detect_communities(
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str, float | None]],
    *,
    target: int,
    min_size: int,
    resolution: float | None = None,
    seed: int = 42,
    resolution_min: float = 0.05,
    resolution_max: float = 3.0,
    max_iter: int = 24,
) -> PartitionResult:
    """完整划分流程：建图 → Leiden（固定或自动分辨率）→ 小社区归并 → 稳定 ID。"""
    fingerprint = graph_fingerprint(node_ids, edges)
    g = build_graph(node_ids, edges)
    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()
    if node_count == 0:
        return PartitionResult([], [], 0.0, int(target), 0.0, fingerprint, 0, 0)

    if resolution is not None:
        used_resolution = float(resolution)
        communities = leiden_communities(g, used_resolution, seed)
    else:
        used_resolution, communities, _ = search_resolution(
            g,
            target=int(target),
            min_size=int(min_size),
            seed=seed,
            resolution_min=resolution_min,
            resolution_max=resolution_max,
            max_iter=max_iter,
        )

    try:
        modularity = float(nx.community.modularity(g, communities, weight="weight"))
    except Exception:  # 退化图（空社区等）不应中断流程
        modularity = 0.0

    big, unclustered = merge_small_communities(g, communities, int(min_size))
    assigned = assign_community_ids(fingerprint, big)
    out = [
        Community(community_id=cid, members=sorted(members), size=len(members))
        for cid, members in assigned
    ]
    return PartitionResult(
        communities=out,
        unclustered=unclustered,
        resolution=used_resolution,
        target=int(target),
        modularity=modularity,
        fingerprint=fingerprint,
        node_count=node_count,
        edge_count=edge_count,
    )


def enrich_communities(
    communities: Sequence[Community],
    node_meta: dict[str, dict[str, Any]],
    degrees: dict[str, int] | None = None,
    *,
    top_entities: int = 30,
) -> list[Community]:
    """补全社区的类型分布、覆盖文档与代表实体（按度数降序）。"""
    degree_map = degrees or {}
    out: list[Community] = []
    for community in communities:
        types: dict[str, int] = {}
        docs: list[str] = []
        seen_docs: set[str] = set()
        for member in community.members:
            meta = node_meta.get(member) or {}
            etype = str(meta.get("type") or "").strip() or "unknown"
            types[etype] = types.get(etype, 0) + 1
            raw_fp = str(meta.get("file_path") or "").strip()
            for part in raw_fp.split(MEMBER_SEP):
                part = part.strip()
                if part and part not in seen_docs:
                    seen_docs.add(part)
                    docs.append(part)
        ranked = sorted(
            community.members,
            key=lambda m: (-int(degree_map.get(m, 0)), m),
        )[: int(top_entities)]
        out.append(
            Community(
                community_id=community.community_id,
                members=list(community.members),
                size=community.size,
                entity_types=dict(sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))),
                docs=docs,
                top_entities=ranked,
            )
        )
    return out
