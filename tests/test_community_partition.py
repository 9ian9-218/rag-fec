"""社区划分：目标数自适应、可复现、小社区归并与稳定 ID。"""

from __future__ import annotations

import networkx as nx

from src.community import partition as P


def _clustered_graph() -> tuple[list[str], list[tuple[str, str, float]]]:
    """三个团（各 8 个节点，团内全连通）+ 一条连到团 1 的小社区 + 一对孤立点。"""
    nodes = [f"c{i}-{j}" for i in range(3) for j in range(8)]
    edges: list[tuple[str, str, float]] = []
    for i in range(3):
        group = [f"c{i}-{j}" for j in range(8)]
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                edges.append((group[a], group[b], 1.0))
    # 小社区（3 个节点）挂在团 0 上
    edges += [
        ("small-1", "small-2", 1.0),
        ("small-2", "small-3", 1.0),
        ("small-1", "small-3", 1.0),
        ("small-1", "c0-0", 5.0),
    ]
    nodes += ["small-1", "small-2", "small-3", "iso-1", "iso-2"]
    return nodes, edges


def test_target_count_is_clamped() -> None:
    assert P.compute_target_count(0, per_nodes=200, target_min=6, target_max=20) == 6
    assert P.compute_target_count(200, per_nodes=200, target_min=6, target_max=20) == 6
    assert P.compute_target_count(2046, per_nodes=200, target_min=6, target_max=20) == 10
    # 语料放大 10 倍仍被封顶
    assert P.compute_target_count(20000, per_nodes=200, target_min=6, target_max=20) == 20
    assert P.compute_target_count(30000, per_nodes=500, target_min=6, target_max=20) == 20


def test_detect_communities_is_deterministic_and_targeted() -> None:
    nodes, edges = _clustered_graph()
    first = P.detect_communities(nodes, edges, target=3, min_size=5)
    second = P.detect_communities(nodes, edges, target=3, min_size=5)
    assert [c.community_id for c in first.communities] == [c.community_id for c in second.communities]
    assert [c.members for c in first.communities] == [c.members for c in second.communities]
    assert len(first.communities) == 3
    assert first.modularity > 0.5


def test_small_community_merges_into_neighbour_and_isolated_goes_unclustered() -> None:
    nodes, edges = _clustered_graph()
    result = P.detect_communities(nodes, edges, target=3, min_size=5)
    merged_ids = set(result.unclustered)
    # 挂在团 0 上的小社区被并入大社区，不与大社区连通的孤立点记为 unclustered
    assert {"small-1", "small-2", "small-3"} & merged_ids == set()
    assert {"iso-1", "iso-2"} <= merged_ids
    member_sets = [set(c.members) for c in result.communities]
    assert any({"small-1", "small-2", "small-3"} <= members for members in member_sets)


def test_fingerprint_changes_with_edges() -> None:
    nodes, edges = _clustered_graph()
    base = P.graph_fingerprint(nodes, edges)
    assert base == P.graph_fingerprint(nodes, list(edges))
    assert base != P.graph_fingerprint(nodes, edges + [("iso-1", "iso-2", 1.0)])
    # 权重参与指纹（用于判断是否需要重算）
    heavier = [(a, b, w + 1.0) for a, b, w in edges]
    assert base != P.graph_fingerprint(nodes, heavier)


def test_merge_prefers_highest_cross_weight_and_updates_membership() -> None:
    g = nx.Graph()
    for prefix in ("a", "b"):
        g.add_edge(f"{prefix}1", f"{prefix}2", weight=1.0)
        g.add_edge(f"{prefix}2", f"{prefix}3", weight=1.0)
        g.add_edge(f"{prefix}1", f"{prefix}3", weight=1.0)
    g.add_edge("s1", "s2", weight=1.0)
    g.add_edge("s1", "a1", weight=9.0)   # 与 a 组跨边更重
    g.add_edge("s1", "b1", weight=1.0)
    big, unclustered = P.merge_small_communities(
        g,
        [{"a1", "a2", "a3"}, {"b1", "b2", "b3"}, {"s1", "s2"}],
        min_size=3,
    )
    assert unclustered == []
    assert {"s1", "s2"} <= big[0]


def test_enrich_communities_fills_types_docs_and_representatives() -> None:
    nodes, edges = _clustered_graph()
    result = P.detect_communities(nodes, edges, target=3, min_size=5)
    meta = {n: {"type": "MethodOrAlgorithm", "file_path": "a.md<SEP>b.md"} for n in nodes}
    meta["c0-0"] = {"type": "Metric", "file_path": "a.md"}
    degrees = dict(P.build_graph(nodes, edges).degree())
    enriched = P.enrich_communities(result.communities, meta, degrees)
    assert enriched and all(cm.docs == ["a.md", "b.md"] for cm in enriched)
    assert any("Metric" in cm.entity_types for cm in enriched)
    assert all(len(cm.top_entities) <= 30 for cm in enriched)


def test_leiden_keeps_all_nodes_in_one_flat_partition() -> None:
    nodes, edges = _clustered_graph()
    graph = P.build_graph(nodes, edges)
    first = P.leiden_communities(graph, resolution=1.0, seed=42)
    second = P.leiden_communities(graph, resolution=1.0, seed=42)
    assert P.ALGORITHM == "leiden"
    assert first == second
    assert set.union(*first) == set(nodes)
    assert sum(len(group) for group in first) == len(nodes)
    assert all(nx.is_connected(graph.subgraph(group)) for group in first)


def test_leiden_handles_empty_and_edgeless_graphs() -> None:
    assert P.leiden_communities(nx.Graph(), resolution=1.0) == []
    for resolution in (None, 1.0):
        result = P.detect_communities(
            ["a", "b"], [], target=1, min_size=2, resolution=resolution,
        )
        assert result.communities == []
        assert result.unclustered == ["a", "b"]
