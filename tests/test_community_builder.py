"""社区构建编排：跳过、复用、限流、失败降级（不连 Neo4j、不调真实 LLM）。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from config.settings import Settings
from src.community import builder, store


def _graph(weight_bump: float = 0.0):
    nodes = [f"c{i}-{j}" for i in range(3) for j in range(6)]
    edges: list[tuple[str, str, float]] = []
    for i in range(3):
        group = [f"c{i}-{j}" for j in range(6)]
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                edges.append((group[a], group[b], 1.0 + weight_bump))
    meta = {
        n: {"type": "MethodOrAlgorithm", "file_path": "a.md", "description": f"实体 {n} 的说明"}
        for n in nodes
    }
    relations = [
        {"src": a, "tgt": b, "weight": w, "description": f"{a} → {b}"} for a, b, w in edges
    ]
    return nodes, edges, meta, relations


@pytest.fixture()
def env(tmp_path, monkeypatch):
    s = Settings()
    s.paths.lightrag_working_dir = str(tmp_path)
    s.community.enabled = True
    s.community.min_size = 3
    s.community.target_min = 3
    s.community.target_max = 3
    s.community.target_per_nodes = 6
    store.reset_version_cache()

    state = {"llm_calls": 0, "graph": _graph(), "fail_first": False}

    async def fake_load_graph(settings=None):
        return state["graph"]

    async def fake_summarize(payload, sample, *, settings=None):
        state["llm_calls"] += 1
        if state["fail_first"] and state["llm_calls"] == 1:
            raise RuntimeError("模拟摘要失败")
        return {
            "title": f"主题 {payload['community_id'][-3:]}",
            "summary": "该社区围绕某个研究主题展开，包含方法与复杂度讨论。",
            "key_points": ["要点1", "要点2"],
            "entities": ["实体A"],
            "model": "fake",
        }

    async def fake_embed(texts, settings):
        return np.ones((len(texts), 4), dtype=np.float32)

    monkeypatch.setattr(builder, "load_graph", fake_load_graph)
    monkeypatch.setattr(builder, "summarize_community", fake_summarize)
    monkeypatch.setattr(builder, "_embed_texts", fake_embed)
    return s, state


async def test_disabled_without_force_skips(tmp_path, env):
    s, state = env
    s.community.enabled = False
    out = await builder.build_communities(mode="auto", settings=s)
    assert out["skipped"] is True
    assert out["reason"] == "disabled_by_server"
    assert state["llm_calls"] == 0
    assert not store.reports_path(s).exists()


async def test_dry_run_writes_nothing(env):
    s, state = env
    out = await builder.build_communities(mode="full", force=True, dry_run=True, settings=s)
    assert out["dry_run"] is True
    assert out["estimated_llm_calls"] == 3
    assert sorted(out["sizes"]) == [6, 6, 6]
    assert state["llm_calls"] == 0
    assert not store.reports_path(s).exists()
    assert not store.index_path(s).exists()


async def test_full_build_then_auto_skip(env):
    s, state = env
    built = await builder.build_communities(mode="full", force=True, settings=s)
    assert built["communities"] == 3
    assert built["built"] == 3
    assert built["reused"] == 0
    assert state["llm_calls"] == 3
    assert store.index_path(s).exists()

    again = await builder.build_communities(mode="auto", settings=s)
    assert again["skipped"] is True
    assert again["reason"] == "up_to_date"
    assert state["llm_calls"] == 3  # 未新增调用


async def test_membership_unchanged_reuses_all_summaries(env):
    s, state = env
    await builder.build_communities(mode="full", force=True, settings=s)
    # 仅权重变化：指纹改变（触发重算），但社区成员完全一致 → 全部复用
    state["graph"] = _graph(weight_bump=1.0)
    out = await builder.build_communities(mode="auto", settings=s)
    assert out["reused"] == 3
    assert out["built"] == 0
    assert state["llm_calls"] == 3

    data = store.load_reports(s)
    assert all(rep.get("reused_from") for rep in data["reports"].values())


async def test_limit_marks_pending(env):
    s, state = env
    out = await builder.build_communities(mode="full", force=True, limit=1, settings=s)
    assert out["built"] == 1
    assert out["pending"] == 2
    assert state["llm_calls"] == 1
    reports = store.load_reports(s)["reports"]
    assert sum(1 for r in reports.values() if r.get("pending")) == 2
    # pending 的社区不进向量索引
    ids, vectors = store.load_index(s)
    assert len(ids) == 1 and vectors.shape[0] == 1


async def test_summary_failure_marks_report_stale_without_crashing(env):
    s, state = env
    state["fail_first"] = True
    out = await builder.build_communities(mode="full", force=True, settings=s)
    assert out["failed"] == 1
    assert out["built"] == 2
    data = store.load_reports(s)
    assert data["meta"]["stale"] is False
    stale = [r for r in data["reports"].values() if r.get("stale")]
    assert len(stale) == 1 and not stale[0].get("summary")


async def test_refresh_respects_refresh_mode(env):
    s, state = env
    s.community.refresh_mode = "manual"
    out = await builder.refresh_communities(reason="incremental", settings=s)
    assert out == {"skipped": True, "reason": "disabled_by_server"}
    assert state["llm_calls"] == 0


@pytest.mark.parametrize("previous_algorithm", ["louvain", None])
async def test_algorithm_change_repartitions_even_when_graph_is_unchanged(env, previous_algorithm):
    s, state = env
    await builder.build_communities(mode="full", force=True, settings=s)
    previous = store.load_reports(s)
    previous["meta"]["algorithm"] = previous_algorithm
    store.save_reports(previous["meta"], previous["reports"], s)

    out = await builder.build_communities(mode="auto", settings=s)

    assert not out.get("skipped")
    assert out["algorithm"] == "leiden"
    assert out["communities"] == 3
    assert out["reused"] == 3
    assert state["llm_calls"] == 3
    assert store.load_reports(s)["meta"]["algorithm"] == "leiden"


async def test_all_summary_failures_preserve_previous_reports_and_index(env, monkeypatch):
    s, state = env
    await builder.build_communities(mode="full", force=True, settings=s)
    previous_reports = store.reports_path(s).read_bytes()
    previous_index = store.index_path(s).read_bytes()

    async def fail_summary(*args, **kwargs):
        raise RuntimeError("LLM returned 403")

    monkeypatch.setattr(builder, "summarize_community", fail_summary)
    out = await builder.build_communities(mode="full", force=True, settings=s)

    assert out["skipped"] is True
    assert out["reason"] == "summaries_unavailable"
    assert out["failed"] == 3
    assert store.reports_path(s).read_bytes() == previous_reports
    assert store.index_path(s).read_bytes() == previous_index
