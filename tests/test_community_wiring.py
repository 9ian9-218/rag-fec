"""社区能力接线：缓存 key、溯源隔离、遥测字段与 retriever 挂载。"""

from __future__ import annotations

import sqlite3

import numpy as np
from config.settings import Settings
from src.community import store
from src.evaluation import online_monitor as om
from src.retrieval.mode_config import is_macro_question, suggest_mode_from_question
from src.retrieval.result_processor import extract_sources
from src.retrieval.retriever import GraphRAGRetriever
from src.storage import redis_cache


def test_retrieval_cache_key_varies_with_community_extra():
    base = dict(question="q", mode="mix", top_k=8, chunk_top_k=9)
    off = redis_cache.build_retrieval_cache_key(**base, extra="community=off|v=none")
    on = redis_cache.build_retrieval_cache_key(**base, extra="community=on|v=none")
    none = redis_cache.build_retrieval_cache_key(**base)
    assert len({off, on, none}) == 3


def test_query_cache_key_varies_with_extra():
    base = dict(question="q", mode="mix", top_k=8, multimodal=False)
    a = redis_cache.build_query_cache_key(**base, extra="community=on|v=1")
    b = redis_cache.build_query_cache_key(**base, extra="community=off|v=1")
    assert a != b


def test_extract_sources_ignores_community_context():
    bundle = {
        "data": {"chunks-1": {"file_path": "a.md", "score": 0.9}},
        "community_context": {"text": "概览内容", "ids": ["com-x-001"], "tokens": 42},
        "community": {"requested": True, "applied": True, "reason": "ok"},
    }
    sources = extract_sources(bundle)
    assert [s["file_path"] for s in sources] == ["a.md"]
    assert all("com-x-001" not in str(s) for s in sources)


def test_build_telemetry_records_community_fields():
    te = om.build_telemetry(
        question="整体对比",
        mode="mix",
        bundle={"data": {"chunks": [{"content": "x"}]}},
        latency_ms=12.5,
        community={
            "requested": True,
            "applied": True,
            "reason": "ok",
            "ids": ["com-a-001", "com-a-002"],
            "tokens": 321,
            "similarities": [0.71, 0.55],
        },
    )
    d = te.to_dict()
    assert d["community_requested"] is True
    assert d["community_applied"] is True
    assert d["community_reason"] == "ok"
    assert d["community_ids"] == "com-a-001,com-a-002"
    assert d["community_tokens"] == 321
    assert d["community_similarity"] == 0.71


def test_telemetry_sqlite_accepts_community_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(om, "_sqlite_path", lambda: tmp_path / "telemetry.sqlite3")
    te = om.build_telemetry(
        question="整体对比",
        mode="mix",
        bundle={},
        latency_ms=1.0,
        community={"requested": True, "applied": False, "reason": "low_similarity", "tokens": 0},
    )
    om._write_batch_to_sqlite([("query", te.to_dict())])
    conn = sqlite3.connect(str(tmp_path / "telemetry.sqlite3"))
    try:
        row = conn.execute(
            "SELECT community_requested, community_applied, community_reason, community_tokens "
            "FROM query_telemetry"
        ).fetchone()
    finally:
        conn.close()
    assert row == (1, 0, "low_similarity", 0)


async def test_attach_community_context_sets_block_and_status(tmp_path):
    s = Settings()
    s.paths.lightrag_working_dir = str(tmp_path)
    s.community.enabled = True
    store.reset_version_cache()
    store.save_reports(
        {"graph_fingerprint": "fpwiring00001", "built_at": "t1", "stale": False},
        {
            "com-fpwiring-001": {
                "title": "整体主题",
                "summary": "该主题覆盖多篇文档的共同方法。",
                "key_points": ["要点A"],
                "docs": ["a.md"],
                "members": ["m1"],
                "stale": False,
            }
        },
        s,
    )
    store.save_index(["com-fpwiring-001"], np.array([[1.0, 0.0]], dtype=np.float32), s)

    retriever = GraphRAGRetriever()
    retriever._settings = s

    class FakeRag:
        async def embedding_func(self, texts):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    bundle = await retriever._attach_community_context(
        {"data": {}}, "整体总结这几篇文档", "mix", True, FakeRag()
    )
    assert bundle["community"]["applied"] is True
    assert bundle["community"]["reason"] == "ok"
    assert bundle["community_context"]["ids"] == ["com-fpwiring-001"]
    assert "整体主题" in bundle["community_context"]["text"]
    assert retriever.last_community_status["applied"] is True


async def test_attach_community_context_degrades_on_error(tmp_path):
    s = Settings()
    s.paths.lightrag_working_dir = str(tmp_path)
    s.community.enabled = True
    retriever = GraphRAGRetriever()
    retriever._settings = s

    class BrokenRag:
        async def embedding_func(self, texts):
            raise RuntimeError("boom")

    # 索引缺失 → 直接拿到 index_missing 状态，不抛异常
    bundle = await retriever._attach_community_context({"data": {}}, "整体总结", "mix", True, BrokenRag())
    assert bundle["community"]["applied"] is False
    assert bundle["community"]["reason"] == "index_missing"


def test_macro_question_helper_matches_router():
    assert is_macro_question("整体总结一下")
    assert is_macro_question("Compare these methods")
    assert not is_macro_question("什么是循环码")
    assert suggest_mode_from_question("整体总结一下") == "global"
    assert suggest_mode_from_question("什么是循环码") == "naive"


async def test_update_manager_refresh_hook(monkeypatch):
    """入库后自动刷新钩子：能力关闭/手动模式/无变更时都不触发。"""
    from src.incremental import update_manager as um
    from src.community import builder

    calls: list[str] = []

    async def fake_refresh(*, reason: str = "incremental", settings=None):
        calls.append(reason)
        return {"built": 1}

    monkeypatch.setattr(builder, "refresh_communities", fake_refresh)
    mgr = um.UpdateManager.__new__(um.UpdateManager)  # 避开 Neo4j/KV 初始化

    s = Settings()
    monkeypatch.setattr(um, "get_settings", lambda: s)

    # 1) 能力关闭 → 不触发
    s.community.enabled = False
    await mgr._refresh_communities_if_needed({"added": 2})
    assert calls == []

    # 2) 能力开启但模式为 manual → 不触发
    s.community.enabled = True
    s.community.refresh_mode = "manual"
    await mgr._refresh_communities_if_needed({"added": 2})
    assert calls == []

    # 3) auto 但本次无任何变更 → 不触发
    s.community.refresh_mode = "auto"
    await mgr._refresh_communities_if_needed({"added": 0, "modified": 0, "removed": 0})
    assert calls == []

    # 4) auto 且有新增 → 触发一次
    await mgr._refresh_communities_if_needed({"added": 1})
    assert calls == ["incremental"]


async def test_update_manager_refresh_hook_swallows_errors(monkeypatch):
    """社区刷新失败不得影响入库结果。"""
    from src.incremental import update_manager as um
    from src.community import builder

    async def boom(*, reason: str = "incremental", settings=None):
        raise RuntimeError("模拟刷新失败")

    monkeypatch.setattr(builder, "refresh_communities", boom)
    mgr = um.UpdateManager.__new__(um.UpdateManager)
    s = Settings()
    s.community.enabled = True
    s.community.refresh_mode = "auto"
    monkeypatch.setattr(um, "get_settings", lambda: s)

    await mgr._refresh_communities_if_needed({"added": 1})  # 不应抛出
