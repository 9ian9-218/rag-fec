"""Rerank 语义回归：本地/线上后端判定、假能力消除、遥测口径。

背景：修复前 `relation_optimizer` 读 `settings.models.rerank_batch_size`（字段不存在）
导致关系重排恒失败，且 `mode_config` 用「sentence-transformers 是否安装」判断是否开启
chunk 精排 —— 已安装但无权重时 LightRAG 只 warning 后原序返回。这些测试锁住修复后的语义。
"""

from __future__ import annotations

from config.model_paths import rerank_backend_available, resolve_local_reranker_dir
from config.settings import Settings
from src.evaluation.online_monitor import build_telemetry, clear_rerank_stats, set_rerank_stats
from src.storage.bge_rerank import build_local_rerank_model_func


def _settings_without_backend() -> Settings:
    """线上 rerank 关闭、本地无权重的最小配置。"""
    s = Settings()
    s.models.rerank_api_enabled = False
    s.models.rerank_api_key = ""
    s.models.rerank_api_base_url = None
    s.models.rerank_api_model_name = ""
    s.models.reranker_local_path = "/nonexistent/reranker"
    return s


def _settings_with_online() -> Settings:
    s = _settings_without_backend()
    s.models.rerank_api_enabled = True
    s.models.rerank_api_key = "k"
    s.models.rerank_api_base_url = "https://example.invalid"
    s.models.rerank_api_model_name = "BAAI/bge-reranker-v2-m3"
    return s


def test_rerank_batch_size_field_exists() -> None:
    """`rerank_batch_size` 必须真实存在于 ModelsSettings（关系重排曾因此 AttributeError）。"""
    assert int(Settings().models.rerank_batch_size) >= 1


def test_backend_unavailable_without_online_or_local_weights() -> None:
    assert rerank_backend_available(_settings_without_backend()) is False
    assert resolve_local_reranker_dir(_settings_without_backend()) is None


def test_backend_available_with_complete_online_config() -> None:
    assert rerank_backend_available(_settings_with_online()) is True


def test_backend_unavailable_when_online_config_incomplete() -> None:
    """开关打开但缺 KEY/BASE_URL/MODEL_NAME 时不算可用（线上函数会返回 None）。"""
    s = _settings_with_online()
    s.models.rerank_api_key = ""
    assert rerank_backend_available(s) is False


def test_local_rerank_func_is_none_without_weights() -> None:
    """无本地权重时返回 None —— 不得在查询路径里触发 Hub 下载。"""
    assert build_local_rerank_model_func(_settings_without_backend()) is None


async def test_local_rerank_func_satisfies_lightrag_contract(tmp_path, monkeypatch) -> None:
    """有权重时：按分数降序返回 [{index, relevance_score}]，并写入 rerank 遥测。

    注意必须用 async 测试（而非 asyncio.run）：rerank 遥测走 ContextVar，
    asyncio.run 会在新 context 里执行，set 的值不会传回调用方。
    """
    from src.evaluation.online_monitor import rerank_stats_ctx
    from src.storage import bge_rerank

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    s = _settings_without_backend()
    s.models.reranker_local_path = str(tmp_path)

    class FakeCrossEncoder:
        def predict(self, pairs, batch_size=None, show_progress_bar=None):
            # 第 2 篇最相关、第 1 篇最不相关（输入顺序 ≠ 输出顺序）
            return [0.1, 0.9, 0.5][: len(pairs)]

    monkeypatch.setattr(bge_rerank, "_get_cross_encoder", lambda *a, **k: FakeCrossEncoder())

    fn = build_local_rerank_model_func(s)
    assert fn is not None

    clear_rerank_stats()
    out = await fn(query="q", documents=["最不相关", "最相关", "中间"], top_n=3)

    assert [o["index"] for o in out] == [1, 2, 0]  # 0.9 > 0.5 > 0.1
    assert out[0]["relevance_score"] == 1.0  # min-max 归一化后最高分为 1
    stats = rerank_stats_ctx.get()
    assert stats and stats["candidates"] == 3 and stats["returned"] == 3
    clear_rerank_stats()


def test_query_param_disables_rerank_when_no_backend(monkeypatch) -> None:
    """没有可用后端时必须显式 enable_rerank=False，而不是留 True 让 LightRAG 空跑。"""
    from src.retrieval import mode_config

    monkeypatch.setattr(mode_config, "rerank_backend_available", lambda *a, **k: False)
    assert mode_config.build_query_param("mix", top_k=8).enable_rerank is False

    monkeypatch.setattr(mode_config, "rerank_backend_available", lambda *a, **k: True)
    assert mode_config.build_query_param("mix", top_k=8).enable_rerank is True


def test_telemetry_rerank_filter_rate_not_from_chunk_truncation() -> None:
    """rerank 未运行时过滤率必须为 0，不能把 chunk 截断率（merged→final）当成重排过滤率。"""
    clear_rerank_stats()
    bundle = {
        "status": "success",
        "data": {
            "entities": [{"entity_name": "RM 码"}],
            "relationships": [{"src_id": "RM 码", "tgt_id": "译码"}],
            "chunks": [{"chunk_id": "c1", "content": "RM 码译码流程"}],
        },
        "metadata": {
            "query_mode": "mix",
            "processing_info": {"merged_chunks_count": 18, "final_chunks_count": 9},
        },
    }
    t = build_telemetry(question="RM 码译码", mode="mix", bundle=bundle, latency_ms=10.0)
    assert t.merged_chunks_count == 18
    assert t.final_chunks_count == 9
    assert t.chunk_truncation_rate == 0.5
    assert t.rerank_filter_rate == 0.0
    assert t.rerank_candidates == 0


def test_telemetry_rerank_filter_rate_from_real_rerank_stats() -> None:
    """rerank 真跑过时用 (candidates-returned)/candidates。"""
    set_rerank_stats(candidates=9, returned=4, below_min_score=0)
    try:
        bundle = {
            "status": "success",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [{"chunk_id": "c1", "content": "x"}],
            },
            "metadata": {
                "query_mode": "mix",
                "processing_info": {"merged_chunks_count": 9, "final_chunks_count": 4},
            },
        }
        t = build_telemetry(question="q", mode="mix", bundle=bundle, latency_ms=10.0)
        assert t.rerank_candidates == 9
        assert t.rerank_returned == 4
        # 遥测对过滤率做 round(4)，故按 4 位小数比较
        assert abs(t.rerank_filter_rate - round((9 - 4) / 9, 4)) < 1e-9
    finally:
        clear_rerank_stats()
