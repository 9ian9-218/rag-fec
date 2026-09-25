"""社区摘要注入：三态开关、门控原因、相似度阈值与 token 预算。"""

from __future__ import annotations

import numpy as np
import pytest

from config.settings import Settings
from src.community import store
from src.community.injector import render_reports, select_community_context


@pytest.fixture()
def community_env(tmp_path):
    """构造一个已构建好的社区索引（2 个摘要 + 正交向量）。"""
    s = Settings()
    s.paths.lightrag_working_dir = str(tmp_path)
    store.reset_version_cache()
    store.save_reports(
        {"graph_fingerprint": "fingerprint0001", "built_at": "2026-01-01T00:00:00Z", "stale": False},
        {
            "com-fingerpr-001": {
                "title": "RPA 译码",
                "summary": "递归投影聚合译码把长码递归分解为短码处理，降低复杂度。",
                "key_points": ["投影-聚合两步", "列表译码版本接近 ML"],
                "docs": ["Recursive_RPA.md"],
                "members": ["m1", "m2"],
                "stale": False,
            },
            "com-fingerpr-002": {
                "title": "RS 编码复杂度",
                "summary": "基于 Reed-Muller 变换的 RS 编码达到近线性计算复杂度。",
                "key_points": [],
                "docs": ["A_Class.md"],
                "members": ["m3"],
                "stale": False,
            },
        },
        s,
    )
    store.save_index(
        ["com-fingerpr-001", "com-fingerpr-002"],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        s,
    )
    s.community.enabled = True
    s.community.default_enabled = False
    return s


def _embed(vector):
    async def _fn(texts):
        return np.asarray([vector], dtype=np.float32)

    return _fn


async def test_server_disabled_wins_even_if_requested(community_env):
    community_env.community.enabled = False
    out = await select_community_context(
        "整体总结一下", use_community=True, mode="global",
        settings=community_env, embed_fn=_embed([1.0, 0.0]),
    )
    assert out.requested is True
    assert out.applied is False
    assert out.reason == "disabled_by_server"
    assert out.text == ""


async def test_explicit_off_overrides_server_default(community_env):
    community_env.community.default_enabled = True
    out = await select_community_context(
        "整体总结一下", use_community=False, mode="global",
        settings=community_env, embed_fn=_embed([1.0, 0.0]),
    )
    assert out.applied is False
    assert out.reason == "disabled_by_request"


async def test_default_none_requires_macro_question(community_env):
    community_env.community.default_enabled = True
    factual = await select_community_context(
        "T 是什么？", use_community=None, mode="mix",
        settings=community_env, embed_fn=_embed([1.0, 0.0]),
    )
    assert factual.applied is False
    assert factual.reason == "not_macro_question"
    macro = await select_community_context(
        "这几篇论文整体对比如何？", use_community=None, mode="mix",
        settings=community_env, embed_fn=_embed([1.0, 0.0]),
    )
    assert macro.applied is True
    assert macro.reason == "ok"


async def test_explicit_on_skips_macro_gate(community_env):
    out = await select_community_context(
        "T 是什么？", use_community=True, mode="mix",
        settings=community_env, embed_fn=_embed([1.0, 0.0]),
    )
    assert out.applied is True
    assert out.ids == ["com-fingerpr-001"]


async def test_mode_gate_blocks_naive_and_local(community_env):
    for mode in ("naive", "local", "bypass"):
        out = await select_community_context(
            "整体总结", use_community=True, mode=mode,
            settings=community_env, embed_fn=_embed([1.0, 0.0]),
        )
        assert out.applied is False
        assert out.reason == "mode_not_eligible"


async def test_low_similarity_reason(community_env):
    community_env.community.sim_threshold = 0.99
    out = await select_community_context(
        "整体总结", use_community=True, mode="global",
        settings=community_env, embed_fn=_embed([0.7071, 0.7071]),
    )
    assert out.applied is False
    assert out.reason == "low_similarity"


async def test_top_k_and_token_budget(community_env):
    community_env.community.top_k = 2
    community_env.community.max_tokens = 400
    out = await select_community_context(
        "整体总结", use_community=True, mode="global",
        settings=community_env, embed_fn=_embed([1.0, 0.05]),
    )
    assert out.applied is True
    assert 1 <= len(out.ids) <= 2
    assert out.tokens <= 400
    assert "全局" not in out.text  # 只渲染社区内容，标题由回答模板加


async def test_index_missing_and_stale(community_env, tmp_path):
    # 目录里没有任何文件 → index_missing
    empty = Settings()
    empty.paths.lightrag_working_dir = str(tmp_path / "empty")
    empty.community.enabled = True
    store.reset_version_cache()
    out = await select_community_context(
        "整体总结", use_community=True, mode="global",
        settings=empty, embed_fn=_embed([1.0, 0.0]),
    )
    assert out.reason == "index_missing"

    # 标记 stale → index_stale
    store.save_reports(
        {"graph_fingerprint": "fp2", "built_at": "t", "stale": True},
        {"com-1": {"title": "x", "summary": "y", "members": ["a"]}},
        community_env,
    )
    store.reset_version_cache()
    out2 = await select_community_context(
        "整体总结", use_community=True, mode="global",
        settings=community_env, embed_fn=_embed([1.0, 0.0]),
    )
    assert out2.reason == "index_stale"


def test_render_reports_respects_token_budget():
    reps = [
        ("com-1", {"title": f"主题{i}", "summary": "内容" * 100, "key_points": ["a", "b"], "docs": ["x.md"]}, 0.9)
        for i in range(3)
    ]
    text, tokens = render_reports(reps, 200)
    assert text
    assert tokens <= 200
    # 预算极小时不返回内容
    assert render_reports(reps, 0) == ("", 0)
