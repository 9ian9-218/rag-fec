"""T02 查询与插入的并发配额分离：phase 上下文 + 独立信号量。"""

from __future__ import annotations

import pytest

from src.utils import concurrency


def test_phase_defaults_to_query() -> None:
    assert concurrency.get_phase() == "query"


@pytest.mark.asyncio
async def test_insert_phase_restores_previous() -> None:
    prev = concurrency.get_phase()
    async with concurrency.insert_phase():
        assert concurrency.get_phase() == "insert"
    assert concurrency.get_phase() == prev


def test_semaphores_separated_per_phase() -> None:
    concurrency.reset_semaphores()
    q = concurrency.get_semaphore("llm:query", 8)
    i = concurrency.get_semaphore("llm:insert", 4)
    assert q is not i


def test_phase_scoped_semaphore_rebuilt_on_limit_change() -> None:
    concurrency.reset_semaphores()
    s1 = concurrency.get_semaphore("llm:insert", 4)
    s2 = concurrency.get_semaphore("llm:insert", 6)
    assert s1 is not s2


@pytest.mark.asyncio
async def test_semaphore_not_rebuilt_while_held() -> None:
    """持有许可期间再次 get_semaphore 必须返回同一实例（限流才有效）。"""
    import asyncio

    concurrency.reset_semaphores()
    sem = concurrency.get_semaphore("embedding:insert", 2)

    async with sem:  # 剩余许可变为 1
        again = concurrency.get_semaphore("embedding:insert", 2)
        assert again is sem, "持有期间信号量被重建，限流失效"


def test_settings_default_insert_concurrency_limits() -> None:
    from config.settings import get_settings

    s = get_settings()
    assert s.llm.max_concurrent_insert_calls >= 1
    assert s.embedding.max_concurrent_insert_calls >= 1
    # 插入配额必须低于查询配额，否则隔离无意义
    assert s.llm.max_concurrent_insert_calls <= s.llm.max_concurrent_calls


@pytest.mark.asyncio
async def test_llm_insert_phase_concurrency_capped_by_insert_limit(monkeypatch) -> None:
    """插入期 8 路并发 LLM 调用，实际并发峰值不得超过插入配额。"""
    import asyncio

    from src.storage import lightrag_init as li

    ctl = {"active": 0, "peak": 0}

    async def fake_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        ctl["active"] += 1
        ctl["peak"] = max(ctl["peak"], ctl["active"])
        await asyncio.sleep(0.02)
        ctl["active"] -= 1
        return "done"

    monkeypatch.setattr("lightrag.llm.openai.openai_complete", fake_complete)
    llm = li._build_llm_func(li.get_settings())
    insert_limit = int(li.get_settings().llm.max_concurrent_insert_calls)

    async with concurrency.insert_phase():
        outs = await asyncio.gather(*[llm(f"q{i}") for i in range(8)])

    assert outs == ["done"] * 8
    assert 1 <= ctl["peak"] <= insert_limit


@pytest.mark.asyncio
async def test_llm_query_phase_concurrency_not_capped_by_insert_limit(monkeypatch) -> None:
    """查询期并发不受插入配额限制（默认 query phase）。"""
    import asyncio

    from src.storage import lightrag_init as li

    ctl = {"active": 0, "peak": 0}

    async def fake_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        ctl["active"] += 1
        ctl["peak"] = max(ctl["peak"], ctl["active"])
        await asyncio.sleep(0.02)
        ctl["active"] -= 1
        return "done"

    monkeypatch.setattr("lightrag.llm.openai.openai_complete", fake_complete)
    llm = li._build_llm_func(li.get_settings())
    insert_limit = int(li.get_settings().llm.max_concurrent_insert_calls)

    assert concurrency.get_phase() == "query"
    outs = await asyncio.gather(*[llm(f"q{i}") for i in range(6)])

    assert outs == ["done"] * 6
    # 查询期并发峰值应超过插入配额（插桩环境无真实 API 延迟，6 路全部并发）
    assert ctl["peak"] > insert_limit