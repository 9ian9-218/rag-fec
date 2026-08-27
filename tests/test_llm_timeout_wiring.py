"""T06 480s Worker 超时根因：LLM_TIMEOUT 传导 + 请求级超时传递。"""

from __future__ import annotations

import os

import pytest

from config.settings import apply_settings_to_environ, get_settings


def test_apply_settings_writes_llm_timeout_with_retry_headroom() -> None:
    """LLM_TIMEOUT 必须显式传导（缺省时 LightRAG 用 240s→480s 掐断），
    且要为 openai_complete 的 3 次重试预留预算。"""
    apply_settings_to_environ(get_settings())
    timeout = int(get_settings().llm.timeout)
    llm_timeout = int(os.environ["LLM_TIMEOUT"])
    # worker execution 超时 = 2×LLM_TIMEOUT，必须容纳 3 次请求尝试
    assert 3 * timeout <= 2 * llm_timeout
    assert llm_timeout >= 240


def test_apply_settings_writes_max_async_llm() -> None:
    apply_settings_to_environ(get_settings())
    assert os.environ["MAX_ASYNC_LLM"] == str(get_settings().llm.max_concurrent_calls)


@pytest.mark.asyncio
async def test_llm_call_passes_request_timeout(monkeypatch) -> None:
    """请求级超时必须传给 openai_complete，避免 SDK 默认 600s 与 Worker 超时脱节。"""
    from src.storage import lightrag_init as li

    seen: dict[str, object] = {}

    async def fake_complete(prompt, system_prompt=None, history_messages=None, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return "ok"

    monkeypatch.setattr("lightrag.llm.openai.openai_complete", fake_complete)
    llm = li._build_llm_func(get_settings())
    await llm("hello")
    assert seen.get("timeout") == int(get_settings().llm.timeout)