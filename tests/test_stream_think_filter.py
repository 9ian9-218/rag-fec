"""流式 <think> 推理块过滤回归。"""

from __future__ import annotations

from src.utils.stream_text import strip_think_stream


async def _collect(chunks: list[str]) -> str:
    async def _gen():
        for c in chunks:
            yield c

    return "".join([c async for c in strip_think_stream(_gen())])


async def test_strips_single_chunk_think_block() -> None:
    assert await _collect(["<think>推理</think>正式答案"]) == "正式答案"


async def test_strips_think_block_split_across_chunks() -> None:
    # 逐字分块，模拟真实流式
    chunks = list("<think>The user asks something</think>最终答案文本")
    assert await _collect(chunks) == "最终答案文本"


async def test_passes_through_plain_answer() -> None:
    assert await _collect(["直接就是答案"]) == "直接就是答案"


async def test_probe_prefix_is_buffered_not_lost() -> None:
    assert await _collect(["<thi", "nk>xx</thi", "nk>答案"]) == "答案"


async def test_unclosed_think_block_is_dropped() -> None:
    # 宁可空回复，也不把推理内容当答案输出
    assert await _collect(["<think>只有推理，没有闭合"]) == ""


async def test_mid_text_closing_tag_is_kept() -> None:
    assert await _collect(["答案里提到 </think> 这个词"]) == "答案里提到 </think> 这个词"


async def test_leading_whitespace_before_think() -> None:
    assert await _collect(["\n  <think>x</think>答案"]) == "答案"
