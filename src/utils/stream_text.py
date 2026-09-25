"""流式输出的思考块过滤。

部分 OpenAI 兼容网关（如本项目 .env 配置的 opencode zen + deepseek-v4.1-flash）
会把模型的 ``<think>...</think>`` 推理内容**混在正文流里**一起返回。若原样转发，
用户看到的答案会先出现一大段英文推理。本模块提供跨 chunk 的剥离逻辑：

- 只处理**开头**的思考块；正文中间出现的 ``</think>`` 不触发剥离；
- 思考块未闭合（模型被截断）时整段丢弃，宁可空回复也不泄露推理；
- 探测窗口有界，不会随思考块长度无限增长内存。
"""

from __future__ import annotations

from typing import AsyncIterator

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TAIL_KEEP = len(_THINK_CLOSE) - 1  # 跨 chunk 拆分的闭合标签最多需要保留这么多字符

_PROBE = "probe"
_PASSTHROUGH = "passthrough"
_IN_THINK = "in_think"


async def strip_think_stream(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """过滤流式输出开头的 ``<think>...</think>``，其余内容原样透传。"""
    state = _PROBE
    buf = ""

    async for raw in chunks:
        text = str(raw)
        if not text:
            continue

        if state == _PASSTHROUGH:
            yield text
            continue

        buf += text

        if state == _IN_THINK:
            idx = buf.find(_THINK_CLOSE)
            if idx == -1:
                buf = buf[-_TAIL_KEEP:]
                continue
            rest = buf[idx + len(_THINK_CLOSE):]
            buf = ""
            state = _PASSTHROUGH
            if rest:
                yield rest
            continue

        # state == _PROBE：判断开头是不是思考块（或它的前缀）
        probe = buf.lstrip()
        if probe.startswith(_THINK_OPEN):
            buf = probe[len(_THINK_OPEN):]
            idx = buf.find(_THINK_CLOSE)
            if idx == -1:
                state = _IN_THINK
                buf = buf[-_TAIL_KEEP:]
                continue
            rest = buf[idx + len(_THINK_CLOSE):]
            buf = ""
            state = _PASSTHROUGH
            if rest:
                yield rest
            continue

        if _THINK_OPEN.startswith(probe):
            # 还可能是 "<thi" 这类被打散的标签前缀，继续缓冲
            continue

        out = buf
        buf = ""
        state = _PASSTHROUGH
        yield out
