"""并发控制工具：为不同资源提供可复用的 asyncio.Semaphore。

查询期与索引插入期（insert phase）使用相互独立的信号量配额，
避免插入管线的 LLM/Embedding 调用挤占在线查询的并发额度。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import AsyncIterator

_semaphores: dict[str, tuple[asyncio.Semaphore, int]] = {}

# 当前执行阶段："query"（查询/生成期）或 "insert"（索引插入期）。
# 上下文传播规则：asyncio.create_task / async with 会复制 ContextVar，
# 因此嵌套在插入任务中的 LLM/Embedding 调用能看到正确的 phase。
_phase_ctx: ContextVar[str] = ContextVar("resource_phase", default="query")


def get_phase() -> str:
    """返回当前资源阶段（"query" 或 "insert"）。"""
    return _phase_ctx.get()


def set_phase(phase: str) -> None:
    """设置当前任务（及派生子任务）的资源阶段。"""
    _phase_ctx.set(phase)


@asynccontextmanager
async def insert_phase() -> AsyncIterator[None]:
    """将协程块的资源配额切到索引插入期，退出后恢复原阶段。"""
    prev = get_phase()
    set_phase("insert")
    try:
        yield
    finally:
        set_phase(prev)


def get_semaphore(key: str, limit: int) -> asyncio.Semaphore:
    """按 key 返回共享信号量；limit 变化时重建（一般启动后不变）。

    注意：不可用 ``sem._value != limit`` 判断重建——``_value`` 是运行时剩余许可，
    只要有任何协程正在持有许可就必然小于 limit，会导致每个调用方拿到全新的
    未占用信号量，让限流完全失效。这里显式记录初始 limit 作比较。
    """
    entry = _semaphores.get(key)
    if entry is None or entry[1] != limit:
        sem = asyncio.Semaphore(limit)
        _semaphores[key] = (sem, limit)
        return sem
    return entry[0]


def reset_semaphores() -> None:
    """测试清理用。"""
    _semaphores.clear()