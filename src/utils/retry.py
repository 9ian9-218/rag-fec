"""外部 API 调用重试工具：有限指数退避，避免重试风暴。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    min_wait: float = 0.5,
    max_wait: float = 5.0,
) -> T:
    """执行 ``fn`` 并在失败时按指数退避重试最多 ``max_retries`` 次。"""
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            return await fn()
    raise RuntimeError("unreachable")  # pragma: no cover
