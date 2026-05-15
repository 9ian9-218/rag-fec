"""通用小工具函式。"""

from __future__ import annotations

import asyncio
import socket
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def ensure_dir(path: Path) -> Path:
    """建立目錄（含父層）並回傳路徑。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def wait_for_tcp(host: str, port: int, timeout_sec: float = 120.0, interval: float = 2.0) -> bool:
    """輪詢 TCP 連線直到成功或逾時；成功回傳 True。"""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return True
        except OSError:
            time.sleep(interval)
    return False


async def run_sync(fn: Callable[[], T]) -> T:
    """在預設 executor 執行同步函式，避免阻塞事件迴圈。"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn)


async def gather_limited(
    coros: list[Awaitable[T]],
    limit: int = 4,
) -> list[T | BaseException]:
    """以有限併發度執行多個 awaitable。"""
    sem = asyncio.Semaphore(limit)

    async def _wrap(c: Awaitable[T]) -> T:
        async with sem:
            return await c

    return await asyncio.gather(*(_wrap(c) for c in coros), return_exceptions=True)
