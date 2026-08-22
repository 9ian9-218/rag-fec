"""并发控制工具：为不同资源提供可复用的 asyncio.Semaphore。"""

from __future__ import annotations

import asyncio
from typing import Any


_semaphores: dict[str, asyncio.Semaphore] = {}


def get_semaphore(key: str, limit: int) -> asyncio.Semaphore:
    """按 key 返回共享信号量；limit 变化时重建（一般启动后不变）。"""
    sem = _semaphores.get(key)
    if sem is None or sem._value != limit:
        sem = asyncio.Semaphore(limit)
        _semaphores[key] = sem
    return sem


def reset_semaphores() -> None:
    """测试清理用。"""
    _semaphores.clear()
