"""限流工具：进程内令牌桶 + Redis 分布式固定窗口。"""

from __future__ import annotations

import threading
import time
from typing import Any

from src.storage.redis_client import get_redis_client


class TokenBucket:
    """简单的进程内令牌桶，线程安全。"""

    def __init__(self, rate_per_second: float, capacity: int) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._rate = float(rate_per_second)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self._capacity,
                self._tokens + (now - self._updated) * self._rate,
            )
            self._updated = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class RedisRateLimiter:
    """基于 Redis 的固定窗口限流，可跨多实例共享。"""

    def __init__(self, max_requests: float, window_seconds: int = 1) -> None:
        self._max_requests = float(max_requests)
        self._window = max(1, int(window_seconds))

    async def acquire(self, key: str = "global") -> bool:
        client = get_redis_client()
        if client is None:
            return True
        try:
            now = int(time.time())
            window_key = f"rag:rl:{key}:{now // self._window}"
            pipe = client.pipeline()
            pipe.incr(window_key)
            pipe.expire(window_key, self._window + 1)
            count, _ = await pipe.execute()
            return int(count) <= self._max_requests
        except Exception:
            # Redis 不可用时放行，避免限流组件拖垮主链路
            return True
