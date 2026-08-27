"""Redis 分布式锁工具。Redis 未启用或异常时退化为进程内 asyncio 互斥锁。"""

from __future__ import annotations

import asyncio
import uuid

from src.storage.redis_client import get_redis_client
from src.utils.logger import get_logger

logger = get_logger("storage.redis_lock")

# 进程内互斥锁回退：Redis 不可用时保证同一进程内同 key 任务不并发。
_local_locks: dict[str, asyncio.Lock] = {}


async def _acquire_local(lock_key: str) -> str | None:
    """进程内互斥：已被占用返回 None，否则持有并返回 token。"""
    lock = _local_locks.setdefault(lock_key, asyncio.Lock())
    if lock.locked():
        return None
    await lock.acquire()
    return f"local:{id(lock)}"


async def _release_local(lock_key: str, token: str) -> bool:
    lock = _local_locks.get(lock_key)
    if lock is None:
        return True
    if f"local:{id(lock)}" != token:
        return False
    if lock.locked():
        lock.release()
    return True


async def acquire_lock(lock_key: str, timeout: int = 30) -> str | None:
    """尝试获取锁；成功返回 token，失败返回 None。

    - Redis 可用：普通 SET NX EX；已占用则拒绝（跨进程语义）。
    - Redis 不可用/异常：退化为进程内互斥锁（同进程语义）。
    """
    client = get_redis_client()
    if client is None:
        return await _acquire_local(lock_key)
    token = uuid.uuid4().hex
    try:
        ok = await client.set(lock_key, token, nx=True, ex=timeout)
        return token if ok else None
    except Exception:
        logger.warning("Redis 锁不可用，降级为进程内锁: %s", lock_key)
        return await _acquire_local(lock_key)


async def release_lock(lock_key: str, token: str | None) -> bool:
    """释放锁；只有持有对应 token 才删除。"""
    if not token:
        return True
    if isinstance(token, str) and token.startswith("local:"):
        return await _release_local(lock_key, token)
    client = get_redis_client()
    if client is None:
        return True
    try:
        value = await client.get(lock_key)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if value == token:
            await client.delete(lock_key)
            return True
        return False
    except Exception:
        return False