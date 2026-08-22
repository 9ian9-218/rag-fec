"""Redis 分布式锁工具。Redis 未启用时退化为本地无锁（直接成功）。"""

from __future__ import annotations

import uuid
from typing import Any

from src.storage.redis_client import get_redis_client

async def acquire_lock(lock_key: str, timeout: int = 30) -> str | None:
    """尝试获取锁；成功返回 token，失败或 Redis 未启用返回 None。"""
    client = get_redis_client()
    if client is None:
        return ""
    token = uuid.uuid4().hex
    try:
        ok = await client.set(lock_key, token, nx=True, ex=timeout)
        return token if ok else None
    except Exception:
        return None


async def release_lock(lock_key: str, token: str | None) -> bool:
    """释放锁；只有持有对应 token 才删除。"""
    client = get_redis_client()
    if client is None or not token:
        return True
    try:
        value = await client.get(lock_key)
        if value == token:
            await client.delete(lock_key)
            return True
        return False
    except Exception:
        return False
