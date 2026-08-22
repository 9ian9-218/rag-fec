"""Redis 客户端封装：懒加载单例、健康检查、关闭。"""

from __future__ import annotations

import threading
from typing import Any

from config.settings import Settings, get_settings

_redis_client: Any = None
_redis_lock = threading.Lock()

_sync_redis_client: Any = None
_sync_redis_lock = threading.Lock()


def create_redis_client(settings: Settings | None = None) -> Any:
    """根据配置创建 redis.asyncio 客户端；未启用时返回 None。"""
    import redis.asyncio as aioredis

    s = settings or get_settings()
    if not s.redis.enabled:
        return None
    return aioredis.Redis(
        host=s.redis.host,
        port=s.redis.port,
        db=s.redis.db,
        password=s.redis.password or None,
        socket_timeout=s.redis.socket_timeout,
        decode_responses=True,
    )


def get_redis_client(settings: Settings | None = None) -> Any:
    """返回全局 Redis 异步客户端单例；未启用时返回 None。"""
    global _redis_client
    s = settings or get_settings()
    if not s.redis.enabled:
        return None
    with _redis_lock:
        if _redis_client is None:
            _redis_client = create_redis_client(s)
        return _redis_client


def create_sync_redis_client(settings: Settings | None = None) -> Any:
    """创建同步 redis.Redis 客户端（供后台线程使用）；未启用时返回 None。"""
    import redis

    s = settings or get_settings()
    if not s.redis.enabled:
        return None
    return redis.Redis(
        host=s.redis.host,
        port=s.redis.port,
        db=s.redis.db,
        password=s.redis.password or None,
        socket_timeout=s.redis.socket_timeout,
        decode_responses=True,
    )


def get_sync_redis_client(settings: Settings | None = None) -> Any:
    """返回全局同步 Redis 客户端单例（后台线程使用）。"""
    global _sync_redis_client
    s = settings or get_settings()
    if not s.redis.enabled:
        return None
    with _sync_redis_lock:
        if _sync_redis_client is None:
            _sync_redis_client = create_sync_redis_client(s)
        return _sync_redis_client


async def ping_redis(settings: Settings | None = None) -> bool:
    client = get_redis_client(settings)
    if client is None:
        return False
    try:
        return bool(await client.ping())
    except Exception:
        return False


async def close_redis() -> None:
    global _redis_client
    with _redis_lock:
        client = _redis_client
        _redis_client = None
    if client is not None:
        try:
            await client.aclose()
        except Exception:
            pass

    global _sync_redis_client
    with _sync_redis_lock:
        sync_client = _sync_redis_client
        _sync_redis_client = None
    if sync_client is not None:
        try:
            sync_client.close()
        except Exception:
            pass
