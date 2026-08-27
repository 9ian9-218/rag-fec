"""T03 锁生命周期：长任务锁超时 + Redis 不可用时的进程内互斥回退。"""

from __future__ import annotations

import pytest

from src.storage import redis_lock
from src.storage.redis_lock import acquire_lock, release_lock


class _StubRedis:
    """行为级 stub：模拟 redis-py 语义——set 存 str，get 返回 bytes。"""

    def __init__(self, pre_occupied: bool = False) -> None:
        self.store: dict[str, str] = {}
        if pre_occupied:
            self.store["rag:lock:doc:a.md"] = "other-token"

    async def set(self, key, value, nx=True, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key):
        v = self.store.get(key)
        return v.encode() if isinstance(v, str) else v

    async def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0


@pytest.mark.asyncio
async def test_local_lock_acquire_release(monkeypatch) -> None:
    """Redis 不可用时退化为进程内互斥：并发获取失败，释放后可重获。"""
    monkeypatch.setattr(redis_lock, "get_redis_client", lambda: None)
    t1 = await acquire_lock("rag:lock:incremental", 60)
    assert t1 is not None
    assert await acquire_lock("rag:lock:incremental", 60) is None
    assert await release_lock("rag:lock:incremental", t1) is True
    t2 = await acquire_lock("rag:lock:incremental", 60)
    assert t2 is not None
    await release_lock("rag:lock:incremental", t2)


@pytest.mark.asyncio
async def test_local_lock_release_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(redis_lock, "get_redis_client", lambda: None)
    t = await acquire_lock("k", 60)
    assert t is not None
    assert await release_lock("k", t) is True
    # 重复释放不应抛错
    assert await release_lock("k", t) is True


@pytest.mark.asyncio
async def test_redis_occupied_rejects_same_key(monkeypatch) -> None:
    """Redis 明确返回失败（被其它进程持有）时必须拒绝，不降级为本地锁。"""
    monkeypatch.setattr(redis_lock, "get_redis_client", lambda: _StubRedis(pre_occupied=True))
    assert await acquire_lock("rag:lock:doc:a.md", 60) is None


@pytest.mark.asyncio
async def test_redis_acquired_releases_with_bytes_token(monkeypatch) -> None:
    """Redis 正常流程：拿到 token 后可释放（get 返回 bytes 也要比较成功）。"""
    stub = _StubRedis()
    monkeypatch.setattr(redis_lock, "get_redis_client", lambda: stub)
    t = await acquire_lock("rag:lock:doc:a.md", 60)
    assert t is not None
    assert t != ""
    # 模拟真实 redis-py：get 返回 bytes
    assert await release_lock("rag:lock:doc:a.md", t) is True
    assert "rag:lock:doc:a.md" not in stub.store


def test_default_lock_timeout_covers_long_docs() -> None:
    """锁默认超时必须覆盖长文档处理（曾有 480s+ 处理），不得再固定 60s。"""
    from config.settings import get_settings

    assert get_settings().service.lock_timeout_seconds >= 3600