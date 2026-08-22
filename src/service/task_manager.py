"""轻量异步任务管理器：内存队列 + 可选 Redis Stream 生产。"""

from __future__ import annotations

import asyncio
import queue
import uuid
from typing import Any, Awaitable, Callable

from src.storage.redis_client import get_redis_client

TaskHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]


class AsyncTaskManager:
    def __init__(self, redis_stream: str = "rag:tasks") -> None:
        self._queue: queue.Queue[tuple[str, str, dict[str, Any]]] = queue.Queue()
        self._statuses: dict[str, dict[str, Any]] = {}
        self._redis_stream = redis_stream

    async def submit(self, kind: str, payload: dict[str, Any]) -> str:
        task_id = uuid.uuid4().hex
        self._statuses[task_id] = {
            "task_id": task_id,
            "kind": kind,
            "status": "pending",
            "payload": payload,
        }
        self._queue.put((task_id, kind, payload))

        client = get_redis_client()
        if client is not None:
            try:
                await client.xadd(
                    self._redis_stream,
                    {
                        "task_id": task_id,
                        "kind": kind,
                        "payload_json": __import__("json").dumps(payload, ensure_ascii=False),
                    },
                )
            except Exception:
                # Redis Stream 写入失败不阻塞内存队列
                pass
        return task_id

    def get_status(self, task_id: str) -> dict[str, Any] | None:
        return self._statuses.get(task_id)

    async def process_next(self, handler: TaskHandler) -> bool:
        try:
            task_id, kind, payload = self._queue.get_nowait()
        except queue.Empty:
            return False
        self._statuses[task_id]["status"] = "running"
        try:
            result = await handler(kind, payload)
            self._statuses[task_id].update({"status": "done", "result": result})
        except Exception as e:
            self._statuses[task_id].update(
                {"status": "failed", "error": str(e)}
            )
        return True

    async def worker_loop(self, handler: TaskHandler, interval: float = 0.05) -> None:
        while True:
            processed = await self.process_next(handler)
            if not processed:
                await asyncio.sleep(interval)
