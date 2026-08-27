"""T01 异步文档处理：任务状态机与默认开启行为。"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.service.task_manager import AsyncTaskManager


@pytest.mark.asyncio
async def test_submit_puts_pending_and_process_marks_done() -> None:
    tm = AsyncTaskManager()
    tid = await tm.submit("add_document", {"path": "/tmp/a.md"})

    st = tm.get_status(tid)
    assert st is not None
    assert st["status"] == "pending"
    assert st["kind"] == "add_document"

    async def handler(kind: str, payload: dict) -> str:
        return f"ok:{kind}:{payload['path']}"

    processed = await tm.process_next(handler)
    assert processed is True
    assert tm.get_status(tid)["status"] == "done"
    assert tm.get_status(tid)["result"] == "ok:add_document:/tmp/a.md"


@pytest.mark.asyncio
async def test_failed_task_records_error() -> None:
    tm = AsyncTaskManager()
    tid = await tm.submit("incremental_update", {})

    async def handler(kind: str, payload: dict):
        raise RuntimeError("boom")

    assert await tm.process_next(handler) is True
    st = tm.get_status(tid)
    assert st["status"] == "failed"
    assert "boom" in st["error"]


@pytest.mark.asyncio
async def test_worker_loop_consumes_multiple_tasks() -> None:
    tm = AsyncTaskManager()
    tids = [await tm.submit("add_document", {"path": f"/tmp/{i}.md"}) for i in range(3)]

    seen: list[str] = []

    async def handler(kind: str, payload: dict) -> str:
        seen.append(payload["path"])
        return "ok"

    # 手动驱动 worker_loop 有限次，避免无限 sleep
    loop_task = asyncio.create_task(tm.worker_loop(handler, interval=0.001))
    await asyncio.sleep(0.2)
    loop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await loop_task

    assert sorted(seen) == ["/tmp/0.md", "/tmp/1.md", "/tmp/2.md"]
    assert all(tm.get_status(t)["status"] == "done" for t in tids)


@pytest.mark.asyncio
async def test_empty_queue_returns_false() -> None:
    tm = AsyncTaskManager()

    async def handler(kind: str, payload: dict):
        raise AssertionError("should not be called")

    assert await tm.process_next(handler) is False


def test_default_document_processing_is_async() -> None:
    """没有显式配置时，文档上传/增量更新必须默认走异步 202 路径。"""
    from config.settings import get_settings

    s = get_settings()
    assert s.service.async_document_processing_enabled is True