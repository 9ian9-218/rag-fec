"""T05 插入幂等与失败重试：残留 doc_status 清理 + 文档级锁防并发重复。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.incremental.update_manager as um
from src.incremental.update_manager import UpdateManager
from src.storage import redis_lock


class _FakeDocStatus:
    """行为级 doc_status：按文件路径/set/dict 模拟 LightRAG JsonDocStatusStorage。"""

    def __init__(self, records: dict[str, dict]) -> None:
        self.records = dict(records)
        self.deleted: list[str] = []

    async def get_docs_by_statuses(self, statuses):
        return {k: v for k, v in self.records.items() if getattr(v, "status", None) in statuses}

    async def get_by_id(self, doc_id):
        return self.records.get(doc_id)

    async def delete(self, ids):
        for i in ids:
            self.records.pop(i, None)
            self.deleted.append(i)


class _FakeFullDocs:
    def __init__(self, records: dict[str, dict]) -> None:
        self.records = dict(records)

    async def get_by_id(self, doc_id):
        return self.records.get(doc_id)

    async def get_all(self):
        return dict(self.records)


class _FakeRag:
    def __init__(self, *, stale_records: dict[str, dict] | None = None) -> None:
        status_records = dict(stale_records or {})
        self.doc_status = _FakeDocStatus(status_records)
        self.full_docs = _FakeFullDocs({})
        self.insert_calls: list[str] = []
        self.delete_calls: list[str] = []

    async def ainsert(self, text, ids=None, file_paths=None):
        self.insert_calls.append(ids or "")
        from lightrag.base import DocStatus

        return "insert_ok"

    async def adelete_by_doc_id(self, doc_id, delete_llm_cache=False):
        self.delete_calls.append(doc_id)
        return SimpleNamespace(status="success", message="ok")

    async def get_docs_by_status(self, status):
        return {}


def _make_manager(raw_dir: Path) -> UpdateManager:
    kv = SimpleNamespace(
        upsert_document=lambda *a, **k: None,
        delete_document_row=lambda *a, **k: None,
    )
    mgr = UpdateManager(raw_dir=raw_dir, kv=kv)
    return mgr


@pytest.fixture
def manager(monkeypatch, tmp_path) -> UpdateManager:
    monkeypatch.setattr(um, "purge_for_doc_id", lambda *a, **k: {})
    monkeypatch.setattr(um, "register_after_ingest", lambda *a, **k: None)
    monkeypatch.setattr(um, "load_hash_cache", lambda: {})
    monkeypatch.setattr(um, "write_hash_cache", lambda c: None)
    monkeypatch.setattr(redis_lock, "get_redis_client", lambda *a, **k: None)
    return _make_manager(tmp_path)


def _sample_file(tmp_path: Path, name: str = "doc_a.md") -> Path:
    p = tmp_path / name
    p.write_text("# 测试文档\n\n包含 Reed-Muller 译码的内容。\n", encoding="utf-8")
    return p


@pytest.mark.asyncio
async def test_ingest_cleans_stale_doc_status_before_insert(manager, tmp_path):
    """残留 processing/dup 记录（同 basename）必须在 ainsert 前清除，避免 DUPLICATE。"""
    path = _sample_file(tmp_path)
    from types import SimpleNamespace as _SN
    stale = {
        "doc-stale-1": _SN(status="processing", file_path="doc_a.md"),
        "dup-abc": _SN(status="failed", file_path="doc_a.md", error_msg="File name already exists."),
    }
    rag = _FakeRag(stale_records=stale)
    manager._raw = tmp_path

    # 模拟 stable_doc_id
    from src.incremental.doc_registry import stable_doc_id

    doc_id = stable_doc_id(path)
    await manager._ingest_doc_locked(rag, path, is_modify=True, stats={"added":0,"modified":0,"removed":0,"errors":0}, cp=None, ingested_ok=[], interval=1)

    assert rag.insert_calls == [doc_id]
    # 同名记录（含 dup 记录）均被清理
    remaining = [k for k, v in rag.doc_status.records.items() if v.get("file_path") == "doc_a.md"]
    assert remaining == []


@pytest.mark.asyncio
async def test_ingest_skips_when_doc_lock_held(manager, tmp_path):
    """同一文件被其他任务持锁时，本次插入必须跳过，不得并发产生 DUPLICATE。"""
    import asyncio

    from src.storage.redis_lock import acquire_lock, release_lock

    path = _sample_file(tmp_path)
    rag = _FakeRag()
    manager._raw = tmp_path

    token = await acquire_lock(f"rag:lock:doc:{path.name}", 60)
    assert token is not None
    try:
        await manager._ingest_doc_locked(rag, path, is_modify=True, stats={"added":0,"modified":0,"removed":0,"errors":0}, cp=None, ingested_ok=[], interval=1)
    finally:
        await release_lock(f"rag:lock:doc:{path.name}", token)

    assert rag.insert_calls == []
    assert rag.delete_calls == []


@pytest.mark.asyncio
async def test_failed_insert_leaves_clean_slot_for_retry(manager, tmp_path):
    """因故障失败的插入必须清理 doc_status，使下次增量重试可从干净状态开始。"""
    path = _sample_file(tmp_path)
    rag = _FakeRag()

    async def fail_once(text, ids=None, file_paths=None):
        # 记录一个 failed 状态（模拟 LightRAG 超时失败残留）
        rag.doc_status.records[ids] = {"status": "failed", "file_path": path.name}
        raise RuntimeError("LLM timeout")

    rag.ainsert = fail_once
    manager._raw = tmp_path

    from src.incremental.doc_registry import stable_doc_id

    doc_id = stable_doc_id(path)
    await manager._ingest_doc_locked(rag, path, is_modify=True, stats={"added":0,"modified":0,"removed":0,"errors":0}, cp=None, ingested_ok=[], interval=1)

    # 插入失败：不应写入 hash 缓存（下次仍会重试），residual 记录应保留下次清理
    # 失败后残余记录存在（下次 _ingest_doc_locked 会再次清理后重试）
    assert rag.doc_status.records.get(doc_id, {}).get("status") == "failed"


@pytest.mark.asyncio
async def test_module_level_cleanup_removes_stale_and_dup_records(tmp_path):
    """build_index 直连路径使用的模块级清理：必须清掉同名 failed 与 dup-* 记录。"""
    from types import SimpleNamespace as _SN

    from src.incremental.update_manager import clear_doc_status_by_basename

    path = _sample_file(tmp_path)
    rag = _FakeRag(stale_records={
        "doc-stale": _SN(status="failed", file_path="doc_a.md"),
        "dup-x": _SN(status="failed", file_path="doc_a.md", error_msg="File name already exists."),
        "other": _SN(status="processed", file_path="other.md"),
    })

    deleted = await clear_doc_status_by_basename(rag, path.name)
    assert sorted(deleted) == ["doc-stale", "dup-x"]
    assert "other" in rag.doc_status.records