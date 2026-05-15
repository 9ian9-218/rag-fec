"""增量更新流程：變更偵測 → 刪除舊索引 → 寫入新內容 → 更新快取與斷點。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import get_settings
from src.data_processing.change_detector import (
    ChangeReport,
    detect_changes,
    load_hash_cache,
    rebuild_hash_cache_for_directory,
    write_hash_cache,
)
from src.data_processing.document_loader import load_document
from src.incremental.cascade_cleaner import cascade_delete_document
from src.incremental.checkpoint_manager import CheckpointManager
from src.incremental.doc_registry import stable_doc_id
from src.storage.kv_client import KVClient
from src.storage.lightrag_init import get_lightrag
from src.utils.hash_utils import md5_file
from src.utils.logger import get_logger

logger = get_logger("incremental.update_manager")


class UpdateManager:
    """以目錄為單位驅動 LightRAG 增量管線。"""

    def __init__(self, raw_dir: Path | None = None, kv: KVClient | None = None) -> None:
        s = get_settings()
        root = Path(s.paths.project_root).resolve()
        self._raw = Path(raw_dir or s.paths.data_raw)
        if not self._raw.is_absolute():
            self._raw = (root / self._raw).resolve()
        self._kv = kv or KVClient()
        self._cp = CheckpointManager()

    async def run_incremental(self) -> dict[str, Any]:
        """偵測 ``data/raw`` 變更並套用至索引。"""
        s = get_settings()
        if not s.incremental.enabled:
            logger.info("增量更新已停用（INCREMENTAL_ENABLED=false）")
            return {"skipped": True}

        report = detect_changes([self._raw], recursive=True)
        rag = await get_lightrag()
        stats = {"added": 0, "modified": 0, "removed": 0, "errors": 0}
        ingested_ok: list[Path] = []
        interval = max(1, s.incremental.checkpoint_interval)
        cp = self._cp.load()
        tick = 0

        async def _maybe_checkpoint() -> None:
            nonlocal tick, cp
            tick += 1
            if tick % interval == 0:
                self._cp.save(cp)

        for path_str in report.removed:
            doc_id = stable_doc_id(Path(path_str))
            try:
                await cascade_delete_document(rag, doc_id, self._kv)
                stats["removed"] += 1
            except Exception as e:
                logger.error("刪除索引失敗 %s: %s", path_str, e)
                stats["errors"] += 1
            await _maybe_checkpoint()

        async def _ingest_one(path: Path, *, is_modify: bool) -> None:
            nonlocal stats, cp
            doc_id = stable_doc_id(path)
            try:
                if is_modify:
                    await cascade_delete_document(rag, doc_id, self._kv)
                loaded = load_document(path)
                text = loaded.text
                if not text.strip():
                    logger.warning("文件為空，略過索引: %s", path)
                    stats["errors"] += 1
                    return
                await rag.ainsert(text, ids=doc_id, file_paths=str(path))
                from lightrag.base import DocStatus

                failed = await rag.get_docs_by_status(DocStatus.FAILED)
                if doc_id in failed:
                    raise RuntimeError(
                        f"LightRAG 文件處理失敗（FAILED）: {path} — {failed[doc_id]}"
                    )
                self._kv.upsert_document(
                    doc_id,
                    str(path.resolve()),
                    loaded.metadata.get("md5_content"),
                    loaded.metadata.get("size_bytes"),
                )
                if is_modify:
                    stats["modified"] += 1
                else:
                    stats["added"] += 1
                cp.processed_paths.append(str(path.resolve()))
                ingested_ok.append(path)
            except Exception:
                logger.exception("處理文件失敗: %s", path)
                stats["errors"] += 1
            await _maybe_checkpoint()

        for path in report.modified:
            await _ingest_one(path, is_modify=True)

        for path in report.added:
            await _ingest_one(path, is_modify=False)

        cache = load_hash_cache()
        for path_str in report.removed:
            cache.pop(path_str, None)
        for path in ingested_ok:
            try:
                cache[str(path.resolve())] = md5_file(path)
            except OSError as e:
                logger.warning("無法更新 hash 快取 %s: %s", path, e)
        write_hash_cache(cache)

        cp.stats.update(stats)
        self._cp.save(cp)
        logger.info("增量更新完成: %s", stats)
        return {"stats": stats, "report": _serialize_report(report)}

    async def ingest_path(self, path: Path, *, replace: bool = False) -> dict[str, Any]:
        """單一路徑入庫（API 上傳用）。"""
        path = path.expanduser().resolve()
        rag = await get_lightrag()
        doc_id = stable_doc_id(path)
        if replace:
            await cascade_delete_document(rag, doc_id, self._kv)
        loaded = load_document(path)
        await rag.ainsert(loaded.text, ids=doc_id, file_paths=str(path))
        from lightrag.base import DocStatus

        failed = await rag.get_docs_by_status(DocStatus.FAILED)
        if doc_id in failed:
            raise RuntimeError(
                f"LightRAG 文件處理失敗（FAILED）: {path} — {failed[doc_id]}"
            )
        self._kv.upsert_document(
            doc_id,
            str(path),
            loaded.metadata.get("md5_content"),
            loaded.metadata.get("size_bytes"),
        )
        if self._raw.exists():
            c = load_hash_cache()
            try:
                c[str(path)] = md5_file(path)
            except OSError as e:
                logger.warning("無法更新 hash 快取 %s: %s", path, e)
            write_hash_cache(c)
        return {"doc_id": doc_id, "metadata": loaded.metadata}


def _serialize_report(r: ChangeReport) -> dict[str, list[str]]:
    return {
        "added": [str(p) for p in r.added],
        "modified": [str(p) for p in r.modified],
        "removed": list(r.removed),
    }
