"""增量更新流程：變更偵測 → 刪除舊索引 → 寫入新內容 → 更新快取與斷點。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import get_settings
from src.data_processing.change_detector import (
    ChangeReport,
    detect_changes,
    detect_conversion_changes,
    detect_index_changes,
    load_conversion_cache,
    load_hash_cache,
    rebuild_hash_cache_for_directory,
    write_conversion_cache,
    write_hash_cache,
)
from src.incremental.conversion_manager import markdown_path_for_pdf
from src.data_processing.document_loader import load_document
from src.incremental.cascade_cleaner import cascade_delete_document
from src.incremental.checkpoint_manager import CheckpointManager
from src.incremental.document_manifest import (
    legacy_cleanup_markdown_sidecars,
    purge_for_doc_id,
    register_after_ingest,
)
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

        s_doc = get_settings().document
        stats = {"added": 0, "modified": 0, "removed": 0, "errors": 0}
        # two_stage：先依 conversion_cache 處理已刪 PDF的側車，否則僅刪 .md 時 images 仍殘留
        if s_doc.is_two_stage():
            from src.data_processing.mineru_convert import remove_mineru_sidecars_for_pdf

            conv_report = detect_conversion_changes([self._raw], recursive=True)
            for path_str in conv_report.removed:
                try:
                    remove_mineru_sidecars_for_pdf(Path(path_str))
                except Exception as e:
                    logger.error("刪除 PDF 側車失敗 %s: %s", path_str, e)
                    stats["errors"] += 1
            conv_cache = load_conversion_cache()
            for path_str in conv_report.removed:
                conv_cache.pop(path_str, None)
            write_conversion_cache(conv_cache)

        if s_doc.is_two_stage():
            report = detect_index_changes([self._raw], recursive=True)
        else:
            report = detect_changes([self._raw], recursive=True)
        rag = await get_lightrag()

        # 數據完整性修復：對標記為 unchanged 的文件，若數據庫中無有效數據則視為新增
        for path_str in list(report.unchanged):
            doc_id = stable_doc_id(Path(path_str))
            existing = await rag.full_docs.get_by_id(doc_id) if rag.full_docs else None
            if existing is None:
                logger.info("文件 %s 在 hash_cache 中但數據庫無數據，視為新增重新入庫", path_str)
                report.added.append(Path(path_str))
                report.unchanged.remove(path_str)
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
                m_out = purge_for_doc_id(doc_id)
                p = Path(path_str)
                suf = p.suffix.lower()
                if m_out.get("skipped"):
                    if suf == ".pdf" and not get_settings().document.is_two_stage():
                        from src.data_processing.mineru_convert import remove_mineru_sidecars_for_pdf

                        remove_mineru_sidecars_for_pdf(p)
                    elif suf in (".md", ".markdown"):
                        legacy_cleanup_markdown_sidecars(path_str)
                stats["removed"] += 1
            except Exception as e:
                logger.error("刪除索引失敗 %s: %s", path_str, e)
                stats["errors"] += 1
            await _maybe_checkpoint()

        async def _ingest_one(path: Path, *, is_modify: bool) -> None:
            nonlocal stats, cp
            doc_id = stable_doc_id(path)
            try:
                loaded = load_document(path)
                text = loaded.text
                if not text.strip():
                    logger.warning("文件為空，略過索引: %s", path)
                    stats["errors"] += 1
                    return
                # 檢查是否需要強制重新入庫（數據庫無有效數據但 doc_status 有殘留）
                if not is_modify:
                    existing = await rag.full_docs.get_by_id(doc_id) if rag.full_docs else None
                    if existing is None:
                        logger.info("文件 %s 在數據庫中無有效數據，清理後重新入庫", path)
                        await cascade_delete_document(rag, doc_id, self._kv)
                        purge_for_doc_id(doc_id)
                        cleared = await self._clear_doc_status_by_file_path(rag, path.name)
                        if cleared:
                            logger.info("清理 doc_status 同名舊記錄: %s", cleared)
                        is_modify = True
                if is_modify:
                    await cascade_delete_document(rag, doc_id, self._kv)
                    purge_for_doc_id(doc_id)
                await rag.ainsert(text, ids=doc_id, file_paths=str(loaded.lightrag_file_path()))
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
                register_after_ingest(doc_id, loaded, path)
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

    async def _clear_doc_status_by_file_path(self, rag, file_path: str) -> list[str]:
        """清理 doc_status 中所有与指定文件路径匹配的旧记录（基于 basename），避免 LightRAG 去重失败。"""
        if rag.doc_status is None:
            return []
        from lightrag.base import DocStatus
        basename = Path(file_path).name
        all_statuses = [
            DocStatus.PENDING, DocStatus.PARSING, DocStatus.ANALYZING,
            DocStatus.PROCESSING, DocStatus.PREPROCESSED, DocStatus.PROCESSED, DocStatus.FAILED,
        ]
        docs = await rag.doc_status.get_docs_by_statuses(all_statuses)
        deleted_ids = []
        for doc_id, doc_data in docs.items():
            if getattr(doc_data, 'file_path', None) == basename:
                await rag.doc_status.delete([doc_id])
                deleted_ids.append(doc_id)
        return deleted_ids

    def _resolve_index_path(self, path: Path) -> Path:
        """兩階段模式下上傳 PDF 時，改為索引同目錄已轉好的 Markdown。"""
        path = path.expanduser().resolve()
        s = get_settings().document
        if path.suffix.lower() == ".pdf" and s.is_two_stage():
            md = markdown_path_for_pdf(path)
            if not md.is_file():
                raise FileNotFoundError(
                    f"兩階段模式須先轉檔 PDF：請執行 python scripts/convert.py incremental，"
                    f"預期 Markdown: {md}"
                )
            return md
        return path

    async def ingest_path(self, path: Path, *, replace: bool = False) -> dict[str, Any]:
        """單一路徑入庫（API 上傳用）。"""
        path = self._resolve_index_path(path)
        rag = await get_lightrag()
        doc_id = stable_doc_id(path)

        # 若數據庫中無該文檔有效數據（可能之前因 DUPLICATE 失敗且已被清理），強制重新入庫
        if not replace:
            existing = await rag.full_docs.get_by_id(doc_id) if rag.full_docs else None
            if existing is None:
                logger.info("文檔 %s 在數據庫中無有效數據，強制清理後重新入庫", doc_id)
                await cascade_delete_document(rag, doc_id, self._kv)
                purge_for_doc_id(doc_id)
                cleared = await self._clear_doc_status_by_file_path(rag, path.name)
                if cleared:
                    logger.info("清理 doc_status 同名舊記錄: %s", cleared)
                replace = True

        if replace:
            await cascade_delete_document(rag, doc_id, self._kv)
            purge_for_doc_id(doc_id)
        loaded = load_document(path)
        await rag.ainsert(loaded.text, ids=doc_id, file_paths=str(loaded.lightrag_file_path()))
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
        register_after_ingest(doc_id, loaded, path)
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
