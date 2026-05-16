"""文檔關聯清單：入庫時記錄側車路徑，刪除索引後按清單刪除磁碟上的 images、MinerU 元數據等。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import Settings, get_settings
from src.data_processing.document_loader import LoadedDocument
from src.retrieval.image_refs import extract_image_refs, resolve_local_image_path
from src.utils.logger import get_logger

logger = get_logger("incremental.document_manifest")


def manifest_file_path(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    root = Path(s.paths.project_root).resolve()
    return (root / s.paths.document_manifest_path).resolve()


def load_manifest(settings: Settings | None = None) -> dict[str, dict[str, Any]]:
    p = manifest_file_path(settings)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("讀取 document_manifest 失敗 %s: %s", p, e)
        return {}
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def save_manifest(data: dict[str, dict[str, Any]], settings: Settings | None = None) -> None:
    p = manifest_file_path(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_artifact_paths(loaded: LoadedDocument, settings: Settings | None = None) -> set[str]:
    """收集與該次入庫相關、刪除文檔時應一併刪除的磁碟路徑（絕對路徑字串）。"""
    s = settings or get_settings()
    root = Path(s.paths.project_root).resolve()
    out: set[str] = set()
    idx = loaded.lightrag_file_path().resolve()
    parent = idx.parent
    stem = idx.stem
    meta = parent / f".{stem}.mineru.json"
    if meta.is_file():
        out.add(str(meta.resolve()))
        try:
            meta_obj = json.loads(meta.read_text(encoding="utf-8"))
            idir_raw = meta_obj.get("images_dir")
            if idir_raw:
                idir = Path(str(idir_raw)).expanduser()
                if not idir.is_absolute():
                    idir = (parent / idir).resolve()
                else:
                    idir = idir.resolve()
                if idir.is_dir():
                    for fp in idir.rglob("*"):
                        if fp.is_file():
                            out.add(str(fp.resolve()))
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.debug("掃描 MinerU images_dir 失敗: %s", e)
    for ref in extract_image_refs(loaded.text):
        lp = resolve_local_image_path(ref, str(idx), root)
        if lp is not None and lp.is_file():
            out.add(str(lp.resolve()))
    return out


def register_after_ingest(
    doc_id: str,
    loaded: LoadedDocument,
    ingest_input_path: Path,
    *,
    settings: Settings | None = None,
) -> None:
    """成功寫入索引後登記關聯路徑（覆寫同 doc_id 舊條目）。"""
    s = settings or get_settings()
    files = _collect_artifact_paths(loaded, s)
    data = load_manifest(s)
    data[doc_id] = {
        "indexed_path": str(loaded.lightrag_file_path().resolve()),
        "ingest_input_path": str(ingest_input_path.expanduser().resolve()),
        "artifact_files": sorted(files),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(data, s)
    logger.info("document_manifest 已登記 doc_id=%s 側車 %d 個路徑", doc_id, len(files))


def purge_for_doc_id(doc_id: str, *, settings: Settings | None = None) -> dict[str, Any]:
    """依清單刪除側車檔案並移除 manifest 條目；不刪除 ingest 主檔（.md/.pdf 等由使用者或增量邏輯自行處理）。"""
    s = settings or get_settings()
    data = load_manifest(s)
    entry = data.pop(doc_id, None)
    if not entry:
        return {"skipped": True, "doc_id": doc_id, "deleted_files": [], "errors": []}
    deleted: list[str] = []
    errors: list[str] = []
    dirs_to_try: set[str] = set()
    for fp in entry.get("artifact_files") or []:
        p = Path(str(fp))
        try:
            if p.is_file():
                p.unlink()
                deleted.append(str(p))
                if p.parent.name == "images":
                    dirs_to_try.add(str(p.parent.resolve()))
        except OSError as e:
            errors.append(f"{fp}: {e}")
    for d in sorted(dirs_to_try, key=len, reverse=True):
        try:
            dp = Path(d)
            if dp.is_dir() and not any(dp.iterdir()):
                dp.rmdir()
                deleted.append(f"{d}/")
        except OSError as e:
            errors.append(f"rmdir {d}: {e}")
    save_manifest(data, s)
    logger.info("document_manifest 已清理 doc_id=%s 刪除 %d 項 err=%d", doc_id, len(deleted), len(errors))
    return {"skipped": False, "doc_id": doc_id, "deleted_files": deleted, "errors": errors}


def wipe_manifest_file(settings: Settings | None = None) -> None:
    """清空關聯檔（例如 clear_index --all）。"""
    p = manifest_file_path(settings)
    if p.is_file():
        p.unlink(missing_ok=True)


def legacy_cleanup_markdown_sidecars(path_str: str) -> None:
    """無 manifest 條目時的後備清理（.md / .markdown）。"""
    p = Path(path_str)
    suf = p.suffix.lower()
    if suf not in (".md", ".markdown"):
        return
    from src.data_processing.mineru_convert import remove_mineru_sidecars_for_markdown

    try:
        remove_mineru_sidecars_for_markdown(p)
    except OSError as e:
        logger.warning("後備 Markdown 側車清理失敗 %s: %s", path_str, e)
