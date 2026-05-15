"""批次建索引：全量（重建 hash 快取 + 逐文件入庫）或增量。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from src.data_processing.change_detector import rebuild_hash_cache_for_directory
from src.data_processing.document_loader import iter_documents
from src.incremental.doc_registry import stable_doc_id
from src.incremental.update_manager import UpdateManager
from src.storage.lightrag_init import get_lightrag
from src.utils.logger import setup_logging


async def _full(raw: Path) -> None:
    setup_logging()
    s = get_settings()
    raw = raw.resolve()
    rebuild_hash_cache_for_directory(raw, recursive=True)
    rag = await get_lightrag()
    from src.storage.kv_client import KVClient

    kv = KVClient()
    for doc in iter_documents(raw, recursive=True):
        doc_id = stable_doc_id(doc.path)
        await rag.ainsert(doc.text, ids=doc_id, file_paths=str(doc.path))
        kv.upsert_document(
            doc_id,
            str(doc.path.resolve()),
            doc.metadata.get("md5_content"),
            doc.metadata.get("size_bytes"),
        )
        print("indexed:", doc.path)


async def _incremental() -> None:
    setup_logging()
    mgr = UpdateManager()
    r = await mgr.run_incremental()
    print(r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("full", "incremental"), default="incremental")
    p.add_argument("--raw", type=Path, default=None, help="原始文件目錄，預設為設定中的 data/raw")
    args = p.parse_args()
    raw = args.raw or Path(get_settings().paths.data_raw)
    if not raw.is_absolute():
        raw = (Path(get_settings().paths.project_root).resolve() / raw).resolve()
    if args.mode == "full":
        asyncio.run(_full(raw))
    else:
        asyncio.run(_incremental())


if __name__ == "__main__":
    main()
