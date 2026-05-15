"""清空索引：Neo4j、LightRAG 儲存、SQLite 元資料。"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from src.storage.lightrag_init import get_lightrag, reset_lightrag_singleton
from src.utils.logger import setup_logging


async def _drop_lightrag_storages() -> None:
    rag = await get_lightrag()
    storages = [
        rag.full_docs,
        rag.text_chunks,
        rag.full_entities,
        rag.full_relations,
        rag.entity_chunks,
        rag.relation_chunks,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunks_vdb,
        rag.chunk_entity_relation_graph,
        rag.llm_response_cache,
        rag.doc_status,
    ]
    for st in storages:
        if st is not None:
            await st.drop()


async def _async_clear(*, wipe_working_dir: bool, wipe_sqlite: bool) -> None:
    s = get_settings()
    root = Path(s.paths.project_root).resolve()
    if wipe_working_dir:
        await _drop_lightrag_storages()
        reset_lightrag_singleton()
        wd = root / s.paths.lightrag_working_dir
        if wd.exists():
            shutil.rmtree(wd)
    if wipe_sqlite:
        dbp = root / s.paths.sqlite_path
        if dbp.is_file():
            dbp.unlink()


def main() -> None:
    p = argparse.ArgumentParser(description="清空索引資料")
    p.add_argument("--neo4j", action="store_true", help="清空 Neo4j 圖（DETACH DELETE）")
    p.add_argument("--working-dir", action="store_true", help="呼叫各儲存 drop 並刪除 working_dir")
    p.add_argument("--sqlite", action="store_true", help="刪除應用層 SQLite 檔")
    p.add_argument("--all", action="store_true", help="Neo4j + working_dir + sqlite")
    args = p.parse_args()
    setup_logging()

    if not (args.all or args.neo4j or args.working_dir or args.sqlite):
        p.print_help()
        return

    wipe_wd = args.all or args.working_dir
    wipe_sql = args.all or args.sqlite
    wipe_neo = args.all or args.neo4j

    if wipe_neo:
        from src.storage.neo4j_client import Neo4jClient

        Neo4jClient().detach_delete_all_nodes()

    if wipe_wd or wipe_sql:
        asyncio.run(_async_clear(wipe_working_dir=wipe_wd, wipe_sqlite=wipe_sql))


if __name__ == "__main__":
    main()
