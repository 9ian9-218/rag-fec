"""校验增量：manifest / SQLite / LightRAG KV / Milvus / Neo4j 与 data/raw 一致。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.incremental.document_manifest import load_manifest
from src.storage.kv_client import KVClient
from src.storage.lightrag_init import get_lightrag


def _raw_index_paths(raw: Path) -> set[str]:
    suf = {".md", ".markdown", ".pdf", ".txt", ".docx"}
    out: set[str] = set()
    if not raw.is_dir():
        return out
    for p in raw.rglob("*"):
        if p.is_file() and p.suffix.lower() in suf:
            out.add(str(p.resolve()))
    return out


async def snapshot() -> dict:
    s = get_settings()
    root = Path(s.paths.project_root).resolve()
    raw = (root / s.paths.data_raw).resolve()
    manifest = load_manifest(s)
    kv_rows = KVClient().list_documents()
    rag = await get_lightrag()
    fd = await rag.full_docs.get_all() if rag.full_docs else {}
    ds = await rag.doc_status.get_all() if rag.doc_status else {}
    milvus_n = 0
    try:
        from pymilvus import MilvusClient

        mc = MilvusClient(uri=s.milvus.uri, timeout=30)
        for coll in ("chunks", "entities", "relationships"):
            if mc.has_collection(coll):
                milvus_n += mc.get_collection_stats(coll).get("row_count", 0) or 0
    except Exception as e:
        milvus_n = -1
        milvus_err = str(e)
    else:
        milvus_err = None
    neo4j_n = 0
    try:
        from src.storage.neo4j_client import Neo4jClient

        with Neo4jClient().session() as sess:
            neo4j_n = sess.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    except Exception as e:
        neo4j_n = -1
        neo4j_err = str(e)
    else:
        neo4j_err = None
    return {
        "raw_files": sorted(_raw_index_paths(raw)),
        "manifest_ids": sorted(manifest.keys()),
        "sqlite_ids": sorted(r["doc_id"] for r in kv_rows),
        "lightrag_full_docs": len(fd) if isinstance(fd, dict) else 0,
        "lightrag_doc_status": len(ds) if isinstance(ds, dict) else 0,
        "milvus_row_count_sum": milvus_n,
        "milvus_err": milvus_err,
        "neo4j_nodes": neo4j_n,
        "neo4j_err": neo4j_err,
    }


def assert_consistent(snap: dict, *, expect_raw: int, label: str) -> list[str]:
    errs: list[str] = []
    n_raw = len(snap["raw_files"])
    n_man = len(snap["manifest_ids"])
    n_sql = len(snap["sqlite_ids"])
    n_fd = snap["lightrag_full_docs"]
    n_ds = snap["lightrag_doc_status"]
    if n_raw != expect_raw:
        errs.append(f"{label}: raw 文件数 {n_raw} != 期望 {expect_raw}")
    if n_man != expect_raw:
        errs.append(f"{label}: manifest 条数 {n_man} != 期望 {expect_raw}")
    if n_sql != expect_raw:
        errs.append(f"{label}: sqlite 条数 {n_sql} != 期望 {expect_raw}")
    if n_fd != expect_raw:
        errs.append(f"{label}: lightrag full_docs {n_fd} != 期望 {expect_raw}")
    if n_ds != expect_raw:
        errs.append(f"{label}: lightrag doc_status {n_ds} != 期望 {expect_raw}")
    if expect_raw == 0:
        if snap["milvus_row_count_sum"] not in (0, "0"):
            if snap["milvus_row_count_sum"] != 0:
                errs.append(f"{label}: milvus 仍有数据 {snap['milvus_row_count_sum']}")
        if snap["neo4j_nodes"] not in (0, -1) and snap["neo4j_nodes"] > 0:
            errs.append(f"{label}: neo4j 仍有节点 {snap['neo4j_nodes']}")
    else:
        if snap["milvus_row_count_sum"] == 0:
            errs.append(f"{label}: milvus 无向量数据")
        if snap["neo4j_nodes"] == 0:
            errs.append(f"{label}: neo4j 无节点")
    return errs


async def main() -> int:
    apply_settings_to_environ(get_settings())
    snap = await snapshot()
    print(json.dumps(snap, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
