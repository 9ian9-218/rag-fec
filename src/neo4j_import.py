#!/usr/bin/env python3
"""
将 fec_lightrag_extract 产出的 graphml 经 manifest 导入 Neo4j。

**``-s`` 只能指向抽取产物目录**：须含 ``manifest.json``。可写 ``data/<主名>`` 或 ``data/cypher/<主名>``（脚本会互为回退查找）。

不接受原始 .md 路径；不设默认数据路径；未传 ``-s`` 则 argparse 报错退出。

Neo4j（与 data/docker-compose.yml 一致；单机建议 bolt://）：
  export NEO4J_URI=bolt://localhost:7687
  export NEO4J_USERNAME=neo4j
  export NEO4J_PASSWORD=all-in-rag

可选：--dry-run 只打印统计，不写库。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env", override=False)


def _candidate_base_paths(raw: Path) -> list[Path]:
    """相对路径：先试当前工作目录，再试仓库根。"""
    raw = raw.expanduser()
    out: list[Path] = []
    if raw.is_absolute():
        out.append(raw.resolve())
    else:
        out.append((Path.cwd() / raw).resolve())
        out.append((_REPO_ROOT / raw).resolve())
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        k = str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _candidate_dirs_for_import(target: Path) -> list[Path]:
    """
    解析 ``-s`` 时尝试的目录列表：含 cwd/仓库根的原始路径，以及
    ``data/cypher/<末级目录名>``、``data/<末级目录名>``（兼容简写 ``data/差错控制编码_第05章`` 与抽取输出 ``data/cypher/...``）。
    """
    bases = _candidate_base_paths(target)
    name = Path(target).name
    extras: list[Path] = []
    if name and name not in (".", "..", "data", "cypher"):
        extras.append((_REPO_ROOT / "data" / "cypher" / name).resolve())
        extras.append((_REPO_ROOT / "data" / name).resolve())
    seen: set[str] = set()
    out: list[Path] = []
    for p in bases + extras:
        k = str(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def resolve_intermediate_dir(target: Path) -> Path:
    """
    由 ``-s`` 解析出含 ``manifest.json`` 的**抽取产物目录**（不接受 .md 文件路径）。
    """
    manifest_name = "manifest.json"
    for base in _candidate_dirs_for_import(target):
        if not base.exists():
            continue
        if base.is_dir() and (base / manifest_name).is_file():
            return base
        if base.is_file():
            raise FileNotFoundError(
                f"-s 须为抽取产物目录（含 {manifest_name}），不能是文件: {base}"
            )
    tried = ", ".join(str(p) for p in _candidate_dirs_for_import(target))
    raise FileNotFoundError(
        f"-s 不是有效目录或缺少 {manifest_name}: {target}（已解析尝试: {tried}）"
    )


def _maybe_decode(v: Any) -> Any:
    if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
        return v.strip('"')
    return v


def _neo4j_flat_props(d: dict[str, Any]) -> dict[str, Any]:
    """仅将值转为 Neo4j 支持的标量或 JSON 字符串，不改键名、不增删语义字段。"""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        key = str(k)
        if isinstance(v, (str, int, float, bool)):
            out[key] = v
        else:
            out[key] = json.dumps(v, ensure_ascii=False)
    return out


def _load_graphml(path: Path) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, dict[str, Any]]]]:
    import networkx as nx

    g = nx.read_graphml(path)
    nodes: dict[str, dict[str, Any]] = {}
    for node_id, data in g.nodes(data=True):
        sid = str(node_id)
        nodes[sid] = {str(k): _maybe_decode(v) for k, v in dict(data).items()}
    edges: list[tuple[str, str, dict[str, Any]]] = []
    for u, v, data in g.edges(data=True):
        edges.append(
            (str(u), str(v), {str(k): _maybe_decode(v) for k, v in dict(data).items()})
        )
    return nodes, edges


def _merge_entity_tx(tx, entity_id: str, props: dict[str, Any]) -> None:
    q = "MERGE (n:LREntity {entity_id: $eid}) SET n += $props"
    tx.run(q, eid=entity_id, props=props)


def _merge_rel_tx(tx, eid_src: str, eid_dst: str, props: dict[str, Any]) -> None:
    q = """
    MATCH (x:LREntity {entity_id: $ea}), (y:LREntity {entity_id: $eb})
    MERGE (x)-[r:LR_REL]->(y)
    SET r += $props
    """
    tx.run(q, ea=eid_src, eb=eid_dst, props=props)


def run_import(
    intermediate_dir: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    manifest_path = intermediate_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    graphml = Path(manifest["artifacts"]["graphml"])
    export = manifest.get("fec_structured_export")
    if isinstance(export, dict):
        alt = export.get("graph_chunk_entity_relation.graphml")
        if alt:
            alt_p = Path(alt)
            if alt_p.is_file():
                graphml = alt_p
    if not graphml.is_file():
        raise FileNotFoundError(f"缺少 graphml: {graphml}")

    nodes, edges = _load_graphml(graphml)

    prepared: list[tuple[str, dict[str, Any]]] = []
    for entity_id, raw in nodes.items():
        prepared.append((entity_id, _neo4j_flat_props(dict(raw))))

    stats = {
        "intermediate_dir": str(intermediate_dir),
        "doc_id": manifest.get("doc_id"),
        "graphml": str(graphml),
        "nodes": len(prepared),
        "edges_from_graph": len(edges),
        "dry_run": dry_run,
        "neo4j_node_label": "LREntity",
        "neo4j_merge_key": "entity_id",
        "neo4j_rel_type": "LR_REL",
    }

    if dry_run:
        stats["sample_entity_id"] = [eid for eid, _ in prepared[:5]]
        return stats

    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError("请安装: pip install neo4j") from e

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "all-in-rag")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:

        def write_all(tx):
            for eid, props in prepared:
                _merge_entity_tx(tx, eid, props)
            n_rel = 0
            for u, v, edata in edges:
                if u not in nodes or v not in nodes:
                    continue
                _merge_rel_tx(tx, u, v, _neo4j_flat_props(dict(edata)))
                n_rel += 1
            return n_rel

        with driver.session() as session:
            rel_written = session.execute_write(write_all)
        stats["lr_rel_written"] = rel_written
    finally:
        driver.close()

    return stats


def main() -> int:
    p = argparse.ArgumentParser(
        description="LightRAG graphml → Neo4j；**必须**使用 -s 目录（含 manifest.json）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python3 src/fec_neo4j_import.py -s xxx\n"
        ),
    )
    p.add_argument(
        "-s",
        dest="target",
        type=Path,
        required=True,
        metavar="DIR",
        help="必填。fec_lightrag_extract 输出目录（内含 manifest.json），非原始 markdown 文件",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计节点/边数量，不写 Neo4j",
    )
    args = p.parse_args()

    try:
        intermediate_dir = resolve_intermediate_dir(args.target)
        stats = run_import(
            intermediate_dir,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as e:
        print(f"[fec_neo4j_import] {e}", file=sys.stderr)
        return 2
    except Exception as e:
        err = str(e).lower()
        if "unauthorized" in err or "authentication failure" in err:
            print(
                "[fec_neo4j_import] Neo4j 认证失败：请检查 password",
                file=sys.stderr,
            )
        raise

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
