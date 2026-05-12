#!/usr/bin/env python3
"""
2) 将 fec_lightrag_extract.py 产出的中间数据导入本地 Docker Neo4j。

读取 data/fec_lightrag_intermediate/manifest.json 与 graph_chunk_entity_relation.graphml，
将 LightRAG 的 entity_type 映射为 FEC 标签（Document / Chapter / Concept / Entity / Attribute / Case），
边默认映射为手册中的七种关系之一（由 keywords/description 启发式推断，未命中则为 RELATES_TO）。

Neo4j（与 data/docker-compose.yml 一致）：
  export NEO4J_URI=neo4j://localhost:7687
  export NEO4J_USERNAME=neo4j
  export NEO4J_PASSWORD=all-in-rag

可选：--dry-run 只打印统计，不写库。
"""
from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_INTERMEDIATE = _REPO_ROOT / "data" / "fec_lightrag_intermediate"

FEC_LABELS = frozenset(
    {"Document", "Chapter", "Concept", "Entity", "Attribute", "Case"}
)
FEC_REL_TYPES = frozenset(
    {
        "CONTAINS",
        "DEFINES",
        "COMPOSES",
        "DEPENDS_ON",
        "RELATES_TO",
        "HAS_ATTRIBUTE",
        "BELONGS_TO",
    }
)

# 手册七种关系：关键词子串（小写）→ Neo4j 关系类型
_REL_RULES: list[tuple[tuple[str, ...], str]] = [
    (("包含", "contain", "consist", "组成结构"), "CONTAINS"),
    (("定义", "define", "指称", "含义是"), "DEFINES"),
    (("组成", "compose", "构成", "由…组成"), "COMPOSES"),
    (("依赖", "depend", "需要", "基于", "用到"), "DEPENDS_ON"),
    (("属于", "belong", "归类", "子类"), "BELONGS_TO"),
    (("属性", "参数", "has_attr", "取值", "特征"), "HAS_ATTRIBUTE"),
    (("关联", "相关", "relate", "联系", "相似", "对比"), "RELATES_TO"),
]


def _norm_label(entity_type: str | None) -> str:
    t = (entity_type or "").strip()
    if t in FEC_LABELS:
        return t
    if t.lower() == "other":
        return "Entity"
    return "Entity"


def _node_id_for(label: str, name: str) -> str:
    """与建图手册分段大致一致的可复现 nodeId（字符串数字）。"""
    bases = {
        "Document": 100_000_000,
        "Chapter": 200_000_000,
        "Concept": 300_000_000,
        "Entity": 400_000_000,
        "Attribute": 600_000_000,
        "Case": 700_000_000,
    }
    base = bases.get(label, 400_000_000)
    h = int(hashlib.sha256(f"{label}|{name}".encode()).hexdigest()[:12], 16)
    return str(base + (h % 50_000_000))


def _infer_rel_type(keywords: str, description: str) -> str:
    blob = f"{keywords or ''} {description or ''}".lower()
    for keys, rel in _REL_RULES:
        if any(k.lower() in blob for k in keys):
            return rel
    return "RELATES_TO"


def _ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


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


def _maybe_decode(v: Any) -> Any:
    if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
        return v.strip('"')
    return v


def _merge_node_tx(tx, label: str, node_id: str, props: dict[str, Any]) -> None:
    # 仅允许白名单标签，防止注入
    if label not in FEC_LABELS:
        label = "Entity"
    q = f"MERGE (n:`{label}` {{nodeId: $nodeId}}) SET n += $props"
    tx.run(q, nodeId=node_id, props=props)


def _merge_rel_tx(
    tx, rel_type: str, src_id: str, tgt_id: str, props: dict[str, Any]
) -> None:
    if rel_type not in FEC_REL_TYPES:
        rel_type = "RELATES_TO"
    q = f"""
    MATCH (a {{nodeId: $src}}), (b {{nodeId: $tgt}})
    MERGE (a)-[r:`{rel_type}`]->(b)
    SET r += $props
    """
    tx.run(q, src=src_id, tgt=tgt_id, props=props)


def run_import(
    intermediate_dir: Path,
    *,
    dry_run: bool,
    skip_doc_contains: bool,
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
    doc_title = manifest.get("document_title") or "Document"
    doc_key = manifest.get("doc_id") or "fec_doc"

    doc_node_id = _node_id_for("Document", doc_key)
    doc_props: dict[str, Any] = {
        "name": doc_title,
        "title": doc_title,
        "uri": manifest.get("markdown_path", ""),
        "sourceRef": f"book:{doc_key}",
        "fecSource": "lightrag_extract",
    }

    # 实体节点：name 为图节点 id（与 LightRAG entity_id 一致）
    prepared: list[tuple[str, str, dict[str, Any]]] = []
    for entity_name, raw in nodes.items():
        et = _norm_label(raw.get("entity_type"))
        nid = _node_id_for(et, entity_name)
        desc = raw.get("description") or raw.get("entity_description") or ""
        props: dict[str, Any] = {
            "name": entity_name,
            "description": str(desc)[:8000] if desc else "",
            "sourceRef": f"book:{doc_key}",
            "fecSource": "lightrag_extract",
        }
        if et == "Entity":
            ek = (raw.get("entity_type") or "general").strip()
            props["entityKind"] = ek.lower().replace(" ", "_")[:120] or "general"
        prepared.append((nid, et, props))

    nid_by_name = {p[2]["name"]: p[0] for p in prepared}
    label_by_name = {p[2]["name"]: p[1] for p in prepared}

    stats = {
        "nodes": len(prepared) + 1,
        "edges_from_graph": len(edges),
        "contains_from_document": 0,
        "dry_run": dry_run,
    }

    if dry_run:
        stats["sample_labels"] = {
            k: label_by_name.get(k) for k in list(label_by_name)[:8]
        }
        return stats

    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError("请安装: pip install neo4j") from e

    uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "all-in-rag")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:

        def write_all(tx):
            contains_count = 0
            _merge_node_tx(tx, "Document", doc_node_id, doc_props)
            for nid, label, props in prepared:
                _merge_node_tx(tx, label, nid, props)

            rel_no = 0
            for u, v, edata in edges:
                if u not in nid_by_name or v not in nid_by_name:
                    continue
                uid, vid = nid_by_name[u], nid_by_name[v]
                kw = str(edata.get("keywords") or "")
                desc = str(edata.get("description") or "")
                rtype = _infer_rel_type(kw, desc)
                lu, lv = label_by_name.get(u), label_by_name.get(v)

                if rtype == "HAS_ATTRIBUTE" or (
                    rtype == "RELATES_TO"
                    and ("属性" in kw or "参数" in kw)
                    and ("Attribute" in (lu, lv))
                ):
                    rtype = "HAS_ATTRIBUTE"
                    if lv == "Attribute":
                        src_id, tgt_id = uid, vid
                    elif lu == "Attribute":
                        src_id, tgt_id = vid, uid
                    else:
                        src_id, tgt_id = _ordered_pair(uid, vid)
                elif rtype == "RELATES_TO":
                    src_id, tgt_id = _ordered_pair(uid, vid)
                else:
                    src_id, tgt_id = _ordered_pair(uid, vid)

                rel_no += 1
                rid = f"FEC_REL_LR_{rel_no}_{hashlib.md5((u + v + kw).encode()).hexdigest()[:10]}"
                _merge_rel_tx(
                    tx,
                    rtype,
                    src_id,
                    tgt_id,
                    {
                        "relationshipId": rid,
                        "keywords": kw[:2000],
                        "description": desc[:8000],
                        "sourceRef": f"book:{doc_key}",
                    },
                )

            if not skip_doc_contains:
                for nid, label, _props in prepared:
                    if label == "Document":
                        continue
                    rid = f"FEC_REL_DOC_{hashlib.md5((doc_node_id + nid).encode()).hexdigest()[:12]}"
                    _merge_rel_tx(
                        tx,
                        "CONTAINS",
                        doc_node_id,
                        nid,
                        {
                            "relationshipId": rid,
                            "keywords": "document,contains",
                            "description": "Document CONTAINS entity (import bridge from LightRAG)",
                        },
                    )
                    contains_count += 1
            return contains_count

        with driver.session() as session:
            contains_written = session.execute_write(write_all)
        stats["contains_from_document"] = contains_written
    finally:
        driver.close()

    return stats


def main() -> int:
    p = argparse.ArgumentParser(description="FEC: 中间结果 → Neo4j")
    p.add_argument(
        "--intermediate-dir",
        type=Path,
        default=DEFAULT_INTERMEDIATE,
        help="含 manifest.json 与 graphml 的目录",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计节点/边数量，不写 Neo4j",
    )
    p.add_argument(
        "--skip-doc-contains",
        action="store_true",
        help="不创建 Document→各实体的 CONTAINS（图较大时可开）",
    )
    args = p.parse_args()

    stats = run_import(
        args.intermediate_dir.resolve(),
        dry_run=args.dry_run,
        skip_doc_contains=args.skip_doc_contains,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
