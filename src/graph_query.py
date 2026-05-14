#!/usr/bin/env python3
"""
FEC 图检索：LightRAG 工作区（JsonKV/NetworkX）+ Neo4j（LREntity / LR_REL），LangChain 编排。

1) LangChain ChatOpenAI 将自然语言解析为结构化 GraphQuery，再通过 Cypher 匹配种子并取邻域。
2) 可选：对同一文档的 ``lightrag_workdir`` 调用 LightRAG ``aquery_data``（与 Neo4j 结果一并返回）。
3) 若未配置 LLM 环境或调用/解析失败，回退为本地子串/词元打分；失败时向 stderr 打印异常与回溯。

环境：NEO4J_URI、NEO4J_USERNAME、NEO4J_PASSWORD；LLM 从进程环境读取（由 .env 注入）。
可选：FEC_GRAPH_MARKDOWN 指向与入库一致的源 .md，链输入未传 ``markdown_path`` 时使用。

程序化串联：``build_fec_graph_retrieval_chain()`` 返回 LangChain Runnable（plan → Neo4j + LightRAG → 组装 JSON），
依赖 ``langchain-core``、``langchain-openai``。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env", override=False)

NODE_LABEL = "LREntity"
REL_TYPE = "LR_REL"
ENTITY_KEY = "entity_id"
ENTITY_ID_SCORE_WEIGHT = 2.5


class QueryType(str, Enum):
    ENTITY_RELATION = "entity_relation"
    MULTI_HOP = "multi_hop"
    SUBGRAPH = "subgraph"
    PATH_FINDING = "path_finding"
    KEYWORD_FALLBACK = "keyword_fallback"


@dataclass
class GraphQuery:
    """与 all-in-rag 图 RAG 类似：由 LLM 产出，再驱动 Neo4j 检索。"""

    query_type: QueryType = QueryType.SUBGRAPH
    source_entities: list[str] = field(default_factory=list)
    target_entities: list[str] = field(default_factory=list)
    relation_types: list[str] = field(default_factory=list)
    max_depth: int = 2
    max_nodes: int = 50
    constraints: dict[str, Any] = field(default_factory=dict)


def _tokenize_question(q: str) -> list[str]:
    q = (q or "").strip()
    if not q:
        return []
    seen: set[str] = set()
    if len(q) >= 2:
        seen.add(q)
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9._+-]{1,}", q):
        seen.add(m.group(0))
    for m in re.finditer(r"\d[\d.\s]*[\d.]?", q):
        t = m.group(0).strip()
        if len(t) >= 2:
            seen.add(t)
    for m in re.finditer(r"[\u4e00-\u9fff]+", q):
        s = m.group(0)
        if len(s) <= 8:
            seen.add(s)
        for w in (2, 3, 4):
            if len(s) < w:
                continue
            for i in range(len(s) - w + 1):
                seen.add(s[i : i + w])
    out = [t for t in seen if len(t.strip()) >= 2]
    out.sort(key=len, reverse=True)
    return out[:80]


def _body_text(props: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in props.items():
        if k == ENTITY_KEY:
            continue
        if isinstance(v, str) and v.strip():
            parts.append(v.lower())
    return "\n".join(parts)


def _score_text_against_tokens(text: str, tokens: list[str]) -> float:
    if not text:
        return 0.0
    score = 0.0
    for t in tokens:
        tl = t.lower()
        if not tl:
            continue
        if tl in text:
            score += 3.0 + min(len(tl), 12) * 0.15
        c = text.count(tl)
        if c > 1:
            score += (c - 1) * 0.4
    return score


def _score_entity_retrieval(
    entity_id: str, props: dict[str, Any], tokens: list[str]
) -> float:
    eid = entity_id.lower()
    body = _body_text(props)
    return ENTITY_ID_SCORE_WEIGHT * _score_text_against_tokens(
        eid, tokens
    ) + _score_text_against_tokens(body, tokens)


def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _strip_json_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        if "```" in text:
            text = text.split("```", 1)[0].strip()
    return text


def _llm_parse_graph_query(question: str) -> GraphQuery:
    """LangChain ChatOpenAI：自然语言 → GraphQuery JSON。"""
    base = (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()
    if not key or not model:
        return GraphQuery(query_type=QueryType.KEYWORD_FALLBACK)

    prompt = f"""你是信息检索规划器。用户图数据库模型如下（Neo4j）：
- 节点标签：{NODE_LABEL}；主键属性：{ENTITY_KEY}（字符串，与书中实体名/公式名/表号等一致）。
- 其它常用属性：entity_type、description、source_id、file_path 等。
- 关系：仅一种有向关系 `{REL_TYPE}`，连接两个 {NODE_LABEL}（LightRAG 抽取的实体关联）。

请把下面「用户问题」解析为 JSON 对象（不要 Markdown，不要解释），字段必须齐全：
- query_type: 字符串，取值之一：
  entity_relation（两实体是否直接相关/有无边）、
  multi_hop（需经多跳关联推理）、
  subgraph（围绕若干核心概念看局部子图）、
  path_finding（从概念 A 到概念 B 的关联路径）。
- source_entities: 字符串数组，图中可能出现的实体名/术语/表号/「公式(5-n)」等锚点，2～8 个为宜，优先具体名词。
- target_entities: 字符串数组；仅在 path_finding / multi_hop 需要终点时填写，否则 []。
- relation_types: 固定填 ["{REL_TYPE}"] 或 []（库中仅 {REL_TYPE}）。
- max_depth: 整数 1～4，表示图上扩展深度（步数）。
- max_nodes: 整数 10～120，控制候选节点规模上限。
- constraints: 对象，可 {{}}；若有「仅限某章/某码型」等可写 {{ "notes": "..." }}。

用户问题：
{question}
"""
    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        llm_kwargs: dict[str, Any] = {
            "model": model,
            "api_key": key,
            "temperature": 0.1,
            "max_tokens": 1200,
        }
        if base:
            llm_kwargs["base_url"] = base
        llm = ChatOpenAI(**llm_kwargs)
        if _env_truthy("OPENAI_JSON_OBJECT_MODE"):
            llm = llm.bind(response_format={"type": "json_object"})
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = (resp.content or "").strip()
        raw = _strip_json_fence(raw)
        obj = json.loads(raw)
    except Exception as e:
        print(
            f"[graph_query] LLM GraphQuery 解析失败，将使用关键词检索: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        return GraphQuery(query_type=QueryType.KEYWORD_FALLBACK)

    qt_raw = (obj.get("query_type") or "subgraph").strip().lower()
    try:
        qt = QueryType(qt_raw)
    except ValueError:
        qt = QueryType.SUBGRAPH

    def _str_list(x: Any) -> list[str]:
        if not isinstance(x, list):
            return []
        out: list[str] = []
        for it in x:
            s = str(it).strip()
            if s and s not in out:
                out.append(s)
        return out[:12]

    md = int(obj.get("max_depth") or 2)
    md = max(1, min(4, md))
    mn = int(obj.get("max_nodes") or 50)
    mn = max(10, min(120, mn))
    cons = obj.get("constraints")
    if not isinstance(cons, dict):
        cons = {}

    return GraphQuery(
        query_type=qt,
        source_entities=_str_list(obj.get("source_entities")),
        target_entities=_str_list(obj.get("target_entities")),
        relation_types=_str_list(obj.get("relation_types")) or [REL_TYPE],
        max_depth=md,
        max_nodes=mn,
        constraints=cons,
    )


def _fetch_entities(driver: Any) -> list[tuple[str, dict[str, Any]]]:
    q = f"MATCH (n:`{NODE_LABEL}`) RETURN n.{ENTITY_KEY} AS eid, properties(n) AS props"
    with driver.session() as session:
        rows = session.run(q)
        return [(str(r["eid"]), dict(r["props"])) for r in rows]


def _cypher_resolve_seeds(
    driver: Any, terms: list[str], per_term_limit: int
) -> dict[str, tuple[dict[str, Any], float]]:
    """用 Cypher 将 LLM 给出的锚点短语落到 entity_id 上（CONTAINS 匹配）。"""
    found: dict[str, tuple[dict[str, Any], float]] = {}
    cy = f"""
    MATCH (n:`{NODE_LABEL}`)
    WHERE toLower(n.{ENTITY_KEY}) CONTAINS toLower($term)
       OR (n.description IS NOT NULL AND toLower(toString(n.description)) CONTAINS toLower($term))
    RETURN n.{ENTITY_KEY} AS eid, properties(n) AS props
    LIMIT $lim
    """
    for term in terms:
        t = term.strip()
        if len(t) < 2:
            continue
        with driver.session() as session:
            for r in session.run(cy, term=t, lim=per_term_limit):
                eid = str(r["eid"])
                props = dict(r["props"])
                pri = 2.5 if t.lower() in eid.lower() else 1.0
                if eid not in found or found[eid][1] < pri:
                    found[eid] = (props, pri)
    return found


def _cypher_entity_ids_within_hops(
    driver: Any, seed_ids: list[str], depth: int, cap: int
) -> list[str]:
    """从种子出发沿 LR_REL 扩展 depth 步内的其它 entity_id（去重）。"""
    if depth < 2 or not seed_ids:
        return []
    d = max(1, min(depth, 4))
    cy = f"""
    UNWIND $sids AS sid
    MATCH (s:`{NODE_LABEL}` {{ {ENTITY_KEY}: sid }})
    MATCH (s)-[:`{REL_TYPE}`*1..{d}]-(n:`{NODE_LABEL}`)
    WHERE n.{ENTITY_KEY} <> sid
    RETURN DISTINCT n.{ENTITY_KEY} AS eid
    LIMIT $cap
    """
    out: list[str] = []
    with driver.session() as session:
        for r in session.run(cy, sids=seed_ids[:20], cap=cap):
            eid = str(r["eid"])
            if eid not in out:
                out.append(eid)
    return out


def _neighbors_for(
    driver: Any, entity_ids: list[str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    if not entity_ids:
        return {}
    cypher = f"""
    MATCH (s:`{NODE_LABEL}`)
    WHERE s.{ENTITY_KEY} IN $ids
    OPTIONAL MATCH (s)-[r_out:`{REL_TYPE}`]->(o:`{NODE_LABEL}`)
    WITH s,
      collect(DISTINCT CASE WHEN o IS NULL THEN NULL ELSE {{
        direction: 'out',
        rel: properties(r_out),
        neighbor_id: o.{ENTITY_KEY},
        neighbor: properties(o)
      }} END) AS outs
    OPTIONAL MATCH (i:`{NODE_LABEL}`)-[r_in:`{REL_TYPE}`]->(s)
    RETURN s.{ENTITY_KEY} AS sid,
      outs,
      collect(DISTINCT CASE WHEN i IS NULL THEN NULL ELSE {{
        direction: 'in',
        rel: properties(r_in),
        neighbor_id: i.{ENTITY_KEY},
        neighbor: properties(i)
      }} END) AS ins
    """
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    with driver.session() as session:
        for r in session.run(cypher, ids=entity_ids):
            sid = str(r["sid"])
            outs = [x for x in r["outs"] if x is not None]
            ins = [x for x in r["ins"] if x is not None]
            result[sid] = {"neighbors_out": outs, "neighbors_in": ins}
    return result


def retrieve_graph_hits_with_plan(
    question: str,
    gq: GraphQuery,
    *,
    top_k: int = 5,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    在已有 GraphQuery 下执行 Neo4j 检索。返回 (hits, extras)；不含再次调用 LLM。
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError("请安装: pip install neo4j") from e

    uri = uri or os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    user = user or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = password or os.environ.get("NEO4J_PASSWORD", "all-in-rag")

    extras: dict[str, Any] = {}
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        tokens = _tokenize_question(question)
        ranked_llm: list[tuple[str, dict[str, Any], float]] = []

        if (
            gq.query_type != QueryType.KEYWORD_FALLBACK
            and (gq.source_entities or gq.target_entities)
        ):
            terms = list(gq.source_entities)
            for t in gq.target_entities:
                if t not in terms:
                    terms.append(t)
            resolved = _cypher_resolve_seeds(driver, terms, per_term_limit=12)
            for eid, (props, pri) in resolved.items():
                ranked_llm.append((eid, props, pri))
            ranked_llm.sort(key=lambda x: x[2], reverse=True)

        ranked_kw: list[tuple[str, dict[str, Any], float]] = []
        if tokens:
            for eid, props in _fetch_entities(driver):
                sc = _score_entity_retrieval(eid, props, tokens)
                if sc > 0:
                    ranked_kw.append((eid, props, sc))
            ranked_kw.sort(key=lambda x: x[2], reverse=True)

        merged: list[tuple[str, dict[str, Any], float]] = []
        seen: set[str] = set()
        for row in ranked_llm:
            eid, props, sc = row
            if eid not in seen:
                seen.add(eid)
                merged.append((eid, props, sc))
        for row in ranked_kw:
            eid, props, sc = row
            if eid not in seen:
                seen.add(eid)
                merged.append((eid, props, sc))

        if not merged:
            return [], extras

        top = merged[: max(1, top_k)]
        ids = [eid for eid, _p, _s in top]

        if gq.max_depth >= 2 and gq.query_type != QueryType.KEYWORD_FALLBACK:
            hop_ids = _cypher_entity_ids_within_hops(
                driver, ids, gq.max_depth, cap=min(gq.max_nodes, 100)
            )
            extras["within_depth_entity_ids"] = hop_ids

        nb_map = _neighbors_for(driver, ids)
        hits: list[dict[str, Any]] = []
        for eid, props, sc in top:
            nbs = nb_map.get(eid, {"neighbors_out": [], "neighbors_in": []})
            hits.append(
                {
                    "seed_entity_id": eid,
                    "score": round(sc, 4),
                    "seed_properties": props,
                    "neighbors_out": nbs["neighbors_out"],
                    "neighbors_in": nbs["neighbors_in"],
                    "match_tokens": tokens[:20],
                }
            )
        return hits, extras
    finally:
        driver.close()


def retrieve_graph_hits(
    question: str,
    *,
    top_k: int = 5,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> tuple[list[dict[str, Any]], GraphQuery, dict[str, Any]]:
    """
    返回 (hits, graph_query, extras)。
    extras 在 max_depth>=2 时可能含 within_depth_entity_ids（--full 使用）。
    """
    gq = _llm_parse_graph_query(question)
    hits, extras = retrieve_graph_hits_with_plan(
        question, gq, top_k=top_k, uri=uri, user=user, password=password
    )
    return hits, gq, extras


def _neighbor_sort_key(edge: dict[str, Any]) -> tuple[float, str]:
    r = edge.get("rel") or {}
    w = r.get("weight")
    try:
        wf = float(w) if w is not None else 0.0
    except (TypeError, ValueError):
        wf = 0.0
    adj = edge.get("neighbor") or {}
    eid = str(edge.get("neighbor_id") or adj.get(ENTITY_KEY) or "")
    return (-wf, eid)


def apply_neighbor_limit(
    hits: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    if limit is None or limit < 0:
        return hits
    out: list[dict[str, Any]] = []
    for h in hits:
        d = dict(h)
        for key in ("neighbors_out", "neighbors_in"):
            edges = list(d.get(key) or [])
            ranked = sorted(edges, key=_neighbor_sort_key)
            d[key] = ranked[:limit]
        out.append(d)
    return out


def _short_text(s: Any, max_len: int) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    one = s.replace("\n", " ").strip()
    if len(one) <= max_len:
        return one
    return one[: max(0, max_len - 1)] + "…"


def _brief_entity(props: dict[str, Any], *, desc_max: int = 200) -> dict[str, Any]:
    return {
        "entity_id": str(props.get(ENTITY_KEY, "")),
        "entity_type": str(props.get("entity_type", "")),
        "description": _short_text(props.get("description"), desc_max),
    }


def _brief_neighbor(
    edge: dict[str, Any],
    *,
    desc_max: int = 120,
    kw_max: int = 80,
    rel_desc_max: int = 160,
) -> dict[str, Any]:
    n = edge.get("neighbor") or {}
    r = edge.get("rel") or {}
    eid = edge.get("neighbor_id") or n.get(ENTITY_KEY) or ""
    rel_desc = str(r.get("description") or "").replace("<SEP>", " ")
    return {
        "direction": edge.get("direction", ""),
        "entity_id": str(eid),
        "entity_type": str(n.get("entity_type", "")),
        "description": _short_text(
            str(n.get("description") or "").replace("<SEP>", " "), desc_max
        ),
        "edge_keywords": _short_text(r.get("keywords"), kw_max),
        "edge_description": _short_text(rel_desc, rel_desc_max),
    }


def format_hits_compact(
    hits: list[dict[str, Any]],
    *,
    seed_desc_max: int = 200,
    neighbor_desc_max: int = 120,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for h in hits:
        props = h.get("seed_properties") or {}
        out.append(
            {
                "score": h.get("score"),
                "seed": _brief_entity(props, desc_max=seed_desc_max),
                "neighbors_out": [
                    _brief_neighbor(x, desc_max=neighbor_desc_max)
                    for x in h.get("neighbors_out") or []
                ],
                "neighbors_in": [
                    _brief_neighbor(x, desc_max=neighbor_desc_max)
                    for x in h.get("neighbors_in") or []
                ],
            }
        )
    return out


def _graph_query_to_json(gq: GraphQuery) -> dict[str, Any]:
    return {
        "query_type": gq.query_type.value,
        "source_entities": list(gq.source_entities),
        "target_entities": list(gq.target_entities),
        "relation_types": list(gq.relation_types),
        "max_depth": gq.max_depth,
        "max_nodes": gq.max_nodes,
        "constraints": dict(gq.constraints),
    }


def _compact_lightrag_payload(
    lr: dict[str, Any],
    *,
    max_entities: int = 24,
    max_relationships: int = 36,
    max_chunks: int = 6,
    chunk_chars: int = 400,
) -> dict[str, Any]:
    """非 --full 时压缩 LightRAG aquery_data 结果体积。"""
    if not isinstance(lr, dict):
        return {}
    st = lr.get("status")
    if st != "success":
        return {k: lr[k] for k in ("status", "message", "metadata") if k in lr} or lr
    data = lr.get("data")
    if not isinstance(data, dict):
        return {"status": st, "metadata": lr.get("metadata")}
    ent = data.get("entities") or []
    rels = data.get("relationships") or []
    chunks = data.get("chunks") or []
    refs = data.get("references") or []

    def _clip_txt(s: Any, n: int) -> str:
        if not isinstance(s, str):
            return ""
        t = s.replace("\n", " ").strip()
        return t if len(t) <= n else t[: max(0, n - 1)] + "…"

    slim_ent = []
    for e in ent[:max_entities]:
        if not isinstance(e, dict):
            continue
        slim_ent.append(
            {
                "entity_name": e.get("entity_name"),
                "entity_type": e.get("entity_type"),
                "description": _clip_txt(e.get("description"), 320),
            }
        )
    slim_rel = []
    for r in rels[:max_relationships]:
        if not isinstance(r, dict):
            continue
        slim_rel.append(
            {
                "src_id": r.get("src_id"),
                "tgt_id": r.get("tgt_id"),
                "keywords": _clip_txt(r.get("keywords"), 120),
                "description": _clip_txt(r.get("description"), 240),
            }
        )
    slim_chunks = []
    for c in chunks[:max_chunks]:
        if not isinstance(c, dict):
            continue
        slim_chunks.append(
            {
                "chunk_id": c.get("chunk_id"),
                "file_path": c.get("file_path"),
                "content": _clip_txt(c.get("content"), chunk_chars),
            }
        )
    return {
        "status": st,
        "metadata": lr.get("metadata"),
        "data": {
            "entities": slim_ent,
            "relationships": slim_rel,
            "chunks": slim_chunks,
            "references": refs[: max_chunks],
        },
    }


def build_fec_graph_retrieval_chain():
    """
    LangChain LCEL：LLM 规划 GraphQuery → Neo4j + LightRAG 检索 → JSON 负载。

    invoke 输入 dict 字段：
      question (str, 必填)
      top_k (int, 默认 5)
      neighbor_limit (int | None，默认 None 表示不截断邻居)
      full (bool，默认 False)
      markdown_path (str | Path，可选)：源 md；缺省读环境变量 FEC_GRAPH_MARKDOWN
      lightrag_workspace (str，可选)
      lightrag_mode (str，默认 mix)：LightRAG QueryParam.mode（local/global/hybrid/mix/…）
      skip_lightrag (bool，默认 False)
      neo4j_uri / neo4j_user / neo4j_password (可选，缺省读环境变量)

    返回 dict：含 graph_query、hits；若启用 LightRAG 则含 lightrag_retrieval。
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as e:
        raise RuntimeError("请安装: pip install langchain-core") from e

    def _plan(d: dict[str, Any]) -> dict[str, Any]:
        q = (d.get("question") or "").strip()
        if not q:
            raise ValueError("chain 输入缺少非空 question")
        gq = _llm_parse_graph_query(q)
        return {**d, "question": q, "graph_query": gq}

    def _retrieve(d: dict[str, Any]) -> dict[str, Any]:
        gq = d["graph_query"]
        assert isinstance(gq, GraphQuery)
        hits, extras = retrieve_graph_hits_with_plan(
            d["question"],
            gq,
            top_k=int(d.get("top_k", 5)),
            uri=d.get("neo4j_uri"),
            user=d.get("neo4j_user"),
            password=d.get("neo4j_password"),
        )
        lr: dict[str, Any] = {}
        if not bool(d.get("skip_lightrag", False)):
            md = d.get("markdown_path")
            if md is None or (isinstance(md, str) and not md.strip()):
                env_md = (os.getenv("FEC_GRAPH_MARKDOWN") or "").strip()
                md = env_md or None
            if md:
                try:
                    from fec_lightrag_extract import query_lightrag_for_markdown_doc

                    mode = str(d.get("lightrag_mode") or "mix").strip() or "mix"
                    ws = str(d.get("lightrag_workspace") or "").strip()
                    lr = query_lightrag_for_markdown_doc(
                        d["question"],
                        Path(md),
                        workspace=ws,
                        mode=mode,
                    )
                except Exception as e:
                    lr = {
                        "status": "failure",
                        "message": f"{type(e).__name__}: {e}",
                        "data": {},
                    }
        return {**d, "hits": hits, "extras": extras, "lightrag_retrieval": lr}

    def _assemble(d: dict[str, Any]) -> dict[str, Any]:
        hits: list[dict[str, Any]] = list(d.get("hits") or [])
        gq: GraphQuery = d["graph_query"]
        nl = d.get("neighbor_limit")
        if nl is not None and int(nl) >= 0:
            hits = apply_neighbor_limit(hits, int(nl))
        top_k = int(d.get("top_k", 5))
        full = bool(d.get("full", False))
        payload: dict[str, Any] = {
            "question": d["question"],
            "top_k": top_k,
            "graph_query": _graph_query_to_json(gq),
            "hits": hits,
        }
        if nl is not None and int(nl) >= 0:
            payload["neighbor_limit"] = int(nl)
        lr = d.get("lightrag_retrieval") or {}
        if lr:
            if full:
                payload["lightrag_retrieval"] = lr
            else:
                payload["lightrag_retrieval"] = _compact_lightrag_payload(lr)
        if not full:
            payload["hits"] = format_hits_compact(hits)
        else:
            ex = d.get("extras") or {}
            if ex.get("within_depth_entity_ids") is not None:
                payload["within_depth_entity_ids"] = ex["within_depth_entity_ids"]
        return payload

    return RunnableLambda(_plan) | RunnableLambda(_retrieve) | RunnableLambda(_assemble)


def main() -> int:
    p = argparse.ArgumentParser(
        description="FEC 图检索：LangChain 规划 GraphQuery + Neo4j；可选 LightRAG aquery_data"
    )
    p.add_argument("question", nargs="?", default="", help="自然语言问题")
    p.add_argument(
        "-q",
        "--query",
        dest="query_alt",
        default="",
        help="与 question 二选一，作为问题文本",
    )
    p.add_argument(
        "--markdown",
        type=Path,
        default=None,
        metavar="MD",
        help="与抽取/导入一致的源 markdown，用于 LightRAG 工作区检索；缺省读 FEC_GRAPH_MARKDOWN",
    )
    p.add_argument(
        "--lightrag-mode",
        default="mix",
        help="LightRAG QueryParam.mode，如 mix、hybrid、local、global",
    )
    p.add_argument(
        "--no-lightrag",
        action="store_true",
        help="不调用 LightRAG，仅 Neo4j",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回命中子图条数（每条含一个 seed 及其一跳邻居）",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="输出完整 JSON（含 graph_query、within_depth_entity_ids、未裁剪的 lightrag_retrieval）",
    )
    p.add_argument(
        "--neighbor-limit",
        type=int,
        default=None,
        metavar="N",
        help="每条 seed 的 neighbors_out / neighbors_in 各最多 N 条；默认不截断",
    )
    args = p.parse_args()
    q = (args.question or "").strip() or (args.query_alt or "").strip()
    if not q:
        p.print_help()
        print(
            "\n请提供问题，例如: python3 src/graph_query.py '循环码的生成多项式是什么'",
            file=sys.stderr,
        )
        return 2

    md_arg = args.markdown
    if md_arg is not None:
        md_arg = md_arg.resolve()

    chain = build_fec_graph_retrieval_chain()
    payload = chain.invoke(
        {
            "question": q,
            "top_k": args.top_k,
            "neighbor_limit": args.neighbor_limit
            if args.neighbor_limit is not None and args.neighbor_limit >= 0
            else None,
            "full": args.full,
            "markdown_path": md_arg,
            "lightrag_mode": args.lightrag_mode,
            "skip_lightrag": args.no_lightrag,
        }
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
