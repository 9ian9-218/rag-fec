"""從 JSONL 構建評估報告（答案 / 檢索 / 圖譜子集）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.answer_metrics import compute_answer_row
from src.evaluation.retrieval_metrics import hit_at_k, reciprocal_rank, retrieval_precision_recall_f1
from src.evaluation.set_metrics import multiset_precision_recall_f1, set_jaccard


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _nested_mean(details: list[dict[str, Any]], path: list[str]) -> float:
    vals: list[float] = []
    for d in details:
        cur: Any = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                cur = None
                break
            cur = cur[p]
        if isinstance(cur, (int, float)):
            vals.append(float(cur))
    return _mean(vals)


def build_report(
    rows: list[dict[str, Any]],
    *,
    rouge_types: tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
    use_stemmer: bool = False,
    hit_ks: tuple[int, ...] = (1, 5, 10),
    include_answer: bool = True,
    include_retrieval: bool | None = None,
    include_graph: bool | None = None,
    max_detail_rows: int = 100,
) -> dict[str, Any]:
    """
    :param rows: JSONL 解析後的 dict 列表。
    :param include_retrieval: None 表示若任一行含 gold_doc_ids+retrieved_doc_ids 則啟用。
    :param include_graph: None 表示若任一行含 gold_entities+retrieved_entities 則啟用。
    """
    if include_retrieval is None:
        include_retrieval = any(
            isinstance(r.get("gold_doc_ids"), list) and isinstance(r.get("retrieved_doc_ids"), list)
            for r in rows
        )
    if include_graph is None:
        include_graph = any(
            (
                isinstance(r.get("gold_entities"), list)
                and isinstance(r.get("retrieved_entities"), list)
            )
            or (
                isinstance(r.get("gold_relations"), list)
                and isinstance(r.get("retrieved_relations"), list)
            )
            for r in rows
        )

    answer_details: list[dict[str, Any]] = []
    retr_details: list[dict[str, Any]] = []
    graph_details: list[dict[str, Any]] = []

    n_answer = 0
    n_retr = 0
    n_graph_ent = 0
    n_graph_rel = 0

    for r in rows:
        q = r.get("question", "")
        ref = str(r.get("reference") or "")
        pred = str(r.get("prediction") or "")

        if include_answer and ref and pred:
            ar = compute_answer_row(ref, pred, rouge_types=rouge_types, use_stemmer=use_stemmer)
            answer_details.append({"question": q, **ar})
            n_answer += 1

        if include_retrieval:
            gd = r.get("gold_doc_ids")
            rd = r.get("retrieved_doc_ids")
            if isinstance(gd, list) and isinstance(rd, list) and len(gd) > 0:
                prf = retrieval_precision_recall_f1([str(x) for x in gd], [str(x) for x in rd])
                ranked = r.get("retrieved_doc_ids_ranked")
                if not isinstance(ranked, list):
                    ranked = [str(x) for x in rd]
                else:
                    ranked = [str(x) for x in ranked]
                rr = reciprocal_rank([str(x) for x in gd], ranked)
                hits = {f"hit@{k}": hit_at_k([str(x) for x in gd], ranked, k) for k in hit_ks}
                retr_details.append({"question": q, "doc_p_r_f1": prf, "mrr": rr, **hits})
                n_retr += 1

        if include_graph:
            ge = r.get("gold_entities")
            re = r.get("retrieved_entities")
            gr = r.get("gold_relations")
            rr = r.get("retrieved_relations")
            row_g: dict[str, Any] | None = None
            if isinstance(ge, list) and isinstance(re, list) and (ge or re):
                ent_ms = multiset_precision_recall_f1([str(x) for x in re], [str(x) for x in ge])
                ent_j = set_jaccard([str(x) for x in re], [str(x) for x in ge])
                row_g = {
                    "question": q,
                    "entity_multiset_f1": ent_ms,
                    "entity_jaccard": ent_j,
                }
                n_graph_ent += 1
            if isinstance(gr, list) and isinstance(rr, list) and (gr or rr):
                if row_g is None:
                    row_g = {"question": q}
                row_g["relation_multiset_f1"] = multiset_precision_recall_f1(
                    [str(x) for x in rr], [str(x) for x in gr]
                )
                row_g["relation_jaccard"] = set_jaccard([str(x) for x in rr], [str(x) for x in gr])
                n_graph_rel += 1
            if row_g is not None:
                graph_details.append(row_g)

    report: dict[str, Any] = {
        "schema": {
            "required_answer": ["reference", "prediction"],
            "optional_retrieval": ["gold_doc_ids", "retrieved_doc_ids", "retrieved_doc_ids_ranked"],
            "optional_graph": ["gold_entities", "retrieved_entities", "gold_relations", "retrieved_relations"],
        },
        "counts": {
            "rows_total": len(rows),
            "answer_evaluated": n_answer,
            "retrieval_evaluated": n_retr,
            "graph_entity_rows": n_graph_ent,
            "graph_relation_rows": n_graph_rel,
        },
    }

    if include_answer and n_answer:
        report["answer"] = {
            "exact_match_mean": _nested_mean(answer_details, ["exact_match"]),
            "token_f1_mean": _nested_mean(answer_details, ["token_f1", "f1"]),
            "char_f1_mean": _nested_mean(answer_details, ["char_f1", "f1"]),
            "rouge_avg": {
                rt: {
                    "f": _nested_mean(answer_details, ["rouge", rt, "f"]),
                    "p": _nested_mean(answer_details, ["rouge", rt, "p"]),
                    "r": _nested_mean(answer_details, ["rouge", rt, "r"]),
                }
                for rt in rouge_types
            },
            "details": answer_details[:max_detail_rows],
        }

    if include_retrieval and n_retr:
        report["retrieval"] = {
            "doc_precision_mean": _nested_mean(retr_details, ["doc_p_r_f1", "precision"]),
            "doc_recall_mean": _nested_mean(retr_details, ["doc_p_r_f1", "recall"]),
            "doc_f1_mean": _nested_mean(retr_details, ["doc_p_r_f1", "f1"]),
            "mrr_mean": _nested_mean(retr_details, ["mrr"]),
            **{f"{hk}_mean": _nested_mean(retr_details, [hk]) for hk in [f"hit@{k}" for k in hit_ks]},
            "details": retr_details[:max_detail_rows],
        }

    if include_graph and graph_details:
        ent_rows = [d for d in graph_details if "entity_multiset_f1" in d]
        gm: dict[str, Any] = {
            "details": graph_details[:max_detail_rows],
        }
        if ent_rows:
            gm["entity_multiset_f1_mean"] = _nested_mean(ent_rows, ["entity_multiset_f1", "f1"])
            gm["entity_jaccard_mean"] = _nested_mean(ent_rows, ["entity_jaccard"])
        rel_rows = [d for d in graph_details if "relation_multiset_f1" in d]
        if rel_rows:
            gm["relation_multiset_f1_mean"] = _nested_mean(rel_rows, ["relation_multiset_f1", "f1"])
            gm["relation_jaccard_mean"] = _nested_mean(rel_rows, ["relation_jaccard"])
        report["graph"] = gm

    return report


def build_report_from_path(
    path: Path,
    *,
    rouge_types: tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
    use_stemmer: bool = False,
    hit_ks: tuple[int, ...] = (1, 5, 10),
    include_answer: bool = True,
    include_retrieval: bool | None = None,
    include_graph: bool | None = None,
    max_detail_rows: int = 100,
) -> dict[str, Any]:
    rows = load_jsonl_rows(path)
    return build_report(
        rows,
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        hit_ks=hit_ks,
        include_answer=include_answer,
        include_retrieval=include_retrieval,
        include_graph=include_graph,
        max_detail_rows=max_detail_rows,
    )
