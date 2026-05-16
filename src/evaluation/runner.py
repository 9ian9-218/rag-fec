"""從 JSONL 構建 KG-RAG 評估報告（7 項核心離線指標 + 可選舊版 ROUGE）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.answer_metrics import compute_answer_row
from src.evaluation.context_utils import resolve_eval_context
from src.evaluation.multihop_metrics import multihop_correct
from src.evaluation.ragas_metrics import compute_ragas_row, compute_ragas_row_llm
from src.evaluation.align_metrics import (
    hit_at_k_docs_aligned,
    ordered_sources_from_context,
    reciprocal_rank_docs_aligned,
    ranked_metrics_bundle_aligned,
    doc_retrieval_prf_aligned,
)
from src.evaluation.retrieval_metrics import hit_at_k, reciprocal_rank, retrieval_precision_recall_f1
from src.evaluation.set_metrics import multiset_precision_recall_f1, set_jaccard


EVAL_SCHEMA = {
    "placement": "data/test/eval_gold.jsonl",
        "alignment": (
            "實體/關係/文檔/Chunk 使用 align_metrics；"
            "RAGAS v2 使用 context_for_llm、chunk-only precision、claim/embedding faithfulness；"
            "多跳以要點/別名/字符級匹配"
        ),
    "required_core": {
        "retrieval_recall_precision_ndcg": [
            "gold_entities + retrieved_entities_ranked",
            "gold_relations + retrieved_relations_ranked",
            "gold_chunk_ids + retrieved_chunk_ids_ranked",
        ],
        "ragas": ["reference", "prediction", "retrieved_context", "gold_evidence_texts (recommended)"],
        "multihop": ["multihop: true", "reference", "prediction", "multihop_answer_aliases (optional)"],
    },
    "optional_legacy": ["gold_doc_ids", "retrieved_doc_ids"],
}


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
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


def _row_k(row: dict[str, Any], default: int = 10) -> int:
    k = row.get("k", default)
    try:
        return max(1, int(k))
    except (TypeError, ValueError):
        return default


def _eval_ranked_rows(
    rows: list[dict[str, Any]],
    *,
    gold_key: str,
    ranked_key: str,
    ks: tuple[int, ...],
    kind: str,
) -> tuple[list[dict[str, Any]], int]:
    details: list[dict[str, Any]] = []
    n = 0
    for r in rows:
        gold = r.get(gold_key)
        ranked = r.get(ranked_key)
        if not isinstance(gold, list) or not isinstance(ranked, list) or not gold:
            continue
        k = _row_k(r)
        ks_use = tuple(sorted(set(ks) | {k}))
        source_paths: list[str] | None = None
        if kind == "chunk":
            stored = r.get("retrieved_chunk_sources_ranked")
            if isinstance(stored, list) and stored:
                source_paths = [str(x) for x in stored]
            else:
                source_paths = ordered_sources_from_context(str(r.get("retrieved_context") or ""))
        metrics = ranked_metrics_bundle_aligned(
            [str(x) for x in gold],
            [str(x) for x in ranked],
            ks_use,
            kind=kind,
            source_paths=source_paths,
        )
        details.append(
            {
                "id": r.get("id"),
                "question": r.get("question", ""),
                "k": k,
                "metrics": metrics.get(f"@{k}", {}),
                "metrics_all_k": metrics,
            }
        )
        n += 1
    return details, n


def build_report(
    rows: list[dict[str, Any]],
    *,
    rouge_types: tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
    use_stemmer: bool = False,
    hit_ks: tuple[int, ...] = (1, 5, 10),
    eval_ks: tuple[int, ...] = (5, 10),
    include_answer: bool = False,
    include_retrieval: bool | None = None,
    include_graph: bool | None = None,
    max_detail_rows: int = 100,
    use_embedding_faithfulness: bool = True,
    ragas_llm_client: Any = None,
    ragas_llm_model: str | None = None,
) -> dict[str, Any]:
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
    ragas_details: list[dict[str, Any]] = []
    multihop_details: list[dict[str, Any]] = []

    n_answer = 0
    n_retr = 0
    n_graph_ent = 0
    n_graph_rel = 0
    n_ragas = 0
    n_multihop = 0

    for r in rows:
        q = r.get("question", "")
        ref = str(r.get("reference") or "")
        pred = str(r.get("prediction") or "")

        if include_answer and ref and pred:
            ar = compute_answer_row(ref, pred, rouge_types=rouge_types, use_stemmer=use_stemmer)
            answer_details.append({"question": q, **ar})
            n_answer += 1

        ctx_full, _ctx_chunks = resolve_eval_context(r)
        kg = str(r.get("kg_text") or "")
        evidences = r.get("gold_evidence_texts")
        ev_list = [str(x) for x in evidences] if isinstance(evidences, list) else None
        bullets = r.get("reference_bullets")
        bl_list = [str(x) for x in bullets] if isinstance(bullets, list) else None
        if ref and pred and ctx_full:
            if ragas_llm_client is not None and ragas_llm_model:
                import asyncio

                rg = asyncio.run(
                    compute_ragas_row_llm(
                        reference=ref,
                        prediction=pred,
                        retrieved_context=ctx_full,
                        question=str(q),
                        client=ragas_llm_client,
                        model=ragas_llm_model,
                    )
                )
            else:
                rg = compute_ragas_row(
                    reference=ref,
                    prediction=pred,
                    retrieved_context=ctx_full,
                    gold_evidence_texts=ev_list,
                    kg_text=kg,
                    reference_bullets=bl_list,
                    use_embedding_faithfulness=use_embedding_faithfulness,
                )
            ragas_details.append({"id": r.get("id"), "question": q, **rg})
            n_ragas += 1

        if r.get("multihop") and ref and pred:
            aliases = r.get("multihop_answer_aliases")
            al = [str(x) for x in aliases] if isinstance(aliases, list) else None
            score = multihop_correct(
                ref,
                pred,
                aliases=al,
                reference_bullets=bl_list,
            )
            multihop_details.append({"id": r.get("id"), "question": q, "correct": score})
            n_multihop += 1

        if include_retrieval:
            gd = r.get("gold_doc_ids")
            rd = r.get("retrieved_doc_ids")
            if isinstance(gd, list) and isinstance(rd, list) and len(gd) > 0:
                prf = doc_retrieval_prf_aligned([str(x) for x in gd], [str(x) for x in rd])
                ranked = r.get("retrieved_doc_ids_ranked")
                if not isinstance(ranked, list):
                    ranked = [str(x) for x in rd]
                else:
                    ranked = [str(x) for x in ranked]
                rr = reciprocal_rank_docs_aligned([str(x) for x in gd], ranked)
                hits = {
                    f"hit@{hk}": hit_at_k_docs_aligned([str(x) for x in gd], ranked, hk)
                    for hk in hit_ks
                }
                retr_details.append({"question": q, "doc_p_r_f1": prf, "mrr": rr, **hits})
                n_retr += 1

        if include_graph:
            ge = r.get("gold_entities")
            re_ = r.get("retrieved_entities")
            gr = r.get("gold_relations")
            rr = r.get("retrieved_relations")
            row_g: dict[str, Any] | None = None
            if isinstance(ge, list) and isinstance(re_, list) and (ge or re_):
                ent_ms = multiset_precision_recall_f1([str(x) for x in re_], [str(x) for x in ge])
                ent_j = set_jaccard([str(x) for x in re_], [str(x) for x in ge])
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

    ent_details, n_ent = _eval_ranked_rows(
        rows,
        gold_key="gold_entities",
        ranked_key="retrieved_entities_ranked",
        ks=eval_ks,
        kind="entity",
    )
    rel_details, n_rel = _eval_ranked_rows(
        rows,
        gold_key="gold_relations",
        ranked_key="retrieved_relations_ranked",
        ks=eval_ks,
        kind="relation",
    )
    chk_details, n_chk = _eval_ranked_rows(
        rows,
        gold_key="gold_chunk_ids",
        ranked_key="retrieved_chunk_ids_ranked",
        ks=eval_ks,
        kind="chunk",
    )

    report: dict[str, Any] = {
        "schema": EVAL_SCHEMA,
        "counts": {
            "rows_total": len(rows),
            "answer_evaluated": n_answer,
            "retrieval_evaluated": n_retr,
            "graph_entity_rows": n_graph_ent,
            "graph_relation_rows": n_graph_rel,
            "retrieval_entity_rows": n_ent,
            "retrieval_relation_rows": n_rel,
            "retrieval_chunk_rows": n_chk,
            "ragas_evaluated": n_ragas,
            "multihop_evaluated": n_multihop,
        },
    }

    def _summarize_ranked(details: list[dict[str, Any]], metric: str) -> float:
        return _nested_mean(details, ["metrics", metric])

    if ent_details or rel_details or chk_details:
        report["retrieval_at_k"] = {
            "entities": {
                "rows": n_ent,
                "recall_mean": _summarize_ranked(ent_details, "recall"),
                "precision_mean": _summarize_ranked(ent_details, "precision"),
                "ndcg_mean": _summarize_ranked(ent_details, "ndcg"),
                "details": ent_details[:max_detail_rows],
            },
            "relations": {
                "rows": n_rel,
                "recall_mean": _summarize_ranked(rel_details, "recall"),
                "precision_mean": _summarize_ranked(rel_details, "precision"),
                "ndcg_mean": _summarize_ranked(rel_details, "ndcg"),
                "details": rel_details[:max_detail_rows],
            },
            "chunks": {
                "rows": n_chk,
                "recall_mean": _summarize_ranked(chk_details, "recall"),
                "precision_mean": _summarize_ranked(chk_details, "precision"),
                "ndcg_mean": _summarize_ranked(chk_details, "ndcg"),
                "details": chk_details[:max_detail_rows],
            },
        }

    if n_ragas:
        report["ragas"] = {
            "mode": "llm_judge" if ragas_llm_client else "v2_heuristic",
            "context_for_eval": "context_for_llm or retrieved_context",
            "context_recall_mean": _nested_mean(ragas_details, ["context_recall"]),
            "context_precision_mean": _nested_mean(ragas_details, ["context_precision"]),
            "faithfulness_mean": _nested_mean(ragas_details, ["faithfulness"]),
            "faithfulness_ngram_legacy_mean": _nested_mean(
                ragas_details, ["faithfulness_ngram_legacy"]
            ),
            "details": ragas_details[:max_detail_rows],
        }

    if n_multihop:
        report["multihop"] = {
            "accuracy": _mean([float(d["correct"]) for d in multihop_details]),
            "details": multihop_details[:max_detail_rows],
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
        gm: dict[str, Any] = {"details": graph_details[:max_detail_rows]}
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
    eval_ks: tuple[int, ...] = (5, 10),
    include_answer: bool = False,
    include_retrieval: bool | None = None,
    include_graph: bool | None = None,
    max_detail_rows: int = 100,
    use_embedding_faithfulness: bool = True,
    ragas_llm: bool = False,
) -> dict[str, Any]:
    rows = load_jsonl_rows(path)
    llm_client = None
    llm_model: str | None = None
    if ragas_llm:
        from openai import AsyncOpenAI

        from config.settings import get_settings

        s = get_settings()
        key = (s.openai_api_key or s.llm.api_key or "").strip()
        if not key:
            raise RuntimeError("ragas_llm 需要 OPENAI_API_KEY / LLM_API_KEY")
        base = (s.openai_base_url or s.llm.base_url or "https://api.openai.com/v1").rstrip("/")
        llm_client = AsyncOpenAI(api_key=key, base_url=base)
        llm_model = s.resolved_llm_model_name()
    return build_report(
        rows,
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
        hit_ks=hit_ks,
        eval_ks=eval_ks,
        include_answer=include_answer,
        include_retrieval=include_retrieval,
        include_graph=include_graph,
        max_detail_rows=max_detail_rows,
        use_embedding_faithfulness=use_embedding_faithfulness,
        ragas_llm_client=llm_client,
        ragas_llm_model=llm_model,
    )
