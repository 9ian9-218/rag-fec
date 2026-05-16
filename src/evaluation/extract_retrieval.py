"""從 LightRAG ``aquery_data`` 結果抽取可寫入評估 JSONL 的排序列表。"""

from __future__ import annotations

from typing import Any

from src.retrieval.multimodal_answer import _build_context_from_chunks, _extract_chunks_and_kg


def _entity_name(e: Any) -> str:
    if isinstance(e, dict):
        return str(e.get("entity_name") or e.get("name") or e.get("id") or "").strip()
    return str(e).strip()


def _relation_key(r: Any) -> str:
    if isinstance(r, dict):
        s = str(r.get("src_id") or r.get("source") or "").strip()
        t = str(r.get("tgt_id") or r.get("target") or "").strip()
        if s and t:
            return f"{s}|{t}"
    return str(r).strip()


def _chunk_id(c: dict[str, Any], fallback_idx: int) -> str:
    for key in ("chunk_id", "id", "reference_id"):
        v = c.get(key)
        if v:
            return str(v).strip()
    fp = str(c.get("file_path") or "").strip()
    if fp:
        return fp
    return f"chunk_{fallback_idx}"


def extract_ranked_from_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    chunks, kg_text = _extract_chunks_and_kg(bundle)
    inner = bundle.get("data") or {}
    if isinstance(inner, dict) and isinstance(inner.get("data"), dict):
        payload = inner["data"]
    else:
        payload = inner if isinstance(inner, dict) else {}

    entities = payload.get("entities") or []
    relations = payload.get("relationships") or []
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    retrieved_entities_ranked = [_entity_name(e) for e in entities if _entity_name(e)]
    retrieved_relations_ranked = [_relation_key(r) for r in relations if _relation_key(r)]
    retrieved_chunk_ids_ranked = [_chunk_id(c, i) for i, c in enumerate(chunks, start=1)]
    retrieved_chunk_sources_ranked = [
        str(c.get("file_path") or "").strip() for c in chunks if isinstance(c, dict)
    ]
    retrieved_context = _build_context_from_chunks(chunks, kg_text, strip_images=True)

    doc_ids: list[str] = []
    seen: set[str] = set()
    for c in chunks:
        fp = str(c.get("file_path") or "").strip()
        if fp and fp not in seen:
            seen.add(fp)
            doc_ids.append(fp)

    return {
        "retrieved_entities_ranked": retrieved_entities_ranked,
        "retrieved_relations_ranked": retrieved_relations_ranked,
        "retrieved_chunk_ids_ranked": retrieved_chunk_ids_ranked,
        "retrieved_chunk_sources_ranked": retrieved_chunk_sources_ranked,
        "retrieved_doc_ids_ranked": doc_ids,
        "retrieved_doc_ids": doc_ids,
        "retrieved_context": retrieved_context,
        "kg_text": kg_text,
    }
