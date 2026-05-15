"""檢索結果後處理：溯源、圖譜轉文字、過濾。"""

from __future__ import annotations

from typing import Any


def extract_sources(data: dict[str, Any]) -> list[dict[str, Any]]:
    """從 ``aquery_data`` 結果整理溯源資訊（盡力而為，欄位隨 LightRAG 版本略有差異）。"""
    refs: list[dict[str, Any]] = []
    chunks = data.get("data") or {}
    if isinstance(chunks, dict):
        for k, v in chunks.items():
            if isinstance(v, dict):
                refs.append(
                    {
                        "id": k,
                        "file_path": v.get("file_path"),
                        "score": v.get("score"),
                    }
                )
    return refs


def kg_dict_to_bullets(entities: list[Any], relationships: list[Any], max_items: int = 40) -> str:
    """將實體與關係列表轉為可讀條列文字。"""
    lines: list[str] = []
    for e in entities[: max_items // 2]:
        if isinstance(e, dict):
            name = e.get("entity_name") or e.get("name") or e.get("id")
            desc = (e.get("description") or "")[:200]
            lines.append(f"- 實體: {name} — {desc}")
        else:
            lines.append(f"- 實體: {e!s}")
    for r in relationships[: max_items // 2]:
        if isinstance(r, dict):
            s = r.get("src_id") or r.get("source")
            t = r.get("tgt_id") or r.get("target")
            d = (r.get("description") or "")[:200]
            lines.append(f"- 關係: {s} → {t} — {d}")
        else:
            lines.append(f"- 關係: {r!s}")
    return "\n".join(lines) if lines else ""


def compact_retrieval_payload(data: dict[str, Any], *, max_chars: int = 12000) -> dict[str, Any]:
    """縮減 payload 方便 API 回傳。"""
    import json

    s = json.dumps(data, ensure_ascii=False)
    if len(s) <= max_chars:
        return data
    return {"truncated": True, "preview": s[:max_chars]}
