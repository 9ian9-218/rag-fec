"""評估用上下文處理：與線上 LLM 一致的截斷上下文、僅 chunk 正文。"""

from __future__ import annotations

import re
from typing import Any

from config.settings import Settings, get_settings


def build_llm_context_from_bundle(bundle: dict[str, Any], settings: Settings | None = None) -> str:
    """與回答階段一致：LightRAG chunks 經 top_k + 字數預算截斷後拼上下文。"""
    from src.retrieval.multimodal_answer import (
        _build_context_from_chunks,
        _extract_chunks_and_kg,
        _reference_trim_params,
        _trim_chunks_for_reference,
    )

    s = settings or get_settings()
    chunks, kg_text = _extract_chunks_and_kg(bundle)
    mc, mch = _reference_trim_params(s)
    trimmed = _trim_chunks_for_reference(chunks, max_chunk_count=mc, max_chars=mch)
    return _build_context_from_chunks(trimmed, kg_text, strip_images=True)


def resolve_eval_context(row: dict[str, Any]) -> tuple[str, str]:
    """
    返回 (用於 RAGAS/忠實度評估的上下文, 僅 chunk 正文用於 precision)。
    優先 ``context_for_llm``，否則 ``retrieved_context``。
    """
    full = str(row.get("context_for_llm") or row.get("retrieved_context") or "")
    chunks_only = str(row.get("context_chunks_only") or "") or extract_chunk_sections(full)
    return full, chunks_only


def extract_chunk_sections(context: str) -> str:
    """從拼裝上下文中只保留【片段 N】正文，排除【知識圖譜摘要】。"""
    if not context.strip():
        return ""
    if "【片段" not in context:
        return context
    parts: list[str] = []
    for m in re.finditer(
        r"【片段\s+\d+】[^\n]*\n(.*?)(?=\n【片段\s+\d+】|\Z)",
        context,
        flags=re.DOTALL,
    ):
        body = m.group(1).strip()
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def split_claim_sentences(text: str) -> list[str]:
    """將參考答案/要點拆為評估用短句。"""
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"[。！？!?；;\n]+", t)
    out = [p.strip() for p in parts if len(p.strip()) >= 4]
    return out
