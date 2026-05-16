"""从 retrieved_context 按线上截断规则回填 context_for_llm / context_chunks_only。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.evaluation.context_utils import extract_chunk_sections
from src.evaluation.runner import load_jsonl_rows
from src.retrieval.multimodal_answer import (
    _build_context_from_chunks,
    _reference_trim_params,
    _trim_chunks_for_reference,
)


def _parse_retrieved_context(ctx: str) -> tuple[str, list[dict[str, Any]]]:
    text = (ctx or "").strip()
    if not text:
        return "", []
    kg = ""
    body = text
    if "【知識圖譜摘要】" in text:
        m = re.match(r"【知識圖譜摘要】\s*\n(.*?)(?=\n【片段\s+\d+】|\Z)", text, flags=re.DOTALL)
        if m:
            kg = m.group(1).strip()
        body = text.split("【知識圖譜摘要】", 1)[-1]
    chunks: list[dict[str, Any]] = []
    for m in re.finditer(
        r"【片段\s+\d+】(?:\s*來源:\s*([^\n]+))?\n(.*?)(?=\n【片段\s+\d+】|\Z)",
        body,
        flags=re.DOTALL,
    ):
        fp = (m.group(1) or "").strip()
        content = (m.group(2) or "").strip()
        if content:
            chunks.append({"file_path": fp, "content": content})
    if not chunks and text and "【片段" not in text:
        chunks.append({"file_path": "", "content": text})
    return kg, chunks


def backfill_row(row: dict[str, Any], settings) -> dict[str, Any]:
    out = dict(row)
    if out.get("context_for_llm") and out.get("context_chunks_only"):
        return out
    ctx = str(row.get("retrieved_context") or "")
    kg, chunks = _parse_retrieved_context(ctx)
    mc, mch = _reference_trim_params(settings)
    trimmed = _trim_chunks_for_reference(chunks, max_chunk_count=mc, max_chars=mch)
    out["context_for_llm"] = _build_context_from_chunks(trimmed, kg, strip_images=True)
    out["context_chunks_only"] = extract_chunk_sections(out["context_for_llm"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="回填评估用 context_for_llm")
    p.add_argument("--input", type=Path, default=Path("data/test/eval_predictions.jsonl"))
    p.add_argument("--out", type=Path, default=None, help="默认覆盖 input")
    args = p.parse_args()
    apply_settings_to_environ(get_settings())
    settings = get_settings()
    out_path = args.out or args.input
    rows = load_jsonl_rows(args.input)
    updated = [backfill_row(r, settings) for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in updated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Backfilled {len(updated)} rows → {out_path}")


if __name__ == "__main__":
    main()
