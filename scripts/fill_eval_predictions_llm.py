"""为 eval_predictions.jsonl 补全 prediction（用 OPENAI_* 主 LLM，不走 LightRAG aquery）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.evaluation.runner import load_jsonl_rows
from src.retrieval.multimodal_answer import answer_with_retrieved_text_only
from src.retrieval.retriever import GraphRAGRetriever


async def main_async(args: argparse.Namespace) -> None:
    apply_settings_to_environ(get_settings())
    settings = get_settings()
    rows = load_jsonl_rows(args.input)
    retriever = GraphRAGRetriever()
    out: list[dict] = []
    for i, row in enumerate(rows):
        q = str(row.get("question") or "").strip()
        rid = row.get("id", i)
        pred = str(row.get("prediction") or "").strip()
        if pred and not args.force:
            print(f"[{i + 1}/{len(rows)}] {rid} skip (has prediction)")
            out.append(row)
            continue
        print(f"[{i + 1}/{len(rows)}] {rid} …", flush=True)
        bundle_wrap = await retriever.retrieve_data(
            q,
            mode=args.mode,  # type: ignore[arg-type]
            use_llm_router=False,
        )
        data = bundle_wrap.get("data") if isinstance(bundle_wrap, dict) else {}
        bundle = data if isinstance(data, dict) else {}
        answer = await answer_with_retrieved_text_only(
            settings=settings,
            question=q,
            bundle=bundle,
        )
        updated = dict(row)
        updated["prediction"] = answer
        if bundle_wrap.get("mode"):
            updated["selected_mode"] = bundle_wrap["mode"]
        from src.evaluation.context_utils import (
            build_llm_context_from_bundle,
            extract_chunk_sections,
        )
        from src.evaluation.extract_retrieval import extract_ranked_from_bundle

        updated.update(extract_ranked_from_bundle(bundle))
        updated["context_for_llm"] = build_llm_context_from_bundle(bundle, settings)
        updated["context_chunks_only"] = extract_chunk_sections(updated["context_for_llm"])
        out.append(updated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} rows → {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/test/eval_predictions.jsonl"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--mode", default="mix")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.out = args.out or args.input
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
