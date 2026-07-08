"""對金標 JSONL 逐題跑檢索+生成，寫入預測欄位供離線評估。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.evaluation.context_utils import build_llm_context_from_bundle, extract_chunk_sections
from src.evaluation.extract_retrieval import extract_ranked_from_bundle
from src.evaluation.runner import load_jsonl_rows
from src.evaluation.text_utils import to_simplified_chinese
from src.retrieval.retriever import GraphRAGRetriever

# 與金標 reference 語言一致，避免繁簡差異拉低生成類指標
_EVAL_SYSTEM_PROMPT = (
    "你是 FEC 领域专家助手。请完全基于检索材料，准确回答问题，使用简体中文，"
    "条理清晰，避免无依据的推测。"
)


async def _run_row(
    retriever: GraphRAGRetriever,
    row: dict,
    *,
    mode: str | None,
    multimodal: bool,
    use_llm_router: bool,
) -> dict:
    q = str(row.get("question") or "").strip()
    if not q:
        return row
    bundle_wrap = await retriever.retrieve_data(
        q,
        mode=mode,  # type: ignore[arg-type]
        use_llm_router=use_llm_router,
    )
    selected_mode = bundle_wrap.get("mode")
    mode_selection = bundle_wrap.get("mode_selection")
    data = bundle_wrap.get("data") if isinstance(bundle_wrap, dict) else {}
    bundle = data if isinstance(data, dict) else {}
    extracted = extract_ranked_from_bundle(bundle)
    settings = get_settings()
    ctx_llm = build_llm_context_from_bundle(bundle, settings)
    out = dict(row)
    out.update(extracted)
    out["context_for_llm"] = ctx_llm
    out["context_chunks_only"] = extract_chunk_sections(ctx_llm)
    if mode_selection:
        out["mode_selection"] = mode_selection
    if selected_mode:
        out["selected_mode"] = selected_mode
    answer = await retriever.query(
        q,
        mode=selected_mode if isinstance(selected_mode, str) else mode,  # type: ignore[arg-type]
        multimodal=multimodal,
        use_llm_router=False,
        system_prompt=_EVAL_SYSTEM_PROMPT,
    )
    out["prediction"] = to_simplified_chinese(str(answer or ""))
    return out


async def main_async(args: argparse.Namespace) -> None:
    apply_settings_to_environ(get_settings())
    rows = load_jsonl_rows(args.input)
    retriever = GraphRAGRetriever()
    retriever._use_llm_mode_router = not args.no_auto_mode
    use_router = not args.no_auto_mode
    out_rows: list[dict] = []
    for i, row in enumerate(rows):
        print(f"[{i + 1}/{len(rows)}] {row.get('id', i)} …", flush=True)
        out_rows.append(
            await _run_row(
                retriever,
                row,
                mode=args.mode,
                multimodal=args.multimodal,
                use_llm_router=use_router,
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_rows)} rows → {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="採集 RAG 預測與檢索排序列表，寫入評估 JSONL")
    p.add_argument("--input", type=Path, default=Path("data/test/eval_gold.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/test/eval_predictions.jsonl"))
    p.add_argument(
        "--mode",
        default=None,
        help="顯式檢索模式；省略則走 LLM 智能路由（與線上問答一致）",
    )
    p.add_argument(
        "--no-auto-mode",
        action="store_true",
        help="關閉智能路由，使用 RETRIEVAL_DEFAULT_MODE",
    )
    p.add_argument("--multimodal", action="store_true")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
