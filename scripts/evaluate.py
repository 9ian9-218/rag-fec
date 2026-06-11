"""離線評估：JSONL → KG-RAG 七項核心指標報告（v2 改進 RAGAS + 對齊檢索）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.getLogger("jieba").setLevel(logging.WARNING)

from src.evaluation.runner import build_report_from_path


def main() -> None:
    p = argparse.ArgumentParser(description="讀取 eval_predictions.jsonl，輸出離線評估報告。")
    p.add_argument("--input", type=Path, required=True, help="輸入 JSONL（建議 eval_predictions.jsonl）")
    p.add_argument("--out", type=Path, default=Path("data/test/eval_report.json"), help="報告 JSON 路徑")
    p.add_argument("--eval-k", default="5,10", help="Recall/Precision/NDCG 的 K 列表")
    p.add_argument("--include-answer", action="store_true", help="額外計算 ROUGE/F1")
    p.add_argument("--rouge", default="rouge1,rouge2,rougeL")
    p.add_argument("--stemmer", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--max-details", type=int, default=100)
    p.add_argument("--print-full", action="store_true")
    p.add_argument(
        "--no-embedding-faithfulness",
        action="store_true",
        help="忠實度僅用 claim 匹配，不載入嵌入模型",
    )
    p.add_argument(
        "--no-ragas",
        action="store_true",
        help="跳過 RAGAS 三項（預設使用 ragas 包 + 專案 LLM 配置）",
    )
    p.add_argument(
        "--ragas-llm",
        action="store_true",
        help="（已棄用，預設即啟用 ragas LLM 評估）",
    )
    args = p.parse_args()

    if args.include_answer:
        try:
            import rouge_score  # noqa: F401
        except ImportError:
            print("請安裝 rouge-score: pip install rouge-score", file=sys.stderr)
            sys.exit(1)

    rouge_types = tuple(x.strip() for x in args.rouge.split(",") if x.strip())
    eval_ks_t: tuple[int, ...] = tuple(int(x.strip()) for x in args.eval_k.split(",") if x.strip())

    report = build_report_from_path(
        args.input,
        rouge_types=rouge_types,
        use_stemmer=args.stemmer,
        eval_ks=eval_ks_t,
        include_answer=args.include_answer,
        include_graph=False if args.no_graph else None,
        max_detail_rows=max(0, args.max_details),
        use_embedding_faithfulness=not args.no_embedding_faithfulness,
        ragas_llm=not args.no_ragas,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print_full:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        snap: dict[str, object] = {"counts": report.get("counts", {})}
        if "retrieval_at_k" in report:
            snap["retrieval_at_k"] = {
                name: {
                    "recall_mean": block.get("recall_mean"),
                    "precision_mean": block.get("precision_mean"),
                    "ndcg_mean": block.get("ndcg_mean"),
                }
                for name, block in report["retrieval_at_k"].items()
                if isinstance(block, dict)
            }
        if "ragas" in report:
            snap["ragas"] = {k: v for k, v in report["ragas"].items() if k != "details"}
        if "multihop" in report:
            snap["multihop"] = {k: v for k, v in report["multihop"].items() if k != "details"}
        print(json.dumps(snap, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
