"""離線評估：JSONL → 多維度報告（答案 ROUGE/F1、文檔檢索、實體/關係集合）。"""

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
    p = argparse.ArgumentParser(
        description=(
            "讀取 JSONL（每行一個 JSON），輸出評估報告。"
            "參考 graph-rag-agent 評估框架：答案質量 + 可選檢索/圖譜欄位。"
        )
    )
    p.add_argument("--input", type=Path, required=True, help="輸入 JSONL 路徑")
    p.add_argument("--out", type=Path, default=Path("data/test/eval_report.json"), help="報告 JSON 路徑")
    p.add_argument(
        "--rouge",
        default="rouge1,rouge2,rougeL",
        help="逗號分隔的 ROUGE 類型（預設 rouge1,rouge2,rougeL）",
    )
    p.add_argument("--stemmer", action="store_true", help="ROUGE 使用 Porter stemmer（英文為主時可開）")
    p.add_argument(
        "--hit-k",
        default="1,5,10",
        help="Hit@K 的 K 列表，逗號分隔（需有 ranked 或順序檢索 ID）",
    )
    p.add_argument("--no-answer", action="store_true", help="不計算答案層指標（僅檢索/圖譜）")
    p.add_argument("--no-retrieval", action="store_true", help="強制跳過文檔 ID 檢索指標")
    p.add_argument("--no-graph", action="store_true", help="強制跳過實體/關係集合指標")
    p.add_argument("--max-details", type=int, default=100, help="報告中每區塊保留的樣本細節條數上限")
    p.add_argument(
        "--print-full",
        action="store_true",
        help="將完整報告 JSON 印到 stdout（預設只印精簡摘要；完整報告始終寫入 --out）",
    )
    args = p.parse_args()

    try:
        import rouge_score  # noqa: F401
    except ImportError:
        print("請安裝 rouge-score: pip install rouge-score", file=sys.stderr)
        sys.exit(1)

    rouge_types = tuple(x.strip() for x in args.rouge.split(",") if x.strip())
    hit_ks_t: tuple[int, ...] = tuple(int(x.strip()) for x in args.hit_k.split(",") if x.strip())

    report = build_report_from_path(
        args.input,
        rouge_types=rouge_types,
        use_stemmer=args.stemmer,
        hit_ks=hit_ks_t,
        include_answer=not args.no_answer,
        include_retrieval=False if args.no_retrieval else None,
        include_graph=False if args.no_graph else None,
        max_detail_rows=max(0, args.max_details),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print_full:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        snap: dict[str, object] = {"counts": report.get("counts", {})}
        if "answer" in report:
            snap["answer"] = {
                "exact_match_mean": report["answer"].get("exact_match_mean"),
                "token_f1_mean": report["answer"].get("token_f1_mean"),
                "char_f1_mean": report["answer"].get("char_f1_mean"),
                "rouge_avg": report["answer"].get("rouge_avg"),
            }
        if "retrieval" in report:
            snap["retrieval"] = {k: v for k, v in report["retrieval"].items() if k != "details"}
        if "graph" in report:
            snap["graph"] = {k: v for k, v in report["graph"].items() if k != "details"}
        print(json.dumps(snap, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
