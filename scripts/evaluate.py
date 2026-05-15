"""讀取問答 JSONL（每行 question / reference / prediction），輸出 ROUGE 等指標。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="簡易 RAG 評估（ROUGE-L）")
    p.add_argument("--input", type=Path, required=True, help="JSONL：每行含 question, reference, prediction")
    p.add_argument("--out", type=Path, default=Path("data/test/eval_report.json"))
    args = p.parse_args()

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("請安裝 rouge-score: pip install rouge-score", file=sys.stderr)
        sys.exit(1)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rows: list[dict[str, str]] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    agg = {"rougeL_f": 0.0, "rougeL_p": 0.0, "rougeL_r": 0.0, "n": 0}
    details: list[dict[str, object]] = []
    for r in rows:
        ref = str(r.get("reference") or "")
        pred = str(r.get("prediction") or "")
        if not ref or not pred:
            continue
        s = scorer.score(ref, pred)
        rl = s["rougeL"]
        agg["rougeL_f"] += rl.fmeasure
        agg["rougeL_p"] += rl.precision
        agg["rougeL_r"] += rl.recall
        agg["n"] += 1
        details.append(
            {
                "question": r.get("question", ""),
                "rougeL": {"f": rl.fmeasure, "p": rl.precision, "r": rl.recall},
            }
        )

    n = max(1, agg["n"])
    report = {
        "count": agg["n"],
        "rougeL_avg": {
            "f": agg["rougeL_f"] / n,
            "p": agg["rougeL_p"] / n,
            "r": agg["rougeL_r"] / n,
        },
        "details": details[:50],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["rougeL_avg"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
