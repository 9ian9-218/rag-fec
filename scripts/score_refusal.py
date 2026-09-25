"""拒答/误答评分 CLI：读取预测 JSONL，按 label 统计「误答率」与「误拒率」。

用法：
  python scripts/score_refusal.py --input data/test/eval_negative_predictions.jsonl

口径与限制见 ``src/evaluation/refusal_metrics.py``；整份评测报告
（``scripts/evaluate.py``）也会在行带 label 时自动输出同一口径的 refusal 区块。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.refusal_metrics import score_rows


def main() -> int:
    p = argparse.ArgumentParser(description="按 label 统计拒答率/误答率")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="报告 JSON 输出路径")
    args = p.parse_args()

    rows: list[dict] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))

    summary, details = score_rows(rows)
    report = {"summary": summary, "details": details}

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
