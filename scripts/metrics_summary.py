"""匯總線上 query_metrics.jsonl 監控摘要。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.online_monitor import DEFAULT_METRICS_LOG, aggregate_metrics_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="匯總線上評估遙測日誌")
    p.add_argument("--log", type=Path, default=DEFAULT_METRICS_LOG)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/test/online_metrics_summary.json"),
        help="寫入 JSON 摘要（默認 data/test/online_metrics_summary.json）",
    )
    args = p.parse_args()
    summary = aggregate_metrics_jsonl(args.log)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
