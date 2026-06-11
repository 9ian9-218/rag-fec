"""合并离线 eval_report 与线上 query_metrics 摘要，输出总览 JSON。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.online_monitor import DEFAULT_METRICS_LOG, aggregate_metrics_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="导出离线+线上全部评估指标总览")
    p.add_argument("--offline", type=Path, default=Path("data/test/eval_report.json"))
    p.add_argument("--online-log", type=Path, default=DEFAULT_METRICS_LOG)
    p.add_argument("--out", type=Path, default=Path("data/test/eval_all_metrics.json"))
    args = p.parse_args()

    offline: dict = {}
    if args.offline.is_file():
        offline = json.loads(args.offline.read_text(encoding="utf-8"))

    online_summary = aggregate_metrics_jsonl(args.online_log)
    online_row_count = online_summary.get("count", 0)

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "offline_gold": "data/test/eval_gold.jsonl",
            "offline_predictions": "data/test/eval_predictions.jsonl",
            "offline_report": str(args.offline),
            "online_raw_log": str(args.online_log),
            "online_summary": "data/test/online_metrics_summary.json",
        },
        "offline": {
            "counts": offline.get("counts"),
            "retrieval_at_k": _strip_details(offline.get("retrieval_at_k")),
            "ragas": _strip_details(offline.get("ragas")),
            "multihop": _strip_details(offline.get("multihop")),
            "answer": _strip_details(offline.get("answer")),
            "graph": _strip_details(offline.get("graph")),
        },
        "online": online_summary,
        "notes": {
            "offline": "需金标 eval_gold.jsonl + collect 生成的 eval_predictions.jsonl，由 evaluate.py 计算",
            "online": "每次 query/retrieve 追加到 query_metrics.jsonl，无需测试集",
            "ragas_mode": (offline.get("ragas") or {}).get("mode"),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    print(f"\nWrote → {args.out} (online rows: {online_row_count})")


def _strip_details(block: object) -> object:
    if not isinstance(block, dict):
        return block
    out: dict = {}
    for k, v in block.items():
        if k == "details":
            continue
        if isinstance(v, dict):
            out[k] = _strip_details(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
