"""社區摘要 A/B 評測：同一批宏觀問題，對比 use_community=false / true。

用法：
  python scripts/eval_community_ab.py                 # 全部問題
  python scripts/eval_community_ab.py --limit 5       # 抽樣
  python scripts/eval_community_ab.py --mode global

輸出：
  - 逐題兩臂答案、關鍵詞覆蓋率、延遲、社區注入狀態
  - 匯總：平均覆蓋率、勝/平/負、平均答案長度、注入步驟耗時 p50/p95
  - 明細落盤 data/test/community_ab_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.service.rag_service import RAGService
from src.utils.logger import setup_logging

QUESTIONS = ROOT / "data/test/community_eval_questions.jsonl"
OUT = ROOT / "data/test/community_ab_results.json"


def load_questions() -> list[dict]:
    items: list[dict] = []
    with QUESTIONS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text = answer.lower()
    hit = sum(1 for k in keywords if str(k).lower() in text)
    return hit / len(keywords)


async def run_arm(rag: RAGService, item: dict, use_community: bool, mode: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        started = time.perf_counter()
        answer = await rag.query(item["question"], mode=mode, use_community=use_community)
        elapsed = time.perf_counter() - started
        return {
            "id": item["id"],
            "question": item["question"],
            "use_community": use_community,
            "answer": str(answer),
            "latency_ms": round(elapsed * 1000, 1),
            "coverage": round(keyword_coverage(str(answer), item.get("keywords") or []), 4),
            "community": rag.last_community_status,
        }


async def measure_injection(questions: list[dict], repeats: int = 20) -> dict:
    """單獨測量"注入步驟"耗時（本地 embedding + numpy 餘弦），與 LLM 生成解耦。"""
    from src.community.injector import select_community_context

    s = get_settings()
    samples: list[float] = []
    for i in range(repeats):
        q = questions[i % len(questions)]["question"]
        t0 = time.perf_counter()
        await select_community_context(q, use_community=True, mode="global", settings=s)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "samples": len(samples),
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[max(0, int(len(samples) * 0.95) - 1)], 2),
        "max_ms": round(max(samples), 2),
    }


async def main_async(args: argparse.Namespace) -> int:
    setup_logging()
    apply_settings_to_environ(get_settings())
    questions = load_questions()[: args.limit or None]
    rag = RAGService()
    sem = asyncio.Semaphore(args.concurrency)

    tasks = []
    for item in questions:
        tasks.append(run_arm(rag, item, False, args.mode, sem))
        tasks.append(run_arm(rag, item, True, args.mode, sem))
    results = await asyncio.gather(*tasks)

    by_id: dict[str, dict[str, dict]] = {}
    for row in results:
        by_id.setdefault(row["id"], {})["on" if row["use_community"] else "off"] = row

    rows = []
    wins = ties = losses = 0
    for item in questions:
        pair = by_id.get(item["id"], {})
        off, on = pair.get("off"), pair.get("on")
        if not off or not on:
            continue
        delta = on["coverage"] - off["coverage"]
        verdict = "win" if delta > 0.05 else ("lose" if delta < -0.05 else "tie")
        wins += verdict == "win"
        ties += verdict == "tie"
        losses += verdict == "lose"
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "coverage_off": off["coverage"],
                "coverage_on": on["coverage"],
                "verdict": verdict,
                "latency_off_ms": off["latency_ms"],
                "latency_on_ms": on["latency_ms"],
                "community": on["community"],
                "answer_off": off["answer"],
                "answer_on": on["answer"],
            }
        )

    injection = await measure_injection(questions, repeats=max(10, len(questions)))
    summary = {
        "questions": len(rows),
        "mode": args.mode,
        "coverage_off": round(statistics.mean([r["coverage_off"] for r in rows]), 4) if rows else 0.0,
        "coverage_on": round(statistics.mean([r["coverage_on"] for r in rows]), 4) if rows else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "avg_answer_chars_off": round(statistics.mean([len(r["answer_off"]) for r in rows]), 1) if rows else 0.0,
        "avg_answer_chars_on": round(statistics.mean([len(r["answer_on"]) for r in rows]), 1) if rows else 0.0,
        "avg_latency_ms_off": round(statistics.mean([r["latency_off_ms"] for r in rows]), 1) if rows else 0.0,
        "avg_latency_ms_on": round(statistics.mean([r["latency_on_ms"] for r in rows]), 1) if rows else 0.0,
        "injection_cost": injection,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n逐題（coverage off → on，verdict）：")
    for r in rows:
        applied = (r.get("community") or {}).get("applied")
        print(
            f"  {r['id']} {r['coverage_off']:.2f} → {r['coverage_on']:.2f}  {r['verdict']:4s} "
            f"injected={applied} reason={(r.get('community') or {}).get('reason')}"
        )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="社區摘要 A/B 評測")
    p.add_argument("--mode", default="global", choices=("global", "hybrid", "mix"))
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 題")
    p.add_argument("--concurrency", type=int, default=3, help="並發問題數（每題兩臂順序無關）")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
