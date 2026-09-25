"""社區摘要（可選能力）命令行：構建 / 增量刷新 / 離線預覽 / 查看。

用法示例：

  # 只看社區劃分與預算估算（不調 LLM、不寫文件），能力關閉時也能跑
  python scripts/build_communities.py --dry-run

  # 增量構建：圖譜指紋未變則 0 次 LLM 調用；成員 Jaccard≥閾值的社區直接複用摘要
  python scripts/build_communities.py

  # 全量重建（忽略緩存；能力關閉時也可用於離線試跑）
  python scripts/build_communities.py --mode full --force

  # 查看當前社區摘要
  python scripts/build_communities.py --list --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.utils.logger import setup_logging


def _print_list(as_json: bool) -> int:
    from src.community import store

    data = store.load_reports()
    meta = (data or {}).get("meta") or {}
    reports = (data or {}).get("reports") or {}
    payload = {
        "enabled": get_settings().community.enabled,
        "version": store.community_version(),
        "meta": meta,
        "communities": [
            {
                "community_id": cid,
                "title": rep.get("title"),
                "size": rep.get("size"),
                "docs": rep.get("docs"),
                "stale": rep.get("stale"),
                "pending": rep.get("pending"),
            }
            for cid, rep in sorted(
                reports.items(),
                key=lambda kv: (-int((kv[1] or {}).get("size") or 0), kv[0]),
            )
        ],
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not reports:
        print("尚未構建社區摘要（可用 --dry-run 先看劃分，或去掉能力開關限制後構建）")
        return 0
    print(
        f"版本={payload['version']} 社區數={len(reports)} "
        f"分辨率={meta.get('resolution')} 目標={meta.get('target')} "
        f"模塊度={meta.get('modularity')} 未歸類={meta.get('unclustered')}"
    )
    for item in payload["communities"]:
        flag = " [stale]" if item["stale"] else (" [pending]" if item["pending"] else "")
        print(f"  {item['community_id']}  size={item['size']:>5}  {item['title']}{flag}")
    return 0


async def _run_build(args: argparse.Namespace) -> int:
    from src.community.builder import build_communities

    result = await build_communities(
        mode=args.mode,
        force=args.force,
        dry_run=args.dry_run,
        limit=args.limit,
        reason="cli",
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if result.get("skipped"):
        reason = result.get("reason")
        if reason == "disabled_by_server":
            print("社區摘要能力未開啟（COMMUNITY_ENABLED=false）；離線試跑可加 --force。")
        elif reason == "up_to_date":
            print(f"圖譜未變化，跳過（現有社區 {result.get('communities')} 個）")
        else:
            print(f"跳過：{reason} {result.get('error') or ''}".strip())
        return 0
    if result.get("dry_run"):
        print(
            f"[dry-run] 節點={result['node_count']} 邊={result['edge_count']} "
            f"目標社區={result['target']} 分辨率={result['resolution']} 模塊度={result['modularity']}"
        )
        print(f"[dry-run] 可摘要社區={result['communities']} 預估 LLM 調用={result['estimated_llm_calls']}")
        print(f"[dry-run] 規模分布={result['sizes']} 未歸類節點={result['unclustered']}")
        return 0
    print(
        f"完成：社區={result['communities']} 新建={result['built']} 複用={result['reused']} "
        f"待生成={result['pending']} 失敗={result['failed']} 向量={result['indexed']} "
        f"耗時={result['duration_s']}s"
    )
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="社區摘要（可選能力）：構建 / 刷新 / 查看")
    p.add_argument("--mode", choices=("auto", "full"), default="auto", help="auto=增量（默認），full=忽略緩存")
    p.add_argument("--force", action="store_true", help="忽略能力開關與緩存，強制執行（離線試跑用）")
    p.add_argument("--dry-run", action="store_true", help="只做劃分與預算估算，不調 LLM、不寫文件")
    p.add_argument("--limit", type=int, default=None, help="本次最多新增多少次 LLM 摘要調用")
    p.add_argument("--list", action="store_true", help="列出當前社區摘要，不執行構建")
    p.add_argument("--json", action="store_true", help="以 JSON 輸出")
    args = p.parse_args()

    setup_logging()
    apply_settings_to_environ(get_settings())

    if args.list:
        raise SystemExit(_print_list(args.json))
    raise SystemExit(asyncio.run(_run_build(args)))


if __name__ == "__main__":
    main()
