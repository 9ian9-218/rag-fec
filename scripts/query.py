"""命令列問答：直接調用 ``RAGService``（無需啟動 HTTP API）。

用法示例：

  python scripts/query.py "什麼是循環碼？"          # 預設：LLM 智能選模式
  python scripts/query.py -q "問題" --mode mix      # 顯式指定模式，跳過路由
  python scripts/query.py "問題" --json --show-mode
  python scripts/query.py --interactive
  echo "簡述第五章重點" | python scripts/query.py
  python scripts/query.py "..." --context --json
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
from src.service.rag_service import RAGService
from src.utils.logger import setup_logging


def _resolve_question(args: argparse.Namespace) -> str | None:
    q = (args.question_opt or args.question_pos or "").strip()
    if q:
        return q
    if not sys.stdin.isatty():
        raw = sys.stdin.read()
        s = raw.strip()
        return s or None
    return None


async def _run_stream(
    rag: RAGService,
    question: str,
    session_id: str | None,
    mode: str | None,
    multimodal: bool,
    *,
    no_auto_mode: bool,
) -> None:
    use_router = not no_auto_mode
    rag.set_llm_mode_router(use_router)
    res = await rag.query(
        question,
        session_id=session_id,
        mode=mode,
        stream=True,
        multimodal=multimodal,
        use_llm_router=use_router if mode is None else False,
    )
    if hasattr(res, "__aiter__"):
        async for chunk in res:  # type: ignore[union-attr]
            if chunk:
                print(str(chunk), end="", flush=True)
        print()
    else:
        print(str(res))


async def _async_main(args: argparse.Namespace) -> int:
    setup_logging()
    apply_settings_to_environ(get_settings())

    rag = RAGService()
    use_router = not args.no_auto_mode
    rag.set_llm_mode_router(use_router)
    mode: str | None = args.mode
    router_kw = use_router if mode is None else False
    session_id: str | None = args.session_id

    if args.interactive:
        sid = session_id or rag.new_session()
        print("互動問答（輸入空行略過，exit / quit / Ctrl-D 結束）", file=sys.stderr)
        print(f"session_id={sid}", file=sys.stderr)
        while True:
            try:
                line = input("Q> ").strip()
            except EOFError:
                print(file=sys.stderr)
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit", "/exit", "/quit"):
                break
            if args.stream:
                await _run_stream(rag, line, sid, mode, args.multimodal, no_auto_mode=args.no_auto_mode)
            else:
                ans = await rag.query(
                    line,
                    session_id=sid,
                    mode=mode,
                    stream=False,
                    multimodal=args.multimodal,
                    use_llm_router=router_kw,
                )
                print(ans)
        return 0

    question = _resolve_question(args)
    if not question:
        print("錯誤：請提供問題（位置參數、-q/--question，或 stdin 管道）", file=sys.stderr)
        return 2

    if args.context_only:
        payload = await rag.query_with_context(
            question,
            mode=mode,
            stream=False,
            use_llm_router=router_kw,
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        elif payload.get("mode_selection"):
            print(json.dumps(payload.get("mode_selection"), ensure_ascii=False), file=sys.stderr)
        else:
            print(json.dumps(payload, ensure_ascii=False, default=str))
        return 0

    if args.stream:
        await _run_stream(rag, question, session_id, mode, args.multimodal, no_auto_mode=args.no_auto_mode)
        return 0

    answer = await rag.query(
        question,
        session_id=session_id,
        mode=mode,
        stream=False,
        multimodal=args.multimodal,
        use_llm_router=router_kw,
    )
    if args.json:
        payload: dict = {"answer": answer}
        if args.show_mode and rag._retriever._last_mode_route is not None:
            payload["mode_selection"] = rag._retriever._last_mode_route.to_dict()
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(answer)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="本地 RAG 問答（直接調用 RAGService，不依賴 HTTP）")
    p.add_argument(
        "question_pos",
        nargs="?",
        default=None,
        metavar="QUESTION",
        help="問題（可與 -q 二選一；若省略且未開 --interactive，可從 stdin 讀取全文）",
    )
    p.add_argument("-q", "--question", dest="question_opt", default=None, help="問題字串")
    p.add_argument(
        "-m",
        "--mode",
        default=None,
        choices=("naive", "local", "global", "hybrid", "mix", "bypass"),
        help="檢索模式；指定後跳過 LLM 自動路由",
    )
    p.add_argument(
        "--no-auto-mode",
        action="store_true",
        help="關閉檢索前 LLM 模式路由（使用 RETRIEVAL_DEFAULT_MODE 或啟發式）",
    )
    p.add_argument(
        "--show-mode",
        action="store_true",
        help="與 --json 同用時輸出 mode_selection（難度/複雜度/選中模式）",
    )
    p.add_argument("--session-id", default=None, help="對話 session（多輪記憶）；不傳則用 default")
    p.add_argument("--stream", action="store_true", help="串流輸出答案（逐塊寫入 stdout）")
    p.add_argument(
        "--context",
        dest="context_only",
        action="store_true",
        help="僅檢索結構化上下文（aquery_data），不生成最終回答",
    )
    p.add_argument("--json", action="store_true", help="以 JSON 輸出（--context 時為美化縮排）")
    p.add_argument("-i", "--interactive", action="store_true", help="REPL 多輪問答（共用 session）")
    p.add_argument(
        "--multimodal",
        action="store_true",
        help="檢索後將 chunk 內圖片一併送入視覺模型（與 --stream 同開時會自動改非串流）",
    )
    args = p.parse_args()
    try:
        raise SystemExit(asyncio.run(_async_main(args)))
    except KeyboardInterrupt:
        print("\n中斷", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
