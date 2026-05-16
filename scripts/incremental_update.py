"""觸發 LightRAG 索引增量（預設僅 .md/.txt/.docx；不含 PDF）。"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.incremental.conversion_manager import ConversionManager
from src.incremental.update_manager import UpdateManager
from src.utils.logger import setup_logging


def _wait_milvus_tcp(*, max_wait_sec: float = 180.0) -> None:
    """Milvus gRPC 就緒前先做 TCP 探測，避免 standalone 尚未監聽時立刻失敗。"""
    s = get_settings()
    parsed = urlparse(s.milvus.uri.strip())
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 19530
    deadline = time.monotonic() + max_wait_sec
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except OSError as e:
            last_err = e
            time.sleep(2.0)
    hint = (
        f"在 {max_wait_sec:.0f}s 內無法連線 Milvus {host}:{port}（{last_err}）。"
        "請在專案根目錄執行: docker compose up -d"
    )
    raise SystemExit(hint)


async def _main() -> None:
    p = argparse.ArgumentParser(description="索引增量（two_stage 下不處理 PDF，請先 convert_documents）")
    p.add_argument(
        "--convert-first",
        action="store_true",
        help="先執行 PDF→MD 轉檔增量，再建 LightRAG 索引",
    )
    args = p.parse_args()

    setup_logging()
    apply_settings_to_environ(get_settings())
    _wait_milvus_tcp()

    if args.convert_first:
        print("=== PDF 轉檔 ===")
        print(ConversionManager().run_incremental())
        print("=== LightRAG 索引 ===")
    print(await UpdateManager().run_incremental())


if __name__ == "__main__":
    asyncio.run(_main())
