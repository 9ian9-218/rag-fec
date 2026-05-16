"""觸發 LightRAG 索引增量（預設僅 .md/.txt/.docx；不含 PDF）。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.incremental.conversion_manager import ConversionManager
from src.incremental.update_manager import UpdateManager
from src.utils.logger import setup_logging


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

    if args.convert_first:
        print("=== PDF 轉檔 ===")
        print(ConversionManager().run_incremental())
        print("=== LightRAG 索引 ===")
    print(await UpdateManager().run_incremental())


if __name__ == "__main__":
    asyncio.run(_main())
