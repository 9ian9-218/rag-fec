#!/usr/bin/env python3
"""PDF → 同目錄 Markdown + images/（MinerU）；獨立於 LightRAG 索引的增量轉檔步驟。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.incremental.conversion_manager import ConversionManager
from src.utils.logger import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(
        description="增量轉檔 data/raw 下的 PDF（寫入 <stem>.md 與 images/，不建索引）"
    )
    p.add_argument("pdf", type=Path, nargs="?", default=None, help="可選：僅轉單個 PDF")
    p.add_argument("--raw", type=Path, default=None, help="原始目錄，預設 PATHS_DATA_RAW")
    p.add_argument("--force", action="store_true", help="強制重轉（覆寫 DOCUMENT_MINERU_FORCE_REFRESH）")
    args = p.parse_args()

    setup_logging()
    apply_settings_to_environ(get_settings())
    mgr = ConversionManager(raw_dir=args.raw)

    if args.pdf is not None:
        out = mgr.convert_path(args.pdf.resolve(), force=args.force or None)
        print(out)
        return

    print(mgr.run_incremental())


if __name__ == "__main__":
    main()
