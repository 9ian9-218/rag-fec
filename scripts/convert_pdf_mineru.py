#!/usr/bin/env python3
"""將單個 PDF 轉為 Markdown 並在同目錄生成 ``images/``（需 MinerU CLI；模型目錄 ``models/``）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import apply_settings_to_environ, get_settings
from src.data_processing.mineru_convert import convert_pdf_to_markdown, mineru_sidecar_paths


def main() -> None:
    p = argparse.ArgumentParser(description="PDF → 同目錄 Markdown（MinerU）+ images/")
    p.add_argument("pdf", type=Path, help="輸入 PDF")
    p.add_argument("-o", "--out", type=Path, default=None, help="輸出 .md（預設與 PDF 同目錄 <stem>.md）")
    p.add_argument("-b", "--backend", default="pipeline", help="MinerU -b 推論後端")
    p.add_argument("--force", action="store_true", help="忽略 .mineru 元資料強制重轉")
    args = p.parse_args()

    apply_settings_to_environ(get_settings())
    pdf = args.pdf.resolve()
    out = args.out
    if out is None:
        out, _, _ = mineru_sidecar_paths(pdf)
    else:
        out = out.resolve()

    if args.force or not out.is_file():
        convert_pdf_to_markdown(pdf, out, infer_backend=args.backend)
    print(out)


if __name__ == "__main__":
    main()
