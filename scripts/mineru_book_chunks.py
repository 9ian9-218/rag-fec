#!/usr/bin/env python3
"""按固定页数分批调用 MinerU 转换 PDF，并合并为单个 Markdown。

用法（项目根目录）::

    python scripts/mineru_book_chunks.py data/raw/某书.pdf
    python scripts/mineru_book_chunks.py data/raw/某书.pdf --chunk-size 16 --dry-run
    python scripts/mineru_book_chunks.py data/raw/某书.pdf --only-merge

分片默认写入 ``data/raw/chunks/<pdf 文件名不含扩展名>/0-15.md`` 等；
合并结果默认 ``data/raw/<pdf 文件名不含扩展名>.md``。
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RAW = ROOT / "data" / "raw"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _prepend_rag_conda_bin() -> None:
    for p in (
        Path.home() / "miniconda3/envs/rag/bin",
        Path.home() / "anaconda3/envs/rag/bin",
    ):
        if (p / "mineru").is_file():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
            break


def _default_mineru_poll_env() -> None:
    os.environ.setdefault("MINERU_HTTP_READ_TIMEOUT_SECONDS", "600")
    os.environ.setdefault("MINERU_TASK_RESULT_TIMEOUT_SECONDS", "14400")


def pdf_page_count(pdf: Path) -> int:
    """读取 PDF 页数（0-based 最大页索引为 返回值 - 1）。"""
    errors: list[str] = []
    for mod_name, reader_factory in (
        ("pypdf", lambda: importlib.import_module("pypdf").PdfReader(str(pdf))),
        ("PyPDF2", lambda: importlib.import_module("PyPDF2").PdfReader(str(pdf))),
        ("fitz", lambda: importlib.import_module("fitz").open(pdf)),
    ):
        try:
            reader = reader_factory()
            if mod_name == "fitz":
                try:
                    return int(reader.page_count)
                finally:
                    reader.close()
            return len(reader.pages)
        except ImportError:
            errors.append(f"未安装 {mod_name}")
        except Exception as exc:
            errors.append(f"{mod_name}: {exc}")
    raise RuntimeError(
        "无法自动检测 PDF 页数。请安装 pypdf（pip install pypdf）或使用 --total-pages 指定。"
        f" 详情: {'; '.join(errors)}"
    )


def plan_chunks(*, last_index: int, chunk_size: int) -> list[tuple[int, int]]:
    """从第 0 页起按 ``chunk_size`` 分页，返回 (start, end) 含端点（与 MinerU -s/-e 一致）。"""
    if chunk_size < 1:
        raise ValueError("chunk_size 须 >= 1")
    if last_index < 0:
        raise ValueError("PDF 无有效页")
    chunks: list[tuple[int, int]] = []
    start = 0
    while start <= last_index:
        end = min(start + chunk_size - 1, last_index)
        chunks.append((start, end))
        start = end + 1
    return chunks


def discover_chunk_mds(chunk_dir: Path) -> list[Path]:
    pat = re.compile(r"^(\d+)-(\d+)\.md$")
    found: list[tuple[int, Path]] = []
    for p in chunk_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


def merge_chunks(chunk_paths: list[Path], out_full: Path) -> None:
    sep = "\n\n<!-- --- chunk --- -->\n\n"
    parts: list[str] = []
    for p in chunk_paths:
        text = p.read_text(encoding="utf-8")
        parts.append(f"<!-- source: {p.name} -->\n{text.strip()}")
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_full.write_text(sep.join(parts) + "\n", encoding="utf-8")


def default_paths(pdf: Path) -> tuple[Path, Path]:
    """分片目录与合并后的 md 路径。"""
    stem = pdf.stem
    chunk_dir = _DEFAULT_RAW / "chunks" / stem
    merge_out = _DEFAULT_RAW / f"{stem}.md"
    return chunk_dir, merge_out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将指定 PDF 按页分批经 MinerU 转为 Markdown 并合并。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf",
        type=Path,
        help="待转换的 PDF 路径",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16,
        help="每批处理的页数",
    )
    parser.add_argument(
        "--total-pages",
        type=int,
        default=None,
        help="PDF 总页数；省略则自动检测",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="分片 md 输出目录（默认 data/raw/chunks/<pdf 主文件名>/）",
    )
    parser.add_argument(
        "--merge-out",
        type=Path,
        default=None,
        help="合并后的 Markdown（默认 data/raw/<pdf 主文件名>.md）",
    )
    parser.add_argument(
        "--engine",
        default="pipeline",
        help="MinerU -b 推理后端",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印分批计划")
    parser.add_argument("--only-merge", action="store_true", help="不转换，只合并已有分片")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若分片 md 已存在且非空则跳过该批转换",
    )
    args = parser.parse_args()

    _prepend_rag_conda_bin()
    _default_mineru_poll_env()

    pdf = args.pdf.expanduser().resolve()
    if not pdf.is_file():
        print(f"找不到 PDF: {pdf}", file=sys.stderr)
        return 1

    default_chunk_dir, default_merge = default_paths(pdf)
    chunk_dir = (args.chunk_dir or default_chunk_dir).resolve()
    merge_out = (args.merge_out or default_merge).resolve()

    total_pages = args.total_pages
    if total_pages is None:
        total_pages = pdf_page_count(pdf)
        print(f"自动检测页数: {total_pages}")
    if total_pages < 1:
        print("总页数须 >= 1", file=sys.stderr)
        return 1

    last_index = total_pages - 1
    chunks = plan_chunks(last_index=last_index, chunk_size=args.chunk_size)

    print(f"PDF: {pdf}")
    print(f"分片目录: {chunk_dir}")
    print(f"合并输出: {merge_out}")
    print(f"计划分片（0-based 含端点，每批最多 {args.chunk_size} 页）:")
    for s, e in chunks:
        print(f"  {s}-{e}  ->  {chunk_dir / f'{s}-{e}.md'}")

    if args.dry_run:
        return 0

    if args.only_merge:
        mds = discover_chunk_mds(chunk_dir)
        if not mds:
            print(f"未在 {chunk_dir} 找到形如 数字-数字.md 的分片。", file=sys.stderr)
            return 1
        print(f"合并 {len(mds)} 个分片 -> {merge_out}")
        merge_chunks(mds, merge_out)
        print("完成。")
        return 0

    from scripts.convert import convert_pdf

    chunk_dir.mkdir(parents=True, exist_ok=True)
    for s, e in chunks:
        out_md = chunk_dir / f"{s}-{e}.md"
        if args.skip_existing and out_md.is_file() and out_md.stat().st_size > 50:
            print(f"跳过已存在: {out_md.name}")
            continue
        print(f"\n>>> 转换页 {s}-{e} -> {out_md.name}")
        convert_pdf(
            pdf,
            out_md,
            infer_backend=args.engine,
            start_page=s,
            end_page=e,
        )
        print(f"    完成: {out_md}")

    mds = discover_chunk_mds(chunk_dir)
    if not mds:
        print("无分片可合并。", file=sys.stderr)
        return 1
    print(f"\n合并 {len(mds)} 个分片 -> {merge_out}")
    merge_chunks(mds, merge_out)
    print("合并完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
