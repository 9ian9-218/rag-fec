"""使用 MinerU CLI（``mineru``）将 PDF 转为 Markdown。

依赖：已安装 MinerU 且 ``mineru`` 在 PATH 中，例如 ``pip install -U "mineru[all]"``。

调用 MinerU 子进程前会：

- 将 ``HF_HOME`` 设为本项目下的 ``models/``，Hub 快照默认落在 ``models/hub/``。
  不再设置 ``TRANSFORMERS_CACHE`` 指向同一目录，以免 Transformers 把 MinerU 的本地权重路径误判为 repo id。
- 若未设置 ``HF_ENDPOINT``，则默认使用 ``https://hf-mirror.com`` 镜像。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

# data/data2md/convert.py → 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HF_MODELS_DIR = _PROJECT_ROOT / "models"
_HF_HUB_CACHE = _HF_MODELS_DIR / "hub"


def hf_project_cache_env() -> dict[str, str]:
    """子进程环境：HF 模型缓存到项目 ``models/``（Hub 使用 ``models/hub/``）。

    只设置 ``HF_HOME``，不显式设置 ``HF_HUB_CACHE`` / ``TRANSFORMERS_CACHE``：
    二者若与 Hub 快照目录相同，Transformers 在加载 MinerU 给出的本地子路径时会触发
    ``HFValidationError``（把路径当成 repo id 校验）。
    """
    _HF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("TRANSFORMERS_CACHE", None)
    env["HF_HOME"] = str(_HF_MODELS_DIR)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return env


def mineru_executable() -> str | None:
    """返回 ``mineru`` 可执行路径；未找到则为 ``None``。"""
    return shutil.which("mineru")


def _pick_largest_markdown(root: Path) -> Path:
    mds = [p for p in root.rglob("*.md") if p.is_file()]
    if not mds:
        raise RuntimeError(f"MinerU 输出目录中未找到 .md 文件: {root}")
    return max(mds, key=lambda p: p.stat().st_size)


def _speed_args_to_cli(
    *,
    start_page: int | None,
    end_page: int | None,
    pdf_method: str | None,
    formula: bool | None,
    table: bool | None,
) -> list[str]:
    """将可选加速/裁剪参数转为 MinerU CLI 片段（页码为 0-based，与 MinerU 一致）。"""
    parts: list[str] = []
    if start_page is not None:
        parts.extend(["-s", str(start_page)])
    if end_page is not None:
        parts.extend(["-e", str(end_page)])
    if pdf_method:
        parts.extend(["-m", pdf_method])
    if formula is False:
        parts.extend(["-f", "false"])
    if table is False:
        parts.extend(["-t", "false"])
    return parts


def convert_pdf(
    pdf_path: str | Path,
    out_path: str | Path | None = None,
    *,
    infer_backend: str = "pipeline",
    start_page: int | None = None,
    end_page: int | None = None,
    pdf_method: str | None = None,
    formula: bool | None = None,
    table: bool | None = None,
    extra_args: Sequence[str] | None = None,
) -> Path:
    """调用 ``mineru -p <pdf> -o <tmp> -b <infer_backend>``，将生成的 Markdown 写入 ``out_path``。"""
    mineru = mineru_executable()
    if not mineru:
        raise RuntimeError(
            '未找到 mineru 命令。请先安装 MinerU，例如: pip install -U "mineru[all]"'
        )

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"需要 .pdf 文件: {pdf_path}")

    if out_path is None:
        out_path = pdf_path.with_suffix(".md")
    out_md = Path(out_path).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cmd: list[str] = [
            mineru,
            "-p",
            str(pdf_path),
            "-o",
            str(tmp_path),
            "-b",
            infer_backend,
        ]
        cmd.extend(
            _speed_args_to_cli(
                start_page=start_page,
                end_page=end_page,
                pdf_method=pdf_method,
                formula=formula,
                table=table,
            )
        )
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(cmd, check=True, env=hf_project_cache_env())
        produced = _pick_largest_markdown(tmp_path)
        out_md.write_bytes(produced.read_bytes())

    return out_md


def _mp_convert(
    args: tuple[str, str, str, tuple[str, ...], int | None, int | None, str | None, bool | None, bool | None],
) -> tuple[str | None, str | None]:
    (
        pdf_s,
        target_s,
        infer_b,
        extra,
        start_p,
        end_p,
        method,
        formula,
        table,
    ) = args
    try:
        out = convert_pdf(
            pdf_s,
            target_s,
            infer_backend=infer_b,
            start_page=start_p,
            end_page=end_p,
            pdf_method=method,
            formula=formula,
            table=table,
            extra_args=extra or None,
        )
        return (str(out), None)
    except Exception as exc:
        return (None, f"{pdf_s}: {exc}")


def convert_pdf_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    glob_pattern: str = "*.pdf",
    infer_backend: str = "pipeline",
    start_page: int | None = None,
    end_page: int | None = None,
    pdf_method: str | None = None,
    formula: bool | None = None,
    table: bool | None = None,
    extra_args: Sequence[str] | None = None,
    recursive: bool = False,
    workers: int = 1,
    continue_on_error: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    """批量转换目录下的 PDF，目录结构与输入相对路径保持一致。"""
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if workers < 1:
        raise ValueError("workers must be >= 1")

    pdfs: Iterable[Path]
    if recursive:
        pdfs = sorted(input_dir.rglob(glob_pattern))
    else:
        pdfs = sorted(input_dir.glob(glob_pattern))

    extra_tuple = tuple(extra_args or ())
    tasks: list[
        tuple[str, str, str, tuple[str, ...], int | None, int | None, str | None, bool | None, bool | None]
    ] = []
    for pdf in pdfs:
        rel = pdf.relative_to(input_dir)
        target = (output_dir / rel).with_suffix(".md")
        tasks.append(
            (
                str(pdf),
                str(target),
                infer_backend,
                extra_tuple,
                start_page,
                end_page,
                pdf_method,
                formula,
                table,
            )
        )

    written: list[Path] = []
    errors: list[str] = []

    def handle_result(path_s: str | None, err: str | None) -> None:
        if err:
            errors.append(err)
            if not continue_on_error:
                raise RuntimeError(err) from None
            return
        assert path_s is not None
        written.append(Path(path_s))

    if workers == 1:
        it = tasks
        if show_progress and tqdm is not None:
            it = tqdm(tasks, desc="PDF→MD (MinerU)", unit="file")  # type: ignore[assignment]
        for (
            pdf_s,
            target_s,
            infer_b,
            extra,
            start_p,
            end_p,
            method,
            formula,
            table,
        ) in it:
            try:
                written.append(
                    convert_pdf(
                        pdf_s,
                        target_s,
                        infer_backend=infer_b,
                        start_page=start_p,
                        end_page=end_p,
                        pdf_method=method,
                        formula=formula,
                        table=table,
                        extra_args=extra or None,
                    )
                )
            except Exception as exc:
                handle_result(None, f"{pdf_s}: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_mp_convert, t) for t in tasks]
            it = as_completed(futures)
            if show_progress and tqdm is not None:
                it = tqdm(it, total=len(futures), desc="PDF→MD (MinerU)", unit="file")  # type: ignore[assignment]
            for fut in it:
                path_s, err = fut.result()
                handle_result(path_s, err)

    written.sort(key=lambda p: str(p).lower())
    return written


def _add_mineru_speed_options(parser: argparse.ArgumentParser) -> None:
    """MinerU 官方 CLI 中与速度/页范围相关的选项（见 mineru --help）。"""
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=None,
        dest="start_page",
        metavar="N",
        help="起始页（0-based，对应 MinerU -s；仅处理部分页可大幅缩短时间）。",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=None,
        dest="end_page",
        metavar="N",
        help="结束页（0-based，对应 MinerU -e）。",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=("auto", "txt", "ocr"),
        default=None,
        dest="pdf_method",
        help="解析方式：txt=纯文本抽取（电子版教材通常最快）；ocr=扫描件；auto=自动。需 pipeline/hybrid 后端。",
    )
    parser.add_argument(
        "--no-formula",
        action="store_true",
        help="关闭公式解析（MinerU -f false），减轻模型负载。",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="关闭表格解析（MinerU -t false），减轻模型负载。",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m data.data2md",
        description="使用 MinerU 将 PDF 转为 Markdown（需已安装 mineru CLI）。",
    )
    sub = p.add_subparsers(dest="command", required=True)

    eng = (
        "MinerU -b 推理后端（默认 pipeline，适合无 GPU / CPU；详见 MinerU 文档）。"
    )

    one = sub.add_parser("one", help="转换单个 PDF。")
    one.add_argument("pdf", type=Path, help="输入 .pdf 路径")
    one.add_argument("-o", "--output", type=Path, default=None, help="输出 .md（默认与 PDF 同目录同名）")
    one.add_argument("--mineru-engine", default="pipeline", help=eng)
    _add_mineru_speed_options(one)

    many = sub.add_parser("batch", help="批量转换目录下的 PDF。")
    many.add_argument("input_dir", type=Path, help="含 PDF 的目录")
    many.add_argument("output_dir", type=Path, help="输出 Markdown 的根目录")
    many.add_argument("--glob", dest="glob_pattern", default="*.pdf", help='glob（默认 "*.pdf"）')
    many.add_argument("-r", "--recursive", action="store_true", help="递归 rglob")
    many.add_argument("--mineru-engine", default="pipeline", help=eng)
    _add_mineru_speed_options(many)
    many.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="并行进程数（MinerU 占 GPU/内存时建议为 1）。",
    )
    many.add_argument(
        "--continue-on-error",
        action="store_true",
        help="跳过失败文件，不中断批量任务。",
    )
    return p


def _formula_table_from_args(ns: argparse.Namespace) -> tuple[bool | None, bool | None]:
    """--no-formula / --no-table 转为 convert_pdf 的 formula/table（None 表示使用 MinerU 默认）。"""
    formula: bool | None = False if getattr(ns, "no_formula", False) else None
    table: bool | None = False if getattr(ns, "no_table", False) else None
    return formula, table


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    formula, table = _formula_table_from_args(args)

    if args.command == "one":
        out = convert_pdf(
            args.pdf,
            args.output,
            infer_backend=args.mineru_engine,
            start_page=args.start_page,
            end_page=args.end_page,
            pdf_method=args.pdf_method,
            formula=formula,
            table=table,
        )
        print(out)
        return 0

    paths = convert_pdf_batch(
        args.input_dir,
        args.output_dir,
        glob_pattern=args.glob_pattern,
        infer_backend=args.mineru_engine,
        start_page=args.start_page,
        end_page=args.end_page,
        pdf_method=args.pdf_method,
        formula=formula,
        table=table,
        recursive=args.recursive,
        workers=args.workers,
        continue_on_error=args.continue_on_error,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
