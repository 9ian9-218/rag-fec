"""使用 MinerU CLI（``mineru``）将 PDF 转为 Markdown（独立步骤，不建 LightRAG 索引）。

依赖：已安装 MinerU 且 ``mineru`` 在 PATH 中，例如 ``pip install -U "mineru[all]"``。

**与增量索引分离**（``DOCUMENT_PIPELINE_MODE=two_stage`` 时）：

- 本脚本：PDF → 同目录 ``<stem>.md`` + ``images/``（及 ``.{stem}.mineru.json`` 元数据）
- ``scripts/incremental_update.py``：仅对 ``.md`` / ``.txt`` / ``.docx`` 等建图与向量索引

子命令：

- ``one``：单个 PDF
- ``batch``：目录内全部 PDF（无变更检测）
- ``incremental``：扫描 ``data/raw``，仅转换新增/变更的 PDF（``conversion_cache.json``）

调用 MinerU 前将 HF 缓存指向项目 ``models/``（见 ``config.model_paths``）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_HF_MODELS_DIR = _PROJECT_ROOT / "models"
_HF_HUB_CACHE = _HF_MODELS_DIR / "hub"


def default_markdown_path(pdf_path: Path) -> Path:
    """与 PDF 同目录的 ``<stem>.md``（与增量管線侧车约定一致）。"""
    return pdf_path.with_suffix(".md")


def hf_project_cache_env() -> dict[str, str]:
    """子进程环境：HF 模型缓存到项目 ``models/``（Hub 使用 ``models/hub/``）。"""
    _HF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("TRANSFORMERS_CACHE", None)
    env["HF_HOME"] = str(_HF_MODELS_DIR)
    env["HF_HUB_CACHE"] = str(_HF_HUB_CACHE)
    env.pop("HF_ENDPOINT", None)
    env.pop("HUGGINGFACE_HUB_URL", None)
    env.setdefault("MINERU_HTTP_READ_TIMEOUT_SECONDS", "600")
    env.setdefault("MINERU_TASK_RESULT_TIMEOUT_SECONDS", "14400")
    return env


def mineru_subprocess_env() -> dict[str, str]:
    """优先使用 ``config.model_paths.mineru_subprocess_environ``，否则回退本地默认。"""
    try:
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from config.model_paths import mineru_subprocess_environ

        env = mineru_subprocess_environ()
    except Exception:
        env = hf_project_cache_env()
    env.setdefault("MINERU_HTTP_READ_TIMEOUT_SECONDS", "600")
    env.setdefault("MINERU_TASK_RESULT_TIMEOUT_SECONDS", "14400")
    return env


def mineru_executable() -> str | None:
    """返回 ``mineru`` 可执行路径；未找到则为 ``None``。"""
    found = shutil.which("mineru")
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / "mineru"
    if candidate.is_file():
        return str(candidate)
    return None


def _pick_largest_markdown(root: Path) -> Path:
    mds = [p for p in root.rglob("*.md") if p.is_file()]
    if not mds:
        raise RuntimeError(f"MinerU 输出目录中未找到 .md 文件: {root}")
    return max(mds, key=lambda p: p.stat().st_size)


def _sync_mineru_assets(mineru_md: Path, out_md: Path) -> None:
    """将 MinerU 与主 Markdown 同目录下的 ``images/`` 复制到 ``out_md`` 所在目录。"""
    src_root = mineru_md.parent.resolve()
    dst_root = out_md.parent.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    src = src_root / "images"
    if not src.is_dir():
        return
    dst = dst_root / "images"
    shutil.copytree(src, dst, dirs_exist_ok=True)


def convert_pdf(
    pdf_path: str | Path,
    out_path: str | Path | None = None,
    *,
    infer_backend: str = "vlm-auto-engine",
    start_page: int | None = None,
    end_page: int | None = None,
    extra_args: Sequence[str] | None = None,
) -> Path:
    """调用 ``mineru -p <pdf> -o <tmp> -b <infer_backend>``，将 Markdown 写入 ``out_path``。

    ``start_page`` / ``end_page`` 为 0-based 含端点，对应 MinerU ``-s`` / ``-e``。
    """
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

    out_md = Path(out_path).resolve() if out_path is not None else default_markdown_path(pdf_path)
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
        if start_page is not None:
            cmd.extend(["-s", str(start_page)])
        if end_page is not None:
            cmd.extend(["-e", str(end_page)])
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(cmd, check=True, env=mineru_subprocess_env())
        produced = _pick_largest_markdown(tmp_path)
        out_md.write_bytes(produced.read_bytes())
        _sync_mineru_assets(produced, out_md)

    return out_md


def _mp_convert(args: tuple[str, str, str, tuple[str, ...]]) -> tuple[str | None, str | None]:
    pdf_s, target_s, infer_b, extra = args
    try:
        out = convert_pdf(pdf_s, target_s, infer_backend=infer_b, extra_args=extra or None)
        return (str(out), None)
    except Exception as exc:
        return (None, f"{pdf_s}: {exc}")


def convert_pdf_batch(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    glob_pattern: str = "*.pdf",
    infer_backend: str = "vlm-auto-engine",
    extra_args: Sequence[str] | None = None,
    recursive: bool = False,
    workers: int = 1,
    continue_on_error: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    """批量转换目录下的 PDF；默认输出到 ``data/raw/``。"""
    input_dir = Path(input_dir).resolve()
    output_dir = (_DEFAULT_RAW_DIR if output_dir is None else Path(output_dir)).resolve()
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
    tasks: list[tuple[str, str, str, tuple[str, ...]]] = []
    for pdf in pdfs:
        rel = pdf.relative_to(input_dir)
        target = (output_dir / rel).with_suffix(".md")
        tasks.append((str(pdf), str(target), infer_backend, extra_tuple))

    written: list[Path] = []

    def handle_result(path_s: str | None, err: str | None) -> None:
        if err:
            if not continue_on_error:
                raise RuntimeError(err) from None
            return
        assert path_s is not None
        written.append(Path(path_s))

    if workers == 1:
        it = tasks
        if show_progress and tqdm is not None:
            it = tqdm(tasks, desc="PDF→MD (MinerU)", unit="file")  # type: ignore[assignment]
        for pdf_s, target_s, infer_b, extra in it:
            try:
                written.append(
                    convert_pdf(
                        pdf_s,
                        target_s,
                        infer_backend=infer_b,
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scripts/convert.py",
        description=(
            "使用 MinerU 将 PDF 完整转为 Markdown（公式/表格/图片均保留）。"
            f"默认输出目录: {_DEFAULT_RAW_DIR}"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    eng = "MinerU -b 推理后端（默认 vlm-auto-engine）。"

    one = sub.add_parser("one", help="转换单个 PDF。")
    one.add_argument("pdf", type=Path, help="输入 .pdf 路径")
    one.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"输出 .md（默认: {_DEFAULT_RAW_DIR}/<pdf 同名>.md）",
    )
    one.add_argument("--mineru-engine", default="vlm-auto-engine", help=eng)
    one.add_argument("--force", action="store_true", help="忽略 .mineru 元数据强制重转")

    many = sub.add_parser("batch", help="批量转换目录下的 PDF。")
    many.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=_DEFAULT_RAW_DIR,
        help=f"含 PDF 的目录（默认: {_DEFAULT_RAW_DIR}）",
    )
    many.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=_DEFAULT_RAW_DIR,
        help=f"输出 Markdown 根目录（默认: {_DEFAULT_RAW_DIR}）",
    )
    many.add_argument("--glob", dest="glob_pattern", default="*.pdf", help='glob（默认 "*.pdf"）')
    many.add_argument("-r", "--recursive", action="store_true", help="递归 rglob")
    many.add_argument("--mineru-engine", default="vlm-auto-engine", help=eng)
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

    inc = sub.add_parser(
        "incremental",
        help="增量转档 data/raw 下 PDF（写 conversion_cache；不建索引）。",
    )
    inc.add_argument("pdf", type=Path, nargs="?", default=None, help="可选：仅转单个 PDF")
    inc.add_argument("--raw", type=Path, default=None, help="原始目录，默认 PATHS_DATA_RAW")
    inc.add_argument("--force", action="store_true", help="强制重转（覆盖 DOCUMENT_MINERU_FORCE_REFRESH）")
    inc.add_argument("--mineru-engine", default=None, help=eng + " 默认读 DOCUMENT_MINERU_INFER_BACKEND。")

    return p


def _apply_project_settings() -> None:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from config.settings import apply_settings_to_environ, get_settings

    apply_settings_to_environ(get_settings())


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "one":
        _apply_project_settings()
        pdf = args.pdf.resolve()
        if args.output is not None:
            out = convert_pdf(pdf, args.output, infer_backend=args.mineru_engine)
        else:
            from src.incremental.conversion_manager import ConversionManager

            out = ConversionManager().convert_path(
                pdf,
                force=args.force or None,
                infer_backend=args.mineru_engine,
            )
        print(out)
        return 0

    if args.command == "incremental":
        _apply_project_settings()
        from src.incremental.conversion_manager import ConversionManager

        mgr = ConversionManager(raw_dir=args.raw)
        if args.pdf is not None:
            backend = args.mineru_engine
            out = mgr.convert_path(
                args.pdf.resolve(),
                force=args.force or None,
                infer_backend=backend,
            )
            print(out)
            return 0
        print(mgr.run_incremental())
        return 0

    paths = convert_pdf_batch(
        args.input_dir,
        args.output_dir,
        glob_pattern=args.glob_pattern,
        infer_backend=args.mineru_engine,
        recursive=args.recursive,
        workers=args.workers,
        continue_on_error=args.continue_on_error,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
