"""PDF → Markdown + 同目錄 ``images/``（MinerU CLI；模型緩存統一在專案 ``models/``）。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

from config.model_paths import apply_models_to_environ, mineru_subprocess_environ
from src.utils.hash_utils import md5_file
from src.utils.logger import get_logger

logger = get_logger("data_processing.mineru_convert")


def mineru_sidecar_paths(pdf_path: Path) -> tuple[Path, Path, Path]:
    """
    與原 PDF 同目錄的產物路徑（支援增量：改動 PDF 即重轉同路徑 md/images）。

    - ``foo.pdf`` → ``foo.md``、``images/``、``.foo.mineru.json`` 元資料
    """
    pdf_path = pdf_path.resolve()
    parent = pdf_path.parent
    out_md = parent / f"{pdf_path.stem}.md"
    images_dir = parent / "images"
    meta_path = parent / f".{pdf_path.stem}.mineru.json"
    return out_md, images_dir, meta_path


def mineru_executable() -> str | None:
    return shutil.which("mineru")


def _pick_largest_markdown(root: Path) -> Path:
    mds = [p for p in root.rglob("*.md") if p.is_file()]
    if not mds:
        raise RuntimeError(f"MinerU 輸出目錄中未找到 .md 文件: {root}")
    return max(mds, key=lambda p: p.stat().st_size)


def _sync_mineru_assets(mineru_md: Path, out_md: Path) -> None:
    """將 MinerU 輸出目錄中的 ``images/`` 同步到 ``out_md`` 所在目錄。"""
    src_root = mineru_md.parent.resolve()
    dst_root = out_md.parent.resolve()
    dst_root.mkdir(parents=True, exist_ok=True)
    src = src_root / "images"
    if not src.is_dir():
        return
    dst = dst_root / "images"
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _speed_args_to_cli(
    *,
    start_page: int | None,
    end_page: int | None,
    pdf_method: str | None,
    formula: bool | None,
    table: bool | None,
) -> list[str]:
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


def convert_pdf_to_markdown(
    pdf_path: Path,
    out_md: Path | None = None,
    *,
    infer_backend: str = "pipeline",
    start_page: int | None = None,
    end_page: int | None = None,
    pdf_method: str | None = None,
    formula: bool | None = None,
    table: bool | None = None,
    extra_args: Sequence[str] | None = None,
) -> Path:
    """調用 ``mineru``；預設輸出與 PDF 同目錄的 ``<stem>.md`` + ``images/``。"""
    mineru = mineru_executable()
    if not mineru:
        raise RuntimeError('未找到 mineru 命令。請安裝 MinerU，例如: pip install -U "mineru[all]"')

    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"需要 .pdf 文件: {pdf_path}")

    if out_md is None:
        out_md, _, _ = mineru_sidecar_paths(pdf_path)
    out_md = out_md.resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)

    apply_models_to_environ()

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
            cmd.extend(list(extra_args))
        logger.info("執行 MinerU: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=mineru_subprocess_environ())
        produced = _pick_largest_markdown(tmp_path)
        out_md.write_bytes(produced.read_bytes())
        _sync_mineru_assets(produced, out_md)

    return out_md


def ensure_pdf_markdown_beside_source(
    pdf_path: Path,
    *,
    infer_backend: str = "pipeline",
    force: bool = False,
) -> Path:
    """
    在 PDF **同目錄** 生成或重用 ``<stem>.md`` 與 ``images/``（不寫入 mineru_cache）。

    增量更新依 ``data/raw`` 下 PDF 路徑與 hash；md/images 與源檔並存，刪改 PDF 時可一併清理。
    """
    pdf_path = pdf_path.resolve()
    out_md, images_dir, meta_path = mineru_sidecar_paths(pdf_path)
    st = pdf_path.stat()
    digest = md5_file(pdf_path)
    meta = {
        "pdf": str(pdf_path),
        "md5_file": digest,
        "size_bytes": st.st_size,
        "mtime": st.st_mtime,
        "infer_backend": infer_backend,
        "markdown": str(out_md),
        "images_dir": str(images_dir),
    }
    if not force and out_md.is_file() and meta_path.is_file():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                old.get("md5_file") == digest
                and int(old.get("size_bytes", -1)) == st.st_size
                and abs(float(old.get("mtime", 0)) - st.st_mtime) < 1e-3
            ):
                logger.info("重用同目錄 MinerU 產物: %s", out_md)
                return out_md
        except (json.JSONDecodeError, OSError):
            pass

    convert_pdf_to_markdown(pdf_path, out_md, infer_backend=infer_backend)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("MinerU 產物已寫入源檔同目錄: %s (+ images/)", out_md)
    return out_md


def remove_mineru_sidecars_for_pdf(pdf_path: Path) -> None:
    """增量刪除 PDF 時，清理同目錄 ``.md``、元資料及該 md 引用的 ``images/`` 檔案。"""
    pdf_path = pdf_path.resolve()
    out_md, images_dir, meta_path = mineru_sidecar_paths(pdf_path)
    if out_md.is_file():
        try:
            from src.retrieval.image_refs import extract_image_refs

            for ref in extract_image_refs(out_md.read_text(encoding="utf-8", errors="replace")):
                img = (out_md.parent / ref).resolve()
                if img.is_file() and images_dir in img.parents:
                    img.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("清理圖片引用失敗 %s: %s", out_md, e)
        out_md.unlink(missing_ok=True)
    if meta_path.is_file():
        meta_path.unlink(missing_ok=True)
    if images_dir.is_dir():
        try:
            if not any(images_dir.iterdir()):
                images_dir.rmdir()
        except OSError:
            pass


def ensure_pdf_markdown_cached(
    pdf_path: Path,
    *,
    project_root: Path | None = None,
    data_processed: Path | None = None,
    infer_backend: str = "pipeline",
    force: bool = False,
) -> Path:
    del project_root, data_processed
    return ensure_pdf_markdown_beside_source(pdf_path, infer_backend=infer_backend, force=force)


def hf_project_cache_env(project_root: Path | None = None) -> dict[str, str]:
    """向後相容：等同 ``mineru_subprocess_environ()``。"""
    del project_root
    return mineru_subprocess_environ()
