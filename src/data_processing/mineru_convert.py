"""PDF → Markdown + 同目錄 ``images/``（MinerU CLI；模型緩存統一在專案 ``models/``）。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from src.utils.hash_utils import md5_file
from src.utils.logger import get_logger

logger = get_logger("data_processing.mineru_convert")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    infer_backend: str = "vlm-auto-engine",
    start_page: int | None = None,
    end_page: int | None = None,
    pdf_method: str | None = None,
    formula: bool | None = None,
    table: bool | None = None,
    extra_args: Sequence[str] | None = None,
) -> Path:
    """調用 ``scripts.convert.convert_pdf``；預設輸出與 PDF 同目錄 ``<stem>.md`` + ``images/``。"""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from scripts.convert import convert_pdf

    if out_md is None:
        out_md, _, _ = mineru_sidecar_paths(pdf_path)
    cli_extra = _speed_args_to_cli(
        start_page=None,
        end_page=None,
        pdf_method=pdf_method,
        formula=formula,
        table=table,
    )
    merged: list[str] = list(cli_extra)
    if extra_args:
        merged.extend(list(extra_args))
    logger.info(
        "執行 MinerU（%s）: %s",
        infer_backend,
        pdf_path.name,
    )
    return convert_pdf(
        pdf_path,
        out_md,
        infer_backend=infer_backend,
        start_page=start_page,
        end_page=end_page,
        extra_args=merged or None,
    )


def ensure_pdf_markdown_beside_source(
    pdf_path: Path,
    *,
    infer_backend: str = "vlm-auto-engine",
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
    """當 PDF 從磁碟消失且曾由 MinerU 轉檔時，清理同目錄 ``.md``、元數據與 ``images/``。

    若不存在 ``.{stem}.mineru.json``，視為**從未由本管線產出該 PDF 的側車**（例如 conversion_cache
    陳項、或使用者僅自行放置 ``.md``），**不刪除** ``.md``／圖片，以免誤刪使用者檔案。
    """
    pdf_path = pdf_path.expanduser()
    try:
        pdf_path = pdf_path.resolve(strict=False)
    except TypeError:
        pdf_path = pdf_path.resolve()
    out_md, images_dir, meta_path = mineru_sidecar_paths(pdf_path)
    if not meta_path.is_file():
        logger.info(
            "略過 PDF 側車檔案刪除（無 MinerU 元數據 %s）；僅同步 conversion_cache",
            meta_path,
        )
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("讀取 MinerU 元數據失敗 %s: %s，僅刪元數據檔", meta_path, e)
        meta_path.unlink(missing_ok=True)
        return
    meta_pdf = meta.get("pdf")
    if meta_pdf:
        mp = Path(str(meta_pdf)).expanduser()
        if not mp.is_absolute():
            mp = (out_md.parent / mp.name).resolve()
        else:
            mp = mp.resolve()
        if mp.stem != pdf_path.stem:
            logger.warning(
                "MinerU 元數據中的 PDF 與當前路徑 stem 不一致（%s vs %s），不刪 Markdown",
                mp.stem,
                pdf_path.stem,
            )
            return
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


def remove_mineru_sidecars_for_markdown(md_path: Path) -> None:
    """刪除/已刪除索引的 Markdown 時，清理同目錄 ``.{stem}.mineru.json`` 與 MinerU ``images/``。

    - 若 ``.md`` 仍在：依正文 ``![](...)`` 刪除 ``images/`` 下對應檔案後刪除 ``.md``。
    - 若使用者已先手刪 ``.md`` 但留有 MinerU 元數據：讀 ``.{stem}.mineru.json`` 的 ``images_dir``，
      僅在該路徑等同於 ``<parent>/images`` 時整目錄移除（避免誤刪同層其他結構）。
    """
    raw = md_path.expanduser()
    try:
        md = raw.resolve(strict=False)
    except TypeError:
        md = raw.resolve()
    parent = md.parent
    stem = md.stem
    meta_path = parent / f".{stem}.mineru.json"
    images_dir = parent / "images"
    md_file = parent / f"{stem}.md"
    target = md_file if md_file.is_file() else md

    if target.is_file():
        try:
            from src.retrieval.image_refs import extract_image_refs

            text = target.read_text(encoding="utf-8", errors="replace")
            for ref in extract_image_refs(text):
                img = (target.parent / ref).resolve()
                if img.is_file() and images_dir in img.parents:
                    img.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("清理 Markdown 圖片引用失敗 %s: %s", target, e)
        target.unlink(missing_ok=True)
        try:
            meta_path.unlink(missing_ok=True)
        except OSError:
            pass
    elif meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            idir_raw = meta.get("images_dir")
            if idir_raw:
                idir = Path(str(idir_raw)).expanduser()
                if not idir.is_absolute():
                    idir = (parent / idir).resolve()
                else:
                    idir = idir.resolve()
                if idir.is_dir():
                    try:
                        if idir.resolve() == images_dir.resolve():
                            shutil.rmtree(idir, ignore_errors=True)
                    except OSError as e:
                        logger.warning("移除 MinerU images 目錄失敗 %s: %s", idir, e)
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning("讀取 MinerU 元數據失敗 %s: %s", meta_path, e)
        try:
            meta_path.unlink(missing_ok=True)
        except OSError:
            pass
    else:
        try:
            meta_path.unlink(missing_ok=True)
        except OSError:
            pass

    if images_dir.is_dir():
        try:
            if not any(images_dir.iterdir()):
                images_dir.rmdir()
        except OSError:
            pass
