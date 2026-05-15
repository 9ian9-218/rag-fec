"""多格式文件載入與文字抽取。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.utils.hash_utils import md5_bytes, md5_file
from src.utils.logger import get_logger

logger = get_logger("data_processing.document_loader")

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".docx", ".txt"}


@dataclass
class LoadedDocument:
    """單一文件載入結果。"""

    path: Path
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_with_unstructured(path: Path) -> str:
    from unstructured.partition.auto import partition

    elements = partition(filename=str(path))
    return "\n\n".join(str(el) for el in elements if str(el).strip())


def _extract_pdf(path: Path) -> str:
    """PDF：優先 unstructured，失敗則嘗試 pypdf。"""
    try:
        return _load_with_unstructured(path)
    except Exception as e:
        logger.warning("unstructured PDF 解析失敗，嘗試 pypdf: %s", e)
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e2:
        logger.error("pypdf 仍無法解析 PDF: %s", e2)
        return ""


def _extract_docx(path: Path) -> str:
    try:
        return _load_with_unstructured(path)
    except Exception as e:
        logger.error("Word 解析失敗: %s", e)
        return ""


def _build_metadata(path: Path, text: str) -> dict[str, Any]:
    stat = path.stat()
    raw = path.read_bytes()
    return {
        "filename": path.name,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "created_hint": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "md5_file": md5_file(path),
        "md5_content": md5_bytes(raw),
        "char_count": len(text),
    }


def load_document(path: Path) -> LoadedDocument:
    """載入單一路徑支援的檔案。"""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支援的副檔名: {suf}")

    if suf in {".md", ".markdown", ".txt"}:
        text = _read_text(path)
    elif suf == ".pdf":
        text = _extract_pdf(path)
    elif suf == ".docx":
        text = _extract_docx(path)
    else:
        text = _read_text(path)

    meta = _build_metadata(path, text)
    return LoadedDocument(path=path, text=text, metadata=meta)


def iter_documents(
    directory: Path,
    *,
    recursive: bool = True,
    filter_fn: Callable[[Path], bool] | None = None,
) -> list[LoadedDocument]:
    """批次載入目錄下所有支援格式。"""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    paths: list[Path] = []
    globber = directory.rglob("*") if recursive else directory.glob("*")
    for p in globber:
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if filter_fn and not filter_fn(p):
            continue
        paths.append(p)

    paths.sort()
    out: list[LoadedDocument] = []
    for p in paths:
        try:
            out.append(load_document(p))
        except Exception as e:
            logger.error("略過無法載入的檔案 %s: %s", p, e)
    return out
