"""多格式文件載入與文字抽取。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config.settings import get_settings
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
    #: 傳給 LightRAG ``file_paths`` 的路徑（MinerU 轉檔後為 .md，否則與 ``path`` 相同）
    index_file_path: Path | None = None

    def lightrag_file_path(self) -> Path:
        return self.index_file_path if self.index_file_path is not None else self.path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_with_unstructured(path: Path) -> str:
    from unstructured.partition.auto import partition

    elements = partition(filename=str(path))
    return "\n\n".join(str(el) for el in elements if str(el).strip())


def _extract_pdf(path: Path) -> tuple[str, dict[str, Any], Path | None]:
    """PDF：two_stage 只讀同目錄 MinerU 產物；coupled 可內聯 MinerU 或 fallback 抽取。"""
    s = get_settings()
    if s.document.is_two_stage():
        from src.data_processing.mineru_convert import mineru_sidecar_paths

        out_md, _, _ = mineru_sidecar_paths(path)
        if not out_md.is_file():
            raise FileNotFoundError(
                f"兩階段模式請先轉檔 PDF（python scripts/convert.py incremental 或 one）：{out_md}"
            )
        text = _read_text(out_md)
        meta_extra = {
            "mineru": True,
            "index_markdown": str(out_md),
            "assets_dir": str(out_md.parent / "images"),
            "source_pdf": str(path.resolve()),
            "pipeline_mode": "two_stage",
        }
        return text, meta_extra, out_md

    if s.document.use_mineru_for_pdf:
        from src.data_processing.mineru_convert import ensure_pdf_markdown_beside_source, mineru_executable

        if mineru_executable():
            try:
                out_md = ensure_pdf_markdown_beside_source(
                    path,
                    infer_backend=s.document.mineru_infer_backend,
                    force=s.document.mineru_force_refresh,
                )
                text = _read_text(out_md)
                meta_extra = {
                    "mineru": True,
                    "index_markdown": str(out_md),
                    "assets_dir": str(out_md.parent / "images"),
                    "source_pdf": str(path.resolve()),
                }
                return text, meta_extra, out_md
            except Exception as e:
                logger.warning("MinerU 轉換失敗，改走一般 PDF 抽取: %s", e)
        else:
            logger.warning("已設定 DOCUMENT_USE_MINERU_FOR_PDF 但未找到 mineru，改走一般 PDF 抽取")

    try:
        return _load_with_unstructured(path), {}, None
    except Exception as e:
        logger.warning("unstructured PDF 解析失敗，嘗試 pypdf: %s", e)
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages), {}, None
    except Exception as e2:
        logger.error("pypdf 仍無法解析 PDF: %s", e2)
        return "", {}, None


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

    index_file: Path | None = None
    meta_extra: dict[str, Any] = {}

    if suf in {".md", ".markdown", ".txt"}:
        text = _read_text(path)
    elif suf == ".pdf":
        text, meta_extra, index_file = _extract_pdf(path)
    elif suf == ".docx":
        text = _extract_docx(path)
    else:
        text = _read_text(path)

    meta = _build_metadata(path, text)
    meta.update(meta_extra)
    return LoadedDocument(path=path, text=text, metadata=meta, index_file_path=index_file)


def iter_documents(
    directory: Path,
    *,
    recursive: bool = True,
    filter_fn: Callable[[Path], bool] | None = None,
    suffixes: set[str] | None = None,
) -> list[LoadedDocument]:
    """批次載入目錄下所有支援格式。"""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    allowed = suffixes if suffixes is not None else SUPPORTED_SUFFIXES
    paths: list[Path] = []
    globber = directory.rglob("*") if recursive else directory.glob("*")
    for p in globber:
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed:
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
