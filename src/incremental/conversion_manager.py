"""PDF → Markdown 轉檔增量管線（與 LightRAG 索引分離）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.settings import get_settings
from src.data_processing.change_detector import (
    detect_conversion_changes,
    load_conversion_cache,
    write_conversion_cache,
)
from src.data_processing.mineru_convert import (
    ensure_pdf_markdown_beside_source,
    mineru_executable,
    mineru_sidecar_paths,
    remove_mineru_sidecars_for_pdf,
)
from src.utils.hash_utils import md5_file
from src.utils.logger import get_logger

logger = get_logger("incremental.conversion_manager")


class ConversionManager:
    """掃描 ``data/raw`` 中的 PDF，產出同目錄 ``<stem>.md`` + ``images/``。"""

    def __init__(self, raw_dir: Path | None = None) -> None:
        s = get_settings()
        root = Path(s.paths.project_root).resolve()
        self._raw = Path(raw_dir or s.paths.data_raw)
        if not self._raw.is_absolute():
            self._raw = (root / self._raw).resolve()

    def convert_path(
        self,
        pdf_path: Path,
        *,
        force: bool | None = None,
        infer_backend: str | None = None,
    ) -> Path:
        """轉換單一 PDF；回傳生成的 Markdown 路徑。"""
        s = get_settings()
        pdf_path = pdf_path.expanduser().resolve()
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"需要 .pdf 文件: {pdf_path}")
        if not s.document.use_mineru_for_pdf:
            raise RuntimeError("DOCUMENT_USE_MINERU_FOR_PDF=false，無法執行 PDF 轉檔")
        if not mineru_executable():
            raise RuntimeError("未找到 mineru 命令，請安裝 MinerU")
        refresh = s.document.mineru_force_refresh if force is None else force
        backend = (infer_backend or s.document.mineru_infer_backend).strip()
        return ensure_pdf_markdown_beside_source(
            pdf_path,
            infer_backend=backend,
            force=refresh,
        )

    def run_incremental(self) -> dict[str, Any]:
        """偵測 PDF 變更並套用 MinerU 轉檔（不寫入 LightRAG）。"""
        s = get_settings()
        if not s.document.use_mineru_for_pdf:
            logger.info("PDF 轉檔已停用（DOCUMENT_USE_MINERU_FOR_PDF=false）")
            return {"skipped": True, "reason": "mineru_disabled"}

        report = detect_conversion_changes([self._raw], recursive=True)
        stats = {"converted": 0, "removed": 0, "errors": 0}
        converted_ok: list[Path] = []

        for path_str in report.removed:
            try:
                remove_mineru_sidecars_for_pdf(Path(path_str))
                stats["removed"] += 1
            except Exception as e:
                logger.error("清理 PDF 產物失敗 %s: %s", path_str, e)
                stats["errors"] += 1

        for pdf in report.modified + report.added:
            try:
                out_md = self.convert_path(pdf)
                converted_ok.append(pdf)
                logger.info("已轉檔: %s → %s", pdf.name, out_md.name)
                stats["converted"] += 1
            except Exception:
                logger.exception("PDF 轉檔失敗: %s", pdf)
                stats["errors"] += 1

        cache = load_conversion_cache()
        for path_str in report.removed:
            cache.pop(path_str, None)
        for pdf in converted_ok:
            try:
                cache[str(pdf.resolve())] = md5_file(pdf)
            except OSError as e:
                logger.warning("無法更新 conversion_cache %s: %s", pdf, e)
        write_conversion_cache(cache)

        logger.info("PDF 轉檔增量完成: %s", stats)
        return {
            "stats": stats,
            "report": {
                "added": [str(p) for p in report.added],
                "modified": [str(p) for p in report.modified],
                "removed": list(report.removed),
            },
        }


def markdown_path_for_pdf(pdf_path: Path) -> Path:
    """PDF 對應的同目錄 Markdown 路徑。"""
    out_md, _, _ = mineru_sidecar_paths(pdf_path)
    return out_md
