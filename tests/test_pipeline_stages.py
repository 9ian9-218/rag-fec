from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from config.settings import get_settings
from src.data_processing.change_detector import (
    CONVERSION_SUFFIXES,
    INDEX_SUFFIXES,
    detect_conversion_changes,
    detect_index_changes,
)
from src.incremental.conversion_manager import markdown_path_for_pdf


def test_index_suffixes_exclude_pdf() -> None:
    assert ".pdf" not in INDEX_SUFFIXES
    assert ".pdf" in CONVERSION_SUFFIXES


def test_detect_index_changes_ignores_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "a.pdf"
    md = tmp_path / "a.md"
    pdf.write_bytes(b"%PDF-1.4")
    md.write_text("# hi", encoding="utf-8")

    with patch(
        "src.data_processing.change_detector._read_cache",
        return_value={},
    ):
        report = detect_index_changes([tmp_path])
    paths = {str(p.name) for p in report.added}
    assert "a.md" in paths
    assert "a.pdf" not in paths


def test_markdown_path_for_pdf_beside_source(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"x")
    assert markdown_path_for_pdf(pdf) == tmp_path / "paper.md"


def test_settings_two_stage_default() -> None:
    s = get_settings()
    assert s.document.is_two_stage() is True
