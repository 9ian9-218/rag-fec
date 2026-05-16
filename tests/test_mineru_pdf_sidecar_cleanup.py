"""``remove_mineru_sidecars_for_pdf``：無 MinerU 元數據時不得刪使用者 .md。"""

from __future__ import annotations

import json
from pathlib import Path

from src.data_processing.mineru_convert import remove_mineru_sidecars_for_pdf


def test_remove_pdf_sidecars_skips_md_without_meta(tmp_path: Path) -> None:
    """conversion_cache 陳項：磁碟無 PDF、無 .stem.mineru.json 時不刪同 stem 的 .md。"""
    pdf = tmp_path / "paper.pdf"
    md = tmp_path / "paper.md"
    md.write_text("# 使用者自備內容\n", encoding="utf-8")
    remove_mineru_sidecars_for_pdf(pdf)
    assert md.is_file()
    assert md.read_text(encoding="utf-8").startswith("#")


def test_remove_pdf_sidecars_with_meta_deletes_md(tmp_path: Path) -> None:
    stem = "paper"
    pdf = tmp_path / f"{stem}.pdf"
    md = tmp_path / f"{stem}.md"
    imgdir = tmp_path / "images"
    imgdir.mkdir()
    (imgdir / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    md.write_text("![](images/a.png)\n", encoding="utf-8")
    meta = tmp_path / f".{stem}.mineru.json"
    meta.write_text(
        json.dumps({"pdf": str(pdf), "markdown": str(md), "images_dir": str(imgdir)}),
        encoding="utf-8",
    )
    remove_mineru_sidecars_for_pdf(pdf)
    assert not md.is_file()
    assert not meta.is_file()
    assert not imgdir.is_dir()
