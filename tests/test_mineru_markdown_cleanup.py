from __future__ import annotations

import json
from pathlib import Path

from src.data_processing.mineru_convert import remove_mineru_sidecars_for_markdown


def test_remove_markdown_sidecars_when_md_still_present(tmp_path: Path) -> None:
    stem = "doc"
    md = tmp_path / f"{stem}.md"
    imgdir = tmp_path / "images"
    imgdir.mkdir(parents=True)
    img = imgdir / "p.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    md.write_text(f"![](images/p.png)\n", encoding="utf-8")
    meta = tmp_path / f".{stem}.mineru.json"
    meta.write_text(
        json.dumps({"markdown": str(md), "images_dir": str(imgdir)}),
        encoding="utf-8",
    )
    remove_mineru_sidecars_for_markdown(md)
    assert not md.is_file()
    assert not meta.is_file()
    assert not img.is_file()
    assert not imgdir.is_dir()


def test_remove_markdown_sidecars_orphan_meta_after_md_gone(tmp_path: Path) -> None:
    stem = "paper"
    imgdir = tmp_path / "images"
    imgdir.mkdir(parents=True)
    (imgdir / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    meta = tmp_path / f".{stem}.mineru.json"
    meta.write_text(
        json.dumps(
            {
                "markdown": str(tmp_path / f"{stem}.md"),
                "images_dir": str(imgdir),
            }
        ),
        encoding="utf-8",
    )
    gone = tmp_path / f"{stem}.md"
    remove_mineru_sidecars_for_markdown(gone)
    assert not meta.is_file()
    assert not imgdir.is_dir()
