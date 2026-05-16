from __future__ import annotations

from pathlib import Path

from src.retrieval.image_refs import extract_image_refs, resolve_local_image_path


def test_extract_markdown_images() -> None:
    t = "正文 ![](images/ab.jpg) 尾 ![x](http://x/y.png) ![](rel/z.webp)"
    r = extract_image_refs(t)
    assert "images/ab.jpg" in r
    assert "rel/z.webp" in r
    assert len([x for x in r if x.startswith("http")]) == 0


def test_resolve_relative_to_chunk_file(tmp_path: Path) -> None:
    root = tmp_path
    md = tmp_path / "doc" / "a.md"
    imgdir = md.parent / "images"
    imgdir.mkdir(parents=True)
    f = imgdir / "x.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    p = resolve_local_image_path("images/x.png", str(md), root)
    assert p is not None and p.is_file()
