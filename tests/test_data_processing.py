from __future__ import annotations

from pathlib import Path

from src.data_processing.preprocessor import preprocess
from src.data_processing.text_splitter import semantic_split_plain


def test_preprocess_collapses_whitespace() -> None:
    raw = "a  \n\n\n  b\t\tc"
    out = preprocess(raw)
    assert "a" in out and "b" in out and "c" in out
    assert "\n\n" in out


def test_semantic_split_plain(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    long_text = "段落一。\n\n" + ("句子。" * 200)
    p.write_text(long_text, encoding="utf-8")
    chunks = semantic_split_plain(long_text, chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all(c.chunk_id.startswith("chk-") for c in chunks)
