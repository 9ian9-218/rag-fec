from __future__ import annotations

from pathlib import Path

from src.incremental.doc_registry import stable_doc_id


def test_stable_doc_id_stable(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("x", encoding="utf-8")
    assert stable_doc_id(p) == stable_doc_id(p)
