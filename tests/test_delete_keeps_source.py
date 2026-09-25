"""防回归：删除索引文档时不得删除 ingest 源 .md（历史事故：源文件被当侧车删除）。"""

from __future__ import annotations

import inspect

from src.data_processing.mineru_convert import remove_mineru_metadata_sidecar
from src.service.rag_service import RAGService


def test_metadata_cleanup_keeps_markdown(tmp_path):
    md = tmp_path / "paper.md"
    md.write_text("# 标题\n\n正文", encoding="utf-8")
    meta = tmp_path / ".paper.mineru.json"
    meta.write_text("{}", encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir()
    (images / "fig1.jpg").write_bytes(b"x")

    remove_mineru_metadata_sidecar(md)

    assert md.is_file(), "源 .md 不能被删除"
    assert not meta.exists(), "MinerU 元数据侧车应被清理"
    assert (images / "fig1.jpg").is_file(), "共享 images/ 不应被删除"


def test_delete_path_does_not_use_legacy_markdown_cleanup():
    src = inspect.getsource(RAGService.delete_document_by_id)
    assert "remove_mineru_metadata_sidecar" in src
    assert "legacy_cleanup_markdown_sidecars" not in src
