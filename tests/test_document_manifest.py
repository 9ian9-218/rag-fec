from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import DocumentConversionSettings, IncrementalSettings, PathsSettings, Settings
from src.data_processing.document_loader import LoadedDocument
from src.incremental import document_manifest as dm


@pytest.fixture
def manifest_settings(tmp_path: Path) -> Settings:
    return Settings(
        paths=PathsSettings(
            project_root=str(tmp_path),
            document_manifest_path="data/meta/document_manifest.json",
        ),
        incremental=IncrementalSettings(),
        document=DocumentConversionSettings(),
    )


def test_register_then_purge_deletes_mineru_and_images(
    tmp_path: Path, manifest_settings: Settings
) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    md = raw / "paper.md"
    imgdir = raw / "images"
    imgdir.mkdir(parents=True)
    img = imgdir / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    md.write_text("![](images/a.png)\n", encoding="utf-8")
    meta = raw / ".paper.mineru.json"
    meta.write_text(
        json.dumps({"markdown": str(md), "images_dir": str(imgdir)}),
        encoding="utf-8",
    )
    loaded = LoadedDocument(path=md, text=md.read_text(encoding="utf-8"), metadata={}, index_file_path=md)
    doc_id = "doc-test-1"
    dm.register_after_ingest(doc_id, loaded, md, settings=manifest_settings)
    out = dm.purge_for_doc_id(doc_id, settings=manifest_settings)
    assert not out["skipped"]
    assert not meta.is_file()
    assert not img.is_file()
    assert not imgdir.exists()
    assert dm.load_manifest(manifest_settings) == {}
