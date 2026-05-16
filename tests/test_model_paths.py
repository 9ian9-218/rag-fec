from __future__ import annotations

from pathlib import Path

from config.model_paths import (
    resolve_embedding_model_load_path,
    resolve_hf_hub_dir,
    resolve_hub_model_dir,
    resolve_hub_snapshot,
    resolve_models_root,
    resolve_reranker_model_load_path,
)
from config.settings import Settings
from src.data_processing.mineru_convert import mineru_sidecar_paths


def test_mineru_sidecar_paths_colocated() -> None:
    pdf = Path("/data/raw/chapter/foo.pdf")
    md, images, meta = mineru_sidecar_paths(pdf)
    assert md == Path("/data/raw/chapter/foo.md")
    assert images == Path("/data/raw/chapter/images")
    assert meta.name == ".foo.mineru.json"


def test_resolve_models_root(tmp_path: Path) -> None:
    from config.settings import ModelsSettings, PathsSettings, Settings

    s = Settings(
        paths=PathsSettings(project_root=str(tmp_path)),
        models=ModelsSettings(dir="models"),
    )
    root = resolve_models_root(s)
    assert root == (tmp_path / "models").resolve()
    assert resolve_hf_hub_dir(s) == root / "hub"


def test_resolve_reranker_path_matches_hub_or_id(tmp_path: Path) -> None:
    from config.settings import ModelsSettings, PathsSettings, Settings

    s = Settings(
        paths=PathsSettings(project_root=str(tmp_path)),
        models=ModelsSettings(dir="models", rerank_model_name="BAAI/bge-reranker-v2-m3"),
    )
    p = resolve_reranker_model_load_path(s)
    assert p == "BAAI/bge-reranker-v2-m3" or Path(p).is_dir()


def test_resolve_hub_model_dir_flat_layout(tmp_path: Path) -> None:
    """展平 Hub 目錄（無 snapshots 子目錄）仍可解析。"""
    from config.settings import ModelsSettings, PathsSettings, Settings

    s = Settings(
        paths=PathsSettings(project_root=str(tmp_path)),
        models=ModelsSettings(dir="models"),
    )
    hub = resolve_hf_hub_dir(s)
    repo = hub / "models--BAAI--bge-reranker-v2-m3"
    repo.mkdir(parents=True)
    (repo / "config.json").write_text("{}", encoding="utf-8")
    d = resolve_hub_model_dir("BAAI/bge-reranker-v2-m3", s)
    assert d is not None and d.resolve() == repo.resolve()


def test_resolve_hub_snapshot_bge_m3() -> None:
    snap = resolve_hub_snapshot("BAAI/bge-m3")
    if snap is not None:
        assert snap.is_dir()
        load = resolve_embedding_model_load_path()
        assert Path(load).is_dir() or load == "BAAI/bge-m3"
