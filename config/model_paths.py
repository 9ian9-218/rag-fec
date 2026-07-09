"""專案內本地模型路徑（統一使用 ``<project_root>/models``）。"""

from __future__ import annotations

import os
from pathlib import Path

from config.settings import Settings, get_settings


def resolve_project_root(settings: Settings | None = None) -> Path:
    s = settings or get_settings()
    root = Path(s.paths.project_root).expanduser()
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    return root.resolve()


def resolve_models_root(settings: Settings | None = None) -> Path:
    """``models/`` 根目錄；HF Hub 快照在 ``models/hub/``。"""
    s = settings or get_settings()
    rel = (s.models.dir or "models").strip() or "models"
    p = Path(rel)
    if not p.is_absolute():
        p = resolve_project_root(s) / p
    p.mkdir(parents=True, exist_ok=True)
    (p / "hub").mkdir(parents=True, exist_ok=True)
    return p.resolve()


def resolve_hf_hub_dir(settings: Settings | None = None) -> Path:
    return (resolve_models_root(settings) / "hub").resolve()


def _hub_repo_dir(hub: Path, repo_id: str) -> Path:
    safe = repo_id.replace("/", "--")
    return hub / f"models--{safe}"


def resolve_hub_snapshot(repo_id: str, settings: Settings | None = None) -> Path | None:
    """若 ``models/hub/models--{org}--{name}/snapshots/<rev>/`` 存在且含 config.json 則回傳最新有效快照。"""
    snaps_root = _hub_repo_dir(resolve_hf_hub_dir(settings), repo_id) / "snapshots"
    if not snaps_root.is_dir():
        return None
    candidates = [
        p
        for p in snaps_root.iterdir()
        if p.is_dir() and (p / "config.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_hub_model_dir(repo_id: str, settings: Settings | None = None) -> Path | None:
    """
    解析 Hub 快取目錄：優先 ``snapshots/<rev>/``；否則若 ``models--org--name/config.json``
    直接位於 repo 根（部分下載工具展平目錄）則回傳該根目錄。
    """
    snap = resolve_hub_snapshot(repo_id, settings)
    if snap is not None:
        return snap
    root = _hub_repo_dir(resolve_hf_hub_dir(settings), repo_id)
    if (root / "config.json").is_file():
        return root.resolve()
    return None





def mineru_subprocess_environ(settings: Settings | None = None) -> dict[str, str]:
    """MinerU 子進程：模型目錄指向 ``models/``，但允許 Hub 解析本地快照（不強制離線）。"""
    apply_models_to_environ(settings, offline=False)
    # hf-mirror 等鏡像常返回不完整元數據（commit_hash 為空），導致 huggingface_hub 報
    # FileMetadataError；MinerU 內建 snapshot_download 須走官方 Hub API，檔案仍寫入 HF_HUB_CACHE。
    os.environ.pop("HF_ENDPOINT", None)
    os.environ.pop("HUGGINGFACE_HUB_URL", None)
    return os.environ.copy()
