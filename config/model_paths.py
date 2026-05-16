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


def resolve_reranker_model_load_path(settings: Settings | None = None) -> str:
    """
    Reranker（CrossEncoder）載入路徑：優先 ``MODELS_RERANK_LOCAL_PATH``，其次 Hub 快照，最後為模型 id。
    """
    s = settings or get_settings()
    override = (s.models.rerank_local_path or "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = resolve_project_root(s) / p
        if p.is_dir():
            return str(p.resolve())
    hub_dir = resolve_hub_model_dir(s.models.rerank_model_name, s)
    if hub_dir is not None:
        return str(hub_dir)
    return s.models.rerank_model_name


def resolve_embedding_model_load_path(settings: Settings | None = None) -> str:
    """
    嵌入模型載入路徑：優先本地 Hub 快照，否則回傳 Hub 模型 id（如 ``BAAI/bge-m3``）。
    """
    s = settings or get_settings()
    override = (s.models.embedding_local_path or "").strip()
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = resolve_project_root(s) / p
        if p.is_dir():
            return str(p.resolve())
    snap = resolve_hub_snapshot(s.embedding.model_name, s)
    if snap is not None:
        return str(snap)
    return s.embedding.model_name


def apply_models_to_environ(settings: Settings | None = None, *, offline: bool | None = None) -> None:
    """啟動前注入 HF 路徑；``offline`` 預設取自 ``MODELS_OFFLINE``（僅建議用於嵌入等已完整快照的組件）。"""
    s = settings or get_settings()
    models_root = resolve_models_root(s)
    hub = resolve_hf_hub_dir(s)
    os.environ["HF_HOME"] = str(models_root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    if s.models.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", s.models.hf_endpoint.strip())
    use_offline = s.models.offline if offline is None else offline
    if use_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)


def mineru_subprocess_environ(settings: Settings | None = None) -> dict[str, str]:
    """MinerU 子進程：模型目錄指向 ``models/``，但允許 Hub 解析本地快照（不強制離線）。"""
    apply_models_to_environ(settings, offline=False)
    # hf-mirror 等鏡像常返回不完整元數據（commit_hash 為空），導致 huggingface_hub 報
    # FileMetadataError；MinerU 內建 snapshot_download 須走官方 Hub API，檔案仍寫入 HF_HUB_CACHE。
    os.environ.pop("HF_ENDPOINT", None)
    os.environ.pop("HUGGINGFACE_HUB_URL", None)
    return os.environ.copy()
