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
    rel = (s.paths.models_dir or "models").strip() or "models"
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





DEFAULT_RERANKER_REPO_ID = "BAAI/bge-reranker-v2-m3"


def resolve_local_reranker_dir(settings: Settings | None = None) -> Path | None:
    """本地 CrossEncoder 重排權重目錄；僅查磁盤，不觸發下載。

    返回 ``None`` 表示本機沒有權重 —— 呼叫方據此避免在查詢路徑中觸發數 GB 的
    Hub 下載，也避免「設定看起來有重排、實際沒跑」的假能力狀態。
    """
    s = settings or get_settings()
    # 若配置中指定了本地路徑則優先使用
    custom_path = (getattr(s.models, "reranker_local_path", None) or "").strip()
    if custom_path:
        p = Path(custom_path).expanduser()
        if p.is_dir() or p.is_file():
            return p.resolve()
    return resolve_hub_model_dir(DEFAULT_RERANKER_REPO_ID, settings=s)


def resolve_reranker_model_load_path(settings: Settings | None = None) -> str:
    """解析本地 CrossEncoder 重排模型加載路徑。

    優先檢查本地 HF Hub 快取；若無則回傳預設 HuggingFace 模型 ID，
    由 sentence-transformers 自動從 Hub 下載（呼叫方須自行確認允許下載）。
    """
    local = resolve_local_reranker_dir(settings)
    return str(local) if local is not None else DEFAULT_RERANKER_REPO_ID


def rerank_backend_available(settings: Settings | None = None) -> bool:
    """是否存在**可實際生效**的 rerank 後端：線上 API 配置完整，或本地已有權重。

    只用於「是否開啟 rerank」的判斷；與「sentence-transformers 是否安裝」無關
    （已安裝但無權重時，LightRAG 只會 warning 後原序返回，等於沒重排）。
    """
    s = settings or get_settings()
    m = s.models
    online_ready = bool(
        m.rerank_api_enabled
        and (m.rerank_api_key or "").strip()
        and (m.rerank_api_base_url or "").strip()
        and (m.rerank_api_model_name or "").strip()
    )
    return online_ready or resolve_local_reranker_dir(s) is not None


def mineru_subprocess_environ(settings: Settings | None = None) -> dict[str, str]:
    """MinerU 子進程：模型目錄指向 ``models/``，但允許 Hub 解析本地快照（不強制離線）。"""
    # 原实现调用全仓不存在的 apply_models_to_environ()，每次必抛 NameError 并被
    # scripts/convert.py 静默吞掉；这里按等价语义显式实现，与 convert.py 的
    # hf_project_cache_env() 回退保持一致。
    s = settings or get_settings()
    models_root = resolve_models_root(s)
    hub_root = models_root / "hub"
    models_root.mkdir(parents=True, exist_ok=True)
    hub_root.mkdir(parents=True, exist_ok=True)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HF_HOME"] = str(models_root)
    os.environ["HF_HUB_CACHE"] = str(hub_root)
    # hf-mirror 等鏡像常返回不完整元數據（commit_hash 為空），導致 huggingface_hub 報
    # FileMetadataError；MinerU 內建 snapshot_download 須走官方 Hub API，檔案仍寫入 HF_HUB_CACHE。
    os.environ.pop("HF_ENDPOINT", None)
    os.environ.pop("HUGGINGFACE_HUB_URL", None)
    return os.environ.copy()
