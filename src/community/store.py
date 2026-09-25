"""社区摘要的轻量文件存储：JsonKV 风格 JSON + numpy 向量索引（原子写）。

存储位置与 LightRAG 的 KV 并列（``data/lightrag_workdir``），不引入 Neo4j/Milvus schema 变更：

- ``kv_store_community_reports.json``：``{"meta": {...}, "reports": {community_id: {...}}}``
- ``community_index.npz``：``ids`` (N,) + ``vectors`` (N, dim) —— 供查询期做余弦检索
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings, get_settings
from src.utils.logger import get_logger

logger = get_logger("community.store")

REPORTS_FILENAME = "kv_store_community_reports.json"
INDEX_FILENAME = "community_index.npz"
NO_VERSION = "none"


def workdir(settings: Settings | None = None) -> Path:
    """社区摘要文件所在目录（与 LightRAG JsonKV 同目录）。"""
    s = settings or get_settings()
    root = Path(s.paths.project_root).resolve()
    return root / s.paths.lightrag_working_dir


def reports_path(settings: Settings | None = None) -> Path:
    return workdir(settings) / REPORTS_FILENAME


def index_path(settings: Settings | None = None) -> Path:
    return workdir(settings) / INDEX_FILENAME


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        except OSError:
            logger.warning("社区摘要备份失败（继续写入）: %s", path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def load_reports(settings: Settings | None = None) -> dict[str, Any] | None:
    """读取社区摘要文件；缺失或损坏返回 None（调用方须降级）。"""
    path = reports_path(settings)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("社区摘要文件不可读，按缺失处理: %s", e)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("reports"), dict):
        logger.warning("社区摘要文件结构异常，按缺失处理: %s", path)
        return None
    data.setdefault("meta", {})
    return data


def save_reports(meta: dict[str, Any], reports: dict[str, Any], settings: Settings | None = None) -> Path:
    payload = {"meta": meta, "reports": reports}
    return _atomic_write_text(
        reports_path(settings),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def load_index(settings: Settings | None = None) -> tuple[list[str], np.ndarray] | None:
    """读取社区向量索引；缺失、损坏或维度异常返回 None。"""
    path = index_path(settings)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            ids = [str(x) for x in data["ids"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as e:
        logger.warning("社区向量索引不可读，按缺失处理: %s", e)
        return None
    if not ids or vectors.ndim != 2 or vectors.shape[0] != len(ids):
        logger.warning("社区向量索引形状异常，按缺失处理: %s", path)
        return None
    return ids, vectors


def save_index(ids: list[str], vectors: np.ndarray, settings: Settings | None = None) -> Path:
    import io

    buf = io.BytesIO()
    # 注意：必须用 unicode 数组而不是 object dtype——object 数组读取时需要
    # allow_pickle=True，与 load_index 的安全加载策略冲突。
    np.savez(
        buf,
        ids=np.asarray([str(i) for i in ids], dtype="U"),
        vectors=np.asarray(vectors, dtype=np.float32),
    )
    return _atomic_write_bytes(index_path(settings), buf.getvalue())


_version_cache: tuple[tuple[int, int], str] | None = None


def community_version(settings: Settings | None = None) -> str:
    """检索缓存用的社区版本号：图谱指纹 + 构建时间；缺失返回 ``none``。

    以文件 mtime/size 做一层缓存，避免每次查询都重新解析 JSON。
    """
    global _version_cache
    path = reports_path(settings)
    try:
        stat = path.stat()
    except OSError:
        return NO_VERSION
    key = (stat.st_mtime_ns, stat.st_size)
    if _version_cache is not None and _version_cache[0] == key:
        return _version_cache[1]
    data = load_reports(settings)
    meta = (data or {}).get("meta") or {}
    fingerprint = str(meta.get("graph_fingerprint") or "").strip()
    built_at = str(meta.get("built_at") or "").strip()
    value = f"{fingerprint[:12]}:{built_at or '0'}" if fingerprint else NO_VERSION
    _version_cache = (key, value)
    return value


def cache_extra(use_community: bool | None) -> str:
    """缓存 key 的社区维度：开关有效值 + 索引版本（供检索/回答缓存复用）。"""
    flag = "auto" if use_community is None else ("on" if use_community else "off")
    return f"community={flag}|v={community_version()}"


def reset_version_cache() -> None:
    """测试辅助：清空版本缓存。"""
    global _version_cache
    _version_cache = None


def is_usable(settings: Settings | None = None) -> tuple[bool, str]:
    """索引是否可用于注入：返回 ``(可用, 原因)``。"""
    data = load_reports(settings)
    if not data:
        return False, "index_missing"
    meta = data.get("meta") or {}
    if bool(meta.get("stale")):
        return False, "index_stale"
    reports = data.get("reports") or {}
    if not any(str(r.get("summary") or "").strip() for r in reports.values() if isinstance(r, dict)):
        return False, "index_missing"
    if load_index(settings) is None:
        return False, "index_missing"
    return True, "ok"
