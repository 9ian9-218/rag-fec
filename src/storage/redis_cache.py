"""Redis 缓存工具：查询结果、路由、Embedding 等共用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.storage.redis_client import get_redis_client

DEFAULT_TTL_SECONDS = 300


def get_index_version() -> str:
    """以文档 manifest 的 mtime 作为索引版本，文档变化后缓存自然失效。"""
    from config.settings import get_settings

    s = get_settings()
    root = Path(s.paths.project_root).resolve()
    manifest = root / s.paths.document_manifest_path
    try:
        if manifest.is_file():
            return str(manifest.stat().st_mtime_ns)
    except OSError:
        pass
    return "default"


def build_query_cache_key(
    *,
    question: str,
    mode: str | None,
    top_k: int,
    multimodal: bool,
    use_llm_router: bool,
) -> str:
    """构造查询结果缓存 key。"""
    import hashlib

    parts = "|".join(
        [
            question.strip(),
            mode or "auto",
            str(top_k),
            str(multimodal),
            str(use_llm_router),
            get_index_version(),
        ]
    )
    digest = hashlib.sha1(parts.encode("utf-8")).hexdigest()
    return f"rag:query:{digest}"


async def get_cache(key: str) -> str | None:
    client = get_redis_client()
    if client is None:
        return None
    try:
        value = await client.get(key)
        return str(value) if value is not None else None
    except Exception:
        return None


async def set_cache(key: str, value: str, ttl: int = DEFAULT_TTL_SECONDS) -> bool:
    client = get_redis_client()
    if client is None:
        return False
    try:
        await client.set(key, value, ex=ttl)
        return True
    except Exception:
        return False


async def get_json_cache(key: str) -> Any | None:
    raw = await get_cache(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def set_json_cache(key: str, obj: Any, ttl: int = DEFAULT_TTL_SECONDS) -> bool:
    return await set_cache(key, json.dumps(obj, ensure_ascii=False), ttl=ttl)


async def delete_cache(key: str) -> bool:
    client = get_redis_client()
    if client is None:
        return False
    try:
        await client.delete(key)
        return True
    except Exception:
        return False
