"""HTTP 客户端复用工具：避免高并发下每次请求都新建 AsyncOpenAI / httpx.AsyncClient。"""

from __future__ import annotations

import threading
from typing import Any

import httpx
from openai import AsyncOpenAI

_openai_clients: dict[tuple[str, str, float | None], AsyncOpenAI] = {}
_openai_lock = threading.Lock()

_httpx_clients: dict[tuple[float | None], httpx.AsyncClient] = {}
_httpx_lock = threading.Lock()


def get_openai_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float | None = None,
) -> AsyncOpenAI:
    """返回可复用的 AsyncOpenAI 客户端（按 api_key/base_url/timeout 缓存）。"""
    key = (api_key, base_url, timeout)
    with _openai_lock:
        client = _openai_clients.get(key)
        if client is None:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            _openai_clients[key] = client
        return client


def get_httpx_client(*, timeout: float | None = None) -> httpx.AsyncClient:
    """返回可复用的 httpx.AsyncClient（按 timeout 缓存）。"""
    key = (timeout,)
    with _httpx_lock:
        client = _httpx_clients.get(key)
        if client is None:
            client = httpx.AsyncClient(timeout=timeout)
            _httpx_clients[key] = client
        return client


async def aclose_clients() -> None:
    """关闭所有缓存的客户端（FastAPI 关闭时调用）。"""
    with _openai_lock:
        clients = list(_openai_clients.values())
        _openai_clients.clear()
    for c in clients:
        try:
            await c.close()
        except Exception:
            pass

    with _httpx_lock:
        h_clients = list(_httpx_clients.values())
        _httpx_clients.clear()
    for c in h_clients:
        try:
            await c.aclose()
        except Exception:
            pass
