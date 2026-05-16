"""LightRAG 建立 MilvusClient 時未传 ``timeout``，pymilvus 会走 gRPC 默认约 10s，standalone 冷启动易超时。

在导入 ``lightrag`` 之前调用 ``ensure_pymilvus_connection_timeout()``，当调用方未显式传入
``timeout`` 时注入 ``MILVUS_CLIENT_TIMEOUT``（由 ``apply_settings_to_environ`` 从配置写入）。
"""

from __future__ import annotations

import os
from typing import Any

_PATCHED = False


def _default_timeout_seconds() -> float:
    raw = (os.environ.get("MILVUS_CLIENT_TIMEOUT") or "").strip()
    if not raw:
        return 60.0
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 60.0


def ensure_pymilvus_connection_timeout() -> None:
    global _PATCHED
    if _PATCHED:
        return
    timeout_sec = _default_timeout_seconds()
    from pymilvus.milvus_client import milvus_client as mc

    _orig = mc.MilvusClient.__init__

    def _init_with_default_timeout(
        self: Any,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        db_name: str = "",
        token: str = "",
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        if timeout is None:
            timeout = timeout_sec
        _orig(self, uri, user, password, db_name, token, timeout=timeout, **kwargs)

    mc.MilvusClient.__init__ = _init_with_default_timeout  # type: ignore[method-assign]
    _PATCHED = True
