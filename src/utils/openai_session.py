"""OpenCode 網關要求的會話頭（``x-opencode-session``）統一管理。

OpenCode Go / Zen 端點要求每個請求帶穩定的會話 ID，否則返回 400 MissingSessionID。
會話 ID 可用 ``OPENAI_SESSION_ID`` 覆寫；未設置時按進程生成一次並全程復用。
"""

from __future__ import annotations

import os
from uuid import uuid4

_session_id: str | None = None


def session_id() -> str:
    """返回當前進程使用的會話 ID（穩定、可配置）。"""
    global _session_id
    if _session_id is None:
        _session_id = (os.getenv("OPENAI_SESSION_ID") or "").strip() or f"rag-fec-{uuid4().hex[:12]}"
    return _session_id


def session_headers() -> dict[str, str]:
    """返回需要注入的請求頭。"""
    return {"x-opencode-session": session_id()}
