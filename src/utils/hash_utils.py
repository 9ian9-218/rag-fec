"""檔案與位元組內容的 MD5 等雜湊工具。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def md5_bytes(data: bytes) -> str:
    """計算位元組串的 MD5 十六進位字串。"""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """以串流方式計算檔案 MD5，避免大檔一次載入記憶體。"""
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def md5_text(text: str, encoding: str = "utf-8") -> str:
    """計算文字內容 MD5。"""
    return md5_bytes(text.encode(encoding))
