"""穩定文件 ID：相同絕對路徑對應相同 LightRAG doc id。"""

from __future__ import annotations

from pathlib import Path

from lightrag.utils import compute_mdhash_id


def stable_doc_id(path: Path) -> str:
    """依正規化路徑產生 ``doc-`` 前綴的穩定 id。"""
    key = str(path.expanduser().resolve())
    return compute_mdhash_id(key, prefix="doc-")
