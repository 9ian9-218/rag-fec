"""從 Markdown chunk 中抽取圖片引用並解析為本地路徑。"""

from __future__ import annotations

import re
from pathlib import Path

# ![](images/foo.jpg) 或 ![alt](images/foo.jpg)
_MD_IMG = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
_HTML_IMG = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.I)


def extract_image_refs(text: str) -> list[str]:
    """回傳原始路徑字串（可為相對路徑 ``images/...`` 或 URL）。"""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for pat in (_MD_IMG, _HTML_IMG):
        for m in pat.finditer(text):
            u = (m.group(1) or "").strip()
            if not u or u.startswith(("http://", "https://", "data:")):
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def resolve_local_image_path(ref: str, chunk_file_path: str | None, project_root: Path) -> Path | None:
    """
    將 ``images/xxx`` 或絕對路徑解析為 ``Path``；若無法對應本地檔則回傳 ``None``。
    ``chunk_file_path`` 為入庫時傳給 LightRAG 的 Markdown 路徑（其同目錄或上級含 ``images/``）。
    """
    ref = ref.strip()
    if not ref or ref.startswith(("http://", "https://", "data:")):
        return None
    p = Path(ref)
    if p.is_absolute():
        return p if p.is_file() else None
    base: Path | None = None
    if chunk_file_path and str(chunk_file_path).strip():
        fp = Path(chunk_file_path).expanduser()
        if not fp.is_absolute():
            fp = (project_root / fp).resolve()
        else:
            fp = fp.resolve()
        base = fp.parent
    if base is None:
        cand = (project_root / ref).resolve()
        return cand if cand.is_file() else None
    cand = (base / ref).resolve()
    if cand.is_file():
        return cand
    if (base / "images" / Path(ref).name).is_file():
        return (base / "images" / Path(ref).name).resolve()
    return None
