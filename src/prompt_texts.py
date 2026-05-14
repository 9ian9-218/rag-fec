"""从仓库根目录 ``prompts/`` 加载提示词模板（UTF-8 文本）。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROMPTS_DIR = _REPO_ROOT / "prompts"


@lru_cache(maxsize=64)
def _read_raw(rel: str) -> str:
    path = _PROMPTS_DIR / rel
    if not path.is_file():
        raise FileNotFoundError(f"缺少提示词文件: {path}")
    return path.read_text(encoding="utf-8")


def load_prompt(rel: str, /, **kwargs: str | int) -> str:
    raw = _read_raw(rel)
    if not kwargs:
        return raw
    return raw.format(**kwargs)
