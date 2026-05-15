"""基於 MD5 的文件變更偵測與 hash_cache.json 維護。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config.settings import get_settings
from src.utils.hash_utils import md5_file
from src.utils.logger import get_logger

logger = get_logger("data_processing.change_detector")


@dataclass
class ChangeReport:
    """目錄掃描後的變更摘要。"""

    added: list[Path]
    modified: list[Path]
    removed: list[str]
    unchanged: list[str]


def _read_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        logger.warning("hash_cache 損毀，將重建: %s", path)
    return {}


def _write_cache(path: Path, mapping: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_changes(
    scan_roots: Iterable[Path],
    *,
    recursive: bool = True,
    suffixes: set[str] | None = None,
) -> ChangeReport:
    """比對目前檔案與快取，回傳新增 / 修改 / 刪除清單。"""
    s = get_settings()
    cache_path = Path(s.paths.project_root).resolve() / s.incremental.hash_cache_path
    old = _read_cache(cache_path)

    suf = suffixes or {".pdf", ".md", ".markdown", ".docx", ".txt"}
    current: dict[str, str] = {}

    for root in scan_roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        it = root.rglob("*") if recursive else root.glob("*")
        for p in it:
            if not p.is_file():
                continue
            if p.suffix.lower() not in suf:
                continue
            key = str(p.resolve())
            try:
                current[key] = md5_file(p)
            except OSError as e:
                logger.warning("無法雜湊檔案 %s: %s", p, e)

    added: list[Path] = []
    modified: list[Path] = []
    removed: list[str] = []

    for path_str, h in current.items():
        if path_str not in old:
            added.append(Path(path_str))
        elif old[path_str] != h:
            modified.append(Path(path_str))

    for path_str in old:
        if path_str not in current:
            removed.append(path_str)

    unchanged = [k for k in current if k in old and old[k] == current[k]]
    return ChangeReport(
        added=sorted(added, key=lambda x: str(x)),
        modified=sorted(modified, key=lambda x: str(x)),
        removed=sorted(removed),
        unchanged=unchanged,
    )


def write_hash_cache(mapping: dict[str, str]) -> None:
    """寫入完整路徑 -> MD5 對照表。"""
    s = get_settings()
    cache_path = Path(s.paths.project_root).resolve() / s.incremental.hash_cache_path
    _write_cache(cache_path, mapping)


def rebuild_hash_cache_for_directory(
    directory: Path,
    *,
    recursive: bool = True,
    suffixes: set[str] | None = None,
) -> dict[str, str]:
    """掃描目錄並覆寫 hash_cache（全量重建）。"""
    suf = suffixes or {".pdf", ".md", ".markdown", ".docx", ".txt"}
    directory = directory.expanduser().resolve()
    mapping: dict[str, str] = {}
    if not directory.is_dir():
        write_hash_cache({})
        return {}
    it = directory.rglob("*") if recursive else directory.glob("*")
    for p in it:
        if not p.is_file() or p.suffix.lower() not in suf:
            continue
        try:
            mapping[str(p.resolve())] = md5_file(p)
        except OSError as e:
            logger.warning("略過 %s: %s", p, e)
    write_hash_cache(mapping)
    return mapping


def load_hash_cache() -> dict[str, str]:
    s = get_settings()
    cache_path = Path(s.paths.project_root).resolve() / s.incremental.hash_cache_path
    return _read_cache(cache_path)


def merge_cache_after_success(old: dict[str, str], report: ChangeReport) -> dict[str, str]:
    """處理完成後合併快取：更新新增與修改、移除刪除。"""
    new_map = dict(old)
    for p in report.added + report.modified:
        key = str(p.resolve())
        try:
            new_map[key] = md5_file(p)
        except OSError:
            logger.warning("合併快取時略過 %s", p)
    for path_str in report.removed:
        new_map.pop(path_str, None)
    return new_map
