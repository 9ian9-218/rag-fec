"""增量處理斷點：JSON 檔案儲存已處理清單與游標。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("incremental.checkpoint_manager")


@dataclass
class IncrementalCheckpoint:
    """斷點資料結構。"""

    processed_paths: list[str] = field(default_factory=list)
    cursor: int = 0
    stats: dict[str, int] = field(
        default_factory=lambda: {"added": 0, "modified": 0, "removed": 0, "errors": 0}
    )
    updated_at: float = field(default_factory=lambda: time.time())


class CheckpointManager:
    """讀寫 ``incremental_checkpoint.json``。"""

    def __init__(self, path: str | None = None) -> None:
        s = get_settings()
        root = Path(s.paths.project_root).resolve()
        self._path = Path(path or s.incremental.checkpoint_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> IncrementalCheckpoint:
        if not self._path.is_file():
            return IncrementalCheckpoint()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return IncrementalCheckpoint(
                processed_paths=list(raw.get("processed_paths") or []),
                cursor=int(raw.get("cursor") or 0),
                stats=dict(raw.get("stats") or {}),
                updated_at=float(raw.get("updated_at") or time.time()),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("斷點檔損毀，將重置: %s", e)
            return IncrementalCheckpoint()

    def save(self, cp: IncrementalCheckpoint) -> None:
        cp.updated_at = time.time()
        self._path.write_text(
            json.dumps(asdict(cp), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        if self._path.is_file():
            self._path.unlink(missing_ok=True)
