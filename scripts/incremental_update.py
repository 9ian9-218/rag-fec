"""觸發增量更新（命令列）。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.incremental.update_manager import UpdateManager
from src.utils.logger import setup_logging


async def _main() -> None:
    setup_logging()
    mgr = UpdateManager()
    print(await mgr.run_incremental())


if __name__ == "__main__":
    asyncio.run(_main())
