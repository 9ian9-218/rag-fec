"""專案入口：啟動 FastAPI（uvicorn）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from src.utils.logger import setup_logging


def main() -> None:
    setup_logging()
    s = get_settings()
    import uvicorn

    uvicorn.run(
        "src.service.api:app",
        host=s.service.host,
        port=s.service.port,
        reload=s.service.debug,
    )


if __name__ == "__main__":
    main()
