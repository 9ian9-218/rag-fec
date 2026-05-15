"""專案內統一日誌初始化。"""

from __future__ import annotations

import logging
import logging.config
import logging.handlers
from pathlib import Path

from config.settings import get_settings


def setup_logging() -> None:
    """載入 logging.conf（僅終端機），若 ``data/logs`` 可寫則附加輪替檔案日誌。"""
    s = get_settings()
    root = Path(s.paths.project_root).resolve()
    conf_path = root / s.paths.logging_conf
    if conf_path.is_file():
        logging.config.fileConfig(
            conf_path,
            disable_existing_loggers=False,
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    log_dir = root / "data" / "logs"
    log_file = log_dir / "app.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8"):
            pass
        fh = logging.handlers.RotatingFileHandler(
            str(log_file),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        g = logging.getLogger("graph_rag_project")
        if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in g.handlers):
            g.addHandler(fh)
    except OSError as e:
        logging.getLogger("graph_rag_project.utils").warning(
            "無法寫入檔案日誌 %s（例如目錄屬主為 root），僅使用終端機輸出: %s",
            log_file,
            e,
        )

    logging.getLogger("neo4j").setLevel(logging.ERROR)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """取得命名 logger（建議使用 graph_rag_project.* 前綴）。"""
    return logging.getLogger(f"graph_rag_project.{name}")
