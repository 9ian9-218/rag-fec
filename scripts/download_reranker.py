"""將 ``BAAI/bge-reranker-v2-m3`` 下載到專案 ``models/hub/``（與嵌入模型一致）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from config.model_paths import resolve_hf_hub_dir
from config.settings import apply_settings_to_environ, get_settings
from src.utils.logger import get_logger, setup_logging

logger = get_logger("scripts.download_reranker")


def main() -> int:
    p = argparse.ArgumentParser(description="Download BGE reranker into models/hub")
    p.add_argument(
        "--repo",
        default=None,
        help="HuggingFace repo id（預設取設定 MODELS_RERANK_MODEL_NAME / BAAI/bge-reranker-v2-m3）",
    )
    args = p.parse_args()

    setup_logging()
    s = get_settings()
    apply_settings_to_environ(s, offline=False)
    repo = (args.repo or s.models.rerank_model_name or "BAAI/bge-reranker-v2-m3").strip()
    hub = resolve_hf_hub_dir(s)
    logger.info("Downloading %s -> HF_HUB_CACHE=%s", repo, hub)
    snapshot_download(repo_id=repo, local_files_only=False)
    logger.info("Done. 啟用離線時請確保 MODELS_OFFLINE=true 且快照已完整。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
