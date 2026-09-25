"""將 ``BAAI/bge-reranker-v2-m3`` 下載到專案 ``models/hub/``（與嵌入模型一致）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from config.model_paths import DEFAULT_RERANKER_REPO_ID, resolve_hf_hub_dir
from config.settings import apply_settings_to_environ, get_settings
from src.utils.logger import get_logger, setup_logging

logger = get_logger("scripts.download_reranker")


def main() -> int:
    p = argparse.ArgumentParser(description="Download BGE reranker into models/hub")
    p.add_argument(
        "--repo",
        default=None,
        help=f"HuggingFace repo id（預設 {DEFAULT_RERANKER_REPO_ID}）",
    )
    args = p.parse_args()

    setup_logging()
    s = get_settings()
    apply_settings_to_environ(s)
    repo = (args.repo or DEFAULT_RERANKER_REPO_ID).strip()
    hub = resolve_hf_hub_dir(s)
    logger.info("Downloading %s -> HF_HUB_CACHE=%s", repo, hub)
    # 必须显式传 cache_dir：apply_settings_to_environ() 在 huggingface_hub 已 import 之后
    # 才设环境变量，HF_HUB_CACHE 不生效，权重会落到 ~/.cache/huggingface/hub，
    # 导致 resolve_local_reranker_dir() 永远找不到权重。
    snapshot_download(repo_id=repo, local_files_only=False, cache_dir=str(hub))
    logger.info("Done. 快照位於 models/hub/models--%s；本地重排將自動啟用。", repo.replace("/", "--"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
