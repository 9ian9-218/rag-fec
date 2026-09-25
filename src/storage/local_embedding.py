"""本地 GPU embedding（sentence-transformers + BAAI/bge-m3）。

用途：在没有外网额度时，用本机显卡跑嵌入。返回对象与
``remote_embedding.build_remote_embedding_func`` 一致（LightRAG ``EmbeddingFunc``），
因此上层检索/索引逻辑无需改动。

显存参考（GTX 1660 Ti 6GB 实测，bge-m3）：
- float32 + batch 8（约 1200 token/条）峰值约 2.5GB，batch 32 约 3.4GB
- 该卡（TU116，无 Tensor Core）float32 比 float16 快约 4 倍，故默认 float32
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import Settings, get_settings
from src.utils.concurrency import get_semaphore
from src.utils.logger import get_logger

logger = get_logger("storage.local_embedding")

_SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")

_model: Any = None
_model_key: tuple[str, str, str, int] | None = None
_load_lock = threading.Lock()


def resolve_model_path(s: Settings) -> Path:
    """把配置里的本地模型路径解析为绝对路径（相对路径以项目根目录为基准）。"""
    raw = (s.embedding.local_model_path or "").strip()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path(s.paths.project_root).resolve() / p
    return p


def _check_device_dtype(device: str, dtype_name: str) -> None:
    """提前拦住这块卡不支持的组合，避免运行到一半才炸。"""
    if not device.startswith("cuda"):
        return
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("EMBEDDING_LOCAL_DEVICE=cuda 但当前环境 torch.cuda.is_available() 为 False")
        if dtype_name == "bfloat16" and torch.cuda.get_device_capability(0)[0] < 8:
            raise RuntimeError(
                "bfloat16 需要算力 8.0 及以上的显卡（Ampere+），当前显卡不支持，请改用 float32 或 float16"
            )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("本地 embedding 需要安装 torch") from exc


def _load_model(s: Settings) -> Any:
    """加载（并缓存）本地 SentenceTransformer 模型。"""
    global _model, _model_key

    path = resolve_model_path(s)
    device = (s.embedding.local_device or "cuda").strip()
    dtype_name = (s.embedding.local_dtype or "float32").strip().lower()
    max_seq = int(s.embedding.local_max_seq_length)

    if dtype_name not in _SUPPORTED_DTYPES:
        raise ValueError(f"EMBEDDING_LOCAL_DTYPE 仅支持 {_SUPPORTED_DTYPES}，当前为 {dtype_name}")

    key = (str(path), device, dtype_name, max_seq)
    with _load_lock:
        if _model is not None and _model_key == key:
            return _model
        if not path.is_dir():
            raise FileNotFoundError(
                f"本地 embedding 模型目录不存在: {path}。请先下载模型，例如：\n"
                f"  huggingface-cli download BAAI/bge-m3 --local-dir {path}"
            )
        _check_device_dtype(device, dtype_name)

        import torch
        from sentence_transformers import SentenceTransformer

        logger.info(
            "加载本地 embedding 模型: path=%s device=%s dtype=%s max_seq=%d",
            path,
            device,
            dtype_name,
            max_seq,
        )
        model = SentenceTransformer(
            str(path),
            device=device,
            model_kwargs={"dtype": getattr(torch, dtype_name)},
        )
        model.max_seq_length = max_seq
        model.eval()
        _model = model
        _model_key = key
        return model


def _encode(texts: list[str], s: Settings) -> np.ndarray:
    """同步编码（由线程池调用，避免阻塞事件循环）。"""
    import torch

    model = _load_model(s)
    with torch.inference_mode():
        vectors = model.encode(
            texts,
            batch_size=int(s.embedding.local_batch_size),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def build_local_embedding_func(settings: Settings | None = None):
    """建立基于本地模型的 embedding 函数。

    返回的函數簽名：``async def _embed(texts: list[str]) -> np.ndarray``（已归一化，float32）。
    """
    s = settings or get_settings()
    dimension = int(s.embedding.dimension)
    max_tokens = int(s.embedding.local_max_seq_length)
    concurrency = int(s.embedding.local_max_concurrency)

    # 启动即加载：模型缺失 / 精度不支持等问题在启动阶段就暴露，而不是跑到一半才失败
    _load_model(s)

    async def _embed(texts: list[str], **_kwargs: Any) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, dimension)
        async with get_semaphore("embedding:local", concurrency):
            arr = await asyncio.to_thread(_encode, list(texts), s)
        if arr.shape[1] != dimension:
            raise ValueError(
                f"本地 embedding 输出维度 {arr.shape[1]} 与 EMBEDDING_DIMENSION={dimension} 不一致"
            )
        return arr

    from lightrag.utils import EmbeddingFunc

    # 集合名由 model_name 决定：沿用线上用的同名模型（如 BAAI/bge-m3），
    # 这样 Milvus 集合仍是 entities_baai_bge_m3_1024d，已入库的向量继续可用。
    model_name = (s.embedding.api_model_name or "").strip() or f"local:{resolve_model_path(s).name}"
    return EmbeddingFunc(
        embedding_dim=dimension,
        max_token_size=max_tokens,
        func=_embed,
        model_name=model_name,
    )


def reset_local_embedding_singleton() -> None:
    """释放已加载的本地模型（测试或切换模型时使用）。"""
    global _model, _model_key
    with _load_lock:
        _model = None
        _model_key = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
