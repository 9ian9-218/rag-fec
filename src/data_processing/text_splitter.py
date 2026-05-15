"""語意分塊：章節 / 標題優先，其次遞迴字元切分。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.data_processing.preprocessor import preprocess
from src.utils.logger import get_logger

logger = get_logger("data_processing.text_splitter")


@dataclass
class TextChunk:
    """單一分塊。"""

    chunk_id: str
    content: str
    order_index: int
    level_path: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


def semantic_split_markdown(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Markdown：先依標題切，再對過長區塊做遞迴切分。"""
    text = preprocess(text)
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    try:
        parent_docs: list[Document] = md_splitter.split_text(text)
    except Exception:
        parent_docs = []
    if not parent_docs:
        parent_docs = [Document(page_content=text, metadata={})]

    rc = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    chunks: list[TextChunk] = []
    order = 0
    for doc in parent_docs:
        content = doc.page_content
        meta = doc.metadata or {}
        level_keys = tuple(f"{k}:{v}" for k, v in sorted(meta.items()))
        for piece in rc.split_text(content):
            cid = f"chk-{uuid.uuid4().hex}"
            chunks.append(
                TextChunk(
                    chunk_id=cid,
                    content=piece,
                    order_index=order,
                    level_path=level_keys,
                    metadata={"structure": meta},
                )
            )
            order += 1
    return chunks


def semantic_split_plain(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """純文字：雙換行優先，其次遞迴切分。"""
    text = preprocess(text)
    rc = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    return [
        TextChunk(
            chunk_id=f"chk-{uuid.uuid4().hex}",
            content=c,
            order_index=i,
            level_path=(),
            metadata={},
        )
        for i, c in enumerate(rc.split_text(text))
    ]
