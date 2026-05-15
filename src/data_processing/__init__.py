"""資料處理子套件。"""

from src.data_processing.change_detector import (
    ChangeReport,
    detect_changes,
    load_hash_cache,
    merge_cache_after_success,
    rebuild_hash_cache_for_directory,
    write_hash_cache,
)
from src.data_processing.document_loader import LoadedDocument, iter_documents, load_document
from src.data_processing.preprocessor import preprocess
from src.data_processing.text_splitter import TextChunk, semantic_split_markdown, semantic_split_plain

__all__ = [
    "ChangeReport",
    "LoadedDocument",
    "TextChunk",
    "detect_changes",
    "iter_documents",
    "load_document",
    "load_hash_cache",
    "merge_cache_after_success",
    "preprocess",
    "rebuild_hash_cache_for_directory",
    "semantic_split_markdown",
    "semantic_split_plain",
    "write_hash_cache",
]
