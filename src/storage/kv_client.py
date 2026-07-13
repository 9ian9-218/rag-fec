"""應用層 SQLite KV：文件註冊、斷點與任意鍵值。"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("storage.kv")


class KVClient:
    """執行緒安全的 SQLite 存取。"""

    def __init__(self, db_path: str | None = None) -> None:
        s = get_settings()
        root = Path(s.paths.project_root).resolve()
        self._path = Path(db_path or s.paths.sqlite_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_kv (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    content_hash TEXT,
                    size_bytes INTEGER,
                    updated_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(source_path);
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self._path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def get(self, key: str, default: str | None = None) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT v FROM app_kv WHERE k = ?", (key,)).fetchone()
            if row is None:
                return default
            return str(row["v"])

    def set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO app_kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (key, value),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM app_kv WHERE k = ?", (key,))
            conn.commit()

    def get_json(self, key: str, default: Any) -> Any:
        raw = self.get(key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON for key %s", key)
            return default

    def set_json(self, key: str, obj: Any) -> None:
        self.set(key, json.dumps(obj, ensure_ascii=False))

    def upsert_document(
        self,
        doc_id: str,
        source_path: str,
        content_hash: str | None,
        size_bytes: int | None,
    ) -> None:
        import time

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(doc_id, source_path, content_hash, size_bytes, updated_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path=excluded.source_path,
                    content_hash=excluded.content_hash,
                    size_bytes=excluded.size_bytes,
                    updated_at=excluded.updated_at
                """,
                (doc_id, source_path, content_hash, size_bytes, time.time()),
            )
            conn.commit()

    def delete_document_row(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()

    def _row_with_file_name(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["file_name"] = Path(str(d.get("source_path") or "")).name or d.get("doc_id")
        return d

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, source_path, content_hash, size_bytes, updated_at FROM documents"
            ).fetchall()
            return [self._row_with_file_name(r) for r in rows]

    def get_doc_by_path(self, source_path: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id, source_path, content_hash, size_bytes, updated_at FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            return self._row_with_file_name(row) if row else None

    def get_doc_by_id(self, doc_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id, source_path, content_hash, size_bytes, updated_at FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            return self._row_with_file_name(row) if row else None
