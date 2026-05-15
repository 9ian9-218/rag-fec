"""Neo4j 連線與常用維運操作（級聯刪除建議優先使用 LightRAG ``adelete_by_doc_id``）。"""

from __future__ import annotations

from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, GraphDatabase

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger("storage.neo4j")


class Neo4jClient:
    """輕量 Neo4j 包裝：同步驅動用於腳本、非同步驅動可選。"""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        s = get_settings()
        self._uri = uri or s.neo4j.uri
        self._user = username or s.neo4j.username
        self._password = password or s.neo4j.password
        self._sync_driver: Any = None

    def sync_driver(self):
        """延遲建立同步 Driver。"""
        if self._sync_driver is None:
            self._sync_driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
        return self._sync_driver

    def close(self) -> None:
        if self._sync_driver is not None:
            self._sync_driver.close()
            self._sync_driver = None

    def run_write(self, cypher: str, **params: Any) -> None:
        """執行單筆寫入交易。"""
        drv = self.sync_driver()
        with drv.session(database=get_settings().neo4j.database) as session:
            session.execute_write(lambda tx: tx.run(cypher, **params))

    def detach_delete_all_nodes(self) -> None:
        """清空圖譜（維運用，慎用）。"""
        logger.warning("Executing DETACH DELETE on all nodes — destructive operation")
        self.run_write("MATCH (n) DETACH DELETE n")

    @staticmethod
    def async_driver() -> AsyncDriver:
        """建立非同步 Driver（呼叫端負責生命週期）。"""
        s = get_settings()
        return AsyncGraphDatabase.driver(
            s.neo4j.uri,
            auth=(s.neo4j.username, s.neo4j.password),
        )
