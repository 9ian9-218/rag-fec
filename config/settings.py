"""使用 pydantic-settings 管理全域設定，支援 .env 與環境變數覆寫。"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Neo4jSettings(BaseSettings):
    """Neo4j 連線設定（Docker 內建議主機名為 neo4j）。"""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", env_file=".env", extra="ignore")

    uri: str = Field(
        default="bolt://127.0.0.1:7687",
        description="Bolt URI；本機腳本請用 127.0.0.1，Compose 內 rag 服務請用環境覆寫為 neo4j",
    )
    username: str = Field(default="neo4j")
    password: str = Field(default="changeme")
    database: str = Field(default="neo4j")


class MilvusSettings(BaseSettings):
    """Milvus 連線設定（Docker 內建議主機名為 milvus）。"""

    model_config = SettingsConfigDict(env_prefix="MILVUS_", env_file=".env", extra="ignore")

    uri: str = Field(
        default="http://127.0.0.1:19530",
        description="MilvusClient URI；本機腳本請用 127.0.0.1，Compose 內 rag 服務請覆寫為 http://milvus:19530",
    )
    db_name: str = Field(default="default")
    collection_name: str = Field(
        default="lightrag_chunks",
        description="業務展示用集合名稱；LightRAG 會依 namespace 自行建 collection",
    )


class LLMSettings(BaseSettings):
    """OpenAI 相容 LLM 設定。"""

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )

    api_key: str = Field(default="")
    base_url: str | None = Field(default=None, description="例如 http://host.docker.internal:11434/v1")
    model_name: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class EmbeddingSettings(BaseSettings):
    """嵌入模型：本機 sentence-transformers（預設 BAAI/bge-m3，1024 維）。"""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")

    model_name: str = Field(default="BAAI/bge-m3")
    dimension: int = Field(default=1024, ge=32, le=4096)
    batch_size: int = Field(default=16, ge=1, le=256)
    lightrag_embedding_timeout: int = Field(
        default=300,
        ge=30,
        le=7200,
        description="傳入 LightRAG default_embedding_timeout（秒）；首次載入模型時須足夠大",
    )
    max_async: int = Field(
        default=2,
        ge=1,
        le=16,
        description="嵌入並發上限（對應 LightRAG embedding_func_max_async）；建議較低以避免重複載入模型",
    )


class ChunkSettings(BaseSettings):
    """分塊參數（對應 LightRAG 的 CHUNK_SIZE / CHUNK_OVERLAP_SIZE）。"""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", env_file=".env", extra="ignore")

    chunk_size: int = Field(default=1200, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)


class RetrievalSettings(BaseSettings):
    """檢索預設行為。"""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="ignore")

    default_mode: str = Field(default="mix")
    top_k: int = Field(default=10, ge=1, le=100)
    max_hop: int = Field(
        default=2,
        ge=1,
        le=8,
        description="語意參數；LightRAG 內部以 max_graph_nodes 等控制圖規模",
    )
    enable_bm25: bool = Field(default=True)


class IncrementalSettings(BaseSettings):
    """增量更新與斷點。"""

    model_config = SettingsConfigDict(env_prefix="INCREMENTAL_", env_file=".env", extra="ignore")

    enabled: bool = Field(
        default=True,
        description="對應環境變數 INCREMENTAL_ENABLED",
    )
    hash_cache_path: str = Field(default="data/hash_cache.json")
    checkpoint_interval: int = Field(default=1, ge=1, description="每處理 N 個文件寫入斷點")
    checkpoint_path: str = Field(default="data/processed/incremental_checkpoint.json")


class ServiceSettings(BaseSettings):
    """FastAPI 服務。"""

    model_config = SettingsConfigDict(env_prefix="SERVICE_", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = Field(default=False)
    cors_origins: str = Field(
        default="*",
        description="逗號分隔來源，或 *",
    )


class PathsSettings(BaseSettings):
    """專案路徑。"""

    model_config = SettingsConfigDict(env_prefix="PATHS_", env_file=".env", extra="ignore")

    project_root: str = Field(default=".", description="專案根目錄（相對 cwd）")
    data_raw: str = Field(default="data/raw")
    data_processed: str = Field(default="data/processed")
    lightrag_working_dir: str = Field(default="data/lightrag_workdir")
    sqlite_path: str = Field(default="data/meta/app_kv.sqlite3")
    logging_conf: str = Field(default="config/logging.conf")


class FecDomainSettings(BaseSettings):
    """FEC 領域：LightRAG 實體抽取所用 ``entity_types`` 與摘要語言。"""

    model_config = SettingsConfigDict(env_prefix="FEC_", env_file=".env", extra="ignore")

    summary_language: str = Field(
        default="Chinese",
        description="對應 LightRAG ``SUMMARY_LANGUAGE`` / ``addon_params['language']``",
    )
    entity_types_json: str | None = Field(
        default=None,
        description="可選：JSON 字串陣列，覆寫 ``config/fec_defaults.FEC_DEFAULT_ENTITY_TYPES``",
    )

    def resolve_entity_types(self) -> list[str]:
        """回傳實體類型列表（FEC 預設或可選 JSON 覆寫）。"""
        import json

        from config.fec_defaults import FEC_DEFAULT_ENTITY_TYPES

        raw = self.entity_types_json
        if raw and str(raw).strip():
            try:
                v = json.loads(raw)
                if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    return list(v)
            except json.JSONDecodeError:
                pass
        return list(FEC_DEFAULT_ENTITY_TYPES)


class Settings(BaseSettings):
    """聚合設定單例。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    incremental: IncrementalSettings = Field(default_factory=IncrementalSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    fec: FecDomainSettings = Field(default_factory=FecDomainSettings)

    lightrag_workspace: str = Field(default="", validation_alias="WORKSPACE")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_BASE", "OPENAI_BASE_URL"),
    )
    openai_model: str | None = Field(default=None, validation_alias="OPENAI_MODEL")
    openai_temperature: float | None = Field(default=None, validation_alias="OPENAI_TEMPERATURE")

    @field_validator("lightrag_workspace", mode="before")
    @classmethod
    def _strip_ws(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    def resolved_llm_model_name(self) -> str:
        """頂層 ``OPENAI_MODEL`` 優先，否則使用 ``LLM_MODEL_NAME``。"""
        m = (self.openai_model or "").strip()
        return m if m else self.llm.model_name

    def resolved_llm_temperature(self) -> float:
        if self.openai_temperature is not None:
            return float(self.openai_temperature)
        return self.llm.temperature


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """回傳已快取的全域設定單例。"""
    return Settings()


def apply_settings_to_environ(settings: Settings | None = None) -> None:
    """將關鍵設定寫入 os.environ，供 LightRAG 各儲存後端讀取。"""
    import json

    s = settings or get_settings()
    os.environ.setdefault("NEO4J_URI", s.neo4j.uri)
    os.environ.setdefault("NEO4J_USERNAME", s.neo4j.username)
    os.environ.setdefault("NEO4J_PASSWORD", s.neo4j.password)
    os.environ.setdefault("NEO4J_DATABASE", s.neo4j.database)

    os.environ.setdefault("MILVUS_URI", s.milvus.uri)
    os.environ.setdefault("MILVUS_DB_NAME", s.milvus.db_name)

    api_key = (s.openai_api_key or s.llm.api_key or "").strip()
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    base_url = s.openai_base_url or s.llm.base_url
    if base_url:
        os.environ.setdefault("OPENAI_API_BASE", base_url.strip())
    llm_model = s.resolved_llm_model_name()
    if llm_model:
        os.environ.setdefault("OPENAI_MODEL", llm_model)

    os.environ["CHUNK_SIZE"] = str(s.chunk.chunk_size)
    os.environ["CHUNK_OVERLAP_SIZE"] = str(s.chunk.chunk_overlap)
    os.environ["TOP_K"] = str(s.retrieval.top_k)
    os.environ["EMBEDDING_BATCH_NUM"] = str(s.embedding.batch_size)
    os.environ["EMBEDDING_FUNC_MAX_ASYNC"] = str(s.embedding.max_async)

    root = os.path.abspath(s.paths.project_root)
    os.environ.setdefault("LIGHTRAG_WORKDIR", os.path.join(root, s.paths.lightrag_working_dir))

    # FEC 領域：實體類型與摘要語言（若使用者已在環境中設定 ENTITY_TYPES / SUMMARY_LANGUAGE 則不覆寫）
    os.environ.setdefault("SUMMARY_LANGUAGE", s.fec.summary_language)
    if not os.environ.get("ENTITY_TYPES"):
        os.environ["ENTITY_TYPES"] = json.dumps(s.fec.resolve_entity_types(), ensure_ascii=False)
