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
    client_timeout: float = Field(
        default=120.0,
        ge=5.0,
        le=600.0,
        description="pymilvus MilvusClient 的 gRPC 連線就緒超時（秒）；LightRAG 未傳 timeout 時由專案 patch 注入",
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
    rerank_min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="對 rerank 分數 min-max 歸一化後的低分過濾；0 表示僅依 chunk_top_k 截斷（對應 MIN_RERANK_SCORE）",
    )


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
    document_manifest_path: str = Field(
        default="data/meta/document_manifest.json",
        description="以 doc_id 為鍵的側車/資源路徑清單；刪除文檔索引後依此刪除 images、.mineru.json 等",
    )


class ModelsSettings(BaseSettings):
    """本機模型根目錄（預設專案下 ``models/``，Hub 快照在 ``models/hub/``）。"""

    model_config = SettingsConfigDict(env_prefix="MODELS_", env_file=".env", extra="ignore")

    dir: str = Field(default="models", description="相對 project_root 的模型根目錄")
    hf_endpoint: str | None = Field(
        default="https://hf-mirror.com",
        description="Hugging Face 鏡像；未設則不覆寫環境變數",
    )
    offline: bool = Field(
        default=True,
        description="為 True 時設 HF_HUB_OFFLINE，優先使用 models/hub 已有快照",
    )
    embedding_local_path: str | None = Field(
        default=None,
        description="可選：嵌入模型目錄，覆寫自動解析的 Hub 快照路徑",
    )
    rerank_enabled: bool = Field(
        default=True,
        description="是否啟用本機 CrossEncoder rerank；關閉時不注入 rerank_model_func 並關閉 RERANK_BY_DEFAULT",
    )
    rerank_model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="HuggingFace 模型 id；權重下載至 models/hub 後自動解析快照路徑",
    )
    rerank_local_path: str | None = Field(
        default=None,
        description="可選：reranker 目錄，覆寫 Hub 快照（與 EMBEDDING 的 *_LOCAL_PATH 一致）",
    )
    rerank_batch_size: int = Field(
        default=8,
        ge=1,
        le=128,
        description="CrossEncoder.predict 批次大小",
    )


class DocumentConversionSettings(BaseSettings):
    """PDF 等轉 Markdown（MinerU）與圖片落地。"""

    model_config = SettingsConfigDict(env_prefix="DOCUMENT_", env_file=".env", extra="ignore")

    pipeline_mode: str = Field(
        default="two_stage",
        description="two_stage：PDF 僅在轉檔步驟處理，索引只讀 .md；coupled：入庫時內聯 MinerU/抽取",
    )
    conversion_cache_path: str = Field(
        default="data/conversion_cache.json",
        description="PDF→MD 增量快取（與索引 hash_cache 分離）",
    )
    use_mineru_for_pdf: bool = Field(
        default=True,
        description="轉檔步驟（或 coupled 入庫）是否使用 MinerU",
    )
    mineru_infer_backend: str = Field(
        default="vlm-auto-engine",
        description="MinerU -b 推論後端",
    )
    mineru_force_refresh: bool = Field(default=False, description="忽略 MinerU 緩存強制重轉")

    def is_two_stage(self) -> bool:
        return (self.pipeline_mode or "two_stage").strip().lower() in {
            "two_stage",
            "two-stage",
            "2",
            "separate",
        }


class MultimodalSettings(BaseSettings):
    """檢索後結合圖片的多模態回答（OpenAI 相容 vision）。

    ``--multimodal`` 時使用本組 ``MULTIMODAL_*``；純文字問答與 LightRAG 圖譜抽取仍用頂層 ``OPENAI_*``。
    """

    model_config = SettingsConfigDict(env_prefix="MULTIMODAL_", env_file=".env", extra="ignore")

    api_key: str = Field(default="", description="多模態 API Key（如本機可填 none）")
    base_url: str | None = Field(default=None, description="多模態 OpenAI 相容端點")
    model_name: str = Field(default="", description="視覺模型名，如 Qwen3.5-9B")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    vision_model: str | None = Field(
        default=None,
        description="已廢止別名，請用 MULTIMODAL_MODEL_NAME；未設 model_name 時才讀此欄",
    )
    max_images_per_query: int = Field(
        default=3,
        ge=0,
        le=32,
        description="多模態：從「最終參考片段」中解析的本地圖片數上限；達到後不再掃描後續片段/地址",
    )
    reference_context_max_chars: int = Field(
        default=28_000,
        ge=2_000,
        le=500_000,
        description="多模態/純檢索上下文：在 LightRAG 已 rerank、去噪與 token 截斷後，再對 chunks 做字數預算（近似 token 上限）",
    )
    max_image_bytes: int = Field(default=4_000_000, ge=50_000, le=20_000_000)
    system_prompt: str | None = Field(
        default=None,
        description="多模態一輪回答的 system 提示；未設則使用內建預設",
    )


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
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    document: DocumentConversionSettings = Field(default_factory=DocumentConversionSettings)
    multimodal: MultimodalSettings = Field(default_factory=MultimodalSettings)
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

    def resolved_multimodal_model_name(self) -> str:
        """``--multimodal`` 專用視覺模型（MULTIMODAL_MODEL_NAME / 舊版 VISION_MODEL）。"""
        m = (self.multimodal.model_name or "").strip()
        if m:
            return m
        v = (self.multimodal.vision_model or "").strip()
        return v if v else self.resolved_llm_model_name()

    def resolved_multimodal_temperature(self) -> float:
        return float(self.multimodal.temperature)

    def rerank_runtime_available(self) -> bool:
        """本機 rerank 是否應開啟（關閉開關、離線且無快照時為 False）。"""
        if not self.models.rerank_enabled:
            return False
        from pathlib import Path

        from config.model_paths import resolve_hub_model_dir, resolve_project_root

        lp = (self.models.rerank_local_path or "").strip()
        if lp:
            p = Path(lp).expanduser()
            if not p.is_absolute():
                p = resolve_project_root(self) / p
            return p.is_dir()
        if resolve_hub_model_dir(self.models.rerank_model_name, self) is not None:
            return True
        return not self.models.offline


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """回傳已快取的全域設定單例。"""
    return Settings()


def apply_settings_to_environ(settings: Settings | None = None) -> None:
    """將關鍵設定寫入 os.environ，供 LightRAG 各儲存後端讀取。"""
    import json

    s = settings or get_settings()
    from config.model_paths import apply_models_to_environ

    apply_models_to_environ(s)

    os.environ.setdefault("NEO4J_URI", s.neo4j.uri)
    os.environ.setdefault("NEO4J_USERNAME", s.neo4j.username)
    os.environ.setdefault("NEO4J_PASSWORD", s.neo4j.password)
    os.environ.setdefault("NEO4J_DATABASE", s.neo4j.database)

    os.environ.setdefault("MILVUS_URI", s.milvus.uri)
    os.environ.setdefault("MILVUS_DB_NAME", s.milvus.db_name)
    os.environ.setdefault("MILVUS_CLIENT_TIMEOUT", str(s.milvus.client_timeout))

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
    os.environ["MIN_RERANK_SCORE"] = str(s.retrieval.rerank_min_score)

    root = os.path.abspath(s.paths.project_root)
    os.environ.setdefault("LIGHTRAG_WORKDIR", os.path.join(root, s.paths.lightrag_working_dir))

    # FEC 領域：實體類型與摘要語言（若使用者已在環境中設定 ENTITY_TYPES / SUMMARY_LANGUAGE 則不覆寫）
    os.environ.setdefault("SUMMARY_LANGUAGE", s.fec.summary_language)
    if not os.environ.get("ENTITY_TYPES"):
        os.environ["ENTITY_TYPES"] = json.dumps(s.fec.resolve_entity_types(), ensure_ascii=False)
