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
    """嵌入模型：仅支持第三方线上 API。"""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")

    # 线上 API 配置（必填）
    api_enabled: bool = Field(
        default=True,
        description="是否启用第三方线上 embedding API（必须为 true）",
    )
    api_key: str = Field(
        default="",
        description="第三方 embedding API 密鑰（如 SiliconFlow、智谱等）",
    )
    api_base_url: str | None = Field(
        default=None,
        description="第三方 embedding API 端点（如 https://api.siliconflow.cn）",
    )
    api_model_name: str = Field(
        default="",
        description="第三方 embedding API 模型名称（如 BAAI/bge-m3、text-embedding-3-large）",
    )
    api_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="API 请求超时时间（秒）",
    )

    # 模型参数
    dimension: int = Field(default=1024, ge=32, le=4096)
    batch_size: int = Field(default=16, ge=1, le=256)
    lightrag_embedding_timeout: int = Field(
        default=300,
        ge=30,
        le=7200,
        description="传入 LightRAG default_embedding_timeout（秒）",
    )
    max_async: int = Field(
        default=2,
        ge=1,
        le=16,
        description="嵌入并发上限（对应 LightRAG embedding_func_max_async）",
    )


class ChunkSettings(BaseSettings):
    """分塊參數（對應 LightRAG 的 CHUNK_SIZE / CHUNK_OVERLAP_SIZE）。"""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", env_file=".env", extra="ignore")

    chunk_size: int = Field(default=1200, ge=100)
    chunk_overlap: int = Field(default=100, ge=0)


class LightRAGRuntimeSettings(BaseSettings):
    """LightRAG 查詢期參數（對應環境變數 LIGHTRAG_* / MAX_*_TOKENS 等）。"""

    model_config = SettingsConfigDict(env_prefix="LIGHTRAG_", env_file=".env", extra="ignore")

    max_entity_tokens: int = Field(
        default=8000,
        ge=512,
        le=32000,
        description="實體上下文 token 上限（原庫默認 6000，放寬以利多跳）",
    )
    max_relation_tokens: int = Field(
        default=10000,
        ge=512,
        le=32000,
        description="關係上下文 token 上限（原庫默認 8000）",
    )
    max_total_tokens: int = Field(
        default=20000,
        ge=4000,
        le=64000,
        description="實體+關係+chunk 總 token 上限（原庫默認 30000，下調降延遲）",
    )
    related_entity_chunk_number: int = Field(
        default=6,
        ge=1,
        le=32,
        description="每個實體關聯的 chunk 數（LightRAG related_chunk_number）",
    )
    related_relation_chunk_number: int = Field(
        default=4,
        ge=1,
        le=32,
        description="每個關係關聯的 chunk 數（運行期 patch，低於實體以降噪）",
    )
    relation_top_k: int = Field(
        default=7,
        ge=1,
        le=100,
        description="關係向量庫查詢 top_k（低於 RETRIEVAL_TOP_K，減少冗餘邊）",
    )
    relation_rerank_enabled: bool = Field(
        default=True,
        description="檢索後對關係描述做 CrossEncoder 重排過濾",
    )
    relation_min_rerank_score: float = Field(
        default=0.32,
        ge=0.0,
        le=1.0,
        description="關係重排 min-max 歸一化後的最低分",
    )
    relation_keyword_min_score: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="關係描述與高/低層關鍵詞的最小重合度（0 表示不過濾）",
    )
    kg_chunk_pick_method: str = Field(
        default="VECTOR",
        description="關聯 chunk 選擇：VECTOR 用向量相似度降噪，WEIGHT 為權重",
    )
    max_graph_nodes: int = Field(
        default=128,
        ge=32,
        le=2000,
        description="Neo4j 子圖遍歷節點上限（原公式 max_hop*48≈96 或庫默認 1000）",
    )
    chunk_top_k: int | None = Field(
        default=9,
        ge=1,
        le=100,
        description="覆寫 CHUNK_TOP_K；None 時與 RETRIEVAL_TOP_K 一致",
    )
    cosine_better_than_threshold: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="向量檢索餘弦閾值（原 0.2）",
    )
    keyword_fallback_enabled: bool = Field(
        default=True,
        description="LLM 關鍵詞抽取失敗或為空時使用 FEC 啟發式回退",
    )
    entity_extract_max_gleaning: int = Field(
        default=2,
        ge=0,
        le=5,
        description="索引期實體抽取補全輪次（原庫默認 1，提高以補全圖譜實體）",
    )


class RetrievalSettings(BaseSettings):
    """檢索預設行為。"""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="ignore")

    default_mode: str = Field(default="mix")
    top_k: int = Field(default=8, ge=1, le=100)
    max_hop: int = Field(
        default=2,
        ge=1,
        le=8,
        description="語意參數；Neo4j 規模主要由 LIGHTRAG_MAX_GRAPH_NODES 控制",
    )
    enable_bm25: bool = Field(default=True)
    rerank_min_score: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
        description="對 rerank 分數 min-max 歸一化後的低分過濾；0 表示僅依 chunk_top_k 截斷（對應 MIN_RERANK_SCORE）",
    )
    llm_mode_router_enabled: bool = Field(
        default=True,
        description="檢索前調用 LLM 評估問題並在 naive/local/global/hybrid/mix 中自動選模式",
    )
    llm_mode_router_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="模式路由 LLM 溫度（建議 0 以穩定選 mode）",
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
    models_dir: str = Field(default="models", description="本地模型存放目錄（相對 project_root）")
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
    """Rerank 线上 API 配置。"""

    model_config = SettingsConfigDict(env_prefix="MODELS_", env_file=".env", extra="ignore")

    # 线上 Rerank API 配置
    rerank_api_enabled: bool = Field(
        default=False,
        description="是否启用第三方线上 rerank API",
    )
    rerank_api_key: str = Field(
        default="",
        description="第三方 rerank API 密钥",
    )
    rerank_api_base_url: str | None = Field(
        default=None,
        description="第三方 rerank API 端点",
    )
    rerank_api_model_name: str = Field(
        default="",
        description="第三方 rerank API 模型名称（如 BAAI/bge-reranker-v2-m3）",
    )
    rerank_api_timeout: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Rerank API 请求超时时间（秒）",
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
        default=18_000,
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
    lightrag: LightRAGRuntimeSettings = Field(default_factory=LightRAGRuntimeSettings)
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
        """檢查本地 CrossEncoder 運行時是否可用（sentence-transformers 是否已安裝）。"""
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False


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
    lr = s.lightrag
    chunk_top_k = lr.chunk_top_k if lr.chunk_top_k is not None else s.retrieval.top_k
    os.environ["TOP_K"] = str(s.retrieval.top_k)
    os.environ["CHUNK_TOP_K"] = str(chunk_top_k)
    os.environ["MAX_ENTITY_TOKENS"] = str(lr.max_entity_tokens)
    os.environ["MAX_RELATION_TOKENS"] = str(lr.max_relation_tokens)
    os.environ["MAX_TOTAL_TOKENS"] = str(lr.max_total_tokens)
    os.environ["RELATED_CHUNK_NUMBER"] = str(lr.related_entity_chunk_number)
    os.environ["KG_CHUNK_PICK_METHOD"] = str(lr.kg_chunk_pick_method).strip().upper()
    os.environ["EMBEDDING_BATCH_NUM"] = str(s.embedding.batch_size)
    os.environ["EMBEDDING_FUNC_MAX_ASYNC"] = str(s.embedding.max_async)
    os.environ["MIN_RERANK_SCORE"] = str(s.retrieval.rerank_min_score)

    root = os.path.abspath(s.paths.project_root)
    os.environ.setdefault("LIGHTRAG_WORKDIR", os.path.join(root, s.paths.lightrag_working_dir))

    # FEC 領域：摘要語言（entity_types 已透過 addon_params 傳入 LightRAG）
    os.environ.setdefault("SUMMARY_LANGUAGE", s.fec.summary_language)
