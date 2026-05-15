# Graph RAG（LightRAG + Neo4j + Milvus）

生產導向的圖 RAG 專案：文件解析與分塊、LightRAG 管線、Neo4j 圖儲存、Milvus 向量儲存、應用層 SQLite 元資料、增量更新與 FastAPI 服務，並提供 Docker Compose 一鍵啟動依賴服務。

## 技術要點

- **LightRAG**（`lightrag-hku`）：`Neo4JStorage` + `MilvusVectorDBStorage`；KV 使用 **`JsonKVStorage`**（持久化於 `data/lightrag_workdir`）。應用層 **`src/storage/kv_client.py`** 使用 **SQLite** 管理文件註冊與斷點，對應需求中的「元資料 / 註冊表入 SQLite」。
- **FEC 領域實體**：預設使用與 `rag-fec` 的 `FEC_ENTITY_KIND_TYPES` 一致的 12 類（見 `config/fec_defaults.py`），經 `LightRAG.addon_params["entity_types"]` 與環境變數 `ENTITY_TYPES`（JSON 陣列）注入；摘要語言預設 **`FEC_SUMMARY_LANGUAGE=Chinese`**（對應 `SUMMARY_LANGUAGE`）。可用 **`FEC_ENTITY_TYPES_JSON`** 覆寫類型列表。
- **增量更新**：`data/hash_cache.json` 比對 MD5；變更時呼叫 LightRAG `adelete_by_doc_id` 再 `ainsert`；路徑穩定 `doc_id` 見 `src/incremental/doc_registry.py`。
- **嵌入**：本機 **sentence-transformers**（預設 **BAAI/bge-m3**，1024 維）；無需外部嵌入 API。
- **編排**：檢索鏈可選 LangChain `RunnableLambda`（見 `src/retrieval/retriever.py`）。

## 本地開發

1. Python 3.11+，建議虛擬環境或 Conda。
2. 安裝依賴：`pip install -r requirements.txt`
3. 複製環境變數：`cp .env.example .env`，至少填入 **`OPENAI_API_KEY`**（DeepSeek）。
4. 啟動本機 Neo4j、Milvus（或僅跑單元測試，見下）。
5. 啟動 API：`python main.py` 或 `uvicorn src.service.api:app --reload`

常用腳本：

- 全量建索引：`python scripts/build_index.py --mode full --raw data/raw`
- 增量更新：`python scripts/incremental_update.py`
- 清空索引：`python scripts/clear_index.py --all`（會清空 Neo4j 圖、LightRAG 工作目錄檔案與 SQLite；請謹慎）
- 評估：`python scripts/evaluate.py --input data/test/pairs.jsonl --out data/test/eval_report.json`

## Docker Compose 一鍵部署

1. `cp .env.example .env`，至少設定 **`OPENAI_API_KEY`**（DeepSeek）。
2. 在專案根目錄執行：`docker compose up --build`
3. 服務：`rag`（8000）、`neo4j`（7474/7687）、`milvus`（19530），以及 Milvus 依賴的 `etcd`、`minio`。
4. 入口腳本會 **TCP 等待** Neo4j 7687 與 Milvus 19530 後再啟動 Uvicorn。

**注意**：容器內若要連本機 Ollama / vLLM，請將 `LLM_BASE_URL` 設為 `http://host.docker.internal:11434/v1`（Linux 可能需額外 `extra_hosts`）。

## 增量更新說明

- 將 PDF / Markdown / Word / TXT 放入 `data/raw`（可子目錄）。
- 首次或週期執行：`POST /api/rag/incremental-update` 或 `python scripts/incremental_update.py`。
- `data/hash_cache.json` 記錄路徑 → MD5；**修改**會先刪除舊 `doc_id` 再寫入；**刪除**會從索引與快取移除。

## API 摘要

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/api/rag/health` | 健康檢查 |
| POST | `/api/rag/query` | 問答（JSON：`question`, `mode`, `stream`） |
| POST | `/api/rag/documents` | 上傳單檔 |
| POST | `/api/rag/documents/batch` | 多檔上傳 |
| PUT | `/api/rag/documents/{doc_id}` | 依 `doc_id` 覆寫原路徑檔案並重建索引 |
| DELETE | `/api/rag/documents/{doc_id}` | 刪除索引 |
| GET | `/api/rag/documents` | 文件列表 |
| GET | `/api/rag/documents/{doc_id}` | 文件詳情 |
| POST | `/api/rag/incremental-update` | 觸發增量掃描 |

OpenAPI：`/docs`、`/redoc`。

## 測試

```bash
export PYTHONPATH=.
pytest -q
```

## 常見問題

1. **Neo4j 連不上**：確認 `NEO4J_URI` 與防火牆。**在主機執行** `python scripts/...` 時請用 `bolt://127.0.0.1:7687`（或本機 Neo4j 位址）；`neo4j` 主機名只在 **Docker 網路內** 有效，`rag` 容器會由 compose 自動覆寫。
2. **Milvus 連不上**：同上，主機腳本請用 `http://127.0.0.1:19530`；容器內為 `http://milvus:19530`。Standalone 啟動較慢時可看 `rag` 日誌。
3. **嵌入維度錯誤**：`EMBEDDING_DIMENSION` 必須與模型輸出一致（bge-m3 為 1024）。首次載入會下載/載入模型，可調大 `EMBEDDING_LIGHTRAG_EMBEDDING_TIMEOUT`。
4. **`.env` 不存在導致 Compose 失敗**：請先 `cp .env.example .env`。
5. **`PermissionError: data/logs/app.log`**：若曾用 Docker 寫過日誌，`data/logs` 可能屬於 root。可執行 `sudo chown -R "$USER:$USER" data/logs`（或刪除該目錄後再啟動）。程式已改為：**無法寫檔時僅輸出到終端機**，不影響啟動。
6. **瀏覽器打開根路徑 `/`**：已導向 **`/docs`**（Swagger）；亦可直接訪問 `/api/rag/health`。

與需求一致：`config/`、`data/`、`src/`（含 `data_processing`、`storage`、`incremental`、`retrieval`、`service`、`utils`）、`scripts/`、`tests/`、`docker/`、`main.py`。

更多 curl 範例見 `examples/example_usage.py`。
