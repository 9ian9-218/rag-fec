"""
使用範例（請先啟動 API：`python main.py` 或 Docker Compose）。

以下假設服務在 http://127.0.0.1:8000。
"""

# 範例 6：不啟 API，於專案根目錄直接問答（需 Neo4j / Milvus / LLM 等與線上 API 相同）
# python scripts/query.py "請簡述文件重點"
# python scripts/query.py -i   # 互動多輪

# 範例 1：健康檢查
# curl -s http://127.0.0.1:8000/api/rag/health

# 範例 2：上傳 Markdown 並建索引
# curl -s -F "file=@./data/raw/sample.md" http://127.0.0.1:8000/api/rag/documents

# 範例 3：問答（預設 mix 模式）
# curl -s -H "Content-Type: application/json" \
#   -d '{"question":"請簡述文件重點","stream":false}' \
#   http://127.0.0.1:8000/api/rag/query

# 範例 4：觸發增量更新（掃描 data/raw 與 hash_cache）
# curl -s -X POST http://127.0.0.1:8000/api/rag/incremental-update

# 範例 5：列出已註冊文件
# curl -s http://127.0.0.1:8000/api/rag/documents
