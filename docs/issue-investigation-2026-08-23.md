# 三个问题现状整理（2026-08-23，仅调研，未改代码）

> 调研方式：只读检查代码、`.env`、运行中服务（PID 56932）、SQLite、Neo4j、Milvus、
> Redis、LightRAG workdir 与日志。**未修改任何代码。**

---

## 1. 增量更新期间系统不可用（前端无反应）

### 事实链

| 事实 | 证据 |
| --- | --- |
| `SERVICE_ASYNC_DOCUMENT_PROCESSING_ENABLED` 未配置 | 默认 `False`（settings.py L345）→ 增量更新**同步执行** |
| `/api/rag/incremental-update` 同步挂起 | api.py L293-304：async 关闭时不返回 202，直接 `await rag.incremental_update()` |
| 实测一次增量更新耗时 ≈23 分钟 | 日志 00:50:22 开始 → 01:13:44 结束（errors=1） |
| 前端无超时、无任务轮询 | front/app.js `doIncremental()` 直接 `await fetch(...)`，无 AbortController/无 task_id 轮询 |
| 查询与插入共享并发资源 | `_build_llm_func` 用 `get_semaphore("llm", 8)` 限流（插入抽取与查询 LLM 共用）；`EMBEDDING_FUNC_MAX_ASYNC=2`（embedding 并发上限 2） |
| 并发实测劣化 | 增量运行期间一次查询 latency 25070ms（正常 315ms，log 01:01:26） |
| 增量锁 60s 过期 | `acquire_lock(lock_key, timeout=60)`（rag_service.py L109）——单文档插入可超过 60s（日志出现 `Worker execution timeout after 480s`），锁过期后允许重复提交 |
| 服务与代码不同步 | 服务器 00:43 启动（旧代码，带 LLM 模式路由）；`mode_router.py` 01:05、`retriever.py` 01:06、`rag_service.py` 01:08 刚被改为纯规则路由 → **运行中的服务不是当前代码** |

### 结论（暂定）

主要原因不是事件循环被阻塞（LightRAG 1.5.4 存储层为异步实现），而是：
**同步执行 + 无反馈 + 查询与插入争抢 LLM/Embedding 并发 + 前端无超时** 的叠加，
用户体感为"完全无反应"。另有一个待验证疑点：busy 状态下 `get_lightrag()` 的
`_init_lock`（初始化约 2 分钟）会阻塞所有并发查询。

---

## 2. 检索链路：明明有内容却全部为空

### 事实链（根因已在数据层坐实）

| 事实 | 证据 |
| --- | --- |
| **Neo4j 空库** | `MATCH (n) ...` 无任何节点（Entity/Chunk/Relationship 全 0） |
| **Milvus 空库** | 3 个 collection（chunks/entities/relationships_baai_bge_m3_1024d）内部 row_count 均 0 |
| workdir 文档状态混乱 | 6 条记录：1 条 failed（`dup-...`，错误=`File name already exists`，**同一文件重复插入**）、3 条 processing、2 条 pending |
| 曾有插入 FAILED | 日志：`extract LLM func: Worker execution timeout after 480s`（ttc202309_yu.md） |
| 抽取结果其实已生成 | `kv_store_llm_response_cache.json` 已有实体抽取结果（RM 变换等）→ 说明管线干活了，但**从未成功落库/flush** |
| 遥测证实空检索发生过 | SQLite `query_telemetry` 2 条 `graph_empty=1` 且 entities/relations/chunks 全 0：`rs的rmbased译码的流程是什么样的`（25s）、`rm算法的复杂度是多少`（5.8s）；用户反馈 `wrong` |

### 检索代码链路检查（输入→检索→输出）

- 模式路由：当前代码为启发式（`suggest_mode_from_question`），不会选 `bypass`；`query()` 中 bypass 会自动改 mix。
- 关键词：`keyword_extraction` 已被项目拦截为 FEC 启发式（不再调 LLM），有兜底（空关键词时 `ll_keywords=[query]`，仅 >50 字符全空才 fail）→ **关键词环节不构成空检索主因**。
- 缓存：`_get_retrieval_bundle` 会把**空 bundle 也写入 Redis 缓存**（TTL 300s）——空结果"粘住"的机制存在；当前 Redis 中无残留 `rag:retrieval:*` / `rag:query:*` 键（已过期）。
- 结论：**检索代码链路本身未发现"有内容却查不到"的逻辑 bug；空结果与"索引库为空"完全自洽**。真正的问题是写入侧：重复插入冲突（DUPLICATE）+ 480s LLM 超时失败 + 未 flush，导致索引从未建成。

---

## 3. 遥测是否写入 SQLite

### 事实链

| 事实 | 证据 |
| --- | --- |
| **已保存** | `data/meta/app_kv.sqlite3` 内 `query_telemetry` 3 行、`feedback_telemetry` 2 行 |
| 字段完整 | question / mode / latency_ms / graph_empty / 各计数 / token 估算 / timestamp 均已落库 |
| 与用户提问吻合 | 3 条查询即"告诉我什么是rs码 / rs的rmbased译码的流程 / rm算法的复杂度"；2 条反馈 correct/wrong |
| 写入路径 | 内存队列 → 后台线程批量写 SQLite（WAL）→ 与 KVClient 共用同一 sqlite 文件；`/api/rag/telemetry` 可聚合查询 |
| 潜在缺口（待核） | 遥测计数：连 `naive` 成功查询也是 entities/chunks=0、tokens=0——**LightRAG 1.5.4 的 bundle 结构与 `_payload_from_bundle` 假设可能不匹配**，检索计数/引文数可能没有被正确解析 |
| 潜在缺口（待核） | 服务器运行旧代码（01:05 代码改动前），遥测代码版本与当前代码可能不一致 |

---

## 时间线还原（本地时区）

```
08-23 00:43  服务器启动（旧代码，带 LLM 路由）
08-23 00:50  → 01:13  增量更新同步执行（含 ttc202309_yu.md 插入失败：480s LLM 超时；日志 FAILED）
08-23 01:01  查询「rs的rmbased译码的流程是什么样的」25.0s，检索全空（增量进行中）
08-23 01:05  → 01:09  代码被修改（移除 LLM 路由 → 启发式路由）；服务器未重启
08-23 01:13  增量更新结束（errors=1，Neo4j/Milvus 仍无数据）
08-23 01:32  查询「rm算法的复杂度是多少」5.8s，检索仍全空（与"索引库空"一致）
08-23 01:37  → 01:51  另一批文档被强制重入库（doc_status 更新至 01:51，仍 processing/pending）
08-23 01:40  反馈标记「rm算法的复杂度是多少」= wrong
```

---

## 决策收敛（按推荐方向暂定，可逐项调整）

1. **索引处置**：清理混乱 doc_status/断点残留后**全量重建**；重建前必须先解决 480s LLM Worker 超时问题，否则重建仍会失败。
2. **问题 1 修复方向**：增量更新改**异步任务化**（202 + task_id + 前端轮询，`SERVICE_ASYNC_DOCUMENT_PROCESSING_ENABLED=true` 默认开启），并叠加：增量锁超时 > 单文档处理上限；查询与插入的 LLM/Embedding 并发配额**分离**；前端 fetch 增加超时与任务状态展示。
3. **问题 2 修复方向**：修重复插入冲突（DUPLICATE）与失败重试路径（同路径/同 doc_id 并发保护、插入幂等）；空 bundle 不写检索缓存；重建后用验收查询证明"有内容必能检索到"。
4. **问题 3 口径**：重启服务后用新代码实测 bundle 解析（LightRAG 1.5.4 返回 `data:{entities,relationships,chunks,urls}`，与 `_payload_from_bundle` 假设需实测对齐）；确认 API 与 CLI 查询均写遥测（已确认：scripts/query.py 同样走 RAGService）；建立"每查询必有遥测行"核验。
5. **部署口径**：修复完成后**重启服务**加载当前工作区代码（当前运行中服务 ≠ 工作区代码）；验证一律在重启后进行。

## 修复需求清单（整理级，未写代码）

| # | 需求 | 关联问题 | 验收口径 |
| --- | --- | --- | --- |
| R1 | 增量更新/文档写入异步化：202+task_id+前端轮询+超时 | 1 | 批量导入期间 API 正常应答，前端可操作、有任务进度 |
| R2 | 查询与插入并发隔离（LLM/Embedding 配额分离） | 1 | 增量运行期间查询 p50 不劣化超过阈值（如 2s） |
| R3 | 增量锁/文档锁超时大于单文档处理上限，防重复提交 | 1, 2 | 并发触发同路径插入不产生 DUPLICATE failed |
| R4 | 插入失败重试与幂等：doc_status 残留清理、同名冲突根治 | 2 | `verify_incremental_consistency.py` 全绿：manifest/SQLite/KV/Milvus/Neo4j 一致 |
| R5 | 空检索结果不污染缓存（空 bundle 不缓存 / 缓存带空标记） | 2 | 空结果查询 5 分钟内重复执行能重试真实检索 |
| R6 | 全量重建索引（先解决 480s LLM Worker 超时的上游根因） | 2 | Neo4j/Milvus 行数 > 0，验收查询 graph_empty=false 且 chunks>0 |
| R7 | 遥测 bundle 解析与 1.5.4 结构对齐；每查询必有落库行 | 3 | 每个 query 在 app_kv.sqlite3 的 query_telemetry 有一行且计数非 0 |
| R8 | 重启服务（加载当前代码）后再验证 | 全部 | 服务进程与工作区代码一致（git 状态/启动时间核对） |

验证入口（已存在，可复用）：`scripts/verify_incremental_consistency.py`（快照对比 manifest/SQLite/LightRAG KV/Milvus/Neo4j）、`GET /api/rag/telemetry`、`scripts/metrics_summary.py`。

> 说明：CLI 与 API 问答均写遥测（scripts/query.py → RAGService → retriever → append_telemetry）。
> 遥测计数连成功 naive 也全 0 的疑点，需在重启+新代码下实测确认，当前运行旧代码无法判定。