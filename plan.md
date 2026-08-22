# RAG-FEC 检索时延优化专项计划（深度分析）

> 状态：分析/计划阶段，不修改任何代码。
> 目标：在不明显损失检索质量的前提下，降低 `POST /api/rag/query` 的端到端检索时延，重点优化 p50/p95/p99。

---

## 1. 当前检索链路拆解

一次非流式查询的端到端路径大致为：

```text
POST /api/rag/query
  → API 限流 / Redis 查询缓存
  → LLM 模式路由（auto_mode=true 时）
  → LightRAG.aquery_data
      → 关键词抽取（可能再次调用 LLM）
      → Query Embedding（远程 Embedding API）
      → Neo4j 图查询
      → Milvus 向量查询
      → 结果融合 / 去重 / 截断
      → Rerank（远程 Rerank API，可选）
  → 自定义 LLM 生成最终答案
  → 遥测异步入队（已不阻塞主链路）
```

当前项目已经完成的并发/缓存优化包括：

- Redis 查询结果缓存
- Redis 模式路由缓存
- Redis Embedding 缓存
- HTTP 客户端复用
- 并发信号量背压
- 遥测异步写 SQLite
- 多 Worker 配置
- 文档异步任务队列

但这些主要解决“高并发稳定性”，对**单请求检索时延**的优化还不够深入。

---

## 2. 时延热点分析

| 环节 | 预估时延 | 是否可优化 | 说明 |
| --- | --- | --- | --- |
| LLM 模式路由 | 0.5~2s | 高 | 每个 auto_mode 请求多一次 LLM 调用 |
| LightRAG 关键词抽取 | 0.5~2s | 高 | 可能再次调用 LLM，和模式路由叠加 |
| Query Embedding API | 0.2~1s | 中 | 每次查询都要远程 embedding |
| Neo4j 图查询 | 0.1~1s | 中 | 受图规模、索引、遍历深度影响 |
| Milvus 向量查询 | 0.1~0.5s | 中 | 受索引类型、top_k、nprobe 等影响 |
| Rerank API | 0.2~1s | 中 | 远程 rerank，延迟较高，可降级/裁剪 |
| 最终 LLM 生成 | 1~5s | 高 | 通常占端到端时延最大比例 |
| 上下文组装/网络开销 | 0.05~0.3s | 低 | 已做连接复用 |

**结论：主要时延来自多次 LLM 调用 + 远程 Embedding/Rerank + 最终生成。**

---

## 3. 可加速点

### 3.1 减少 LLM 调用次数

1. **关闭或降级 LLM 模式路由**
   - 使用 `RETRIEVAL_LLM_MODE_ROUTER_ENABLED=false`，改用启发式 `suggest_mode_from_question`。
   - 或只对复杂问题启用 LLM 路由，简单问题直接走 `naive`/`local`。
   - 预计节省 0.5~2s/请求。

2. **关闭 LightRAG 的 LLM 关键词抽取**
   - 当前 `keyword_fallback_enabled=true` 仍先尝试 LLM 抽取。
   - 可增加配置强制使用 FEC 启发式关键词，跳过 LLM。
   - 预计节省 0.5~2s/请求。

3. **Redis 缓存关键词抽取结果**
   - `question → high/low keywords` 缓存，相同问题不再调用 LLM。
   - 可与模式路由缓存合并为“检索前决策缓存”。

4. **最终生成模型/参数优化**
   - 使用更快/更小的生成模型。
   - 限制 `max_tokens`，降低 `temperature`。
   - 精简 system prompt 和上下文模板。
   - 预计降低最终生成时延。

### 3.2 降低检索阶段耗时

1. **根据问题复杂度选择轻量模式**
   - 简单问题默认 `naive`，只走向量检索，跳过图查询。
   - 中等问题走 `local`，减少全局关系遍历。
   - 只有复杂问题才走 `hybrid`/`mix`。

2. **下调检索参数**
   - `RETRIEVAL_TOP_K`：8 → 5/6
   - `LIGHTRAG_CHUNK_TOP_K`：9 → 5/6
   - `LIGHTRAG_MAX_GRAPH_NODES`：128 → 64/96
   - `LIGHTRAG_MAX_TOTAL_TOKENS`：20000 → 12000/15000
   - `LIGHTRAG_RELATED_CHUNK_NUMBER`：6 → 3/4
   - 需要在召回质量和时延之间做 A/B。

3. **Rerank 降级/裁剪**
   - 高并发或简单问题可关闭远程 Rerank。
   - 或先只用 embedding 相似度粗排，再对前 N 个调 Rerank。
   - 预计节省 0.2~1s/请求。

4. **存储层调优**
   - Neo4j：为实体名、关系类型建立索引/约束；控制 `max_hop`。
   - Milvus：HNSW 参数调优，如 `M`、`efConstruction`、查询 `ef`/`nprobe`。
   - 使用 `PROFILE` 找出图查询中的全库扫描。

### 3.3 并行化

1. **图查询与向量查询并行**
   - LightRAG `mix` 模式内部可能是串行执行图+向量。
   - 可探索 patch：在 `local/global/mix` 中并行发起 Neo4j 与 Milvus 查询，再 merge。
   - 风险：LightRAG 内部数据结构/状态可能不是线程安全，需要严格测试。

2. **多路召回并行**
   - `hybrid` 的 local/global 两路可并行执行后再合并。
   - 如果 LightRAG 不支持，可在上层拆成多个 `aquery_data` 并发调用再合并。

### 3.4 缓存

1. **检索上下文缓存**
   - 对 `question → retrieval bundle` 做 Redis 缓存，命中后跳过 embedding、图查询、向量查询、rerank。
   - 与最终答案缓存不同，它保留检索结果，可用于调试/二次生成。

2. **Rerank 结果缓存**
   - 对 `(query, doc_ids)` 缓存 rerank 分数，避免重复 rerank。

3. **Query Embedding 缓存**
   - 当前已有文本级 embedding 缓存；可进一步对规范化问题本身做 query embedding 缓存。

### 3.5 输出体验优化

1. **流式输出**
   - `stream=true` 可降低首字延迟感知，虽然总时延不变。
2. **快速失败/兜底**
   - 检索为空时直接返回“无法回答”，不调用最终 LLM。
3. **分级响应**
   - 先返回简短答案，再异步补充详细检索证据（如适用）。

---

## 4. 分阶段实施计划

### Phase A：消除非必要 LLM 调用（收益最大，风险中低）

- A1 增加配置：
  - `RETRIEVAL_LLM_MODE_ROUTER_ENABLED` 默认改为 `false` 或增加“简单问题跳过路由”。
- A2 增加“强制 FEC 启发式关键词”开关：
  - 跳过 LightRAG LLM 关键词抽取，直接使用 `fec_keyword_fallback`。
- A3 实现关键词抽取 Redis 缓存。
- 验证：
  - 对比开关前后 p50/p95、LLM 调用次数、检索质量（离线评估）。

### Phase B：检索参数与存储调优（收益中，风险中）

- B1 对 top_k、chunk_top_k、max_graph_nodes、max_total_tokens 做多组 A/B。
- B2 调优 Neo4j 索引、Milvus 索引参数。
- B3 提供 `ENABLE_RERANK=false` 或“仅复杂问题启用 Rerank”的开关。
- 验证：
  - 压测 QPS/时延变化。
  - 离线评估 recall/precision 不显著下降。

### Phase C：并行化与上下文缓存（收益中高，风险中高）

- C1 探索 LightRAG 图/向量并行。
- C2 增加检索上下文缓存。
- C3 增加 Rerank 结果缓存。
- 验证：
  - 对比串行/并行时延。
  - 验证缓存命中率与一致性。

### Phase D：生成加速（收益高，风险低）

- D1 支持配置更快的生成模型。
- D2 限制 `max_tokens`。
- D3 精简提示词。
- D4 流式输出优化。
- 验证：
  - 端到端时延、首字时延、回答质量抽样。

### Phase E：端到端回归

- 使用 `loadtest/` 跑固定/阶梯压测。
- 使用 `scripts/evaluate.py` 跑离线评估。
- 使用 SQLite 遥测统计 p50/p95/p99、LLM 调用次数、Rerank 调用次数。
- 全量回归测试。

---

## 5. 优先级建议

| 优先级 | 优化项 | 预计收益 | 风险 |
| --- | --- | --- | --- |
| P0 | 关闭/降级 LLM 模式路由 | 高 | 低 |
| P0 | 跳过 LLM 关键词抽取，使用启发式 | 高 | 中 |
| P1 | Redis 关键词/检索上下文缓存 | 高 | 低 |
| P1 | 下调 top_k / max tokens / graph nodes | 中 | 中 |
| P1 | Rerank 降级/裁剪 | 中 | 中 |
| P2 | 图/向量并行 | 中高 | 高 |
| P2 | Milvus/Neo4j 索引调优 | 中 | 低 |
| P3 | 更小生成模型 / 流式输出 | 中 | 低 |

---

## 6. 验证与灰度

所有优化必须可配置、可灰度：

- `ENABLE_LLM_MODE_ROUTER=false`
- `ENABLE_LLM_KEYWORD_EXTRACTION=false`
- `ENABLE_RERANK=false`
- `ENABLE_RETRIEVAL_CONTEXT_CACHE=true/false`
- `RETRIEVAL_TOP_K=5/8`
- `LIGHTRAG_MAX_TOTAL_TOKENS=12000/20000`
- `STREAM_DEFAULT=true/false`

回滚方式：环境变量开关 + 重新压测对比。

---

## 7. 暂不实施

- 本阶段只做分析和计划，不修改任何代码。
- 不盲目删除 Rerank/图检索，必须通过离线评估确认质量不下降。
- 不引入额外重中间件；优先使用现有 Redis 和能力。
