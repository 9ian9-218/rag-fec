# Jev / Laya 在本项目 RAG 流程中的可插入点评估

- **日期**：2026-09-25（Asia/Shanghai）
- **输入**：`docs/jev-laya-source-brief-2026-09-25.md`（事实核查简报）+ 本次对项目代码/配置/遥测/日志的实测核对
- **判据**（用户给定）：只有**成本可控**且**能带来明显使用价值/使用感受提升**的位置才值得做
- **性质**：评估与验证方案，不含代码改动

---

## 0. 结论速览

**一句话**：Jev/Laya 在本项目里**唯一不可替代**的定位是"**消费上下文的低成本结构化判断**"（够不够、支不支持、选哪个）；凡是需要**生成文本**的位置（关键词、查询改写、社区摘要、最终回答）都不适用；凡是**排序**位置，先被本地免费 CrossEncoder 方案占位。

| # | 候选位置 | 代码落点 | 单次成本 | 延迟增量 | 用户可感知 | 结论 |
|---|---|---|---|---|---|---|
| 1 | 离线评测判官（faithfulness/是否支持） | `src/evaluation/ragas_metrics.py` | ≈$0.0005/题 | 离线 | 间接（先有度量） | ✅ **先做**（零用户风险） |
| 2 | 回答前**证据充分性门控**（noul） | `answer_with_retrieved_text_only()` | ≈$0.00025/次 | +0.25~0.4 s | **强**（少编造/该拒答会拒答） | ✅ **值得做**（需先测基线） |
| 3 | 回答后**断言级引用核验**（noul） | 同上 + `sources` | ≈$0.0003/次 | +0.3~0.5 s | 中强（可信度可见） | ⚠️ **第二批**（默认只标注） |
| 4 | Chunk 精排（listwise/noul） | LightRAG `enable_rerank` | $0.0003~0.0004/次 | +0.4~0.6 s | 弱（间接） | ❌ 不做（被本地方案压制） |
| 5 | 关系重排 | `relation_optimizer.filter_relationships` | — | — | 弱 | ❌ 不做（先修本地路径） |
| 6 | 检索模式/难度路由（choice+score） | `src/retrieval/mode_router.py` | ≈$0.00002/次 | +0.25 s | 很弱（用户无感） | ❌ 暂不做（无证据） |
| 7 | 索引期三元组质检（noul） | `src/data_processing` / LightRAG 插入期 | 一次性 ≈$0.02 | 离线 | 弱 | ❌ 暂不做（无金标） |
| 8 | Laya 本地部署（任一位置） | 新进程 + 0.64~0.84 GB 权重 | 0（自付算力） | CPU 0.2~0.6 s/问 | 无 | ❌ 本阶段不做 |

**成本口径**（本项目实测）：单次查询检索上下文 ≈ **10.6k tokens**（9 个 chunk，`query_telemetry` 40 条均值），Chunk 约 1.2k tokens/条。Jev 按输入 $0.042/M 计费 → 送 6k tokens 判断一次 ≈ **$0.00025**；按 1 万次查询/月计，门控约 **$2.5/月**，核验约 **$3/月**。

---

## 1. 对简报的复核修正（本次直连 + 本机实测）

| # | 简报表述 | 复核结果 | 影响 |
|---|---|---|---|
| 1 | "Laya 需要 `transformers` 5.x、`torch` 2.14、`huggingface_hub` 1.x" | **不准确**。`pyproject.toml`/PyPI 0.3.20 的硬约束是 `torch>=2.0.0`、`transformers>=4.48.0`、`safetensors>=0.4.0`、`huggingface_hub>=0.20.0`；5.x/2.14 是官方 README 的**参考验证集** | 本项目 `rag` 环境（torch 2.13.0+cu130 / transformers 4.57.6 / sentence-transformers 5.6.0）**满足安装下限**，install 阶段不必然冲突；但 4.57 未被官方验证，需 smoke test。**"必须升级 transformers"这个劝退理由不成立** |
| 2 | "`laya-serve` 与 Jev 同 wire protocol" | ✅ 官方 README 原文：`laya.serve` 暴露 `POST /v1/systemone`，"…needs its `baseUrl` repointed; nothing else changes" | 可用**同一个客户端实现**接两家 → 试验成本极低、进退自由 |
| 3 | Jev 价格/上下文/限流（$0.042/M、64k/32k、250k tok/s、1200 rpm、`jev-latest`→`jev-1.13.0`） | ✅ 本次直连 `docs.typesafe.ai/models.md` 与 `api.md` 复核一致 | 成本模型可直接采用 |
| 4 | "GTX 16 系 + bge-m3 共存：未知/需实测" | ✅ 判断成立。本机 **GTX 1660 Ti，6 GB，compute_cap 7.5（无 bf16）** → Laya `[fast]`（bf16+CUDA graph）不可用；官方 BENCHMARKS 无 16 系数据 | Laya 只能走 stock 前向或 CPU 回退，且与 bge-m3 抢 6 GB 显存 |
| 5 | Laya 零样本 near-random、必须微调 | ✅ 与官方 "Honest limits" 一致（base 0.362/0.352 vs 随机 0.318、多数类 0.461） | Laya 在本项目**没有可用的一手能力**，除非先做微调（≈30k 问题、2×T4 4–5 h） |
| 6 | 简报未覆盖 | **本项目依赖已就绪**：`sentence-transformers 5.6.0`、`lightrag-hku 1.5.4`、`openai 2.44.0` 均在 `rag` conda 环境 | Jev 接入只需一个 HTTP 客户端；Laya 只需 `pip install laya` + 权重下载 |

---

## 2. 现状核对：本项目的 RAG 流程到底长什么样

**查询链路（实测，`.env` 当前配置）**

```
question
  → resolve_retrieval_mode()        # 纯规则启发式，不调 LLM
  → scholarly_keyword_fallback()    # 关键词抽取也不调 LLM（LightRAG 侧被 patch 成规则）
  → LightRAG aquery_data(mode)      # Milvus 向量 + Neo4j 图；bge-m3 本地嵌入
  → refine_retrieval_bundle()       # 关系：关键词重合过滤（+ 名义上的 CrossEncoder）
  → 可选社区摘要注入（本地 embedding 余弦，无 LLM）
  → answer_with_retrieved_text_only()  # 主 LLM 生成最终答案（唯一生成环节）
  → sources（chunk 级来源）
```

**关键实测数据**（来源：`data/meta/app_kv.sqlite3` `query_telemetry` 40 条，`data/logs/app.log`）

| 指标 | 实测值 | 说明 |
|---|---|---|
| 语料规模 | 5 篇文档 / 86 chunks / 2254 实体 / 3550 关系 | `data/lightrag_workdir/*` |
| 检索上下文 | 均值 **10.6k tokens**（9 chunks，≈1.2k/条） | `tokens_total_estimated` |
| 图检索空率 | 0% | `graph_empty` 全 False |
| Chunk 重排候选 | **0**（40/40 条） | `rerank_candidates=0` → chunk 精排**未运行** |
| 关系重排 | **从未成功执行** | 日志：`'ModelsSettings' object has no attribute 'rerank_batch_size'`（88 次关系优化中反复出现），异常被吞后仅保留关键词过滤 |
| 端到端延迟 | p50 **≈50 s**、p95 ≈75 s（40 题并发跑，含排队） | 40 题 / 19 min ⇒ 吞吐口径 ≈28 s/题 |
| 拒答机制 | 仅 prompt 指令（"材料不足请说明"） | **无独立门控、无断言核验** |

> **测量口径警告**：`chunk_truncation_rate` 与 `rerank_filter_rate` 在 `online_monitor.build_telemetry()` 里共用同一兜底公式；因为 `rerank_candidates=0`，遥测里 0.47 的"重排过滤率"其实是 **LightRAG token 截断率**（18→9），不是重排效果。任何基于该字段的 A/B 结论都会失真，需先修。

---

## 3. 逐位置评估

### 3.1 ✅ 位置 A：离线评测判官（推荐作为第一步）

- **做什么**：用 Jev 的 `noul` 对"答案句 × 检索材料"逐条判定"是否被支持"，替代/补充 `scripts/evaluate.py --ragas-llm` 里那个**与被测模型同源**的 LLM 判官。
- **为什么是它**：① 完全离线，零用户风险；② 同源 LLM 判官有自偏好偏差，独立判官更可信；③ 成本 ≈ 10.6k tokens/题 × 40 题 ≈ $0.02/次全量评估；④ **它产出的 faithfulness/拒答基线，是后面所有决定的依据**——没有它，位置 B/C 都是拍脑袋。
- **验证**：抽 20 条人工标注（支持/不支持/无关）与 Jev 判定比一致率；一致率 ≥85% 才把它当判官用，否则只当"第二意见"。
- **风险**：Jev 在部分基准上校准很差（独立评测：ECE 0.246、forced-uncertainty 仅 49.7%、16% 零概率），所以**只取相对排序/阈值，不直接相信它的概率数值**。

### 3.2 ✅ 位置 B：回答前的证据充分性门控（最有价值的一处）

- **做什么**：在主 LLM 生成之前，问一次 `noul`：「给定问题与以下检索材料，材料是否足以回答该问题？」。不足 → 直接回"材料不足"（或先换个模式重检索一次），不再让主 LLM 自由发挥。
- **为什么值得**：这正是当前系统**最薄弱、用户最能感知**的点。现在把"不要编造"写进 system prompt 就指望模型听话，而在 10.6k tokens 上下文 + 数值/公式密集的 FEC 语料上，模型往往会补齐细节。决策模型给出的**概率化门控**可以做成可标定策略。
- **成本**：state = 问题 + top-5 chunks ≈ 6k tokens → **$0.00025/次**；延迟 +0.25~0.4 s，占当前 p50 的 **<1%**。
- **插入点**：`src/retrieval/multimodal_answer.py::answer_with_retrieved_text_only()` 入口处（API/CLI 共用），或在 `GraphRAGRetriever.query()` 生成前。
- **失败时**：任何异常/超时 → 直接放行到现状（silent degradation），不得使查询失败。
- **先决条件（必须先测，否则不值得写代码）**：
  1. 扩出 **负样本集**（语料外问题，当前 `data/test/` 只有 20 条宏观问题，无负样本）；
  2. 测现状：负样本"编造答案率"、正样本"拒答率"。
- **通过线**（建议）：负样本误答率相对下降 ≥50%，正样本误拒率 ≤5%，P50 增量 ≤1 s，成本 ≤$0.001/次。

### 3.3 ⚠️ 位置 C：回答后断言级引用核验（第二批）

- **做什么**：把答案拆成断言，逐条 `noul`：「该断言是否被所引 chunk 支持？」。≥2 条不支持 → 触发一次重生成；否则在响应里标注"未支持句子"。
- **证据**：`erendikmenn/jev-rag-benchmark` 的 citation verification 消融（100 题子集）报告总成本 +3.7%、触发 1 次重生成。
- **成本**：10 条断言 × (断言+对应 chunk ≈700 tokens) ≈ **$0.0003/次**。
- **为什么排第二批**：重生成的代价是**再等 30–50 s**。若判官精度不够，用户会为误报付出巨大的等待成本。所以：默认只**标注**，且必须先在位置 A 把精度标出来。
- **用户感受**：取决于前端是否展示"该结论未被材料支持"。若不展示，价值接近 0。

### 3.4 ❌ 位置 D/E：Chunk 精排 / 关系重排

- **被谁压制**：`BAAI/bge-reranker-v2-m3` 已在 `requirements.txt`，`sentence-transformers` 已装，`scripts/download_reranker.py` 已有；关系路径本来就写了 CrossEncoder（只是坏了）。本地重排边际成本 $0、延迟几十毫秒、无数据出域、无外部限流。
- **Jev 的证据不覆盖这个对比**：独立评测比的是 Cohere Rerank 3.5（Jev 便宜 60.6%，但 nDCG 低 0.53 点、p50 慢 66 ms），**没有和 bge-reranker-v2-m3 比过**；而且那个结论是土耳其语、作者明确写"不可外推到其他语言/领域"。
- **结论**：先把**免费的本地重排**修好并测出来；只有本地方案在 6 GB 显存上确实跑不动、且实测证明 Jev 明显更强，才重新讨论。当前顺序反了。

### 3.5 ❌ 位置 F：检索模式 / 难度路由

- **现状**：`use_llm_router` 一路从 API（`auto_mode`）透传到 `resolve_retrieval_mode()`，但该参数**被忽略**；`ModeRouteResult` 的 `difficulty/complexity/context_richness` 三个字段恒为空串，而 API 文档还写着"附帶 mode_selection（難度/複雜度/選中模式）"。
- **为什么仍不建议现在做**：这是"**补上已宣称的能力**"，不是"新增用户价值"。模式选择的差异只在检索层，用户很难感知；而且当前 20 题评测集全是宏观问题（遥测里 40/40 是 `global`），**根本无法证明启发式会选错**。
- **若要做，先做这个**：造 40–60 题带人工模式标签的集合，测启发式 vs Jev 的一致率与下游关键词覆盖率；Jev 明显更优再上。成本本身可忽略（$0.00002/次）。

### 3.6 ❌ 位置 G：索引期三元组质检

- **成本极低**（3550 条关系 × ≈500 tokens ≈ 1.8M tokens ≈ **$0.075 一次性**），但**价值无法度量**：现有评测只有 20 条问题的关键词覆盖率，**没有实体/关系金标**，图谱精度改动无法验收。
- **结论**：等有图金标再说；不做。

### 3.7 ❌ 位置 H：Laya 本地部署

- **边际成本优势用不上**：Jev 在本项目的量级是 $2.5–5/月，Laya 要为此付出的固定成本（权重 0.64–0.84 GB、venv、CPU 0.2–0.6 s/问、与 bge-m3 抢 6 GB 显存、温度标定/微调）远高于节省额。
- **能力现状**：零样本 typed-decisions 近随机（0.362/0.352 vs 随机 0.318）；要可用需 ≈30k 问题微调（2×T4 4–5 h）。
- **它真正不可替代的场景是"数据不出域"**——本项目语料是公开 FEC 文献，不成立。
- **保留选项即可**：因为 wire protocol 与 Jev 相同，**用同一个客户端抽象**（`base_url` + `model` + `api_key` 三处配置）就能在未来一行配置切到 `laya-serve`，不必现在部署。

---

## 4. 必须先修的前置项（与 Jev/Laya 无关，但会污染结论）

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| 1 | 关系 CrossEncoder 重排从未生效 | `src/retrieval/relation_optimizer.py:116` 用了不存在的 `settings.models.rerank_batch_size`，异常被吞；`app.log` 反复出现该错误 | 补配置项（或 `getattr` 取默认），并加"重排失败率"遥测；README 的"CrossEncoder 关系重排"当前不成立 |
| 2 | Chunk 精排静默不执行 | `MODELS_RERANK_API_ENABLED=false` → `rerank_model_func=None`，LightRAG 仅 warning 后原序返回；`rerank_candidates=0`（40/40） | 明确开关语义：要么接本地 CrossEncoder，要么显式 `enable_rerank=False`，别再留"看起来在重排"的状态 |
| 3 | 遥测把 token 截断当重排过滤 | `online_monitor.build_telemetry()` 中 `rerank_filter_rate` 的兜底分支 | 拆成 `rerank_filter_rate` 与 `context_truncation_rate` 两个字段，否则所有重排类 A/B 无效 |
| 4 | 文档与实现漂移 | README "LLM 智能检索路由" vs 实际纯规则；README "query_metrics.jsonl" vs 实际 SQLite 表；`models/hub` 只有 bge-m3，无重排权重 | 对齐文档，或标注"计划中" |

---

## 5. 推荐执行路径（含预算与通过线）

**Step 1｜零成本前置（0.5–1 天）**
1. 修上表 1–3；
2. 建 `data/test/eval_negative.jsonl`：20–30 条**语料外/部分可答**问题（含"混合型"边界题）；
3. 用现有 `scripts/collect_eval_predictions.py` + `scripts/evaluate.py --ragas-llm` 跑出**基线**：负样本误答率、正样本拒答率、faithfulness。

**Step 2｜Jev 作为离线判官（预算 ≈$5，半天）**
4. 申请 TypeSafe key，固定 `jev-1.13.0`（不要用滚动别名）；
5. 写一个 `SystemOneClient`（只做 `POST /v1/systemone`，`state` + typed questions，超时/重试/降级齐全）；
6. 抽 20 条人工标注验证判官一致率；达标（≥85%）后用它重跑 Step 1 的全量评估。

**Step 3｜证据充分性门控（预算 ≈$2.5/万次查询，1–2 天）**
7. 阈值在自有数据上标定（不要照搬官方 confidence 定义，Jev `(n·p_max−1)/(n−1)` 与 Laya `1−归一化熵` 不同）；
8. Feature flag 上线，异常静默降级；把 `{决策, 概率, 阈值, 延迟, token}` 写入遥测；
9. A/B：负样本误答率、正样本误拒率、P50 延迟、成本。**不达标就回滚**。

**Step 4（可选）｜断言级核验**：只有在 Step 3 证明判官在自有语料上足够可靠、且前端愿意展示"未支持"标注时才做。

---

## 6. 明确不建议的用法（省得再评估一遍）

| 用法 | 为什么不行 |
|---|---|
| 用 Jev/Laya 做关键词抽取、查询改写、社区摘要、最终回答 | 两家都**不生成文本**（官方明说 `jev-1.13` 非生成模型、拼接 Choice 生成"不会好且非常慢"）；这些位置继续用主 LLM 或现有规则 |
| 用 Jev/Laya 做图片/多模态判定 | 输入仅文本（无图像/音频/视频） |
| 用 Jev/Laya 判"两个问题是否同一语义"（语义缓存） | 本地 bge-m3 向量已在 GPU 上、$0、更合适；且 Redis TTL 仅 300 s，收益本来就小 |
| 把 Jev 放在**无降级**的关键路径 | 限流"动态调整、不预告"，且失败请求是否计费未核实；必须可降级到现状 |
| 一次塞入 >32k state 或 >255 选项 | 官方硬限：32k（state+最长问）/64k（整请求）；≥256 选项直接 HTTP 400 |

---

## 7. 若接入：最小设计约束

1. **一个客户端，两家后端**：`SystemOneClient(base_url, model, api_key)`；Jev = `https://api.typesafe.ai/v1` + `jev-1.13.0`；Laya = 本机 `laya-serve` + `laya-multilingual`。切换只改配置。
2. **固定版本**：`jev-latest`/`jev-preview` 会漂移（当前都指向 1.13.0）；调过阈值后必须固定版本号，并记录响应里的 `model` 字段。
3. **预算护栏**：单次判断 state 上限（建议 ≤8k tokens）、月度预算上限 + kill switch、超时（建议 ≤1.5 s）后直接放行。
4. **降级语义**：任何异常 → 走现状路径，禁止把"判断失败"变成"查询失败"。
5. **可标定交付物**：每次判断必须落库（决策、概率、阈值、耗时、token），否则无法做阈值标定与回归。
6. **数据出域**：判断会把 chunk 文本与用户问题发到 TypeSafe 服务器（官方不用于训练，但确实出域；ZDR 需企业方案）。本项目为公开文献，风险低，但需显式确认。

---

## 8. 需要你拍板的三件事

1. **数据出域**：允许把检索到的 chunk 文本 + 用户问题发给 TypeSafe 吗？（若不允许 → 本项目内 Jev 全部方案作废，只剩 Laya 本地方案，而 Laya 需要先微调）
2. **预算**：Step 1–3 合计 ≈ **$5–10** 一次性（含试错），门控上线后 ≈ **$2.5/万次查询**。是否批准？
3. **是否保留 Laya 选项**：建议在 Step 2 的客户端里就留好 `base_url` 抽象（成本 0），但**本阶段不部署 Laya**。

---

## 附：本次核对使用的一手证据

| 证据 | 位置/命令 |
|---|---|
| 成本/限流/别名/上下文 | `https://docs.typesafe.ai/models.md`（2026-09-25 直连 200） |
| API 请求/响应结构 | `https://docs.typesafe.ai/api.md`（2026-09-25 直连 200） |
| Laya 依赖下限 | `https://pypi.org/pypi/laya/json` + `https://raw.githubusercontent.com/NandhaKishorM/laya/main/pyproject.toml`（0.3.20） |
| Laya 同协议/CPU 部署/校准 | `https://raw.githubusercontent.com/NandhaKishorM/laya/main/README.md`（`laya.serve` 段、Installation、Calibration） |
| 本机硬件 | `nvidia-smi`：GTX 1660 Ti / 6144 MiB / compute_cap 7.5 |
| 依赖版本 | `conda env list` → `/home/z9ian9/miniconda3/envs/rag`：torch 2.13.0+cu130、transformers 4.57.6、sentence-transformers 5.6.0、lightrag-hku 1.5.4 |
| 语料/遥测规模 | `data/lightrag_workdir/kv_store_*.json`、`data/meta/app_kv.sqlite3`（`query_telemetry` 40 条） |
| 重排失效证据 | `data/logs/app.log`：`'ModelsSettings' object has no attribute 'rerank_batch_size'`；`rerank_candidates=0`（40/40） |
| LightRAG 缺 rerank 函数时行为 | `lightrag/utils.py:4271-4279`（warning 后原序返回） |
