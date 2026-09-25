# Jev/Laya 落地 Step 1 结果：前置修复与零成本基线

- **日期**：2026-09-25（Asia/Shanghai）
- **输入**：`docs/jev-laya-integration-assessment-2026-09-25.md` §5 Step 1
- **性质**：代码修复 + 新数据集 + 实测基线；**未接入任何外部付费 API**（Jev 相关花费 $0）
- **结论**：Step 1 全部完成；**基线的误答率为 0%，使 Step 3 门控的通过线（误答率相对下降 ≥50%）不可达** —— 这直接影响是否值得继续投入预算（见 §5）

---

## 1. 一句话

评估文档把「回答前证据充分性门控」列为最值得做的位置，其依据是"当前靠 prompt 指望模型不编造"。**实测推翻了这一前提**：在 5 篇 FEC 文献语料上，现有严格 prompt 已能做到 23/23 负样本不编造、5/5 正样本不误拒。因此 Step 3 的价值需要用**别的口径**重新论证，否则就是花钱修一个没有症状的问题。

---

## 2. 交付物

| 类型 | 文件 | 说明 |
|---|---|---|
| 修复 | `config/settings.py`、`config/model_paths.py` | 补齐 `rerank_batch_size` / `reranker_local_path`；新增 `rerank_backend_available()` |
| 修复 | `src/storage/bge_rerank.py` | 新增 `build_local_rerank_model_func()`（本地 CrossEncoder 接 LightRAG） |
| 修复 | `src/storage/lightrag_init.py`、`src/retrieval/mode_config.py` | chunk 精排判定改为"后端是否真的存在"，无后端时显式 `enable_rerank=False` |
| 修复 | `src/retrieval/relation_optimizer.py` | 修 `rerank_batch_size`（原每次都 AttributeError） |
| 修复 | `src/evaluation/online_monitor.py` | `rerank_filter_rate` 不再复用 chunk 截断率 |
| 修复 | `scripts/download_reranker.py`、`scripts/collect_eval_predictions.py` | 修两处死引用，使脚本可运行 |
| 数据 | `data/test/eval_negative.jsonl` | 30 条：12 语料外 / 8 边界 / 5 伪前提 / 5 语料内对照（含 reference） |
| 数据 | `data/test/eval_negative_hard.jsonl` | 8 条"相邻但缺失"硬负样本 |
| 脚本 | `scripts/score_refusal.py` | 误答率 / 误拒率评分（含 suspect 复核标记） |
| 测试 | `tests/test_rerank_semantics.py`、`tests/test_refusal_scoring.py` | 12 条新测试；全仓 89 条全绿 |
| 结果 | `data/test/eval_negative_report.json`、`eval_negative_hard_report.json` | 评分报告（含逐题明细） |

**未提交任何 git commit**（工作区另有他人在做的 community 改动，按交接约定不与用户确认前不提交）。

---

## 3. 修掉的缺陷（含实测证据）

### 3.1 关系 CrossEncoder 重排从未生效（评估文档 §4 项 1）
- 根因：`settings.models.rerank_batch_size` 字段不存在 → `AttributeError` → 被 `except` 吞掉，只剩关键词过滤。
- 实证（历史日志）：`关系重排失败，保留关键词过滤结果: 'ModelsSettings' object has no attribute 'rerank_batch_size'`（88 次）。
- 修复：补字段（`MODELS_RERANK_BATCH_SIZE`，默认 32）+ 批量写法整理。

### 3.2 Chunk 精排的"假能力"（评估文档 §4 项 2）——真正的根因比文档写的更深
- 文档说根因是 `MODELS_RERANK_API_ENABLED=false`。实际根因是 **`mode_config.py` 用「sentence-transformers 是否安装」判断是否开启精排**：
  ```python
  if not settings.rerank_runtime_available():   # 只检查包是否装
      param = replace(param, enable_rerank=False)
  ```
  本机装了 `sentence-transformers` → 判定"可用" → `enable_rerank=True`，而 `rerank_model_func=None` → LightRAG 只 warning 后**原序返回**。
- 修复：新增 `rerank_backend_available()`（线上 API 配置完整 **或** 本地有权重），三处调用点（chunk 精排门、关系重排门、查询参数）统一用它；无后端时显式 `enable_rerank=False`，并在启动日志明确说明与启用方法。
- 顺带把本地 CrossEncoder 接进了 LightRAG（`rerank_model_func`），**下载权重即生效**，无需改开关：
  `python scripts/download_reranker.py`（约 2.3 GB，本机尚未下载）。

### 3.3 遥测口径混淆（评估文档 §4 项 3）
- 根因：`rerank_filter_rate` 在 `rerank_candidates==0` 时用 `(merged-final)/merged` 兜底 —— 与 `chunk_truncation_rate` 完全同式。
- 实证（历史日志）：`chunk_trunc=0.609 rerank_filter=0.609`、`0.400/0.400`、`0.550/0.550`，逐行相同。
- 修复：去掉兜底分支；rerank 未跑就是 0。**实测验证**（本次运行写入的遥测）：`chunk_trunc` 均值 0.3097，`rerank_filter` 全为 0，`rerank_candidates>0` 的行数 = 0 —— 口径正确反映"重排未运行"。

### 3.4 附带发现（评估文档未列，均属"引用不存在的东西"同一族）
| # | 位置 | 问题 | 处理 |
|---|---|---|---|
| 1 | `scripts/download_reranker.py` | `s.models.rerank_model_name` 不存在 → 脚本第一步就崩 | 已修（改用常量） |
| 2 | 同上 | `apply_settings_to_environ(s, offline=False)` 该函数没有 `offline` 参数 → TypeError | 已修 |
| 3 | `scripts/collect_eval_predictions.py` | `retriever.query(..., system_prompt=...)` 该参数不存在 → **基线采集脚本根本无法运行** | 已修（改走线上默认 prompt，基线才可比） |
| 4 | `config/model_paths.py:95` | `apply_models_to_environ()` 全仓无定义、无导入 → `mineru_subprocess_environ()` 必抛 NameError；被 `convert.py` 的 `except` 吞掉，静默退回默认 env | **未修**（属 PDF 转换子系统，原函数语义已丢失，需你确认后再补） |
| 5 | `tests` 之外 | `import ragas` 直接失败：ragas 0.4.3 需要 `langchain_community.chat_models.vertexai`，而 langchain-community 已升到 0.4.2 删除该模块 | **已修**（见 §7） |

---

## 4. 基线结果（实测）

### 4.1 拒答/误答（`scripts/score_refusal.py`）

| 数据集 | 类别 | 题数 | 拒答率 | 误答率 |
|---|---|---:|---:|---:|
| Set A（30） | 语料外 out_of_corpus | 12 | **100%** | 0% |
| Set A | 伪前提 false_premise | 5 | **100%** | 0% |
| Set A | 边界 partial | 8 | 100%（先声明不足，再给相邻材料） | 不计入 |
| Set A | **负样本合计** | **17** | **100%** | **0.0%** |
| Set A | 语料内对照 in_corpus | 5 | **0%**（误拒率 0%） | — |
| Set B（8 硬） | 负样本合计 | 6 | **100%** | **0.0%** |
| **合计** | **负样本** | **23** | **100%** | **0.0%** |

**人工复核**（不是只看正则）：38 条答案逐条读首段确认拒答真实性；5 条对照题的关键事实逐项核对通过（Plotkin 型分解/置换/候选选择；RM 变换消除线性方程组；投影-递归-聚合；对称函数；Cantor 域塔 + LCH）。8 条 `suspect`（拒答短语 + 长答案）逐条查看，均为"先说不确定、再给语料内相邻信息"的**正确**行为，无编造。

**标注纠错**：Set B 的 h03 我最初标成"语料外"，实测回答给出语料内的运行时间数据（RPA 对 RM(8,2) 为 4.3 ms、Dumer 为 0.85 ms）—— 说明我的初标有误，已改标为 partial 并重算。这提醒：负样本集的标注本身需要复核，不能只看自己写下的 `expect`。

### 4.2 延迟与上下文（本次 38 题串行运行的遥测）
- 检索上下文：均值 **10.3k tokens**（p50 10.7k，max 11.1k）—— 与评估文档的 10.6k 一致。
- 含生成阶段的遥测行：p50 ≈ **12 s**、p95 ≈ 15 s、max ≈ **31 s**。
- 串行吞吐约 10–19 s/题；比交接文档记录的 28 s/题、p50 50 s 更快，原因是**拒答答案很短**且串行无并发排队。
- 图检索空率 0%；模式分布 local 28 / hybrid 18 / global 5。

---

## 5. 对后续决策的直接影响（重点）

### 5.1 Step 3 门控的通过线不可达
评估文档建议的通过线是"负样本误答率**相对下降 ≥50%**、正样本误拒率 ≤5%"。基线已经是 0% 误答 / 0% 误拒 —— **没有可下降的空间**，A/B 拿不到可解释的差异。继续做 Step 3 的前提必须换掉，可选方向：
1. **换口径**：从"少编造"改为"引用可核验"（断言级核验，即评估文档的位置 C），或"用户可见的可信度标注"；
2. **换难度**：造**对抗性负样本**（把语料里真实存在但被打散的细节拼成伪问题、跨论文张冠李戴、要求精确数值而语料只有趋势），当前两套负样本（含"相邻但缺失"硬集）都没能把它问出编造，说明需要专门构造；
3. **接受结论**：当前语料（5 篇、86 chunks、纯公开文献、问题集中）下，prompt 级约束已足够，**不投入 Step 3**，把预算留给位置 C 或不做。

### 5.2 判官能力已修复（原为硬缺口）
- 本节原先的结论是"ragas 在本环境根本跑不起来、`--ragas-llm` 静默跳过、`ragas_evaluated=0`"。**该问题已于同日修复**，详见 §7。修复后 RAGAS 三项指标可正常产出，且失败会被显式标记而不再伪装成 0 分。
- **由此 Step 2 的定位回到评估文档原本的论证**：判官并非"没有"，而是"与被测模型同源、有自偏好偏差"。Jev/Laya 的价值是**独立判官**，不是"唯一能跑的判官"——请按这个口径权衡预算。
- 该 harness 对负样本的**结构性限制仍在**：`runner.py` 的 RAGAS 路径要求每行有 `reference`（金标答案），而负样本的正确行为是拒答、没有 reference。因此 faithfulness 只在"可答题"子集上有效（本语料 5 条对照题）；负样本用同一报告里的 `refusal` 区块度量（见 §7）。
- **判官噪声水平**（复现 Step 3 通过线时必须知道）：同一批 5 条数据、同配置重复运行，`faithfulness_mean` 实测波动约 **±0.007**（temperature=0 后：0.9732 / 0.9665；temperature 未固定前曾达 ±0.04）。**小于 0.01 的差异不可当作信号**。

### 5.3 本地免费重排是当前最高性价比的动作（$0）
按评估文档 §3.4 的判断"先把免费的本地重排修好并测出来"，代码侧已就绪，只差一步：
```
python scripts/download_reranker.py     # 约 2.3 GB
```
风险已明确：GTX 1660 Ti 6 GB，bge-m3 已占 2.28 GB，再加 bge-reranker-v2-m3（fp32 约 2.3 GB）→ 约 4.6 GB，**可能紧张**；且 rerank 遥测走 ContextVar，真实启用后需确认 `rerank_candidates` 能落到遥测（若落不到，说明跨 task 丢失，需要改成显式传参）。这两点都需要一次实测才能确认 —— 这是接 Step 2 前值得先做的零成本验证。

---

## 6. 复现命令

```bash
PY=/home/z9ian9/miniconda3/envs/rag/bin/python
cd /home/z9ian9/myproject/rag-fec

# 1) 起服务（Milvus/Neo4j/etcd/minio/Redis）
docker compose --profile redis up -d

# 2) 采集预测（30 条 + 8 条硬负样本）
$PY scripts/collect_eval_predictions.py --input data/test/eval_negative.jsonl \
      --out data/test/eval_negative_predictions.jsonl
$PY scripts/collect_eval_predictions.py --input data/test/eval_negative_hard.jsonl \
      --out data/test/eval_negative_hard_predictions.jsonl

# 3) 拒答/误答评分
$PY scripts/score_refusal.py --input data/test/eval_negative_predictions.jsonl \
      --out data/test/eval_negative_report.json

# 4) 回归测试
$PY -m pytest tests/ -q
```

---

## 7. 待你拍板（与评估文档 §8 相同，但现在有了数据）

1. **数据出域**：是否允许把 chunk 文本 + 用户问题发给 TypeSafe？
   —— *新增输入*：基线误答率为 0，出域换来的收益从"减少编造"变成"可核验/可标定"，风险收益比变差了。
2. **预算**：Step 1–3 合计 ≈ $5–10 是否批准？
   —— *新增输入*：Step 3 的通过线不可达，建议**先不要批 Step 3 的预算**；若批，建议改投 Step 2（**独立**判官；注意自动判官已恢复，见 §7）或位置 C。
3. **Laya 定位**：是否同意"保留 `base_url` 抽象、本阶段不部署 Laya"？
   —— 立场不变，成本为 0。

**我的建议顺序**（$0 先行）：
① 下载本地 reranker 并实测（显存 + 遥测两项验证）→ ② 若 ① 通过，本地重排即可作为"排序类增强"的免费方案，无需 Jev；
③ 在此之后重新决定是否为"判官/门控/核验"付钱，且需要先把 Step 3 的通过线改成一个基线非零的指标。

---

## 7. 评测机制修复（同日追加）

原状：`import ragas` 直接失败 → `evaluate.py --ragas-llm` 静默跳过 → `ragas_evaluated=0`，等于**没有任何自动判官**。修复分五步，每步都有实测依据：

| # | 问题 | 根因 | 处理 |
|---|---|---|---|
| 1 | `import ragas` 失败 | ragas 0.4.3 顶层导入 `langchain_community.chat_models.vertexai`；该模块在 langchain-community **0.4.2** 被删（0.4.0/0.4.1 仍在） | 装 `langchain-community==0.4.1`（其全部依赖本已满足，`--no-deps` 替换，环境仅此一包变动）；并钉进 `requirements.txt` 防复发 |
| 2 | 指标全为 0.0 | 判官 client 未带 `x-opencode-session` 头 → 网关 400 MissingSessionID；ragas 用 `raise_exceptions=False`，失败被 `NaN→0.0` 吞掉 | `build_ragas_llm` 复用项目既有的 `session_headers()`（与 `client_cache.py` 同一套） |
| 3 | 失败伪装成 0 分 | 整行/单指标失败都变成 0.0，与"答案确实不忠实"无法区分 | 逐指标标记 `<metric>_scored` + 报告输出 `scored_rows` / `scored_by_metric`，均值只统计真正算出来的行；失败时输出告警 |
| 4 | faithfulness 有 2/5 失败 | 判官输出被 `max_tokens` 截断（`IncompleteOutputException`）；该网关模型会用 completion token 做推理 | 新增 `EVAL_RAGAS_MAX_TOKENS`（默认 16384，是上限而非目标值）；8192 实测 3/5，16384 实测 5/5 |
| 5 | 判官随机、评测不可复现 | 未设温度 | 判官固定 `temperature=0`；实测波动由 ±0.04 收到 ±0.007（该模型仍非完全确定） |

**顺手补上的完整性**：拒答/误答口径从独立脚本提为公共模块 `src/evaluation/refusal_metrics.py`，`scripts/evaluate.py` 在行带 `label` 时自动输出 `refusal` 区块——现在一条命令即可拿到检索指标 + RAGAS + 拒答率，且两处口径必然一致（原先两套工具容易给出不同数字）。

**修复后的基线**（同一批 30 条数据，一条命令）：

```
ragas:    scored_by_metric = {context_recall: 5, context_precision: 5, faithfulness: 5}
          context_recall = 1.000, context_precision = 1.000, faithfulness ≈ 0.97
refusal:  负样本误答率 = 0.0, 正样本误拒率 = 0.0（23 负 / 5 正）
```

**回归**：全仓 96 条测试通过（新增 4 条锁住依赖可导入、会话头、失败标记、报告口径）；`langchain_community` 0.4.1 与 `langchain-core` 1.4.9 / `langchain-openai` 1.3.4 / `lightrag-hku` 1.5.4 互操作正常。

**遗留（不影响当前使用）**：
- `ragas.metrics` 导入有 deprecation 警告（ragas 1.0 会迁到 `ragas.metrics.collections`）；项目代码里已注明 collections 版与 `evaluate` 的 isinstance 检查不兼容，故暂留 v1 单例。
- 判官噪声 ±0.007 是**小样本（5 行）**估计；要做 Step 3 的 A/B 通过线判定，应先把对照集扩到几十条再定阈值。
