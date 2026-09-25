# Jev（TypeSafe AI）与 Laya（ConvAI Innovations）事实核查简报

- **核查日期**：2026-09-25（Asia/Shanghai）
- **范围**：仅事实核查与来源整理，不含接入建议、不含代码改动。
- **访问情况**：本次核查中 `docs.typesafe.ai`、`api.github.com`、`huggingface.co`、`openrouter.ai` **均直连成功**（HTTP 200）。因此标注为「官方」的条目是本次抓取的官方页面原文；标注为「官方快照」的是第三方仓库转引的官方页面副本；标注为「独立」的是第三方自建/自费复现。
- **数字口径提示**：文中每个关键数字后给出链接；无法核实的统一写「未核实」，不写确定语气。


## 0. 一页速览

| 维度 | Jev 1.13（TypeSafe AI） | Laya（ConvAI Innovations） |
|---|---|---|
| 形态 | 托管 API，闭源权重（同一权重服务所有账号） | 开源权重（Apache-2.0），本地/离线可跑 |
| 输出 | typed decisions（choice/noul/score + 概率），**不生成文本** | 同构 typed decisions，**非自回归、不生成文本** |
| 上下文 | 64k/请求；`state`+最长单问 32k | `laya` 512；`laya-multilingual`/`typed-decisions` 1024；多语可 `max_len=8192` |
| 价格 | $0.042 / 1M **输入** token，输出免费 | 模型免费，自付算力/工程成本 |
| 公开延迟（独立） | p50 ≈236–276 ms（托管，单请求） | T4 p50 32.8–39.5 ms（1 问）；CPU p50 193–584 ms（1 问） |
| 零样本能力 | 强（AG News 0.910、Banking77 0.870，300 例独立 pilot） | 英文近随机（typed-decisions 0.362/0.352 vs 随机 0.318），须微调 |
| 主要风险 | 数据出域、别名漂移、限流、无解释、上下文/语言相关 | 零样本差、过度自信、中文/长文不稳、>20 选项退化 |
| 许可/商用 | 商业 API（无公开权重） | Apache-2.0，可商用 |


## 1. Jev 是什么

### 1.1 System One / typed decisions
- Jev 是 TypeSafe 的旗舰模型、第一个 System One 模型：输入 `state` + 若干 **typed questions**，直接返回结构化答案与概率分布，**没有文本生成、没有解析环节**（官方原文："No text generation, no parsing."）— [官方 introduction](https://docs.typesafe.ai/introduction)，2026-09-25 直连核对。
- 官方把「大模型生成文本再解析成结构化」称为 mismatch；System One 的定位是「给代码直接消费的判断」— [官方 introduction](https://docs.typesafe.ai/introduction)（同上）。
- 官方明确：`jev-1.13` **不是为生成文本训练的**；强行用串联 Choice 拼文本"不会好且会非常慢"，需要生成请换生成式模型 — [官方 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)，2026-09-25 直连核对。

### 1.2 三类原语（官方）
| 原语 | 语义 | 返回 | 备注（官方原文要点） |
|---|---|---|---|
| Choice | 从选项集中选一个 | `choice`、`probabilities`、`confidence` | 单问**最多 255 个选项**；`probabilities` 之和为 1 |
| Noul | 判断一句话是否成立 | `noul`（0–1 的 P(是)） | **没有单独的 `confidence`**，概率即答案兼置信 |
| Score | 在有序 rubric 上打分 | `score`、`probabilities`、`confidence` | `confidence` 由分布形状导出 |

- 三类问题**可混合在同一个 API 调用**里 — [官方 introduction](https://docs.typesafe.ai/introduction)。
- 来源：[官方 primitives/choice](https://docs.typesafe.ai/primitives/choice)、[官方 primitives/noul](https://docs.typesafe.ai/primitives/noul)、[官方 confidence](https://docs.typesafe.ai/confidence)，均 2026-09-25 直连核对。

### 1.3 API / SDK / 托管方式
- 端点：`POST https://api.typesafe.ai/v1/systemone`，`Authorization: Bearer <API_KEY>`；另有 `GET /v1/models` 列出可用模型/别名 — [官方 api](https://docs.typesafe.ai/api)、[官方 models](https://docs.typesafe.ai/models)，2026-09-25 直连核对。
- 官方 SDK：Python `typesafe-sdk`（`uv add typesafe-sdk`，MIT，仓库 2026-09-21 HEAD `0ffd094c`）— [typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python/blob/main/README.md)；文档同时给出 JS SDK `@typesafe-ai/sdk`（[官方 models](https://docs.typesafe.ai/models)）。另有「用 LLM 复刻 System One 接口」的 `system-one-adapter-python`，用途写明是**对比 TypeSafe 与 LLM 的成本/速度/智能**（对标实现，非 Jev 本体）— [system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python/blob/main/README.md)。
- 托管属性：Jev 只在 TypeSafe 服务器上运行、**非开放权重**，每次调用都会把 `state`（可能包含业务数据）发到其服务器；企业客户可谈零数据保留（ZDR）。官方 models 页确认"Jev is not trained on customer requests or responses"，但**数据仍会出域** — [官方 models](https://docs.typesafe.ai/models) 的 Data handling 段（官方原文）；托管/闭源表述见 [agent-squad 对官方文档的转引](https://github.com/2FastLabs/agent-squad/blob/3d67629ca564eaf35ec7779115c45a42ee23a2bd/docs/src/content/docs/classifiers/built-in/jev-classifier.mdx)（**官方文档快照，经 2FastLabs 仓库转引，2026-09-25 核对**）。
- 官方 Agent Skill 仓库面向编码代理，强调"以在线文档为准"：`docs.typesafe.ai/llms.txt`、页面可加 `.md` 取 Markdown — [typesafe-ai/skills](https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md)，2026-09-25 核对。

### 1.4 模型别名与版本固定
- 官方别名表（2026-09-25 直连）：`jev-latest` → `jev-1.13.0`（最新**稳定**发布，也是官方 SDK 默认名）；`jev-preview` → `jev-1.13.0`（当前**无**预览构建，故指向同一模型）— [官方 models](https://docs.typesafe.ai/models)。
- 官方建议：**别名会随新版本移动**，若已按某一版调过置信阈值，应固定版本号；响应的 `model` 字段回显实际回答的版本 ID — [官方 models](https://docs.typesafe.ai/models)。
- 官方版本号写法为 `jev-1.13.0`；官方 jaggedness 页的示例代码里出现 `TypeSafeClient(model="jev-1.13")`，即 `jev-1.13` 这类版本化 ID 也被接受（官方 models 页："Versioned IDs such as `jev-1.13.0` are accepted ... whether or not they appear in the list"）— [官方 models](https://docs.typesafe.ai/models)、[官方 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)。
- 第三方转引一致：`jev-latest`/`jev-preview` 为移动别名，调过阈值后应固定 `jev-1.13.0` — [pydantic-ai docs/models/typesafe.md](https://github.com/pydantic/pydantic-ai/blob/08f4cc2757d107a37d6697b976cbaa034830a6c9/docs/models/typesafe.md)、[TanStack/ai docs/adapters/typesafe.md](https://github.com/TanStack/ai/blob/7404539c55acf9d7d0b0f7ccb2f16f4dcd75acdb/docs/adapters/typesafe.md)、[bifrost typesafe.mdx](https://github.com/maximhq/bifrost/blob/cefde78ba0e574f03da111e3fdcb9f4286e5ccc4/docs/providers/supported-providers/typesafe.mdx)（**官方文档快照，经第三方仓库转引，2026-09-25 核对**）。


## 2. Jev 上下文与请求限制

| 项目 | 官方数值（2026-09-25 直连） | 来源 |
|---|---|---|
| 单请求总上下文 | **64k tokens**（`state` + 全部 questions 合计） | [官方 models](https://docs.typesafe.ai/models) |
| `state` + 最长单个 question | **32k tokens** | [官方 models](https://docs.typesafe.ai/models) |
| 输入形态 | 仅文本：字符串 / JSON 对象 / 文本数组；无图像、音频、视频 | [官方 models](https://docs.typesafe.ai/models) |
| 单 Choice 选项上限 | 255 个选项 | [官方 primitives/choice](https://docs.typesafe.ai/primitives/choice) |
| 速率限制 | **250,000 tokens/秒** 与 **1,200 请求/分钟**；官方注明"动态调整、可能不预告变化" | [官方 models](https://docs.typesafe.ai/models) |

- **32k/64k/25k 不一致的来源**：官方口径只有 32k（state+最长问）与 64k（整请求）；**25k 不是官方限制**，而是第三方为留余量自设的工程预算：
  - 出处分两类：NousResearch 评测记录 "Jev's 32K window forces the whole history into 25K tokens"（[SCORECARD-2026-09-19-jev.md](https://github.com/NousResearch/hermes-agent/blob/8623cd4a8403f020090c9c3e8feae943b4dac55e/evals/compaction/results/SCORECARD-2026-09-19-jev.md)）；另有项目自设 30k 预算（`maxRequestTokens`，注明 "under Jev's 32k request limit"）并压缩到 ~25k state（[fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction/blob/e3f262a7f4d42bd8dd32ced30d26176f7cb545b0/README.md)、[jev-compactor JEV-API.md](https://github.com/edwardyen724-g/jev-compactor/blob/0a9b64e8481fac186e3dcb478450d8cf133d2772/docs/JEV-API.md)）。均为独立工程记录。
- **一次请求能否并行回答多个问题**：能。官方："Jev ingests the `state` once and **evaluates every question against it in parallel**"；"Adding questions barely changes the response time"，且各问题彼此独立（互不可见）— [官方 models](https://docs.typesafe.ai/models)、[官方 introduction](https://docs.typesafe.ai/introduction)、[官方 fan-out](https://docs.typesafe.ai/patterns/fan-out)，2026-09-25 直连核对。
  - 数量上限：官方未给 questions 个数硬上限；第三方实测"200 questions in one call worked (161 ms)"，并记录 state+最长问超 32k 会返回 HTTP 400 `max_tokens_exceeded` — [jev-compactor docs/JEV-API.md](https://github.com/edwardyen724-g/jev-compactor/blob/0a9b64e8481fac186e3dcb478450d8cf133d2772/docs/JEV-API.md)（独立实测，非官方）。
- **批处理/分片行为**（第三方封装库的官方文档）：
  - `jev-reranker`：超长候选列表**自动切分**，按字符/tokenizer/自定义长度计预算；listwise 分片使用不同共享上下文，**不保证与不分片同分**；pairwise 为 n(n−1)/2 次请求；单文档/单对放不下会抛 `ContextLimitError`；429/5xx 指数退避并遵守 `Retry-After`（上限 60s），鉴权失败不重试 — [hotchpotch/jev-reranker](https://github.com/hotchpotch/jev-reranker/blob/main/README.md)，2026-09-25 核对。
  - `jev-recall`：大记忆库**切成多请求并发**发送；同一 request 里每个记忆一条 yes/no — [samdotmak/jev-recall](https://github.com/samdotmak/jev-recall/blob/main/README.md)，2026-09-25 核对。
- 限流与重试：超限返回 `429 Too Many Requests`（另有 `529 Overloaded`）；官方 SDK **默认自动退避重试**并在响应带 `retry-after` 时遵守；HTTP 直连需自行退避 — [官方 api](https://docs.typesafe.ai/api)、[官方 models](https://docs.typesafe.ai/models)，2026-09-25 直连核对。第三方快照补充了同样的 250k tokens/s + 1,200 req/min 与"动态调整、可能不预告"——[PostHog typesafe README](https://github.com/PostHog/posthog/blob/ba7e9c1ba6f0222582253cade83ce00bc272e87a/posthog/egress/typesafe/README.md)（**官方文档快照，经 PostHog 仓库转引，2026-09-25 核对**）。


## 3. Jev 价格与计费口径

| 口径 | 数值 | 来源 |
|---|---|---|
| 官方直连价 | **$42 / Btok = $0.042 / 1M 输入 token** | [官方 models](https://docs.typesafe.ai/models)，2026-09-25 直连 |
| 输出 token | **免费**（"Output tokens are free."） | 同上 |
| 计费单位 | 按**输入** token 计费；1 Btok = 10 亿 token，1 Mtok = 100 万 token | 同上 |
| OpenRouter 网关价（同日 API 读取） | prompt `$0.000000042`/token = **$0.042/M**，completion `0`（同等价、未见网关加价） | [OpenRouter model endpoints API](https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints)，2026-09-25 |
| OpenRouter 侧上下文/输出上限字段 | `context_length: 32000`、`max_completion_tokens: 28800`（注意：网关暴露的是 **32k**，与官方 64k 整请求上限不同口径） | 同上 |
| OpenRouter 实际解析版本 | 请求 `typesafe/jev-1.13` → 解析为 `typesafe/jev-1.13-20260917`，provider = TypeSafe | [jev-rag-benchmark README](https://github.com/erendikmenn/jev-rag-benchmark/blob/main/README.md)；该仓库 manifest 记录 `price_verified_on: 2026-09-19` |

- **实测成本样例（可复算，独立）**：XQuAD-TR 全量 1,044 问、每问 20 候选，Jev batch Noul 重排**总 $0.410866**（≈$0.000394/问）— [comparison.md](https://github.com/erendikmenn/jev-rag-benchmark/blob/7c7fd8afc30a6fe4e54236e6d65835858cf7393a/reports/generated/xquad-tr-hybrid-fullfusion20-a-d-o/comparison.md)；记忆检索 238 条/18 请求 **$0.00044/请求**（[jev-recall](https://github.com/samdotmak/jev-recall/blob/main/README.md)，2026-09-19，`jev-1.13.0`）；banking 77 类 **$0.07/1,000 次决策**（[nibzard](https://github.com/nibzard/decision-model-benchmark/blob/main/README.md)，自费 $28.34 总预算）。
- **重试是否计费：未核实。** 官方 models/api 页只说明"按输入 token 计费、输出免费"和 429/529 要退避重试，**未找到**关于失败请求、被限流请求或重试是否计费的说明；需要以实际账单或向厂商确认（[官方 models](https://docs.typesafe.ai/models)、[官方 api](https://docs.typesafe.ai/api)，2026-09-25 直连核对）。


## 4. Jev 的公开评测与性能（区分厂商宣称 / 独立测量）

### 4.1 延迟（独立）
| 来源 | 条件 | 结果 |
|---|---|---|
| AbdelStark/jev-benchmarks pilot（独立，自费） | 300 条 held-out，4/6 标签；Jev 为托管调用（从法国发起），GLiNER 本地 M4 Max CPU | Jev p50 **236–256 ms**（4/6 标签）、**246 ms**（72 标签）；GLiNER ~44 ms（4/6 标签）、296 ms（72 标签） |
| nibzard/decision-model-benchmark（独立，自费 $28.34） | 五个 suite、60 个 cell、冻结协议 | Jev p50 **264–276 ms**，且从 2 到 255 选项**基本不变**；比 Cerebras 上的 gpt-oss-120b 快约 1.2×，比 thinking 模式模型快 10–16× |
| Laya 官方 BENCHMARKS 转引 | 引用上述两个独立来源 | "Jev independently measured at 236–276 ms p50" |

来源：[AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks/blob/main/README.md)、[nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark/blob/main/README.md)、[Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)（均 2026-09-25 核对）。

### 4.2 RAG rerank 独立评测：`erendikmenn/jev-rag-benchmark`（XQuAD-TR）
- 协议：**1,044 个唯一土耳其语 XQuAD 问题**、240 篇语料；BM25 + `baai/bge-m3` 混合检索，全量融合后给每个 reranker **同一批 top-20 候选**；回答模型只看最佳 5 篇；Jev 请求为 `typesafe/jev-1.13`（解析 `...-20260917`）；生成阶段关闭以隔离重排效果 — [README](https://github.com/erendikmenn/jev-rag-benchmark/blob/7c7fd8afc30a6fe4e54236e6d65835858cf7393a/README.md)。

| 重排方法 | Recall@5 | Gold in top-5 | nDCG@10 | MRR@10 | 重排 p50 | 总重排成本 | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| 不重排（hybrid 顺序） | 97.605% | 1,019/1,044 | 94.314% | 93.194% | 0.0 ms | $0.000000 | 全量 |
| **Jev 1.13 batch Noul** | **99.425%** | **1,038/1,044** | 98.097% | 97.637% | 532.0 ms | **$0.410866** | 全量 |
| Cohere Rerank 3.5 | **99.425%** | **1,038/1,044** | **98.626%** | **98.348%** | **466.1 ms** | $1.044000 | 全量 |

- Jev 与 Cohere 把相同数量的 gold 拉进 top-5；Jev 便宜 **60.6%**，但 nDCG 低 0.529 点、p50 慢 65.9 ms；相对不重排，Jev 多召回 19 个 top-5 + 3.783 nDCG 点，零 API fallback — 同上（数据源：[comparison.md](https://github.com/erendikmenn/jev-rag-benchmark/blob/7c7fd8afc30a6fe4e54236e6d65835858cf7393a/reports/generated/xquad-tr-hybrid-fullfusion20-a-d-o/comparison.md)）。
- **claim/citation 判定**：该仓库的 "Jev citation verification" 消融（100 题子集）报告"最终所有被引 claim 均通过验证、触发 1 次重生成、总成本 +3.7%"；属于**部分样本**（100/1,044），作者明确标注非全量分数 — [README §4](https://github.com/erendikmenn/jev-rag-benchmark/blob/7c7fd8afc30a6fe4e54236e6d65835858cf7393a/README.md)。
- **作者自陈的局限**：Jev 1.13 在英语最强，**土耳其语结果不可外推到其他语言/领域**；候选顺序会显著影响 Jev 分数（同一问题换序：Spearman 秩相关均值 0.262、top-5 Jaccard 0.288、gold top-5 成员在 3/50 题上变化）；自动 F1 / 引用判定只是代理指标，非人工事实性裁定 — 同上。
- 同一仓库给出的 Jev 策略消融（子集，非全量）：pointwise Jev 无召回提升但延迟 +80.8%、成本 +56.5%；3 阶排列集成 +0.238 nDCG 点但成本 ×3；full-corpus 层级式 Jev 被否（100 题上 Recall@5 76%，劣于 BM25 top-5 的 97%）— 同上。

### 4.3 其它独立证据与反例
- **选择性自动化/校准**：AG News 4 类 0.910、Banking77/BTZSC 72 类 0.870，均显著优于 GLiNER2.5（0.700/0.610）；但 DAIR Emotion 6 类准确率差异不显著（0.480 vs 0.440），且 Jev 校准明显更差：Brier 0.846 vs 0.668、NLL 5.588 vs 1.381，**16% 的样本对真实标签给出 0 概率** — [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks/blob/0d610cc53e79bcbec691312b0c4adb4a0e371642/README.md)（独立 pilot，300 例）。
- **负面结论**：仅 Jev "自认不确定"的比例异常低（forced-uncertainty 题上 49.7%，同批 LLM 为 97.3–100%）；ECE 0.246 为该基准最差（LLM 为 0.039–0.122）；选项置换会改变 **13%** 的选择；选项数 ≥256 时直接 HTTP 400 `Too many choices.`（254/255 时 100% 正确）— [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark/blob/8404980de041f38c2f5cc84cc4a617345091bf40/README.md)（独立）。
- **结构化决策 vs LLM**：238 条记忆、18 个请求的诊断基准中，Jev pointer 模式 17/18 请求、19/20 关键记忆，$0.00044、0.35 s；Claude Sonnet 5 同为 17/18、19/20，$0.020、7.6 s（无 prompt caching；缓存后 $0.0085，差距收窄到约 19×）— [samdotmak/jev-recall](https://github.com/samdotmak/jev-recall/blob/d3e4acfb45c7cf6231f1621b7dafb09d21e95862/README.md)（独立，2026-09-19，`jev-1.13.0`）。
- **中文场景的第三方诊断**：64 条合成中文职场场景（Feishu 风格）中，Jev 1.13.0 choice 模式 **64/64**、四个独立 yes/no 模式 63/64；同场景 Laya multilingual choice 仅 20/64 — [laya/research/benchmarks/feishu_zh/README.md](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/README.md)（**非独立盲测**，作者自陈是合成 fixture 的回归诊断，2026-09-21 归档）。

### 4.4 厂商自述 vs 独立测量对照
- "Jev 快"（厂商宣称，常引 40–200×）：部分成立——托管 p50 236–276 ms，比 thinking 模式 LLM 快 10–16×，但只比 Cerebras 上的 gpt-oss-120b 快约 1.2×（nibzard 认为该口径只对比了"最慢默认模式"的 LLM）。
- "校准好"（厂商定位）：**场景相关**——AG News/Banking77 优势明显；DAIR Emotion 上 ECE/Brier/NLL 更差且会输出 0 概率。
- "支持 255 选项"（官方）：独立复现 254/255 时正确率 100%，256 起 HTTP 400，属**悬崖式硬限**。


## 5. Laya 是什么

- 定位：ConvAI Innovations 出品的**多语言、非自回归 System 1 决策引擎**；"single forward pass — 33 ms"、"No text generation, so nothing to parse and nothing to hallucinate" — [laya README](https://github.com/NandhaKishorM/laya/blob/970dc8c5f63d7b886a68409493f37d569424f933/README.md)，2026-09-25 核对。
- 许可：**Apache-2.0**，作者/组织在 README 末尾写明 "Developed by Convai Innovations"；HF 三张模型卡 license 字段均为 `apache-2.0` — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)、[HF convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)（HF API 读于 2026-09-25）。
- 三个 checkpoint（官方表格，2026-09-25 核对）：

| 名称 | 编码器 | 参数量 | 官方上下文 | 用途 | 权重文件（HF API `?blobs=true`） |
|---|---|---:|---|---|---|
| `laya` | ModernBERT-large | 421M | 512 | 英文 | `model.safetensors` **842.6 MB** + tokenizer 3.6 MB |
| `laya-multilingual` | mmBERT-base | 322M | 1024（可至 8192） | 100+ 语言，快约 2× | `model.safetensors` **643.8 MB** + tokenizer **34.4 MB** |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 | typed-decisions 工作流 | `model.safetensors` **842.6 MB** + tokenizer 3.6 MB |

- 三个 checkpoint 同时存在于 HF 仓库 `convaiinnovations/laya`（根 + `multilingual/` + `typed-decisions/` 子目录），另有同名独立仓库 `laya-multilingual`、`laya-typed-decisions` — HF 文件清单（2026-09-25）。
- **Router**：内置路由会做脚本/语言检测（<0.5 ms 纯 Python），自动把非英文送 `laya-multilingual`；默认常驻 english+multilingual 两个 checkpoint（LRU），支持 `preload=True`、`max_loaded`、`attach`、`default=` 覆盖 — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)。
- 原语与 Jev 同构：`choice` / `score` / `noul`；`noul` 返回 P(true)，无独立 confidence — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)、[Laya docs 索引](https://github.com/NandhaKishorM/laya/tree/main/docs)。
- 交付形态：Python 包 `laya`（PyPI）、`laya-serve` HTTP 服务（**wire protocol 与 Jev 相同**：`POST /v1/systemone`，答案结构与 `{input_tokens, output_tokens}` usage 一致，可让既有 Jev 客户端只改 baseUrl）、MCP server、LangChain/LangGraph 集成、可选 ONNX/TileLang 加速、以及 `laya-ts`（Node/浏览器）— [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)、[laya-ts README](https://github.com/NandhaKishorM/laya/blob/main/laya-ts/README.md)。
- **与 Jev 的接口差异（官方明示）**：选项共享 `head_max_len` 预算（English 192 / 其余 256 tokens，而非 Jev 的 255 选项上限）；`score` 每一档都必须有文字描述（`null` 档返回 422）；`confidence` 是 1 − 归一化熵，**与 Jev 的 `(n·p_max − 1)/(n − 1)` 定义不同**，跨模型的阈值不可直接迁移 — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)（Jev 的公式见 [官方 confidence](https://docs.typesafe.ai/confidence)：`(count × peak − 1) / (count − 1)`，2026-09-25 直连核对）。


## 6. Laya 的上下文、显存、部署与延迟

### 6.1 上下文
- 默认：`laya` 512 ctx（其中 `head_max_len=192`，≈320 tokens 留给 state）；`laya-multilingual` 与 `laya-typed-decisions` 1024 ctx（`head_max_len=256`，≈768 tokens 留给 state）— [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)"Honest limits"节。
- 长文：`laya-multilingual` 支持 `max_len=8192`（RoPE 编码器上限）；官方脚本 `bench_long_context.py` 显示：前置文本 ≤约 4,000 tokens 时 20 题答对 16–18 题，>4,000 tokens 后波动（8–17/20）；短输入用不用 8192 结果一致，速度只随真实长度变化（4,000 token 输入在 Apple GPU 上约 **1.7 s**）— [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)。
- 官方明确建议长英文文本也要显式指定 `model="multilingual"`，否则路由会走英文 checkpoint — 同上。

### 6.2 显存/内存（官方与可信实测）
- CPU 峰值：官方延迟脚本"同时最多加载 5 个 checkpoint 时最大 RSS **9.3 GiB**" — [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)。
- Docker CPU quickstart 建议：**8 GB RAM、10 GB 空闲磁盘**（不装 Python/PyTorch 到宿主机）— [docs/docker.md](https://github.com/NandhaKishorM/laya/blob/main/docs/docker.md)。
- 单 checkpoint 权重：643.8 MB（多语，fp32 safetensors）/ 842.6 MB（英文与 typed-decisions）；多语运行时文件合计约 **678 MB** — [HF 文件体积](https://huggingface.co/convaiinnovations/laya)（API 读于 2026-09-25）、[feishu_zh README](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/README.md)。
- **GTX 16 系（无 Tensor Core）+ bge-m3 共存的显存占用：未核实/需实测。** 官方没有 GTX 16 系数据；其 GPU fast path 依赖 **TileLang + bf16 常驻权重 + CUDA graph**（在 RTX 4070 Ti SUPER、torch 2.11 + CUDA 13 上测得），不装 tilelang 会回退 stock 前向 — [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)。本项目机器上的实际可行性只能实测。

### 6.3 延迟（官方实测表，p50）
| 硬件 | 1 问 | 5 | 10 | 50 | 备注 |
|---|---|---|---|---|---|
| Tesla T4 — `laya` | 39.5 ms | 84.5 ms | 158.6 ms | 771.3 ms | 批量吞吐 103–332 问/秒 |
| Tesla T4 — `laya-multilingual` | **32.8 ms** | 40.1 ms | 72.3 ms | **337.4 ms** | 7.2 ms/问 @batch 10 |
| AMD EPYC 9R14（4 物理核，fp32，CPU） — english | 580 ms | 3,072 ms | 6,244 ms | 35,969 ms | 冷加载 4.4 s；p95≈p50+2% |
| 同上 — multilingual | 193 ms | 912 ms | 1,842 ms | 11,157 ms | 冷加载 2.5 s |
| 同上 — typed-decisions | 584 ms | 2,819 ms | 6,031 ms | 35,653 ms | 冷加载 0.5 s |
| NVIDIA GB10（DGX Spark，CUDA，经 HTTP 回环） — typed-decisions | 100.2 ms | 137.7 ms | 159.3 ms | 443.1 ms | p95 169.3/162.4/243.0/464.6 ms；每多一问 ≈ +7.0 ms |
| Apple M4（MPS，fp32）— 我按公开存档日志重新统计 | 151 ms | — | — | — | 192 次 choice 请求 p50=151 ms（58–286 ms）；four_noul p50=415 ms（164–898 ms） |

- T4/EPYC 数据来自 [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)（官方，2026-09-24 提交）；GB10 与 M4 数据分别来自同上 BENCHMARKS "Other hardware" 一节与 [feishu_zh 归档请求日志](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/results/v1/laya/raw.jsonl)（官方归档的第三方贡献/社区诊断；M4 的 p50 是**本简报自行统计**，非原文给出；同批 Jev 请求 p50 为 253 ms/250 ms，含网络时延，两者硬件与路径不可比）。
- **冷启动**：服务端 CPU 冷加载 0.5–4.4 s（上表）；懒加载 Router 在 `max_loaded=1` 时每次语言切换重建 checkpoint，官方测得中位 **7.4 s（CPU）/ 10.3 s（T4）**；`preload=True` 后无重建 — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)。
- **是否需要 GPU**：不需要。CPU 是官方一等公民（PyTorch CPU wheel、Docker CPU quickstart、CPU 延迟表）；支持 CUDA / MPS；GPU fast path 只是可选加速 — [docs/docker.md](https://github.com/NandhaKishorM/laya/blob/main/docs/docker.md)、[laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)。
- 线程坑（值得记录）：某 CPU 主机 torch 默认（10 intra-op / 5 inter-op）p50 达 **9,396 ms**，设 `torch.set_num_threads(8)` + `set_num_interop_threads(1)` 后降到 **783 ms**（12×）；最佳 intra-op 约为物理核数（8 线程 p50 329 ms）— [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)。

### 6.4 ONNX / WebGPU / 浏览器部署
- 官方提供 **`laya-ts`**（TypeScript，Node + 浏览器）与导出脚本 `laya-ts/scripts/export_onnx.py`（产出 `encoder.onnx` + `head.onnx` + tokenizer/config，并校验 torch↔ONNX 偏差 ≤1e-4）；浏览器路径走 `onnxruntime-web`，执行提供者顺序 `["webgpu", "wasm"]`，WebGPU OOM 时已提示处于 WASM fallback — [laya-ts README](https://github.com/NandhaKishorM/laya/blob/main/laya-ts/README.md)、[laya-ts/src/providers.ts](https://github.com/NandhaKishorM/laya/blob/main/laya-ts/src/providers.ts)，2026-09-25 核对。
- **官方未发布预构建 ONNX/浏览器产物，也未给出其体积**；体积证据来自社区导出（HF API `?blobs=true`，2026-09-25）：

| 形态 | 仓库 | 体积 | 许可 |
|---|---|---|---|
| ONNX（multilingual 合并图） | `mizchi/laya-multilingual-onnx` | `model.onnx` **646.9 MB**（+34.4 MB tokenizer） | apache-2.0 |
| ONNX（English，社区标称 fp16 导出） | `sevenreasons/laya-onnx-fp16` | `model.onnx` **846.3 MB** | apache-2.0 |
| GGUF（English） | `mys/laya-GGUF` | f16 846.1 MB / Q8_0 451.5 MB / Q4_K_M 419.9 MB | apache-2.0 |
| LiteRT（.tflite） | `litert-community/laya-LiteRT` | 1.29–1.69 GB（fp32，按 256/512 长度分档） | apache-2.0 |
| CoreML（fp16，多语） | `FluidInference/laya-coreml` | 权重 ≈644–648 MB/档 | apache-2.0 |
| MLX（Apple） | `aac6fef/laya-mlx` | `model.safetensors` 842.6 MB | apache-2.0 |

> 上表是社区导出体积，不等于官方支持的部署面；官方只对本仓库的 `laya` / `laya-ts` 路径负责。


## 7. Laya 的准确率与校准

### 7.1 官方 benchmark（自测，非独立）
| 任务 | `laya` | `laya-multilingual` | `laya-typed-decisions` | Jev（转引 published） |
|---|---:|---:|---:|---:|
| typed-decisions（400 例/2,000 决策） | 0.362 | 0.352 | **0.766** | **0.727** |
| AG News（4 类） | 0.950 | 0.930 | **0.953** | 0.910 |
| DAIR Emotion（6 类） | 0.595 | 0.530 | **0.600** | 0.480 |
| banking77（77 类） | 0.425 | 0.425 | 0.492 | **0.870** |
| MASSIVE intent，English | **0.783** | 0.657 | — | 无公开 |
| MASSIVE intent，其他 13 语言 | 0.306 | **0.451** | — | 无公开 |
| MASSIVE intent，全部 51 语言 macro | 0.2269 | **0.3661** | — | 无公开 |
| MASSIVE intent，`zh-CN` | 0.620 | **0.630** | — | 无公开 |
| MASSIVE intent，`zh-TW` | 0.460 | **0.540** | — | 无公开 |
| XNLI，English / 其他 14 语言 | **0.860** / 0.521 | 0.843 / **0.731** | — | 无公开 |

来源：[laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)、[Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)（含 51 语言逐语言表），2026-09-25 核对。

- **typed-decisions 的关键口径（务必引用时带上）**：0.766 属于**在该基准自己的训练切分上微调过**的 `laya-typed-decisions`；同基准的随机基线 0.318、多数类基线 0.461、教师自一致上限 0.735；两个基础 checkpoint 0.362/0.352 **低于多数类基线**。作者原话："Treat Laya as a fast base to specialise, not as a zero-shot decision engine." — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)"Honest limits"节。
- 软准确率/校准这次**Jev 更好**：soft accuracy 0.471 vs Jev 0.580；原始 ECE 0.213 vs Jev 0.144（Laya 的 0.081 是温度标定后）— 同上。
- 工作流强弱（400 例/任务，官方自测）：email spam 0.993、phishing 0.993（均在训练混合中，ECE≈0.01）；held-out moderation 仅 0.530（macro-F1 0.400）；jailbreak/guardrail 0.708–0.762；RAG 段落相关性 0.625–0.657；support triage 0.502–0.522 — [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)。

### 7.2 校准与过度自信
- 出厂即**过度自信**：官方 51 语言 sweep 中 English checkpoint macro ECE **0.7331**、平均置信度 0.9582–0.9989；Khmer 0.000 准确率时仍给 **0.952** 置信度，"confidence 无法救你"，所以必须靠前向之前的路由 — [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)、[laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)。
- 温度标定修复：按（问题类型 × 选项数）分桶拟合一个 temperature，官方表把 ECE 从 **0.466 → 0.081**（`laya`）、**0.314 → 0.106**（`laya-multilingual`）；51 语言 macro ECE 0.7331 → 0.5709（服务态）；加载时 temperature 被 clamp 到 `[0.5, 5.0]` — [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)。
- 反方向证据：在 `laya_router` 的 180 条 3 档选择任务上，绝大多数配置**欠**自信（平均 P(chosen) 比准确率低 0.18–0.19）——说明误差方向随任务变化，官方建议在自有数据上重拟合 — [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)。
- 内部置信度定义差异：`action.act_probability` 无信号（396 条上 AUROC 0.30），应改用 `confidence`（同批 AUROC 0.77）— [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)"Honest limits"。

### 7.3 独立/第三方 Laya 证据
- **浏览器 claim judge smoke test（75% accuracy、AUROC 0.80、3-option 下 false-verification 11%、72 条样本）：未核实。** 本次在 Laya 官方仓库（README/BENCHMARKS/docs/research 全量 grep）、官方 Demo Space（`convaiinnovations/laya-demo` 的 `app.py`）、GitHub 代码/仓库检索、Hugging Face spaces/models 检索中均**未找到**该记录的一手来源；如引用，须先补原始链接。（对比：官方可选 `stuntd` 社区工具曾报告 12 类意图任务 89.5% 零样本 → 100% 微调后，属社区自述而非该 smoke test — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md) Community Tools。）
- **中文诊断（官方归档的社区贡献，非独立盲测）**：64 条合成中文场景，Laya multilingual choice 20/64、四连 yes/no 18/64；Jev 1.13.0 为 64/64 与 63/64；作者自陈"AI 辅助合成 fixture、无多标注者一致性，不是通用排名" — [feishu_zh README](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/README.md)。
- **社区报告的短名单增益**：issue #102 报告者称 top-20 零样本短名单把 BANKING77 从 54.3% 提升到 60.8%，官方注明"未复测、属报告者数字" — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)"Honest limits"。


## 8. Laya 的成本与工程成本

| 项 | 事实 | 来源 |
|---|---|---|
| 许可 | Apache-2.0，可商用 | [HF 模型卡 license 字段](https://huggingface.co/convaiinnovations/laya)、[laya LICENSE](https://github.com/NandhaKishorM/laya/blob/main/LICENSE) |
| 本地/离线 | 支持；首次需从 HF 下载 checkpoint，之后加载/推理离线可跑 | [feishu_zh README](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/README.md)（"Loading/inference runs offline after local checkpoint preparation"） |
| 模型体积 | 643.8 MB（322M，fp32）／842.6 MB（421M，fp32）；多语 tokenizer 34.4 MB | [HF API ?blobs=true](https://huggingface.co/api/models/convaiinnovations/laya?blobs=true)（2026-09-25 读取） |
| 运行时依赖 | Python ≥3.10；`transformers` 5.x、`torch` 2.14、`huggingface_hub` 1.x；可选 extras：`serve`/`mcp`/`langchain`/`onnx`/`fast`(tilelang) | [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)、[pyproject.toml](https://github.com/NandhaKishorM/laya/blob/main/pyproject.toml) |
| Docker 资源 | CPU quickstart 建议 8 GB RAM / 10 GB 空闲磁盘；GPU 镜像用 CUDA 12.8 wheels | [docs/docker.md](https://github.com/NandhaKishorM/laya/blob/main/docs/docker.md) |
| 微调数据量 | 官方示例 ≈**30k questions**（4 epoch） | [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md) |
| 微调算力 | Kaggle 免费 **2×T4，4–5 小时**（RLCD：proper scoring rule 奖励 + GRPO 式策略梯度；内置校准拟合与评测）；另有一个"单张 16 GB GPU"的浏览器代理特化实例 | [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)、[notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) |
| 微调收益 | 浏览器代理例子：~45 候选里元素 top-1 从 0.10（零样本）→ 0.66（微调），真实任务成功率 0% → 62%，17–23 ms/步 | [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md) |
| 边际成本 | 推理零 API 费用（自付算力）；分类决策可用 CPU 跑 | [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) |
| 与 GTX 16 系共存（无 Tensor Core、bge-m3 已占显存） | **未知/需实测**：官方无 GTX 16 系任何数据；fast path 依赖 bf16 + CUDA graph；需实测（a）单 checkpoint 与 bge-m3/reranker 同卡的显存上限，（b）无 bf16 时 stock 前向的吞吐，（c）CPU 回退时的 p50 | 结论基于 [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md) 的硬件覆盖范围（T4/GB10/EPYC/6900HX/4070 Ti SUPER），缺 GTX 16 系证据 |


## 9. 两者定位差异与已知短板

### 9.1 定位差异（仅事实对照）
| 维度 | Jev | Laya |
|---|---|---|
| 能力来源 | 云端零样本强（AG News 0.910 / Banking77 0.870，独立 300 例 pilot） | 零样本弱、**微调后**才强（typed-decisions 0.766 需在基准训练切分上微调） |
| 成本结构 | 按输入 token 付费（$0.042/M），有速率上限 | 权重免费，自付算力/工程/微调成本；零边际 API 费用 |
| 数据出域 | 每次调用把 state 发到 TypeSafe；不做训练但会出域，ZDR 需企业方案 | 本地/离线可跑，数据不出域 |
| 版本漂移 | `jev-latest`/`jev-preview` 会移动；官方建议固定 `jev-1.13.0` | 权重按 HF revision 固定；官方也声明 temperature clamp 等运行时变化 |
| 限流 | 250k tokens/s、1,200 req/min，**动态调整不预告**；429/529 需退避 | 无外部限流；瓶颈是本地算力/显存 |
| fallback | 超限/超上下文 → 429/400/422；SDK 自动退避重试；无本地回退 | 无网/超显存可退 CPU（官方 fast path 在 CUDA OOM 后回退 CPU，并先关闭 fast path） |
| 可解释性 | **不能解释**（无推理输出；官方 jaggedness 亦承认会字面理解、易被注入） | 同样无自然语言解释；但本地可看概率分布、可微调、可加 hooks |
| 选项规模 | ≤255 选项；256 起直接 400 | 受 `head_max_len` 预算限制，>20 选项退化明显（77 类 0.425） |

### 9.2 Jev 已知短板（官方 + 独立）
- 官方 jaggedness 列 9 类：字面理解、算术/计数不可靠、日期比较、多层间接、**大 state 含无关细节会掉精度**、对抗性内容（state 不被当作敌意输入）、instruction 与 criteria 冲突、结构不变式不成立（noul vs choice 不可互推）、**不适合生成**；官方还提醒 "context rot" — [官方 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)，2026-09-25 直连核对。
- 上下文：64k 整请求 / 32k state+最长问；语言：英语最好，其他语言（含 CJK）准确率较低 — [官方 models](https://docs.typesafe.ai/models)。
- 独立负面：DAIR Emotion 校准差 + 16% 零概率；forced-uncertainty 只承认 49.7%；选项置换 13% 翻面；≥256 选项硬失败 — [AbdelStark](https://github.com/AbdelStark/jev-benchmarks/blob/main/README.md)、[nibzard](https://github.com/nibzard/decision-model-benchmark/blob/main/README.md)。

### 9.3 Laya 已知短板（官方自陈 + 独立）
- 基础 checkpoint 在 typed-decisions 上**近随机**（0.362/0.352 vs 0.318 随机、0.461 多数类）；真正的 0.766 来自在基准训练切分上微调 — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)。
- 出厂过度自信（macro ECE 0.7331，English 平均置信 0.9582–0.9989），必须温度标定；`laya-multilingual` 甚至没带拟合温度 — [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)。
- 中文/多语：MASSIVE 51 语言 `laya` macro 仅 0.2269（`laya-multilingual` 0.3661），中文本体 0.620/0.630；合成中文职场诊断里 multilingual choice 20/64 — 同上、[feishu_zh](https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/README.md)。
- 长文：≤约 4,000 token 尚可，更长波动明显；官方要求自行验长文准确率 — [laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md)。
- 其它官方自陈：`choice` 选项应 <约 20；布尔词标签（true/false、yes/no）会被跟随；否定语义不安全（5 个取消类例子中 4 个/2 个判错）；ordinal `score` 最弱（SST-5 0.372）；`noul` 可能跟随标签对而非 state；`laya-multilingual` 的 `score` 存在位置偏好；`action.act_probability` 无信号 — 同上"Honest limits"。
- 选项顺序稳定性：20 选项时 Laya 翻面率 0.150/0.230（massive_intent.en），差于 Jev 的 0.13 — [Laya BENCHMARKS.md](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)。


---

## 10. 未核实 / 仍需实测的事项（3 项重点）

1. **浏览器 claim judge smoke test（75% accuracy、AUROC 0.80、3-option false-verification 11%、72 条样本）**：未核实。官方仓库/文档站/HF/GitHub 检索均无一手来源；引用前必须补原始链接与样本构成。
2. **Jev 重试是否计费**：官方文档只写"按输入 token 计费、输出免费"，未说明失败/限流/重试请求的计费口径——需账单实测或问厂商。
3. **GTX 16 系 + bge-m3 共存可行性（显存/吞吐/无 bf16 回退）**：官方无任何 GTX 16 系数据，本项目机器上属未知/需实测。

（其它：`jev-latest` 在核查日解析为 `jev-1.13.0`，但其为移动别名，未来会漂移；`25k` 不是任何官方限制，而是第三方自设预算。）


---

## 来源清单

> 访问/核对日期统一为 **2026-09-25**（Asia/Shanghai），除非另注。类型：官方 / 官方快照 / 独立评测 / 社区导出。

### 官方（本次直连成功）
1. https://docs.typesafe.ai/models — 官方（价格、限流、上下文、别名、数据政策）
2. https://docs.typesafe.ai/api — 官方（端点、鉴权、状态码 401/422/429/529、重试建议）
3. https://docs.typesafe.ai/confidence — 官方（confidence 定义 `(count×peak−1)/(count−1)`、Noul 无 confidence）
4. https://docs.typesafe.ai/introduction — 官方（System One 定义、"No text generation, no parsing"、并行评估）
5. https://docs.typesafe.ai/primitives/choice — 官方（255 选项上限、probabilities 语义）
6. https://docs.typesafe.ai/primitives/noul — 官方（Noul = P(yes)，无独立 confidence）
7. https://docs.typesafe.ai/model-jaggedness/jev-1.13 — 官方（9 类已知缺陷，Last reviewed 2026-09-17）
8. https://docs.typesafe.ai/patterns/fan-out — 官方（单请求塞多问的用法）
9. https://github.com/typesafe-ai/typesafe-sdk-python（HEAD `0ffd094c`，2026-09-21）— 官方 SDK（MIT）
10. https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md（HEAD `65a39f39`）— 官方（"questions run in parallel"、设计指导）
11. https://github.com/typesafe-ai/system-one-adapter-python（HEAD `e1d4cc93`，2026-09-22）— 官方（LLM 对标实现）
12. https://github.com/NandhaKishorM/laya（HEAD `970dc8c5`）— Laya 官方仓库
13. https://github.com/NandhaKishorM/laya/blob/main/README.md — 官方（三个 checkpoint、Router、8192 长文、微调、Honest limits）
14. https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md — 官方（全部延迟/准确率/ECE/顺序稳定性表）
15. https://github.com/NandhaKishorM/laya/blob/main/docs/docker.md — 官方（8 GB RAM / 10 GB 磁盘、CPU/GPU 镜像）
16. https://github.com/NandhaKishorM/laya/blob/main/laya-ts/README.md、/laya-ts/src/providers.ts — 官方（ONNX 导出、webgpu→wasm）
17. https://github.com/NandhaKishorM/laya/blob/main/research/benchmarks/feishu_zh/README.md（及 results/v1 归档日志）— 官方归档（中文合成诊断、离线可跑、外部预训练数据）
18. https://huggingface.co/convaiinnovations/laya、/laya-multilingual、/laya-typed-decisions（含 `?blobs=true`）— 官方模型卡与文件体积
19. https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints — 官方网关侧数据（TypeSafe 直供端点、$0.042/M、context 32000）

### 官方文档快照（经第三方仓库转引，逐条注明）
20. https://github.com/jonathanavis96/jev-kit/blob/main/docs/jev-reference/typesafe-docs/models.md — **官方文档快照**（该文件自注 `fetched: 2026-09-23`；含价格表、限流、上下文、别名）
21. https://github.com/PostHog/posthog/blob/ba7e9c1ba6f0222582253cade83ce00bc272e87a/posthog/egress/typesafe/README.md — **官方文档快照**（1,200 req/min、250k tokens/s、$0.042/M、64k、别名）
22. https://github.com/2FastLabs/agent-squad/blob/3d67629ca564eaf35ec7779115c45a42ee23a2bd/docs/src/content/docs/classifiers/built-in/jev-classifier.mdx — **官方文档快照**（托管闭源、64k/32k、255 选项、语言与对抗风险）
23. https://github.com/pydantic/pydantic-ai/blob/08f4cc2757d107a37d6697b976cbaa034830a6c9/docs/models/typesafe.md、https://github.com/TanStack/ai/blob/7404539c55acf9d7d0b0f7ccb2f16f4dcd75acdb/docs/adapters/typesafe.md、https://github.com/maximhq/bifrost/blob/cefde78ba0e574f03da111e3fdcb9f4286e5ccc4/docs/providers/supported-providers/typesafe.mdx — **官方文档快照**（别名与版本固定）
24. https://github.com/edwardyen724-g/jev-compactor/blob/0a9b64e8481fac186e3dcb478450d8cf133d2772/docs/JEV-API.md — 第三方工程笔记（200 问/请求实测 161 ms；32k 超限返回 400；SDK 退避策略）

### 独立评测
25. https://github.com/erendikmenn/jev-rag-benchmark（HEAD `7c7fd8af`）— 独立（XQuAD-TR 1,044 题、hybrid top-20、Recall@5/nDCG/p50/成本、citation verification 消融）
26. https://github.com/hotchpotch/jev-reranker（HEAD `d58594b3`）— 独立库（分片/listwise 语义、退避、ContextLimitError、p50）
27. https://github.com/samdotmak/jev-recall（HEAD `d3e4acfb`）— 独立（238 记忆/18 请求，Jev vs Claude 成本延迟）
28. https://github.com/AbdelStark/jev-benchmarks（HEAD `0d610cc5`）— 独立（300 例 pilot；Jev p50 236–256 ms；DAIR Emotion 校准反例）
29. https://github.com/nibzard/decision-model-benchmark（HEAD `8404980d`）— 独立（$28.34 自费；p50 264–276 ms；ECE 0.246；255 选项悬崖；13% 顺序翻面）
30. https://github.com/NousResearch/hermes-agent/blob/8623cd4a8403f020090c9c3e8feae943b4dac55e/evals/compaction/results/SCORECARD-2026-09-19-jev.md — 独立（"32K window forces ... 25K tokens" 的 25k 出处）
31. https://github.com/tamaratran/fast-jev-compaction/blob/e3f262a7f4d42bd8dd32ced30d26176f7cb545b0/README.md — 独立（30k 自设预算、并发多请求合并）

### 社区导出/工具（仅体积与存在性证据）
32. https://huggingface.co/mizchi/laya-multilingual-onnx、https://huggingface.co/sevenreasons/laya-onnx-fp16、https://huggingface.co/mys/laya-GGUF、https://huggingface.co/litert-community/laya-LiteRT、https://huggingface.co/FluidInference/laya-coreml、https://huggingface.co/aac6fef/laya-mlx — 社区导出（体积见 §6.4；均 apache-2.0）
33. https://huggingface.co/spaces/convaiinnovations/laya-demo — 官方 Demo Space（本次检查 `app.py` 未见 claim judge 相关实现）
