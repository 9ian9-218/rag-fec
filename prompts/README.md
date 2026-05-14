# 提示词目录说明

本目录位于仓库根下的 `prompts/`，与 `src/prompt_texts.py` 中的加载路径一致：运行时代码通过 `load_prompt("<文件名>")` 从此目录读取 UTF-8 文本；带占位符的模板由调用方传入 `str.format` 参数。

## 文件与用途

| 文件 | 用途 |
|------|------|
| `fec_extraction_system_append.txt` | LightRAG 抽取时在 system 后追加的 FEC 规则 |
| `json_mode_system_suffix.txt` | JSON 模式下的 system 后缀（含占位符 `{fec_json_extraction_example}`） |
| `neo4j_graph_query_planner.txt` | Neo4j 检索规划 LLM 用户提示 |
| `lightrag_retrieval_planner.txt` | LightRAG mode + 改写问句的规划提示 |
| `openai_probe_user.txt` | `--probe-openai` 的探测用户消息 |

## 位置与代码对应关系

| 仓库路径 | 被引用位置（示例） |
|----------|-------------------|
| `prompts/fec_extraction_system_append.txt` | `src/extraction.py` → `FEC_EXTRACTION_SYSTEM_APPEND` |
| `prompts/json_mode_system_suffix.txt` | `src/extraction.py` → `_JSON_MODE_SYSTEM_SUFFIX`（`fec_json_extraction_example` 由代码中的 `_FEC_JSON_EXTRACTION_EXAMPLE` 填入） |
| `prompts/neo4j_graph_query_planner.txt` | `src/light_query.py` → `_llm_parse_graph_query()` |
| `prompts/lightrag_retrieval_planner.txt` | `src/light_query.py` → `_llm_parse_lightrag_retrieval_plan()` |
| `prompts/openai_probe_user.txt` | `src/extraction.py` → `_probe_openai_only()` |

统一加载入口：`src/prompt_texts.py` 中的 `load_prompt()`。
