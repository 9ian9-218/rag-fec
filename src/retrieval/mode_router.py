"""检索前 LLM 路由：评估问题难度/复杂度/上下文需求，并选择 LightRAG 模式。"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from openai import AsyncOpenAI

from config.settings import Settings, get_settings
from src.retrieval.mode_config import MODE_DEFAULTS, RetrievalMode, suggest_mode_from_question
from src.utils.logger import get_logger

logger = get_logger("retrieval.mode_router")

ROUTABLE_MODES: tuple[str, ...] = ("naive", "local", "global", "hybrid", "mix")

_MODE_SPECS: list[dict[str, str]] = [
    {
        "mode": "naive",
        "summary": "仅向量库检索文本 Chunk，不使用知识图谱。",
        "when": "事实简单、表述直接、只需局部段落即可回答；不需实体关系推理。",
        "params": f"预设参数：{MODE_DEFAULTS.get('naive', {})}",
    },
    {
        "mode": "local",
        "summary": "以低层关键词聚焦「实体」及其邻域 Chunk（图+局部上下文）。",
        "when": "问具体概念、定义、某算法/码型的性质、单点事实。",
        "params": f"预设参数：{MODE_DEFAULTS.get('local', {})}",
    },
    {
        "mode": "global",
        "summary": "以高层关键词聚焦「关系」与社群级结构（图谱全局视角）。",
        "when": "问整体趋势、全书/全章总结、跨主题比较、宏观结论。",
        "params": f"预设参数：{MODE_DEFAULTS.get('global', {})}",
    },
    {
        "mode": "hybrid",
        "summary": "local + global 两路图谱结果按 round-robin 合并，再与 Chunk 融合。",
        "when": "需同时用到具体实体细节与关系/结构信息，但可不依赖额外向量补充。",
        "params": f"预设参数：{MODE_DEFAULTS.get('hybrid', {})}",
    },
    {
        "mode": "mix",
        "summary": "知识图谱（实体+关系）与向量 Chunk 双路召回并融合。",
        "when": "问题结构复杂、需同时利用图谱推理和大量原始文本证据；涉及性能对比、多条件综合分析、需要逐字引用原文。仅在问题明显超出其他四种模式能力时选择。",
        "params": f"预设参数：{MODE_DEFAULTS.get('mix', {})}",
    },
]


def mode_catalog_text() -> str:
    lines: list[str] = []
    for spec in _MODE_SPECS:
        lines.append(
            f"- **{spec['mode']}**：{spec['summary']}\n"
            f"  适用：{spec['when']}\n"
            f"  {spec['params']}"
        )
    return "\n".join(lines)


@dataclass
class ModeRouteResult:
    mode: RetrievalMode
    difficulty: str = ""
    complexity: str = ""
    context_richness: str = ""
    reason: str = ""
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _llm_client(settings: Settings) -> tuple[AsyncOpenAI, str, str]:
    api_key = (settings.openai_api_key or settings.llm.api_key or "").strip()
    if not api_key:
        raise RuntimeError("模式路由需要 OPENAI_API_KEY / LLM_API_KEY")
    base = (settings.openai_base_url or settings.llm.base_url or "").strip().rstrip("/")
    if not base:
        base = "https://api.openai.com/v1"
    model = settings.resolved_llm_model_name()
    return AsyncOpenAI(api_key=api_key, base_url=base), base, model


_ROUTER_FEW_SHOT = """
## 路由示例（请严格按照以下格式输出）

Q: "RPA_RM 译码的时间复杂度是多少？"
→ {"difficulty":"medium","complexity":"low","context_richness":"low","mode":"local","reason":"单点事实查询，问具体算法属性"}

Q: "比较 RPA_RM 和 Chase 列表译码的性能差异"
→ {"difficulty":"medium","complexity":"medium","context_richness":"medium","mode":"mix","reason":"需对比两种译码算法，需图谱+原文证据"}

Q: "什么是 Reed-Muller 码？"
→ {"difficulty":"easy","complexity":"low","context_richness":"low","mode":"naive","reason":"基础概念定义，简单直接"}

Q: "本文在译码器设计上的主要贡献是什么？"
→ {"difficulty":"hard","complexity":"high","context_richness":"high","mode":"global","reason":"需全局视角总结全文贡献"}

Q: "给出算法1的伪代码步骤"
→ {"difficulty":"medium","complexity":"low","context_richness":"medium","mode":"naive","reason":"需要原文段落直接回答，不需图谱推理"}
"""


def _build_router_prompt(question: str) -> list[dict[str, str]]:
    catalog = mode_catalog_text()
    system = (
        "你是 KG-RAG 检索模式路由器。任务：分析用户问题，从 naive、local、global、hybrid、mix 中选择最适合的检索模式。\n\n"
        "## 决策优先级（从上到下判断）\n"
        "1. 问题是否简单直接、只需原文片段即可回答？ → naive\n"
        "2. 是否仅围绕单一实体/概念展开，不需要全局结构？ → local\n"
        "3. 是否关注整体趋势、全书总结、宏观关系结构？ → global\n"
        "4. 是否需要同时关注实体细节和关系结构，但不需要大量原文引用？ → hybrid\n"
        "5. 是否问题结构复杂、需图谱+大量原文证据、性能对比？ → mix\n\n"
        "## 输出要求\n"
        "- 禁止输出任何思考过程、解释说明或 <think> 标签\n"
        "- 仅输出一个纯 JSON 对象，不要 markdown 代码块\n"
        "- 字段：difficulty、complexity、context_richness、mode、reason\n"
        "- mode 只能是：naive、local、global、hybrid、mix 之一\n"
        "- reason 用一句中文简要说明选择理由"
    )
    user = (
        f"## LightRAG 五种模式说明\n{catalog}\n"
        f"{_ROUTER_FEW_SHOT}\n\n"
        f"## 用户问题\n{question.strip()}\n\n"
        "请直接输出 JSON（不要任何其他内容）："
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_mode_router_response(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}

    # 先尝试从原始文本（含 think 内外）提取 JSON
    def _try_extract_json(source: str) -> dict[str, Any] | None:
        # 兼容 markdown 代码块
        code_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", source)
        candidate = code_match.group(1).strip() if code_match else source
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(candidate[start : end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return None

    # 1) 先尝试去除 <think> 后的文本
    no_think = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    if no_think:
        parsed = _try_extract_json(no_think)
        if parsed is not None:
            return parsed

    # 2) 若去除 think 后为空，尝试从 <think> 内部提取 JSON（deepseek-v4-flash 可能把 JSON 放在 think 中）
    think_match = re.search(r"<think>([\s\S]*?)</think>", raw)
    if think_match:
        parsed = _try_extract_json(think_match.group(1))
        if parsed is not None:
            return parsed

    # 3) 全文再试一次（兜底）
    parsed = _try_extract_json(raw)
    if parsed is not None:
        return parsed

    # 4) 最后回退：关键词匹配
    lowered = raw.lower()
    for m in ROUTABLE_MODES:
        if re.search(rf"\b{re.escape(m)}\b", lowered):
            return {"mode": m, "reason": raw[:200]}
    return {}


def normalize_mode(value: str | None, *, allow_bypass: bool = False) -> RetrievalMode | None:
    if not value:
        return None
    m = str(value).strip().lower()
    if m == "native":
        m = "naive"
    if allow_bypass and m == "bypass":
        return "bypass"
    if m in ROUTABLE_MODES:
        return m  # type: ignore[return-value]
    return None


async def route_mode_with_llm(
    question: str,
    *,
    settings: Settings | None = None,
) -> ModeRouteResult:
    s = settings or get_settings()
    client, base, model = _llm_client(s)
    temp = float(s.retrieval.llm_mode_router_temperature)
    messages = _build_router_prompt(question)
    logger.info("模式路由 LLM：model=%s base=%s", model, base)
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temp,
        max_tokens=512,
    )
    content = (resp.choices[0].message.content or "").strip()
    parsed = parse_mode_router_response(content)
    mode = normalize_mode(str(parsed.get("mode") or ""))
    if mode is None:
        logger.warning("模式路由 LLM 解析失败，回退启发式: %s", content[:200])
        mode = suggest_mode_from_question(question)
        source = "heuristic_fallback"
    else:
        source = "llm"
    result = ModeRouteResult(
        mode=mode,
        difficulty=str(parsed.get("difficulty") or ""),
        complexity=str(parsed.get("complexity") or ""),
        context_richness=str(parsed.get("context_richness") or ""),
        reason=str(parsed.get("reason") or ""),
        source=source,
    )
    logger.info(
        "模式路由结果 mode=%s difficulty=%s complexity=%s context_richness=%s source=%s reason=%s",
        result.mode,
        result.difficulty,
        result.complexity,
        result.context_richness,
        result.source,
        result.reason,
    )
    return result


async def resolve_retrieval_mode(
    question: str,
    explicit_mode: str | None,
    *,
    settings: Settings | None = None,
    use_llm_router: bool = True,
) -> ModeRouteResult:
    """
    解析最终检索模式：显式 ``mode`` 优先；否则在启用时走 LLM 路由；最后回退默认/启发式。
    """
    s = settings or get_settings()
    forced = normalize_mode(explicit_mode, allow_bypass=True)
    if forced is not None:
        return ModeRouteResult(mode=forced, reason="用户显式指定模式", source="explicit")

    default_m = normalize_mode(s.retrieval.default_mode)
    if use_llm_router and s.retrieval.llm_mode_router_enabled:
        try:
            return await route_mode_with_llm(question, settings=s)
        except Exception as e:
            logger.warning("LLM 模式路由失败，回退: %s", e)

    if default_m is not None:
        return ModeRouteResult(mode=default_m, reason="设置默认模式", source="default")
    mode = suggest_mode_from_question(question)
    return ModeRouteResult(mode=mode, reason="关键词启发式", source="heuristic")
