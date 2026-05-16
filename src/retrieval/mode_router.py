"""檢索前 LLM 路由：評估問題難度/複雜度/上下文需求，並選擇 LightRAG 模式。"""

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
        "summary": "僅向量庫檢索文本 Chunk，不使用知識圖譜。",
        "when": "事實簡單、表述直接、只需局部段落即可回答；不需實體關係推理。",
        "params": f"預設參數：{MODE_DEFAULTS.get('naive', {})}",
    },
    {
        "mode": "local",
        "summary": "以低層關鍵詞聚焦「實體」及其鄰域 Chunk（圖+局部上下文）。",
        "when": "問具體概念、定義、某算法/碼型的性質、單點事實。",
        "params": f"預設參數：{MODE_DEFAULTS.get('local', {})}",
    },
    {
        "mode": "global",
        "summary": "以高層關鍵詞聚焦「關係」與社群級結構（圖譜全局視角）。",
        "when": "問整體趨勢、全書/全章總結、跨主題比較、宏觀結論。",
        "params": f"預設參數：{MODE_DEFAULTS.get('global', {})}",
    },
    {
        "mode": "hybrid",
        "summary": "local + global 兩路圖譜結果按 round-robin 合併，再與 Chunk 融合。",
        "when": "需同時用到具體實體細節與關係/結構信息，但可不依賴額外向量補充。",
        "params": f"預設參數：{MODE_DEFAULTS.get('hybrid', {})}",
    },
    {
        "mode": "mix",
        "summary": "知識圖譜（實體+關係）與向量 Chunk 雙路召回並融合（本專案 KG-RAG 默認推薦）。",
        "when": "問題中等以上複雜、需圖譜+原文證據、性能對比、多條件綜合分析。",
        "params": f"預設參數：{MODE_DEFAULTS.get('mix', {})}",
    },
]


def mode_catalog_text() -> str:
    lines: list[str] = []
    for spec in _MODE_SPECS:
        lines.append(
            f"- **{spec['mode']}**：{spec['summary']}\n"
            f"  適用：{spec['when']}\n"
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


def _build_router_prompt(question: str) -> list[dict[str, str]]:
    catalog = mode_catalog_text()
    system = (
        "你是 KG-RAG 檢索模式路由器。先分析用戶問題，再在 naive、local、global、hybrid、mix 五種模式中選一種。\n"
        "評估維度：\n"
        "1. difficulty（easy/medium/hard）：回答所需領域深度與推理難度\n"
        "2. complexity（low/medium/high）：問題結構複雜度（單跳/多跳/多條件）\n"
        "3. context_richness（low/medium/high）：為答對所需召回上下文的豐富程度\n"
        "僅輸出一個 JSON 對象，不要 markdown，字段：\n"
        '{"difficulty":"...","complexity":"...","context_richness":"...","mode":"naive|local|global|hybrid|mix","reason":"一句中文理由"}'
    )
    user = f"## LightRAG 五種模式說明\n{catalog}\n\n## 用戶問題\n{question.strip()}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_mode_router_response(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
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
        max_tokens=256,
    )
    content = (resp.choices[0].message.content or "").strip()
    parsed = parse_mode_router_response(content)
    mode = normalize_mode(str(parsed.get("mode") or ""))
    if mode is None:
        logger.warning("模式路由 LLM 解析失敗，回退啟發式: %s", content[:200])
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
        "模式路由結果 mode=%s difficulty=%s complexity=%s context_richness=%s source=%s reason=%s",
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
    解析最終檢索模式：顯式 ``mode`` 優先；否則在啟用時走 LLM 路由；最後回退默認/啟發式。
    """
    s = settings or get_settings()
    forced = normalize_mode(explicit_mode, allow_bypass=True)
    if forced is not None:
        return ModeRouteResult(mode=forced, reason="用戶顯式指定模式", source="explicit")

    default_m = normalize_mode(s.retrieval.default_mode)
    if use_llm_router and s.retrieval.llm_mode_router_enabled:
        try:
            return await route_mode_with_llm(question, settings=s)
        except Exception as e:
            logger.warning("LLM 模式路由失敗，回退: %s", e)

    if default_m is not None:
        return ModeRouteResult(mode=default_m, reason="設定默認模式", source="default")
    mode = suggest_mode_from_question(question)
    return ModeRouteResult(mode=mode, reason="關鍵詞啟發式", source="heuristic")
