"""检索前规则路由：不调用 LLM，基于问题关键词选择 LightRAG 模式。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from config.settings import Settings, get_settings
from src.retrieval.mode_config import RetrievalMode, suggest_mode_from_question

ROUTABLE_MODES: tuple[str, ...] = ("naive", "local", "global", "hybrid", "mix")


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


async def resolve_retrieval_mode(
    question: str,
    explicit_mode: str | None,
    *,
    settings: Settings | None = None,
    use_llm_router: bool = True,
) -> ModeRouteResult:
    """
    解析最终检索模式：显式模式优先；否则使用基于规则的启发式路由，不调用 LLM。
    """
    _ = settings or get_settings()
    forced = normalize_mode(explicit_mode, allow_bypass=True)
    if forced is not None:
        return ModeRouteResult(mode=forced, reason="用户显式指定模式", source="explicit")

    mode = suggest_mode_from_question(question)
    return ModeRouteResult(mode=mode, reason="规则关键词路由", source="heuristic")
