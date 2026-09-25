"""查询期社区摘要注入：按需开关 → 门控 → 本地向量检索 → 渲染概览块（零 LLM 调用）。

开关语义（三态）：
- ``use_community=None``：按 ``COMMUNITY_DEFAULT_ENABLED``（默认 false）+ 宏观问题启发式；
- ``use_community=True``：用户意愿优先，跳过宏观启发式（仍受模式、相似度与 token 预算约束）；
- ``use_community=False``：强制不注入，即使服务端默认开启。

无论是否生效，都返回结构化状态（``requested/applied/reason``），便于调用方解释"开了为什么没用上"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

import numpy as np

from config.settings import Settings, get_settings
from src.community import store
from src.community.summarizer import count_tokens, truncate_to_tokens
from src.retrieval.mode_config import is_macro_question
from src.utils.logger import get_logger

logger = get_logger("community.injector")

ELIGIBLE_MODES = ("global", "hybrid", "mix")
EmbedFn = Callable[[list[str]], Awaitable[Any]]

REASON_OK = "ok"
REASON_DISABLED_SERVER = "disabled_by_server"
REASON_DISABLED_REQUEST = "disabled_by_request"
REASON_MODE = "mode_not_eligible"
REASON_NOT_MACRO = "not_macro_question"
REASON_MISSING = "index_missing"
REASON_STALE = "index_stale"
REASON_LOW_SIM = "low_similarity"
REASON_ERROR = "error"


@dataclass(slots=True)
class CommunityInjection:
    """一次查询的社区注入结果。"""

    requested: bool = False
    applied: bool = False
    reason: str = REASON_DISABLED_REQUEST
    ids: list[str] = field(default_factory=list)
    tokens: int = 0
    similarities: list[float] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """响应/遥测用的状态（不含正文，避免响应体膨胀）。"""
        return {
            "requested": bool(self.requested),
            "applied": bool(self.applied),
            "reason": self.reason,
            "ids": list(self.ids),
            "tokens": int(self.tokens),
            "similarities": list(self.similarities),
        }


def _format_block(index: int, community_id: str, rep: dict[str, Any]) -> str:
    title = str(rep.get("title") or "").strip() or community_id
    docs = [str(d) for d in (rep.get("docs") or [])][:5]
    header = f"【主題 {index}｜{title}】（社区 {community_id}"
    if docs:
        header += f"，覆盖：{'、'.join(docs)}"
    header += "）"
    lines = [header, str(rep.get("summary") or "").strip()]
    points = [str(p).strip() for p in (rep.get("key_points") or []) if str(p).strip()]
    if points:
        lines.append("要点：" + "；".join(points[:5]))
    return "\n".join(line for line in lines if line)


def render_reports(
    picked: Sequence[tuple[str, dict[str, Any], float]],
    max_tokens: int,
) -> tuple[str, int]:
    """渲染社区概览块；超出 token 预算时截断最后一篇而不是留半截。"""
    if max_tokens <= 0 or not picked:
        return "", 0
    parts: list[str] = []
    used = 0
    for index, (community_id, rep, _sim) in enumerate(picked, start=1):
        block = _format_block(index, community_id, rep)
        if not block:
            continue
        cost = count_tokens(block) + (2 if parts else 0)
        if used + cost <= max_tokens:
            parts.append(block)
            used += cost
            continue
        remaining = max_tokens - used - (2 if parts else 0)
        if remaining > 50:
            trimmed = truncate_to_tokens(block, remaining)
            parts.append(trimmed)
            used += count_tokens(trimmed) + (2 if len(parts) > 1 else 0)
        break
    text = "\n\n".join(parts).strip()
    return text, count_tokens(text)


async def _default_embed(texts: list[str], settings: Settings) -> Any:
    if settings.embedding.backend == "local":
        from src.storage.local_embedding import build_local_embedding_func

        func = build_local_embedding_func(settings)
    else:
        from src.storage.remote_embedding import build_remote_embedding_func

        func = build_remote_embedding_func(settings)
    return await func(texts)


async def select_community_context(
    question: str,
    *,
    use_community: bool | None = None,
    mode: str | None = None,
    settings: Settings | None = None,
    embed_fn: EmbedFn | None = None,
) -> CommunityInjection:
    """按开关与索引状态选出社区摘要；任何异常都降级为"不注入 + 原因"。"""
    s = settings or get_settings()
    cfg = s.community
    requested = bool(cfg.default_enabled) if use_community is None else bool(use_community)

    if not cfg.enabled:
        return CommunityInjection(requested=requested, applied=False, reason=REASON_DISABLED_SERVER)
    if not requested:
        return CommunityInjection(requested=False, applied=False, reason=REASON_DISABLED_REQUEST)
    if mode is not None and str(mode) not in ELIGIBLE_MODES:
        return CommunityInjection(requested=True, applied=False, reason=REASON_MODE)
    if use_community is None and not is_macro_question(question):
        return CommunityInjection(requested=True, applied=False, reason=REASON_NOT_MACRO)

    usable, reason = store.is_usable(s)
    if not usable:
        return CommunityInjection(requested=True, applied=False, reason=reason)

    reports_data = store.load_reports(s) or {}
    reports: dict[str, Any] = reports_data.get("reports") or {}
    index = store.load_index(s)
    if index is None:
        return CommunityInjection(requested=True, applied=False, reason=REASON_MISSING)
    ids, vectors = index

    try:
        embed = embed_fn or (lambda texts: _default_embed(texts, s))
        raw = await embed([question])
        query_vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        if query_vec.shape[0] != vectors.shape[1]:
            logger.warning(
                "社区检索维度不匹配（query=%d, index=%d），跳过注入",
                query_vec.shape[0],
                vectors.shape[1],
            )
            return CommunityInjection(requested=True, applied=False, reason=REASON_ERROR)
        q_norm = float(np.linalg.norm(query_vec)) or 1.0
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0
        sims = (vectors @ query_vec) / (norms * q_norm)
    except Exception as e:
        logger.warning("社区摘要检索失败，跳过注入: %s", e)
        return CommunityInjection(requested=True, applied=False, reason=REASON_ERROR)

    order = np.argsort(-sims)
    picked: list[tuple[str, dict[str, Any], float]] = []
    seen_sims: list[float] = []
    for idx in order:
        score = float(sims[int(idx)])
        seen_sims.append(round(score, 4))
        if score < float(cfg.sim_threshold):
            break
        community_id = ids[int(idx)]
        rep = reports.get(community_id)
        if not isinstance(rep, dict) or not str(rep.get("summary") or "").strip():
            continue
        picked.append((community_id, rep, score))
        if len(picked) >= int(cfg.top_k):
            break

    if not picked:
        return CommunityInjection(
            requested=True,
            applied=False,
            reason=REASON_LOW_SIM,
            similarities=seen_sims[: int(cfg.top_k)],
        )

    text, tokens = render_reports(picked, int(cfg.max_tokens))
    if not text:
        return CommunityInjection(requested=True, applied=False, reason=REASON_LOW_SIM)
    return CommunityInjection(
        requested=True,
        applied=True,
        reason=REASON_OK,
        ids=[pid for pid, _, _ in picked],
        tokens=tokens,
        similarities=[round(sim, 4) for _, _, sim in picked],
        text=text,
    )
