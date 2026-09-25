"""社区摘要生成：代表实体取样 → 提示词 → LLM → 结构化 JSON。

成本控制：
- 每个社区的输入被裁剪到 ``COMMUNITY_SUMMARY_MAX_INPUT_TOKENS``；
- ``content_hash``（成员集合 + 取样文本 + 提示词版本 + 模型）未变则完全复用旧摘要，不调 LLM；
- 调用复用项目统一的 AsyncOpenAI 客户端（带 ``x-opencode-session`` 头）与插入期并发配额。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

from config.settings import Settings, get_settings
from src.utils.client_cache import get_openai_client
from src.utils.concurrency import get_semaphore
from src.utils.logger import get_logger
from src.utils.retry import call_with_retry

logger = get_logger("community.summarizer")

DESC_MAX_CHARS = 200
DEFAULT_SAMPLE_ENTITIES = 20
DEFAULT_SAMPLE_RELATIONS = 30

_SYMBOL_PATTERNS = (
    re.compile(r"^[A-Za-z]$"),
    re.compile(r"^[A-Za-z]_?\d*$"),
    re.compile(r"^[\W_]+$"),
)

_encoding: Any = None


def _get_encoding():
    global _encoding
    if _encoding is None:
        import tiktoken

        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def truncate_to_tokens(text: str, limit: int) -> str:
    """按 token 上限截断文本（保留原文顺序）。"""
    if limit <= 0:
        return ""
    enc = _get_encoding()
    tokens = enc.encode(text)
    if len(tokens) <= limit:
        return text
    return enc.decode(tokens[:limit])


def is_symbol_like(name: str) -> bool:
    """判断实体名是否为通用符号（单字母/带下标变量/纯标点），摘要取样时降权。"""
    n = (name or "").strip()
    if not n:
        return True
    return any(p.match(n) for p in _SYMBOL_PATTERNS)


def rank_entities(entities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """非符号实体优先、度数降序；符号实体仍保留在尾部。"""
    def key(item: dict[str, Any]) -> tuple[int, int, str]:
        name = str(item.get("name") or "")
        return (1 if is_symbol_like(name) else 0, -int(item.get("degree") or 0), name)

    return sorted(entities, key=key)


def sample_text(
    payload: dict[str, Any],
    *,
    max_tokens: int,
    max_entities: int = DEFAULT_SAMPLE_ENTITIES,
    max_relations: int = DEFAULT_SAMPLE_RELATIONS,
) -> str:
    """把社区结构渲染成受 token 预算约束的取样文本。"""
    lines: list[str] = []
    docs = payload.get("docs") or []
    if docs:
        lines.append("覆盖文档：" + "、".join(str(d) for d in docs[:8]))
    types = payload.get("entity_types") or {}
    if types:
        top_types = list(types.items())[:8]
        lines.append("实体类型分布：" + "、".join(f"{k}×{v}" for k, v in top_types))
    lines.append(f"社区规模：{int(payload.get('size') or 0)} 个实体")

    entities = rank_entities(list(payload.get("entities") or []))[: int(max_entities)]
    if entities:
        lines.append("代表实体：")
        for e in entities:
            desc = str(e.get("description") or "").strip().replace("\n", " ")[:DESC_MAX_CHARS]
            lines.append(f"- {e.get('name')} [{e.get('type') or 'unknown'}]：{desc}")

    relations = sorted(
        (r for r in (payload.get("relations") or []) if isinstance(r, dict)),
        key=lambda r: (-float(r.get("weight") or 1.0), str(r.get("src") or ""), str(r.get("tgt") or "")),
    )[: int(max_relations)]
    if relations:
        lines.append("关键关系：")
        for r in relations:
            desc = str(r.get("description") or "").strip().replace("\n", " ")[:DESC_MAX_CHARS]
            weight = float(r.get("weight") or 1.0)
            lines.append(f"- {r.get('src')} → {r.get('tgt')} (w={weight:g})：{desc}")

    text = "\n".join(lines).strip()
    return truncate_to_tokens(text, int(max_tokens))


def build_messages(payload: dict[str, Any], sample: str, *, language: str = "Chinese") -> list[dict[str, str]]:
    community_id = str(payload.get("community_id") or "")
    system = (
        "你是科研文献知识图谱的分析助手。请把给定社区的实体与关系归纳成一个主题摘要，"
        f"输出语言：{language}。只使用提供的材料，不要引入外部知识或编造数值结论。"
    )
    user = (
        "以下是从 FEC（前向纠错编码）领域文献图谱中划分出的一个实体社区，"
        "请归纳它的主题、核心结论与涉及的关键实体。\n\n"
        f"社区ID：{community_id}\n"
        f"{sample}\n\n"
        "严格输出 JSON（不要 markdown 代码块、不要额外解释），字段：\n"
        '{"title": "不超过20字的主题名", "summary": "150-300字的中文概述", '
        '"key_points": ["要点1", "要点2", "要点3"], "entities": ["关键实体名", "…"]}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_summary_json(text: str) -> dict[str, Any]:
    """解析 LLM 输出；解析失败时降级为"整段作为 summary"。"""
    raw = (text or "").strip()
    if not raw:
        return {"title": "", "summary": "", "key_points": [], "entities": []}
    cleaned = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if start >= 0 and end > start else cleaned
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        head = raw.splitlines()[0].strip()[:40]
        return {"title": head, "summary": raw[:2000], "key_points": [], "entities": []}
    if not isinstance(data, dict):
        return {"title": "", "summary": raw[:2000], "key_points": [], "entities": []}

    def _clean_list(value: Any, limit: int) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            s = str(item).strip()
            if s and s not in out:
                out.append(s)
            if len(out) >= limit:
                break
        return out

    return {
        "title": str(data.get("title") or "").strip()[:80],
        "summary": str(data.get("summary") or "").strip()[:2000],
        "key_points": _clean_list(data.get("key_points"), 5),
        "entities": _clean_list(data.get("entities"), 10),
    }


def content_hash(
    members: Sequence[str],
    sample: str,
    *,
    prompt_version: str,
    model: str,
) -> str:
    h = hashlib.sha1()
    h.update("|".join(sorted(members)).encode("utf-8"))
    h.update(b"\x02")
    h.update(sample.encode("utf-8"))
    h.update(b"\x02")
    h.update(f"{prompt_version}|{model}".encode("utf-8"))
    return h.hexdigest()


def client_and_model(settings: Settings) -> tuple[Any, str]:
    """返回（AsyncOpenAI 客户端, 摘要模型名）。"""
    api_key = (settings.openai_api_key or settings.llm.api_key or "").strip()
    if not api_key:
        raise RuntimeError("社区摘要需要 OPENAI_API_KEY / LLM_API_KEY")
    base = (settings.openai_base_url or settings.llm.base_url or "").strip().rstrip("/")
    if not base:
        base = "https://api.openai.com/v1"
    model = (settings.community.summary_model or "").strip() or settings.resolved_llm_model_name()
    return get_openai_client(api_key=api_key, base_url=base), model


async def summarize_community(
    payload: dict[str, Any],
    sample: str,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """调用 LLM 生成单个社区摘要（失败抛异常，由调用方决定是否跳过）。"""
    s = settings or get_settings()
    client, model = client_and_model(s)
    messages = build_messages(payload, sample)
    concurrency = int(s.community.max_concurrent_summaries)

    async def _call() -> Any:
        async with get_semaphore("community_summary_llm", concurrency):
            return await client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=messages,
            )

    resp = await call_with_retry(_call, max_retries=int(s.community.max_retries))
    text = (resp.choices[0].message.content or "").strip()
    parsed = parse_summary_json(text)
    parsed["model"] = model
    return parsed
