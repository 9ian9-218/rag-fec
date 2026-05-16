"""檢索後解析 chunk 內 ``![](images/...)``，可選送入 vision API；純文字降級走 ``OPENAI_*``。"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from openai import APIError, AsyncOpenAI, BadRequestError

from config.settings import Settings
from src.retrieval.image_refs import (
    extract_image_refs,
    resolve_local_image_path,
    strip_image_markup,
)
from src.retrieval.result_processor import kg_dict_to_bullets
from src.utils.logger import get_logger

logger = get_logger("retrieval.multimodal_answer")


def _mime_for(path: Path) -> str:
    m, _ = mimetypes.guess_type(str(path))
    return m or "application/octet-stream"


def _b64_data_url(path: Path) -> str:
    raw = path.read_bytes()
    mime = _mime_for(path)
    b64 = base64.standard_b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_chunks_and_kg(bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    inner = bundle.get("data") or {}
    if not isinstance(inner, dict):
        inner = {}
    # LightRAG aquery_data：{status, data:{chunks,...}}；retrieve_data 包裝：{data:{status, data:{chunks}}}
    payload = inner
    if isinstance(inner.get("data"), dict) and (
        "chunks" in inner["data"] or "entities" in inner["data"]
    ):
        payload = inner["data"]
    chunks = payload.get("chunks") or []
    if not isinstance(chunks, list):
        chunks = []
    ent = payload.get("entities") or []
    rel = payload.get("relationships") or []
    if not isinstance(ent, list):
        ent = []
    if not isinstance(rel, list):
        rel = []
    kg_text = kg_dict_to_bullets(ent, rel)
    return [c for c in chunks if isinstance(c, dict)], kg_text


def _trim_chunks_for_reference(
    chunks: list[dict[str, Any]],
    *,
    max_chunk_count: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    """在 LightRAG 已輸出最終 chunks 之後，再按條數與字數預算截斷（供參考上下文與抽圖）。"""
    if max_chunk_count <= 0 or not chunks:
        return []
    out: list[dict[str, Any]] = []
    total = 0
    for c in chunks[:max_chunk_count]:
        content = str(c.get("content") or "")
        if not content.strip() and not str(c.get("file_path") or "").strip():
            continue
        L = len(content)
        if not out and L > max_chars:
            c2 = dict(c)
            c2["content"] = content[:max_chars]
            out.append(c2)
            break
        if total + L > max_chars:
            break
        out.append(c)
        total += L
    return out


def _build_context_from_chunks(
    chunks: list[dict[str, Any]],
    kg_prefix: str,
    *,
    strip_images: bool = False,
) -> str:
    parts: list[str] = []
    if kg_prefix.strip():
        parts.append("【知識圖譜摘要】\n" + kg_prefix.strip())
    for i, c in enumerate(chunks, start=1):
        content = str(c.get("content") or "").strip()
        if strip_images:
            content = strip_image_markup(content).strip()
        fp = str(c.get("file_path") or "").strip()
        if not content:
            continue
        head = f"【片段 {i}】"
        if fp:
            head += f" 來源: {fp}"
        parts.append(head + "\n" + content)
    return "\n\n".join(parts).strip()


def _main_llm_client_and_base(settings: Settings) -> tuple[AsyncOpenAI, str]:
    """純文字問答與多模態降級：頂層 ``OPENAI_*``，其次 ``LLM_*``。"""
    api_key = (settings.openai_api_key or settings.llm.api_key or "").strip()
    if not api_key:
        raise RuntimeError("回答需要 OPENAI_API_KEY / LLM_API_KEY")
    base = (settings.openai_base_url or settings.llm.base_url or "").strip().rstrip("/")
    if not base:
        base = "https://api.openai.com/v1"
    return AsyncOpenAI(api_key=api_key, base_url=base), base


def _multimodal_vision_client_and_base(settings: Settings) -> tuple[AsyncOpenAI, str]:
    """``--multimodal`` 視覺請求：僅 ``MULTIMODAL_*``。"""
    api_key = (settings.multimodal.api_key or "").strip()
    if not api_key:
        raise RuntimeError("多模態回答需要 MULTIMODAL_API_KEY")
    base = (settings.multimodal.base_url or "").strip().rstrip("/")
    if not base:
        base = "https://api.openai.com/v1"
    return AsyncOpenAI(api_key=api_key, base_url=base), base


async def _chat_completion_text(
    *,
    settings: Settings,
    model: str,
    user_text: str,
    history_messages: list[dict[str, str]] | None,
    system_prompt: str | None,
) -> str:
    client, _ = _main_llm_client_and_base(settings)
    messages: list[dict[str, Any]] = []
    sp = (system_prompt or "").strip()
    if sp:
        messages.append({"role": "system", "content": sp})
    if history_messages:
        for h in history_messages:
            role = h.get("role")
            content = h.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    resp = await client.chat.completions.create(
        model=model,
        temperature=float(settings.resolved_llm_temperature()),
        messages=messages,
    )
    choice = resp.choices[0].message.content
    return (choice or "").strip()


def _reference_trim_params(settings: Settings) -> tuple[int, int]:
    max_chars = int(settings.multimodal.reference_context_max_chars)
    max_chunks = int(settings.retrieval.top_k)
    return max_chunks, max_chars


async def answer_with_retrieved_text_only(
    *,
    settings: Settings,
    question: str,
    bundle: dict[str, Any],
    history_messages: list[dict[str, str]] | None = None,
) -> str:
    """僅將檢索到的文字上下文送主 LLM（與多模態共用「先截斷參考、再去圖片語法」策略）。"""
    chunks, kg_text = _extract_chunks_and_kg(bundle)
    mc, mch = _reference_trim_params(settings)
    trimmed = _trim_chunks_for_reference(chunks, max_chunk_count=mc, max_chars=mch)
    logger.info(
        "多模態/純文字參考：chunks %d → 截斷後 %d（rerank/去噪/top_k 由 LightRAG 完成後再套字數預算 %d）",
        len(chunks),
        len(trimmed),
        mch,
    )
    context = _build_context_from_chunks(trimmed, kg_text, strip_images=True)
    if not context.strip():
        raise ValueError("純文字回答：檢索上下文為空")
    user_text = (
        "請根據以下檢索到的文字材料回答用戶問題（材料中已移除圖片鏈接語法，請依文字推理）。\n\n"
        f"【問題】\n{question}\n\n【檢索上下文】\n{context}"
    )
    model = settings.resolved_llm_model_name()
    sys_prompt = (
        settings.multimodal.system_prompt
        or "你是專業助手，請基於提供的檢索材料作答，使用繁體中文或使用者語言。"
    ).strip()
    return await _chat_completion_text(
        settings=settings,
        model=model,
        user_text=user_text,
        history_messages=history_messages,
        system_prompt=sys_prompt,
    )


def _collect_local_image_paths(
    *,
    settings: Settings,
    chunks: list[dict[str, Any]],
) -> list[Path]:
    """僅掃描 ``chunks``（須已為截斷後子集）；圖片達 ``max_images_per_query`` 後立即停止正則與解析。"""
    root = Path(settings.paths.project_root).resolve()
    image_paths: list[Path] = []
    seen: set[str] = set()
    max_n = max(0, settings.multimodal.max_images_per_query)
    if max_n == 0:
        return []
    max_bytes = max(10_000, settings.multimodal.max_image_bytes)
    for c in chunks:
        if len(image_paths) >= max_n:
            break
        content = str(c.get("content") or "")
        fp = str(c.get("file_path") or "").strip() or None
        remaining = max_n - len(image_paths)
        for ref in extract_image_refs(content, max_refs=remaining):
            if len(image_paths) >= max_n:
                break
            lp = resolve_local_image_path(ref, fp, root)
            if lp is None or not lp.is_file():
                continue
            try:
                if lp.stat().st_size > max_bytes:
                    logger.warning("跳過過大圖片 %s (%s bytes)", lp, lp.stat().st_size)
                    continue
            except OSError:
                continue
            key = str(lp.resolve())
            if key in seen:
                continue
            seen.add(key)
            image_paths.append(lp)
            if len(image_paths) >= max_n:
                break
    return image_paths


async def answer_with_retrieved_images(
    *,
    settings: Settings,
    question: str,
    bundle: dict[str, Any],
    history_messages: list[dict[str, str]] | None = None,
) -> str:
    """
    使用 ``aquery_data`` 的 ``chunks`` + 圖譜摘要：在 **LightRAG 已完成 rerank、低分去噪與 chunk 截斷**
    之後，再按 ``reference_context_max_chars`` / ``top_k`` 截參考子集；於該子集上解析圖片（有上限且滿額即停）；
    構造送 LLM 的正文時移除圖片 Markdown/HTML，圖檔以 ``image_url`` 單獨附加。
    """
    chunks, kg_text = _extract_chunks_and_kg(bundle)
    mc, mch = _reference_trim_params(settings)
    trimmed = _trim_chunks_for_reference(chunks, max_chunk_count=mc, max_chars=mch)
    logger.info(
        "多模態抽圖：於最終 chunks %d 中截為 %d 條後再正則取圖（每查詢最多 %d 張）",
        len(chunks),
        len(trimmed),
        settings.multimodal.max_images_per_query,
    )

    image_paths = _collect_local_image_paths(settings=settings, chunks=trimmed)
    context = _build_context_from_chunks(trimmed, kg_text, strip_images=True)
    if not context.strip():
        raise ValueError("多模態回答：檢索上下文為空")

    if not image_paths:
        logger.warning(
            "多模態請求：未從檢索 chunk 解析到可讀的本地圖片（請確認含 ![](images/...) 且 file_path 指向對應 .md 同目錄 images/）。"
            "已改為僅將檢索文字送主語言模型作答。"
        )
        return await answer_with_retrieved_text_only(
            settings=settings,
            question=question,
            bundle=bundle,
            history_messages=history_messages,
        )

    vision_model = settings.resolved_multimodal_model_name()
    resolved_paths = [str(p.resolve()) for p in image_paths]
    logger.info(
        "多模態 vision：model=%s 送入 %d 張圖片 -> %s",
        vision_model,
        len(resolved_paths),
        resolved_paths,
    )
    user_parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "請根據以下檢索到的文字與圖片回答用戶問題。若圖片與問題相關請結合圖中內容說明。\n\n"
                f"【問題】\n{question}\n\n【檢索上下文】\n{context}"
            ),
        }
    ]
    for p in image_paths:
        try:
            user_parts.append({"type": "image_url", "image_url": {"url": _b64_data_url(p)}})
        except OSError as e:
            logger.warning("讀取圖片失敗 %s: %s", p, e)

    messages: list[dict[str, Any]] = []
    sys_prompt = (
        settings.multimodal.system_prompt or "你是專業助手，請基於提供的材料與圖片作答，使用繁體中文或使用者語言。"
    ).strip()
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    if history_messages:
        for h in history_messages:
            role = h.get("role")
            content = h.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_parts})

    client, vision_base = _multimodal_vision_client_and_base(settings)
    logger.info("多模態 vision：base=%s", vision_base)
    try:
        resp = await client.chat.completions.create(
            model=vision_model,
            temperature=float(settings.resolved_multimodal_temperature()),
            messages=messages,
        )
        choice = resp.choices[0].message.content
        return (choice or "").strip()
    except (BadRequestError, APIError) as e:
        logger.warning(
            "視覺模型「%s」或當前 API 不接受圖片輸入（不支援多模態或參數被拒絕）：%s。"
            "已降級為僅將檢索文字上下文送主語言模型「%s」作答。",
            vision_model,
            e,
            settings.resolved_llm_model_name(),
        )
        return await answer_with_retrieved_text_only(
            settings=settings,
            question=question,
            bundle=bundle,
            history_messages=history_messages,
        )
    except Exception as e:
        logger.warning(
            "多模態請求呼叫失敗（%s），已降級為僅將檢索文字送主語言模型「%s」。",
            e,
            settings.resolved_llm_model_name(),
        )
        return await answer_with_retrieved_text_only(
            settings=settings,
            question=question,
            bundle=bundle,
            history_messages=history_messages,
        )
