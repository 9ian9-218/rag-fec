#!/usr/bin/env python3
"""
LightRAG 结构化抽取：图 + JsonKV 写入 data/<markdown 主名>/，本地占位向量。

fec_structured_export/：graphml、entities.json、relations.json、full_doc.json；
  --export-text-chunks 时另复制 text_chunks.json。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from prompt_texts import load_prompt

load_dotenv(_REPO_ROOT / ".env", override=False)


def _ensure_lightrag_installed() -> None:
    try:
        import lightrag  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(1) from None


DEFAULT_MD = _REPO_ROOT / "data" / "差错控制编码.md"
_DEFAULT_LLM_TIMEOUT_S = "600"

FEC_ENTITY_KIND_TYPES: tuple[str, ...] = (
    "coding_paradigm",
    "coding_scheme",
    "encoding_methods",
    "decoding_methods",
    "code_instance",
    "modulation_and_demodulation_methods",
    "math_methods_or_math_interpretations",
    "channel_model",
    "channel_phenomena",
    "case_for_illustration",
    "image_asset",
    "table",
)
FEC_ENTITY_TYPES: list[str] = list(FEC_ENTITY_KIND_TYPES)

_TUPLE_DL = "<|#|>"
_COMPLETE_DL = "<|COMPLETE|>"

FEC_EXTRACTION_SYSTEM_APPEND = load_prompt("fec_extraction_system_append.txt")

_FEC_JSON_EXTRACTION_EXAMPLE = json.dumps(
    {
        "extraction": (
            f"entity{_TUPLE_DL}图3-2:images/p.png{_TUPLE_DL}image_asset{_TUPLE_DL}图注一句。\n{_COMPLETE_DL}"
        )
    },
    ensure_ascii=False,
)

_JSON_MODE_SYSTEM_SUFFIX = load_prompt(
    "json_mode_system_suffix.txt",
    fec_json_extraction_example=_FEC_JSON_EXTRACTION_EXAMPLE,
)

_CREATE_EXTRA_KEYS = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "stop",
        "seed",
        "logit_bias",
    }
)


def _keyword_extraction_via_json_object() -> bool:
    """
    OpenAI 官方 API 可用 completions.parse + 结构化 schema；
    DeepSeek 等会报 response_format 不可用，应走 json_object 或纯文本 JSON。
    """
    if _env_truthy("LIGHTRAG_KEYWORD_USE_OPENAI_PARSE"):
        return False
    if _env_truthy("LIGHTRAG_KEYWORD_JSON_OBJECT"):
        return True
    base = (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").lower()
    return "deepseek" in base


async def _openai_keyword_extraction_compat(
    *,
    model: str,
    prompt: str,
    system_prompt: str | None,
    history_messages: list | None,
    base_url: str,
    api_key: str | None,
    http_timeout: int,
    client_configs: dict | None,
    safe_kw: dict[str, Any],
) -> str:
    """关键词抽取：避免 LightRAG 默认的 parse + GPTKeywordExtractionFormat（部分网关不支持）。"""
    from lightrag.llm.openai import create_openai_async_client

    bu = base_url.rstrip("/")
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for rf in ({"type": "json_object"}, None):
        try:
            client = create_openai_async_client(
                api_key=api_key,
                base_url=bu,
                timeout=http_timeout,
                client_configs=client_configs or None,
            )
            kwargs: dict[str, Any] = {**safe_kw, "timeout": http_timeout}
            if rf is not None:
                kwargs["response_format"] = rf
            async with client:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            if rf is None:
                break
            print(
                f"[fec_lightrag_extract] keyword 使用 json_object 失败，将重试无 response_format: {e}",
                file=sys.stderr,
            )
    assert last_err is not None
    raise last_err


def _effective_llm_timeout_seconds() -> int:
    return int(os.getenv("LLM_TIMEOUT", _DEFAULT_LLM_TIMEOUT_S))


def _effective_openai_http_timeout_seconds() -> int:
    return int(os.getenv("OPENAI_TIMEOUT", str(_effective_llm_timeout_seconds())))


def _env_truthy(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _run_paths_and_doc_from_markdown(md_path: Path) -> tuple[Path, str, str]:
    stem = (md_path.stem or "").strip() or "unnamed"
    out_dir = (_REPO_ROOT / "data" / stem).resolve()
    return out_dir, stem, stem


def _looks_like_lightrag_tuple_extraction(text: str) -> bool:
    s = (text or "").strip()
    if _TUPLE_DL not in s:
        return False
    for line in s.splitlines():
        t = line.strip()
        if t.startswith("entity" + _TUPLE_DL) or t.startswith("relation" + _TUPLE_DL):
            return True
    return False


def _ensure_lightrag_extraction_complete(text: str) -> str:
    if not text or not text.strip():
        return text
    if _COMPLETE_DL in text:
        return text
    if not _looks_like_lightrag_tuple_extraction(text):
        return text
    return text.rstrip() + "\n" + _COMPLETE_DL


def _unwrap_json_object_extraction(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        if "```" in text:
            text = text.split("```", 1)[0].strip()
    if not text or not (text.startswith("{") or text.startswith("[")):
        return text or ""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return raw.strip()
    if isinstance(obj, dict):
        for k in ("extraction", "content", "output", "result", "text", "answer"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in obj.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return raw.strip()


async def _probe_openai_only() -> int:
    _apply_lightrag_timeout_env(None)
    _ensure_lightrag_installed()
    from lightrag.llm.openai import create_openai_async_client

    model = (os.getenv("OPENAI_MODEL") or "").strip()
    base_raw = (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").strip()
    base_url = base_raw.rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY")
    timeout = _effective_openai_http_timeout_seconds()
    import time

    t0 = time.perf_counter()
    try:
        client = create_openai_async_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        messages = [{"role": "user", "content": load_prompt("openai_probe_user.txt")}]
        async with client:
            if _env_truthy("OPENAI_JSON_OBJECT_MODE"):
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_tokens=32,
                )
                text = (resp.choices[0].message.content or "").strip()
            else:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=32,
                )
                text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[fec_lightrag_extract] --probe-openai 失败: {e}", file=sys.stderr)
        return 1
    dt = time.perf_counter() - t0
    preview = text[:200].replace("\n", " ")
    print(
        f"[fec_lightrag_extract] --probe-openai 成功: {dt:.2f}s, 预览: {preview!r}",
        file=sys.stderr,
    )
    return 0


def _apply_lightrag_timeout_env(cli_llm_timeout: int | None) -> None:
    if cli_llm_timeout is not None:
        os.environ["LLM_TIMEOUT"] = str(cli_llm_timeout)
    else:
        os.environ.setdefault("LLM_TIMEOUT", _DEFAULT_LLM_TIMEOUT_S)


async def _openai_chat_create_json_object(
    *,
    model: str,
    prompt: str,
    system_prompt: str | None,
    history_messages: list | None,
    base_url: str,
    api_key: str | None,
    http_timeout: int,
    extra: dict,
    client_configs: dict | None,
) -> str:
    from lightrag.llm.openai import create_openai_async_client

    system_eff = (system_prompt or "") + FEC_EXTRACTION_SYSTEM_APPEND + _JSON_MODE_SYSTEM_SUFFIX
    client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        timeout=http_timeout,
        client_configs=client_configs or None,
    )
    messages: list[dict] = [{"role": "system", "content": system_eff}]
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    extra = {**extra, "response_format": {"type": "json_object"}}
    async with client:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            **extra,
        )
    raw = (resp.choices[0].message.content or "").strip()
    return _unwrap_json_object_extraction(raw)


def _build_openai_llm():
    from lightrag.llm.openai import openai_complete_if_cache

    model = (os.getenv("OPENAI_MODEL") or "").strip()
    base_raw = (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").strip()
    base_url = base_raw.rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY")
    http_timeout = _effective_openai_http_timeout_seconds()
    use_json_object = _env_truthy("OPENAI_JSON_OBJECT_MODE")

    async def llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ):
        kwargs.pop("timeout", None)
        kwargs.pop("hashing_kv", None)
        safe_kw = {k: v for k, v in kwargs.items() if k in _CREATE_EXTRA_KEYS}
        client_cfgs = kwargs.get("openai_client_configs")

        if keyword_extraction:
            if _keyword_extraction_via_json_object():
                return await _openai_keyword_extraction_compat(
                    model=model,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    base_url=base_url,
                    api_key=api_key,
                    http_timeout=http_timeout,
                    client_configs=client_cfgs if isinstance(client_cfgs, dict) else None,
                    safe_kw=safe_kw,
                )
            return await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                keyword_extraction=True,
                base_url=base_url,
                api_key=api_key,
                timeout=http_timeout,
                **kwargs,
            )

        system_entity = (system_prompt or "") + FEC_EXTRACTION_SYSTEM_APPEND

        if use_json_object:
            out = await _openai_chat_create_json_object(
                model=model,
                prompt=prompt,
                system_prompt=system_entity,
                history_messages=history_messages or [],
                base_url=base_url,
                api_key=api_key,
                http_timeout=http_timeout,
                extra=safe_kw,
                client_configs=client_cfgs if isinstance(client_cfgs, dict) else None,
            )
            return _ensure_lightrag_extraction_complete(out)

        out = await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_entity,
            history_messages=history_messages or [],
            keyword_extraction=False,
            base_url=base_url,
            api_key=api_key,
            timeout=http_timeout,
            **kwargs,
        )
        return _ensure_lightrag_extraction_complete(out)

    return llm_model_func, model


def _build_stub_embedding():
    import numpy as np
    from lightrag.utils import EmbeddingFunc

    dim = int(os.getenv("STUB_EMBEDDING_DIM", "384"))

    async def _stub(texts: list[str], **_kw):
        n = len(texts)
        if n == 0:
            return np.zeros((0, dim), dtype=np.float32)
        v = np.ones((n, dim), dtype=np.float32) / (dim**0.5)
        return v

    return EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=_stub)


def build_lightrag_instance(
    *,
    working_dir: Path,
    workspace: str = "",
    chunk_token_size: int = 1200,
    chunk_overlap_token_size: int = 100,
):
    """
    与 ``_run`` 中相同的 LightRAG 配置（JsonKV + NetworkX + NanoVectorDB），
    供 Neo4j 图检索侧复用；调用方须 ``await rag.initialize_storages()`` 后再 query。
    """
    _ensure_lightrag_installed()
    from lightrag import LightRAG

    llm_model_func, llm_model_name = _build_openai_llm()
    embedding_func = _build_stub_embedding()
    ws = (workspace or "").strip()
    return LightRAG(
        working_dir=str(working_dir.resolve()),
        workspace=ws,
        llm_model_func=llm_model_func,
        llm_model_name=llm_model_name,
        llm_model_kwargs={},
        embedding_func=embedding_func,
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
        addon_params={"language": "Chinese", "entity_types": FEC_ENTITY_TYPES},
        chunk_token_size=chunk_token_size,
        chunk_overlap_token_size=chunk_overlap_token_size,
    )


async def aquery_lightrag_stores(
    question: str,
    *,
    working_dir: Path,
    workspace: str = "",
    mode: str = "mix",
) -> dict[str, Any]:
    """
    打开工作区存储，执行 ``aquery_data``（图 + 向量上下文，不生成最终 RAG 答案段落）。
    ``mode`` 为 LightRAG 的 ``QueryParam.mode``（如 local / global / hybrid / mix）。
    """
    if not (question or "").strip():
        return {"status": "failure", "message": "empty question", "data": {}}
    wd = working_dir.resolve()
    if not wd.is_dir():
        return {
            "status": "skipped",
            "message": f"LightRAG 工作目录不存在: {wd}",
            "data": {},
        }
    from lightrag.base import QueryParam

    rag = build_lightrag_instance(working_dir=wd, workspace=workspace)
    await rag.initialize_storages()
    try:
        param = QueryParam(mode=mode)
        return await rag.aquery_data(question.strip(), param)
    finally:
        await rag.finalize_storages()


def query_lightrag_stores_sync(
    question: str,
    *,
    working_dir: Path,
    workspace: str = "",
    mode: str = "mix",
) -> dict[str, Any]:
    return asyncio.run(
        aquery_lightrag_stores(
            question, working_dir=working_dir, workspace=workspace, mode=mode
        )
    )


def _dir_looks_like_lightrag_workdir(d: Path) -> bool:
    if not d.is_dir():
        return False
    for name in ("kv_store_full_docs.json", "kv_store_doc_status.json"):
        if (d / name).is_file():
            return True
    return False


def resolve_lightrag_workdir_from_data_spec(spec: str | Path) -> Path:
    """
    将 ``data`` 下的导出目录名或路径解析为 LightRAG ``working_dir``。

    接受例如：

    - ``差错控制编码_第05章`` → ``<repo>/data/差错控制编码_第05章/lightrag_workdir``
    - ``data/差错控制编码_第05章``（相对仓库根）
    - 已指向 ``.../lightrag_workdir`` 的绝对路径

    若目录本身已含 kv_store 等文件（即直接就是 workdir），则原样返回。
    """
    raw = Path(str(spec).strip()).expanduser()
    if not str(raw):
        raise ValueError("lightrag 数据源路径为空")

    if raw.is_absolute():
        base = raw.resolve()
    else:
        parts = raw.parts
        if parts and parts[0] in ("data", "Data"):
            base = (_REPO_ROOT / raw).resolve()
        else:
            base = (_REPO_ROOT / "data" / raw).resolve()

    if base.name == "lightrag_workdir" and base.is_dir():
        return base
    nested = base / "lightrag_workdir"
    if nested.is_dir():
        return nested.resolve()
    if _dir_looks_like_lightrag_workdir(base):
        return base
    raise ValueError(
        "无法解析 LightRAG 工作目录："
        f"期望 {nested} 存在，或 {base} 下已有 kv_store_*；当前 base={base}"
    )


def query_lightrag_for_markdown_doc(
    question: str,
    markdown_path: Path,
    *,
    workspace: str = "",
    mode: str = "mix",
) -> dict[str, Any]:
    """根据源 markdown 定位 ``data/<主名>/lightrag_workdir`` 并查询 LightRAG。"""
    out_dir, _stem, _ = _run_paths_and_doc_from_markdown(markdown_path.resolve())
    work_dir = out_dir / "lightrag_workdir"
    return query_lightrag_stores_sync(
        question, working_dir=work_dir, workspace=workspace, mode=mode
    )


def build_fec_lightrag_extraction_chain():
    """
    LangChain Runnable：与 CLI 相同的异步抽取。

    输入 dict：markdown (str|Path)、workspace、max_chars、chunk_token_size、
    chunk_overlap、clear、llm_timeout、export_text_chunks（均可选，缺省与 CLI 一致）。
    输出 dict：out_dir、manifest、lightrag_workdir、doc_key。
    """
    from langchain_core.runnables import RunnableLambda

    def _go(d: dict[str, Any]) -> dict[str, Any]:
        md = Path(d["markdown"]).resolve()
        args = argparse.Namespace(
            markdown=md,
            workspace=str(d.get("workspace") or ""),
            max_chars=int(d.get("max_chars", 0)),
            chunk_token_size=int(d.get("chunk_token_size", 1200)),
            chunk_overlap=int(d.get("chunk_overlap", 100)),
            clear=bool(d.get("clear", False)),
            llm_timeout=d.get("llm_timeout"),
            export_text_chunks=bool(d.get("export_text_chunks", False)),
            probe_openai=False,
        )
        asyncio.run(_run(args))
        out_dir, stem, _ = _run_paths_and_doc_from_markdown(md)
        return {
            "out_dir": str(out_dir),
            "doc_key": stem,
            "manifest": str(out_dir / "manifest.json"),
            "lightrag_workdir": str(out_dir / "lightrag_workdir"),
        }

    return RunnableLambda(_go)


def _export_structured_copies(
    out_dir: Path, work_dir: Path, rel_ws: Path, *, export_text_chunks: bool
) -> dict[str, str]:
    export_dir = out_dir / "fec_structured_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    copies: dict[str, str] = {}

    def _copy(rel_name: str, dest_name: str) -> None:
        src = work_dir / rel_ws / rel_name
        if src.is_file():
            dst = export_dir / dest_name
            shutil.copy2(src, dst)
            copies[dest_name] = str(dst)

    _copy("graph_chunk_entity_relation.graphml", "graph_chunk_entity_relation.graphml")
    _copy("kv_store_full_entities.json", "entities.json")
    _copy("kv_store_full_relations.json", "relations.json")
    _copy("kv_store_full_docs.json", "full_doc.json")
    if export_text_chunks:
        _copy("kv_store_text_chunks.json", "text_chunks.json")
    return copies


async def _run(args: argparse.Namespace) -> None:
    _apply_lightrag_timeout_env(args.llm_timeout)

    md_path = args.markdown.resolve()
    out_dir, doc_key, document_title = _run_paths_and_doc_from_markdown(md_path)
    work_dir = out_dir / "lightrag_workdir"
    if args.clear and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    text = md_path.read_text(encoding="utf-8")
    if args.max_chars > 0:
        text = text[: args.max_chars]

    _ensure_lightrag_installed()
    rag = build_lightrag_instance(
        working_dir=work_dir,
        workspace=args.workspace,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap,
    )
    llm_model_name = rag.llm_model_name

    await rag.initialize_storages()
    await rag.ainsert(text, ids=[doc_key], file_paths=[str(md_path)])
    await rag.finalize_storages()

    ws = (args.workspace or "").strip()
    rel_ws = Path(ws) if ws else Path()
    graphml = work_dir / rel_ws / "graph_chunk_entity_relation.graphml"

    export_paths = _export_structured_copies(
        out_dir,
        work_dir,
        rel_ws,
        export_text_chunks=args.export_text_chunks,
    )

    manifest = {
        "step": "extract",
        "markdown_path": str(md_path),
        "doc_id": doc_key,
        "document_title": document_title,
        "fec_entity_types": FEC_ENTITY_TYPES,
        "llm_backend": "openai_compatible_api",
        "embedding_backend": "stub_local",
        "openai_json_object_mode": _env_truthy("OPENAI_JSON_OBJECT_MODE"),
        "llm_timeout_seconds": _effective_llm_timeout_seconds(),
        "openai_http_timeout_seconds": _effective_openai_http_timeout_seconds(),
        "llm_model": llm_model_name,
        "max_chars_applied": args.max_chars if args.max_chars > 0 else None,
        "chunking_mode": "lightrag_token_window",
        "chunk_token_size": args.chunk_token_size,
        "chunk_overlap_token_size": args.chunk_overlap,
        "workspace": ws or None,
        "lightrag_working_dir": str(work_dir),
        "artifacts": {
            "graphml": str(graphml),
            "kv_full_docs": str(work_dir / rel_ws / "kv_store_full_docs.json"),
            "kv_text_chunks": str(work_dir / rel_ws / "kv_store_text_chunks.json"),
            "kv_full_entities": str(work_dir / rel_ws / "kv_store_full_entities.json"),
            "kv_full_relations": str(work_dir / rel_ws / "kv_store_full_relations.json"),
            "kv_doc_status": str(work_dir / rel_ws / "kv_store_doc_status.json"),
        },
        "fec_structured_export": export_paths or None,
        "downstream_alignment": {
            "chapter8": "数据准备、分块与索引管线（本书 Milvus 在后续阶段单独构建）",
            "chapter9_sec2": "图数据建模与准备：本目录 entities/relations/graphml 可作为 Neo4j 导入前结构化输入",
            "chapter9_sec3": "Milvus 索引构建：待 Neo4j/图结构稳定后，用正式 embedding 对 text chunk 重编码入库",
        },
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "import_hint": "python3 src/fec_neo4j_import.py -s " + shlex.quote(str(md_path)),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FEC: LightRAG 结构化抽取 → data/<markdown 主文件名>/（占位向量）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    p.add_argument("--workspace", default="", help="LightRAG workspace 子目录，默认空")
    p.add_argument("--max-chars", type=int, default=0, help="0 表示全文")
    p.add_argument("--chunk-token-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=100)
    p.add_argument(
        "--clear",
        action="store_true",
        help="删除 data/<markdown 主文件名>/lightrag_workdir 后重抽",
    )
    p.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help="写入 LLM_TIMEOUT（秒）；不传则默认 600",
    )
    p.add_argument(
        "--export-text-chunks",
        action="store_true",
        help="同时复制 kv_store_text_chunks.json（体积可能很大）",
    )
    p.add_argument(
        "--probe-openai",
        action="store_true",
        help="不做抽取，仅发一条最短 chat 探测 LLM 是否可达",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.probe_openai:
        raise SystemExit(asyncio.run(_probe_openai_only()))
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
