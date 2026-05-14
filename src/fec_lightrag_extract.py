#!/usr/bin/env python3
"""
LightRAG 结构化抽取：图 + JsonKV 写入 data/<markdown 主名>/，本地占位向量。

输出目录、doc_id、manifest 的 document_title 均由 `--markdown` 文件主名推导。
依赖：pip install lightrag-hku networkx nano-vectordb openai

启动时从仓库根目录 `.env` 注入进程环境（不列举变量名；其余由 LightRAG / OpenAI 兼容客户端读取环境）。
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

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
FEC_EXTRACTION_SYSTEM_APPEND = (
    "\n\n【FEC 附图/表】\n"
    "图：遇 `![…](相对路径)` 或紧邻图题「图 m-n」→ 抽成实体，entity_type=image_asset，"
    "entity_name=`图m-n:相对路径`（英文冒号仅一处；无图号时可用路径作名）。\n"
    "`<!-- fec_figure path=… -->` 与指向同一文件的 `![…](path)` 合并为一条实体。\n"
    "禁止：entity_name 仅为「图 m-n」等图题、不含冒号后相对路径（无 `:` 后路径段、无 `images/` 等文件路径）时，"
    "不得使用 entity_type=image_asset。\n"
    "表：正文「表 m-n」→ entity_type=table，entity_name 与书中引用一致。\n"
    "\n"
    "【FEC 实体 description 硬性要求】（tuple 第四段，须严格遵守）\n"
    "1) 公式类：凡实体对应书中显式公式、等式、码多项式、生成/校验关系等数学表达式（含实体名形如「公式(m-n)」、"
    "或类型属数学公式/数学方法等与式子直接相关者），description 中必须写出【完整公式表达式】——"
    "与原文一致的符号、下标、多项式或矩阵形式（可含 LaTeX `$…$` 或书中原有记法），"
    "禁止仅用「公式(m-n)给出某某」之类元叙述而不写出式子本身。\n"
    "2) 专业术语：对编码/信道/译码等专业名词，description 必须包含【完整定义】——"
    "从正文摘录或等价复述：条件、对象、结论/性质须齐全；禁止「某某是重要概念」「详见上文」等空泛句。\n"
    "3) 全体实体：每条 description 须含【具体内容】（事实、数据、式子、步骤、判据等），使读者仅凭该段即可把握该实体；"
    "禁止笼统概括、标题复述、无信息增量的套话。\n"
    "4) entity_type=table：description 必须包含【表内实质内容】——至少列出表头与各行列关键数据/码字/多项式对应关系"
    "（可用 Markdown 表或分行枚举复现正文表格信息），禁止仅写「表m-n列出了…」而不给出表体。"
)

_FEC_JSON_EXTRACTION_EXAMPLE = json.dumps(
    {
        "extraction": (
            f"entity{_TUPLE_DL}图3-2:images/p.png{_TUPLE_DL}image_asset{_TUPLE_DL}图注一句。\n{_COMPLETE_DL}"
        )
    },
    ensure_ascii=False,
)

_JSON_MODE_SYSTEM_SUFFIX = (
    "\n\n【JSON】只输出一个 JSON 对象，且仅含键 extraction（小写）；"
    "值为字符串，与未开 JSON 时 LightRAG 要求的正文逐字相同（含换行与末行结束符）。\n"
    "示例：\n"
    f"{_FEC_JSON_EXTRACTION_EXAMPLE}\n"
    "禁止 Markdown 围栏；禁止 JSON 外任何字符。"
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
        messages = [{"role": "user", "content": "只回复一个字：好"}]
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
    from lightrag import LightRAG

    llm_model_func, llm_model_name = _build_openai_llm()
    embedding_func = _build_stub_embedding()

    rag = LightRAG(
        working_dir=str(work_dir),
        workspace=args.workspace,
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
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap,
    )

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
