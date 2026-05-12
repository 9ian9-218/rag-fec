#!/usr/bin/env python3
"""
1) 使用 LightRAG 对《差错控制编码》等 Markdown 做实体/关系抽取。

图与 KV 均写入本地工作目录（不连 Neo4j），便于与 2) 导入脚本解耦。
中间结果默认目录：data/fec_lightrag_intermediate/

依赖：pip install lightrag-hku networkx nano-vectordb openai

**配置全部来自仓库根目录 `.env`（勿提交版本库）**，不再内置固定网关地址，也不支持 Ollama 本地模型。

必需（抽取 / 嵌入走同一套 OpenAI 兼容 HTTP API 时）：
  - OPENAI_API_BASE 或 OPENAI_BASE_URL：API 根地址（通常以 `/v1` 结尾，如 `https://api.deepseek.com/v1`）
  - OPENAI_API_KEY
  - OPENAI_MODEL：对话与实体抽取所用模型 id

嵌入（非 `--structured-only` 时）：
  - EMBEDDING_MODEL：默认 `text-embedding-3-small`
  - EMBEDDING_DIM：向量维度，默认 `1536`（需与所用嵌入模型一致）

可选：
  - OPENAI_JSON_OBJECT_MODE=1：对 **非 keyword_extraction** 的补全使用
    `response_format={"type":"json_object"}`，并在系统提示中要求模型将原 delimited 正文放在
    JSON 键 `extraction` 的字符串值中，脚本再解包给 LightRAG。若网关不支持会报错，请改回 0。

  - LLM_TIMEOUT / OPENAI_TIMEOUT：见脚本 `--llm-timeout` 说明。

  仅要结构化结果、暂不调用真实嵌入 API：
  python3 src/fec_lightrag_extract.py --structured-only

fec_structured_export/ 下各文件（由 LightRAG 的 JsonKV + NetworkX 导出）：
  entities.json / relations.json / graph_chunk_entity_relation.graphml / full_doc.json
  text_chunks.json（需 --export-text-chunks）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv


def _ensure_lightrag_installed() -> None:
    try:
        import lightrag  # noqa: F401
    except ModuleNotFoundError:
        print(
            "未找到 Python 包 `lightrag`（由 PyPI 的 lightrag-hku 提供）。\n"
            "请在当前环境中执行:\n"
            "  pip install lightrag-hku networkx nano-vectordb\n"
            "若曾用「可编辑安装」链到已删除的 LightRAG/ 目录，请先:\n"
            "  pip uninstall lightrag-hku -y && pip install lightrag-hku\n"
            "若已安装旧包 lightrag==0.1.0b6 且冲突，请先: pip uninstall lightrag -y\n"
            "再: pip install lightrag-hku",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def _merge_dotenv_when_empty() -> None:
    """先 load_dotenv；若 shell 里 export 了空字符串（常见于误配），用 .env 里的非空值补上。"""
    load_dotenv(_REPO_ROOT / ".env", override=False)
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for key, val in dotenv_values(env_path).items():
        if val is None or str(val).strip() == "":
            continue
        cur = os.environ.get(key)
        if cur is None or (isinstance(cur, str) and cur.strip() == ""):
            os.environ[key] = str(val).strip()


_merge_dotenv_when_empty()

FEC_ENTITY_TYPES = [
    "Document",
    "Chapter",
    "Concept",
    "Entity",
    "Attribute",
    "Case",
]

DEFAULT_MD = _REPO_ROOT / "data" / "差错控制编码.md"
DEFAULT_OUT = _REPO_ROOT / "data" / "fec_lightrag_intermediate"

# LightRAG 默认 LLM_TIMEOUT=180，内部 Worker 上限约为 2×该值（秒）
_DEFAULT_LLM_TIMEOUT_S = "600"

_JSON_MODE_SYSTEM_SUFFIX = (
    "\n\n【JSON 输出协议】为便于服务端 json_object 模式校验，你必须只输出一个 JSON 对象，"
    "且仅包含一个字符串键 \"extraction\"（不要使用其它顶层键）。"
    "\"extraction\" 的值必须是本任务原本要求的那种分隔符元组正文，保持 delimiter 与实体类型约束不变；"
    "不要在 JSON 外输出任何字符，也不要使用 Markdown 代码围栏。"
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


def _require_openai_api_base() -> str:
    _merge_dotenv_when_empty()
    base = (os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "").strip()
    if not base:
        raise RuntimeError(
            "请在仓库根目录 `.env` 中设置 OPENAI_API_BASE 或 OPENAI_BASE_URL（OpenAI 兼容 API 根地址，"
            "通常以 /v1 结尾，例如 https://api.deepseek.com/v1）。脚本不再使用内置默认网关。"
        )
    return base.rstrip("/")


def _require_openai_model() -> str:
    _merge_dotenv_when_empty()
    model = (os.getenv("OPENAI_MODEL") or "").strip()
    if not model:
        raise RuntimeError(
            "请在 `.env` 中设置 OPENAI_MODEL（对话与实体抽取所用模型 id，例如 deepseek-chat）。"
        )
    return model


def _unwrap_json_object_extraction(raw: str) -> str:
    """将 json_object 模式下的整段 JSON 尽量还原为 LightRAG 期望的 delimited 正文。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        if "```" in text:
            text = text.split("```", 1)[0].strip()
    if not text:
        return ""
    if not (text.startswith("{") or text.startswith("[")):
        return text
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


def _log_startup_diagnostic(
    *,
    structured_only: bool,
    embedding_backend: str | None,
) -> None:
    base = _require_openai_api_base()
    model = _require_openai_model()
    key = os.getenv("OPENAI_API_KEY") or ""
    key_ok = len(key.strip()) >= 8
    proxy = os.getenv("ALL_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or ""
    json_mode = _env_truthy("OPENAI_JSON_OBJECT_MODE")
    msg = (
        "[fec_lightrag_extract] 配置快照 — "
        f"structured_only={structured_only}, "
        f"embedding={embedding_backend or 'n/a'}, "
        f"OPENAI_API_BASE={base}, OPENAI_MODEL={model}, "
        f"OPENAI_JSON_OBJECT_MODE={'on' if json_mode else 'off'}, "
        f"OPENAI_API_KEY={'已设置' if key_ok else '未设置或为空'}, "
        f"LLM_TIMEOUT={os.getenv('LLM_TIMEOUT', _DEFAULT_LLM_TIMEOUT_S)}, "
        f"proxy={'有' if proxy else '无'}"
    )
    print(msg, file=sys.stderr)
    print(
        "[fec_lightrag_extract] 说明: LightRAG 会先写入 chunk 向量(Stage1)再调用 LLM 抽实体(Stage2)；"
        "structured-only 使用本地占位向量，Stage1 应秒级完成。"
        "若控制台长期无计费，多为请求未到达网关或卡在连接；请运行 --probe-openai。",
        file=sys.stderr,
    )


async def _probe_openai_only() -> int:
    """单次 chat 调用，验证密钥、BASE_URL、模型是否可用。"""
    _merge_dotenv_when_empty()
    _apply_lightrag_timeout_env(None)
    _ensure_lightrag_installed()
    from lightrag.llm.openai import create_openai_async_client

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        print(
            "[fec_lightrag_extract] OPENAI_API_KEY 为空；若已写 .env，请检查 shell 是否 export 了空变量遮蔽 .env。",
            file=sys.stderr,
        )
        return 1
    model = _require_openai_model()
    base_url = _require_openai_api_base()
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

    system_eff = (system_prompt or "") + _JSON_MODE_SYSTEM_SUFFIX
    client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        timeout=http_timeout,
        client_configs=client_configs or None,
    )
    messages: list[dict] = []
    messages.append({"role": "system", "content": system_eff})
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

    model = _require_openai_model()
    base_url = _require_openai_api_base()
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

        if use_json_object:
            return await _openai_chat_create_json_object(
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=base_url,
                api_key=api_key,
                http_timeout=http_timeout,
                extra=safe_kw,
                client_configs=client_cfgs if isinstance(client_cfgs, dict) else None,
            )

        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            keyword_extraction=False,
            base_url=base_url,
            api_key=api_key,
            timeout=http_timeout,
            **kwargs,
        )

    return llm_model_func, model


def _build_stub_embedding():
    """占位嵌入：不访问网络，仅满足 LightRAG 对向量库的写入；后续 Milvus 请用真实模型重算。"""
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


def _build_openai_embedding_func():
    """从 .env 绑定 base_url / api_key / 模型，供 LightRAG Stage1 向量写入。"""
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    base_url = _require_openai_api_base()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    embed_model = (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    embed_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
    max_token = int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192"))

    inner = partial(
        openai_embed.func,
        model=embed_model,
        base_url=base_url,
        api_key=api_key,
    )
    return EmbeddingFunc(
        embedding_dim=embed_dim,
        max_token_size=max_token,
        func=inner,
    )


def _export_structured_copies(
    out_dir: Path, work_dir: Path, rel_ws: Path, *, export_text_chunks: bool
) -> dict[str, str]:
    """将导入 Neo4j / 后续 Milvus 所需的结构化文件复制到 data 下固定子目录。"""
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
    _merge_dotenv_when_empty()
    _apply_lightrag_timeout_env(args.llm_timeout)

    _require_openai_api_base()
    _require_openai_model()

    embedding_backend: str
    if args.structured_only:
        embedding_backend = "stub_local"
    else:
        embedding_backend = "openai_api"

    _log_startup_diagnostic(
        structured_only=args.structured_only,
        embedding_backend=embedding_backend,
    )

    out_dir: Path = args.output_dir.resolve()
    work_dir = out_dir / "lightrag_workdir"
    if args.clear and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    md_path: Path = args.markdown.resolve()
    text = md_path.read_text(encoding="utf-8")
    if args.max_chars > 0:
        text = text[: args.max_chars]

    _ensure_lightrag_installed()
    from lightrag import LightRAG

    llm_model_kwargs: dict = {}

    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        raise RuntimeError(
            "未设置 OPENAI_API_KEY。请在仓库根目录 `.env` 中填写密钥，"
            "或执行: export OPENAI_API_KEY=你的key"
        )

    llm_model_func, llm_model_name = _build_openai_llm()

    if args.structured_only:
        embedding_func = _build_stub_embedding()
    else:
        embedding_func = _build_openai_embedding_func()

    rag = LightRAG(
        working_dir=str(work_dir),
        workspace=args.workspace,
        llm_model_func=llm_model_func,
        llm_model_name=llm_model_name,
        llm_model_kwargs=llm_model_kwargs,
        embedding_func=embedding_func,
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": 0.2},
        addon_params={
            "language": "Chinese",
            "entity_types": FEC_ENTITY_TYPES,
        },
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap,
    )

    await rag.initialize_storages()

    doc_key = args.doc_id
    await rag.ainsert(
        text,
        ids=[doc_key],
        file_paths=[str(md_path)],
    )
    await rag.finalize_storages()

    ws = (args.workspace or "").strip()
    rel_ws = Path(ws) if ws else Path()

    graphml = work_dir / rel_ws / "graph_chunk_entity_relation.graphml"
    export_paths: dict[str, str] = {}
    if args.structured_only or args.export_structured_copies:
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
        "document_title": args.title,
        "fec_entity_types": FEC_ENTITY_TYPES,
        "llm_backend": "openai_compatible_api",
        "embedding_backend": embedding_backend,
        "structured_only": args.structured_only,
        "openai_json_object_mode": _env_truthy("OPENAI_JSON_OBJECT_MODE"),
        "llm_timeout_seconds": _effective_llm_timeout_seconds(),
        "openai_http_timeout_seconds": _effective_openai_http_timeout_seconds(),
        "llm_model": llm_model_name,
        "max_chars_applied": args.max_chars if args.max_chars > 0 else None,
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
        "import_hint": "运行: python3 src/fec_neo4j_import.py --intermediate-dir "
        + str(out_dir),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="FEC: LightRAG 抽取 → data/fec_lightrag_intermediate（仅 OpenAI 兼容 API + .env）",
        epilog=(
            "常见问题:\n"
            "  • Duplicate document — 请加 --clear 或换 --doc-id。\n"
            "  • Worker execution timeout after 360s — LightRAG 默认 LLM_TIMEOUT=180（Worker≈2×）。"
            "本脚本默认将 LLM_TIMEOUT 提到 600；仍慢可在 .env 设 LLM_TIMEOUT=900 或 "
            "命令行 --llm-timeout 900；HTTP 层可用 OPENAI_TIMEOUT（默认跟随 LLM_TIMEOUT）。\n"
            "  • Retrying /chat/completions — 网关抖动或限流，稍候重试或换模型。\n"
            "  • 长时间运行但控制台无 token — 可能卡在连接、代理 SOCKS、或 shell 里空的 OPENAI_API_KEY"
            " 遮蔽了 .env；请执行 --probe-openai。\n"
            "  • .env 须设置 OPENAI_API_BASE（或 OPENAI_BASE_URL）与 OPENAI_MODEL；可选 OPENAI_JSON_OBJECT_MODE=1。\n"
            "使用 python3 src/fec_lightrag_extract.py -h 查看本页。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--doc-id", default="fec_ecc_zh_md", help="LightRAG 文档 id")
    p.add_argument("--title", default="差错控制编码", help="写入 manifest，供导入建 Document")
    p.add_argument("--workspace", default="", help="LightRAG workspace 子目录，默认空")
    p.add_argument("--max-chars", type=int, default=0, help="0 表示全文")
    p.add_argument("--chunk-token-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=100)
    p.add_argument(
        "--clear",
        action="store_true",
        help="删除 output-dir/lightrag_workdir（含 doc_status、KV、graphml），避免 Duplicate document 并从头抽取",
    )
    p.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        metavar="SEC",
        help="写入环境变量 LLM_TIMEOUT（秒）；不传则默认 600。LightRAG Worker 上限约为其 2 倍",
    )
    p.add_argument(
        "--structured-only",
        action="store_true",
        help="不调用远程 embedding API，仅用占位向量跑通 LightRAG；并复制结构化文件到 fec_structured_export/",
    )
    p.add_argument(
        "--export-structured-copies",
        action="store_true",
        help="即使未加 --structured-only，也将 entities/relations/graphml 复制到 fec_structured_export/",
    )
    p.add_argument(
        "--export-text-chunks",
        action="store_true",
        help="同时复制 kv_store_text_chunks.json（体积可能很大）",
    )
    p.add_argument(
        "--probe-openai",
        action="store_true",
        help="不做抽取，仅发一条最短 chat 验证 BASE_URL/密钥/模型是否可达",
    )
    args = p.parse_args()

    if args.probe_openai:
        raise SystemExit(asyncio.run(_probe_openai_only()))

    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
