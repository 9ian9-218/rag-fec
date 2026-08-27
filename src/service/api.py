"""FastAPI REST 介面。"""

from __future__ import annotations

import asyncio
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import get_settings
from src.service.rag_service import RAGService
from src.service.task_manager import AsyncTaskManager
from src.utils.client_cache import aclose_clients
from src.evaluation.online_monitor import stop_telemetry_writer
from src.storage.redis_client import close_redis
from src.utils.logger import get_logger, setup_logging
from src.utils.rate_limit import RedisRateLimiter, TokenBucket

logger = get_logger("service.api")

_task_manager: AsyncTaskManager | None = None
_worker_task: asyncio.Task | None = None


class QueryBody(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    mode: str | None = Field(
        default=None,
        description="顯式指定檢索模式時跳過智能路由；省略則由 LLM 在 naive/local/global/hybrid/mix 中自動選擇",
    )
    auto_mode: bool = Field(
        default=True,
        description="為 True 且未指定 mode 時啟用基于规则的检索路由（不调用 LLM）",
    )
    stream: bool = False
    multimodal: bool = Field(
        default=False,
        description="為 True 時：檢索後解析 chunk 內 ![](images/...) 並送視覺模型（需 API 支援 image_url）",
    )
    include_mode_selection: bool = Field(
        default=False,
        description="為 True 時在 JSON 響應中附帶 mode_selection（難度/複雜度/選中模式）",
    )


class IncrementalBody(BaseModel):
    """可擴充的增量請求體（目前無必填欄位）。"""

    pass


class FeedbackBody(BaseModel):
    """回答反饋請求體。"""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    feedback: str = Field(..., pattern="^(correct|wrong)$")
    session_id: str | None = None


_rag: RAGService | None = None


def get_rag() -> RAGService:
    global _rag
    if _rag is None:
        _rag = RAGService()
    return _rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_task
    setup_logging()
    s = get_settings()

    if s.service.async_document_processing_enabled and _task_manager is not None:
        async def _handle_task(kind: str, payload: dict[str, Any]) -> Any:
            rag = get_rag()
            if kind == "add_document":
                return await rag.add_document(Path(payload["path"]))
            if kind == "update_document":
                return await rag.update_document(Path(payload["path"]))
            if kind == "incremental_update":
                return await rag.incremental_update(
                    convert_first=bool(payload.get("convert_first", False))
                )
            raise ValueError(f"unknown task kind: {kind}")

        _worker_task = asyncio.create_task(
            _task_manager.worker_loop(_handle_task)
        )

    # 预热 LightRAG：把存储初始化（JsonKV/Neo4j/Milvus，约 2 分钟）移到启动阶段。
    # 否则首个查询才惰性初始化，期间事件循环被占用，health 与所有请求长达数分钟无响应
    # （压测预检 "health 失败" / 前端"点击无反应"均由此类冻结导致）。
    try:
        from src.storage.lightrag_init import get_lightrag

        await get_lightrag()
        logger.info("LightRAG 预热初始化完成（请求期不再触发懒加载初始化）")
    except Exception:
        logger.exception("LightRAG 预热初始化失败，将由首个请求重试")

    logger.info("Graph RAG API 啟動")
    yield

    if _worker_task is not None:
        _worker_task.cancel()
        _worker_task = None
    await aclose_clients()
    stop_telemetry_writer()
    await close_redis()
    logger.info("Graph RAG API 關閉")


def create_app() -> FastAPI:
    global _task_manager
    s = get_settings()
    _task_manager = AsyncTaskManager()
    app = FastAPI(title="Graph RAG (LightRAG + Neo4j + Milvus)", lifespan=lifespan)
    rate_limiter = None
    if s.service.rate_limit_enabled:
        if s.redis.enabled:
            rate_limiter = RedisRateLimiter(
                s.service.rate_limit_per_second,
                window_seconds=1,
            )
        else:
            rate_limiter = TokenBucket(
                s.service.rate_limit_per_second,
                s.service.rate_limit_burst,
            )

    origins = [o.strip() for o in s.service.cors_origins.split(",") if o.strip()]
    if origins == ["*"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    front_dir = Path(__file__).resolve().parent.parent.parent / "front"
    if front_dir.is_dir():
        app.mount("/front", StaticFiles(directory=str(front_dir)), name="front")

    @app.get("/", include_in_schema=False)
    async def root():
        index = front_dir / "index.html"
        if index.is_file():
            return FileResponse(str(index))
        return JSONResponse({"message": "Graph RAG API is running", "docs": "/docs"})

    @app.get("/api/rag/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/rag/query")
    async def rag_query(body: QueryBody):
        if rate_limiter is not None:
            if isinstance(rate_limiter, RedisRateLimiter):
                allowed = await rate_limiter.acquire("global")
            else:
                allowed = rate_limiter.acquire()
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Too Many Requests",
                    headers={"Retry-After": "1"},
                )
        rag = get_rag()
        use_router = body.auto_mode if body.mode is None else False

        if body.stream:

            async def gen() -> AsyncIterator[bytes]:
                res = await rag.query(
                    body.question,
                    session_id=body.session_id,
                    mode=body.mode,
                    stream=True,
                    multimodal=body.multimodal,
                    use_llm_router=use_router,
                )
                if hasattr(res, "__aiter__"):
                    async for chunk in res:  # type: ignore[union-attr]
                        if chunk:
                            yield str(chunk).encode("utf-8")
                else:
                    yield str(res).encode("utf-8")

            return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

        text = await rag.query(
            body.question,
            session_id=body.session_id,
            mode=body.mode,
            stream=False,
            multimodal=body.multimodal,
            use_llm_router=use_router,
        )
        payload: dict[str, Any] = {"answer": text}
        if body.include_mode_selection:
            sel = rag.last_mode_selection
            if sel is not None:
                payload["mode_selection"] = sel
                payload["mode"] = sel.get("mode")
        return JSONResponse(payload)

    @app.post("/api/rag/documents")
    async def upload_document(file: UploadFile = File(...)):
        raw = Path(get_settings().paths.data_raw)
        raw.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "upload.bin").name
        dest = raw / safe_name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        if s.service.async_document_processing_enabled and _task_manager is not None:
            task_id = await _task_manager.submit("add_document", {"path": str(dest)})
            return JSONResponse({"task_id": task_id, "status": "pending"}, status_code=202)

        rag = get_rag()
        meta = await rag.add_document(dest)
        return JSONResponse(meta)

    @app.post("/api/rag/documents/batch")
    async def upload_batch(files: list[UploadFile] = File(...)):
        raw = Path(get_settings().paths.data_raw)
        raw.mkdir(parents=True, exist_ok=True)
        rag = get_rag()
        out: list[dict[str, Any]] = []
        if s.service.async_document_processing_enabled and _task_manager is not None:
            for file in files:
                safe_name = Path(file.filename or "upload.bin").name
                dest = raw / safe_name
                with dest.open("wb") as f:
                    shutil.copyfileobj(file.file, f)
                task_id = await _task_manager.submit("add_document", {"path": str(dest)})
                out.append({"task_id": task_id, "status": "pending"})
            return JSONResponse({"items": out}, status_code=202)

        for file in files:
            safe_name = Path(file.filename or "upload.bin").name
            dest = raw / safe_name
            with dest.open("wb") as f:
                shutil.copyfileobj(file.file, f)
            out.append(await rag.add_document(dest))
        return JSONResponse({"items": out})

    @app.put("/api/rag/documents/{doc_id}")
    async def update_document(doc_id: str, file: UploadFile = File(...)):
        rag = get_rag()
        row = rag.get_document(doc_id)
        if not row:
            raise HTTPException(404, detail="找不到文件")
        dest = Path(row["source_path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        if s.service.async_document_processing_enabled and _task_manager is not None:
            task_id = await _task_manager.submit("update_document", {"path": str(dest)})
            return JSONResponse({"task_id": task_id, "status": "pending"}, status_code=202)

        meta = await rag.update_document(dest)
        return JSONResponse(meta)

    @app.delete("/api/rag/documents/{doc_id}")
    async def delete_document(doc_id: str):
        rag = get_rag()
        res = await rag.delete_document_by_id(doc_id)
        return JSONResponse({"result": res})

    @app.get("/api/rag/documents")
    async def list_documents():
        return JSONResponse({"items": get_rag().list_documents()})

    @app.get("/api/rag/documents/{doc_id}")
    async def document_detail(doc_id: str):
        row = get_rag().get_document(doc_id)
        if not row:
            raise HTTPException(404, detail="找不到文件")
        return JSONResponse(row)

    @app.post("/api/rag/incremental-update")
    async def incremental_update(_body: IncrementalBody | None = None):
        if s.service.async_document_processing_enabled and _task_manager is not None:
            task_id = await _task_manager.submit(
                "incremental_update",
                {"convert_first": False},
            )
            return JSONResponse({"task_id": task_id, "status": "pending"}, status_code=202)

        rag = get_rag()
        result = await rag.incremental_update()
        return JSONResponse(result)

    @app.post("/api/rag/feedback")
    async def feedback(body: FeedbackBody):
        from src.evaluation.online_monitor import append_feedback, FeedbackTelemetry

        telem = FeedbackTelemetry(
            question=body.question,
            answer=body.answer,
            feedback=body.feedback,
            session_id=body.session_id,
        )
        append_feedback(telem)
        return JSONResponse({"status": "ok"})

    @app.get("/api/rag/telemetry")
    async def telemetry():
        from src.evaluation.online_monitor import aggregate_feedback_sqlite, aggregate_metrics_sqlite

        query_metrics = aggregate_metrics_sqlite()
        feedback_metrics = aggregate_feedback_sqlite()
        return JSONResponse({"query": query_metrics, "feedback": feedback_metrics})

    @app.get("/api/rag/tasks/{task_id}")
    async def task_status(task_id: str):
        if _task_manager is None:
            raise HTTPException(status_code=404, detail="任务管理器未初始化")
        st = _task_manager.get_status(task_id)
        if st is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return JSONResponse(st)

    return app


app = create_app()
