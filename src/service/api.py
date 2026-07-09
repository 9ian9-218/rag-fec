"""FastAPI REST 介面。"""

from __future__ import annotations

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
from src.utils.logger import get_logger, setup_logging

logger = get_logger("service.api")


class QueryBody(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str | None = None
    mode: str | None = Field(
        default=None,
        description="顯式指定檢索模式時跳過智能路由；省略則由 LLM 在 naive/local/global/hybrid/mix 中自動選擇",
    )
    auto_mode: bool = Field(
        default=True,
        description="為 True 且未指定 mode 時啟用 LLM 智能路由（受 RETRIEVAL_LLM_MODE_ROUTER_ENABLED 約束）",
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
    setup_logging()
    logger.info("Graph RAG API 啟動")
    yield
    logger.info("Graph RAG API 關閉")


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="Graph RAG (LightRAG + Neo4j + Milvus)", lifespan=lifespan)

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
        rag = get_rag()
        use_router = body.auto_mode if body.mode is None else False
        rag.set_llm_mode_router(use_router)

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
        rag = get_rag()
        meta = await rag.add_document(dest)
        return JSONResponse(meta)

    @app.post("/api/rag/documents/batch")
    async def upload_batch(files: list[UploadFile] = File(...)):
        raw = Path(get_settings().paths.data_raw)
        raw.mkdir(parents=True, exist_ok=True)
        rag = get_rag()
        out: list[dict[str, Any]] = []
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
        from src.evaluation.online_monitor import aggregate_metrics_jsonl, DEFAULT_METRICS_LOG, aggregate_feedback_jsonl, DEFAULT_FEEDBACK_LOG

        query_metrics = aggregate_metrics_jsonl(DEFAULT_METRICS_LOG)
        feedback_metrics = aggregate_feedback_jsonl(DEFAULT_FEEDBACK_LOG)
        return JSONResponse({"query": query_metrics, "feedback": feedback_metrics})

    return app


app = create_app()
