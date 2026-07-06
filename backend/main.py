import shutil
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import API_KEY, DB_PATH, UPLOAD_DIR
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    DeleteResponse,
    HealthResponse,
    SourcesResponse,
    UploadResponse,
    UrlRequest,
)
from backend.services.rag_service import ChefAIRAGService
from backend.utils.file_validation import validate_upload_filename


app = FastAPI(
    title="ChefAI RAG API",
    description="FastAPI 后端：封装文档上传、知识入库、RAG 检索问答、知识源管理等能力。",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_service: Optional[ChefAIRAGService] = None


def get_service() -> ChefAIRAGService:
    global _service

    if _service is None:
        try:
            _service = ChefAIRAGService()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return _service


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "vector_db_path": DB_PATH,
        "has_api_key": bool(API_KEY),
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    支持：
    .pdf .txt .md .docx .xlsx .xls .csv .png .jpg .jpeg .bmp

    当前不支持 .doc，因为现有解析器不能稳定处理老 Word .doc。
    """
    try:
        safe_name = validate_upload_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    service = get_service()

    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = upload_dir / temp_name

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = service.ingest_file(str(file_path), safe_name)

        return {
            "message": "文件解析并入库成功",
            "source_name": safe_name,
            **result,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文件处理失败：{exc}") from exc

    finally:
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/api/url", response_model=UploadResponse)
def ingest_url(req: UrlRequest):
    service = get_service()

    try:
        result = service.ingest_url(req.url)

        return {
            "message": "网页解析并入库成功",
            "source_name": req.url,
            **result,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"网页处理失败：{exc}") from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    service = get_service()

    try:
        return service.chat(req.question, req.session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"问答失败：{exc}") from exc


@app.get("/api/sources", response_model=SourcesResponse)
def list_sources():
    service = get_service()
    return {"sources": service.list_sources()}


@app.delete("/api/sources", response_model=DeleteResponse)
def delete_source(source: str = Query(..., description="要删除的数据源名称")):
    service = get_service()
    decoded_source = unquote(source)
    deleted = service.delete_source(decoded_source)

    return {
        "message": "删除完成" if deleted else "未找到该数据源",
        "deleted_chunks": deleted,
    }


@app.delete("/api/session/{session_id}", response_model=DeleteResponse)
def clear_session(session_id: str):
    service = get_service()
    service.clear_session(session_id)

    return {
        "message": "会话已清空",
        "deleted_chunks": 0,
    }