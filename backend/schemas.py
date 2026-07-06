from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field("default", description="会话 ID")


class SourceItem(BaseModel):
    source: str
    page: Optional[int] = None
    file_type: Optional[str] = None
    rerank_score: Optional[float] = None
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    is_confident: bool
    confidence_score: float
    rewritten_query: str
    retrieval_strategy: str = "hybrid"
    sources: List[SourceItem] = Field(default_factory=list)


class UploadResponse(BaseModel):
    message: str
    source_name: str
    parent_chunks: int
    child_chunks: int
    total_chunks: int


class UrlRequest(BaseModel):
    url: str = Field(..., min_length=5)


class SourceInfo(BaseModel):
    source: str
    count: int
    file_type: str
    parent_chunks: int = 0
    child_chunks: int = 0


class SourcesResponse(BaseModel):
    sources: List[SourceInfo] = Field(default_factory=list)


class DeleteResponse(BaseModel):
    message: str
    deleted_chunks: int


class HealthResponse(BaseModel):
    status: str
    vector_db_path: str
    has_api_key: bool