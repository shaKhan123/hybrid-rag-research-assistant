"""
Request/response schemas for the API.

Kept separate from main.py so the API contract is easy to find and review
on its own, and reusable if a second interface (e.g. a CLI or another
route) needs the same shapes.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000,
                        description="The question to ask.")
    use_hyde: bool = Field(default=False,
                            description="Rewrite the query via HyDE before retrieval.")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SourceChunk(BaseModel):
    arxiv_id: str
    chunk_index: int
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    is_grounded: bool
    retry_count: int
    groundedness_report: Optional[str] = None
    sources: List[SourceChunk]
    from_cache: bool = False


class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool
    qdrant_collection: Optional[str] = None
    points_count: Optional[int] = None
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None