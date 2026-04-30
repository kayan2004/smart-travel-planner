from pydantic import BaseModel, Field


class RagRetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    destination_name: str | None = Field(default=None, min_length=1, max_length=255)
    top_k: int = Field(default=5, ge=1, le=10)


class RagRetrievedChunk(BaseModel):
    id: int
    destination_name: str
    source_type: str
    source_title: str
    source_url: str | None
    chunk_index: int
    content: str
    similarity_score: float


class RagRetrievalResponse(BaseModel):
    query: str
    count: int
    results: list[RagRetrievedChunk]
