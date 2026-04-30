from pydantic import BaseModel, Field


class RagSourceDocument(BaseModel):
    destination_name: str = Field(min_length=1, max_length=255)
    source_type: str = Field(min_length=1, max_length=100)
    source_title: str = Field(min_length=1, max_length=255)
    source_url: str = Field(min_length=1, max_length=1000)


class RagFetchedDocument(BaseModel):
    destination_name: str = Field(min_length=1, max_length=255)
    source_type: str = Field(min_length=1, max_length=100)
    source_title: str = Field(min_length=1, max_length=255)
    source_url: str = Field(min_length=1, max_length=1000)
    content: str = Field(min_length=1)


class RagDocumentChunk(BaseModel):
    destination_name: str
    source_type: str
    source_title: str
    source_url: str
    chunk_index: int
    content: str
