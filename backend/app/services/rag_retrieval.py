from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

import httpx

from app.core.config import Settings
from app.db.models.destination_document import DestinationDocument
from app.schemas.rag_retrieval import (
    RagRetrievedChunk,
    RagRetrievalRequest,
    RagRetrievalResponse,
)
from app.services.voyage_embeddings import embed_texts


async def retrieve_destination_context(
    session: AsyncSession,
    http_client: httpx.AsyncClient,
    settings: Settings,
    payload: RagRetrievalRequest,
) -> RagRetrievalResponse:
    query_embedding = (
        await embed_texts(
            http_client,
            settings,
            [payload.query.strip()],
            input_type="query",
        )
    )[0]

    distance_expr = DestinationDocument.embedding.cosine_distance(query_embedding)
    statement: Select[tuple[DestinationDocument, float]] = (
        select(DestinationDocument, distance_expr.label("distance"))
        .order_by(distance_expr)
        .limit(payload.top_k)
    )

    if payload.destination_name is not None:
        statement = statement.where(
            DestinationDocument.destination_name == payload.destination_name
        )

    rows = (await session.execute(statement)).all()
    results = [
        RagRetrievedChunk(
            id=document.id,
            destination_name=document.destination_name,
            source_type=document.source_type,
            source_title=document.source_title,
            source_url=document.source_url,
            chunk_index=document.chunk_index,
            content=document.content,
            similarity_score=round(max(0.0, 1.0 - float(distance)), 4),
        )
        for document, distance in rows
    ]

    return RagRetrievalResponse(
        query=payload.query.strip(),
        count=len(results),
        results=results,
    )
