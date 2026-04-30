from pydantic import BaseModel

from app.agent.tools.base import BaseTool, ToolContext
from app.schemas.rag_retrieval import RagRetrievalRequest, RagRetrievalResponse
from app.services.rag_retrieval import retrieve_destination_context


class DestinationContextRetrieverTool(BaseTool):
    name = "destination_context_retriever"
    description = "Retrieves relevant destination context chunks from the vector store."
    input_model = RagRetrievalRequest

    async def arun(
        self,
        payload: BaseModel,
        context: ToolContext,
    ) -> RagRetrievalResponse:
        if not isinstance(payload, RagRetrievalRequest):
            raise TypeError("DestinationContextRetrieverTool received an invalid payload type.")
        if context.session is None:
            raise RuntimeError("Database session is required for retrieval.")
        if context.http_client is None:
            raise RuntimeError("HTTP client is required for query embeddings.")

        return await retrieve_destination_context(
            context.session,
            context.http_client,
            context.settings,
            payload,
        )

