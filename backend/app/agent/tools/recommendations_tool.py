from pydantic import BaseModel

from app.agent.tools.base import BaseTool, ToolContext
from app.schemas.recommendations import (
    DestinationRecommendationRequest,
    DestinationRecommendationResponse,
)
from app.services.destination_recommendations import recommend_destinations


class DestinationRecommendationsTool(BaseTool):
    name = "destination_recommender"
    description = (
        "Recommends destinations via a structured SQL pre-filter followed by a "
        "pgvector cosine similarity re-rank."
    )
    input_model = DestinationRecommendationRequest

    async def arun(
        self,
        payload: BaseModel,
        context: ToolContext,
    ) -> DestinationRecommendationResponse:
        if not isinstance(payload, DestinationRecommendationRequest):
            raise TypeError("DestinationRecommendationsTool received an invalid payload type.")
        if context.session is None:
            raise RuntimeError("Database session is not available.")
        if context.http_client is None:
            raise RuntimeError("HTTP client is not available.")

        return await recommend_destinations(
            context.session,
            context.http_client,
            context.settings,
            payload,
        )
