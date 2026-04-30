from pydantic import BaseModel

from app.agent.tools.base import BaseTool, ToolContext
from app.schemas.recommendations import (
    DestinationRecommendationRequest,
    DestinationRecommendationResponse,
)
from app.services.recommendations import recommend_destinations


class DestinationRecommendationsTool(BaseTool):
    name = "destination_recommender"
    description = "Recommends destination candidates from the labeled destination catalog."
    input_model = DestinationRecommendationRequest

    async def arun(
        self,
        payload: BaseModel,
        context: ToolContext,
    ) -> DestinationRecommendationResponse:
        if not isinstance(payload, DestinationRecommendationRequest):
            raise TypeError("DestinationRecommendationsTool received an invalid payload type.")

        catalog = context.resources.get("destination_catalog")
        if catalog is None:
            raise RuntimeError("Destination catalog is not loaded.")

        return recommend_destinations(catalog, payload)

