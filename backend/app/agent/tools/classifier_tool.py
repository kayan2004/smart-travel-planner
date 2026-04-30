from pydantic import BaseModel

from app.agent.tools.base import BaseTool, ToolContext
from app.schemas.classifier import (
    TravelStylePredictionRequest,
    TravelStylePredictionResponse,
)
from app.services.classifier import predict_travel_style


class TravelStyleClassifierTool(BaseTool):
    name = "travel_style_classifier"
    description = "Classifies a structured travel profile into a travel style."
    input_model = TravelStylePredictionRequest

    async def arun(
        self,
        payload: BaseModel,
        context: ToolContext,
    ) -> TravelStylePredictionResponse:
        if not isinstance(payload, TravelStylePredictionRequest):
            raise TypeError("TravelStyleClassifierTool received an invalid payload type.")

        model = context.resources.get("travel_style_model")
        if model is None:
            raise RuntimeError("Travel style model is not loaded.")

        return predict_travel_style(model, payload)

