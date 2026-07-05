from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.classifier import TravelStylePredictionRequest
from app.schemas.recommendation_read import RecommendationRead
from app.schemas.tool_logs import ToolLogRead


class AgentRunCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    travel_profile: TravelStylePredictionRequest | None = None
    destination_name: str | None = Field(default=None, min_length=2, max_length=120)
    location_query: str | None = Field(default=None, min_length=2, max_length=120)
    location_country_code: str | None = Field(default=None, min_length=2, max_length=2)
    retrieval_top_k: int = Field(default=3, ge=1, le=8)


class AgentRunRead(BaseModel):
    id: int
    user_id: int
    prompt: str
    response: str
    status: str
    created_at: datetime
    tool_logs: list[ToolLogRead] = []
    recommendations: list[RecommendationRead] = []

    model_config = ConfigDict(from_attributes=True)
