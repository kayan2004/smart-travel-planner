from pydantic import BaseModel, Field

from app.schemas.classifier import TravelStylePredictionRequest


class ClaudeTestRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    predicted_style: str | None = Field(default=None, max_length=80)
    destination_name: str | None = Field(default=None, max_length=120)
    response_sections: list[str] = Field(default_factory=list)
    tool_logs: list[dict[str, str]] = Field(default_factory=list)


class ClaudeTestResponse(BaseModel):
    selected_model: str
    generated_text: str


class ExtractedRequestFields(BaseModel):
    destination_name: str | None = Field(default=None, max_length=120)
    location_query: str | None = Field(default=None, max_length=120)
    location_country_code: str | None = Field(default=None, min_length=2, max_length=2)
    travel_profile: TravelStylePredictionRequest | None = None


class ExtractionTestRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


class ExtractionTestResponse(BaseModel):
    selected_model: str
    extracted_fields: ExtractedRequestFields
