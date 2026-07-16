from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.claude import TravelProfile
from app.schemas.recommendation_read import RecommendationRead
from app.schemas.tool_logs import ToolLogRead


class AgentRunCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    travel_profile: TravelProfile | None = None
    destination_name: str | None = Field(default=None, min_length=2, max_length=120)
    location_query: str | None = Field(default=None, min_length=2, max_length=120)
    location_country_code: str | None = Field(default=None, min_length=2, max_length=2)
    retrieval_top_k: int = Field(default=3, ge=1, le=8)
    # BYOK provider/model selection - only meaningful together with the
    # X-LLM-API-Key header (see app/api/routes/agent_runs.py). Both must be
    # in app/core/llm_allowlist.py's BYOK_ALLOWLIST or the request is
    # rejected with a 400. No max_tokens field here, ever - that stays
    # server-controlled (app/core/config.py), never client-settable.
    llm_provider: str | None = Field(default=None, max_length=32)
    llm_model: str | None = Field(default=None, max_length=64)


class AgentRunRead(BaseModel):
    id: int
    user_id: int
    prompt: str
    response: str
    status: str
    created_at: datetime
    tool_logs: list[ToolLogRead] = []
    recommendations: list[RecommendationRead] = []
    # How many free server-key runs the caller has left after this response.
    # The frontend reveals the BYOK panel when this hits 0 (the "show BYOK
    # once the free prompt is used" behavior). None on rows where it wasn't
    # computed. Not a stored column - populated by the route per request.
    free_runs_remaining: int | None = None

    model_config = ConfigDict(from_attributes=True)


class AgentRunSummary(BaseModel):
    """Lightweight shape for GET /agent-runs (history list) - omits
    tool_logs/recommendations, which GET /agent-runs/{id} (detail) still
    returns in full via AgentRunRead. Keeps the list endpoint cheap
    regardless of how many past runs a user has.
    """

    id: int
    prompt: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
