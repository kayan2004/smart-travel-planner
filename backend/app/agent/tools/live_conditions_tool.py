from pydantic import BaseModel

from app.agent.tools.base import BaseTool, ToolContext
from app.schemas.live_conditions import LiveConditionsRequest, LiveConditionsResponse
from app.services.live_conditions import get_live_conditions


class LiveConditionsTool(BaseTool):
    name = "live_conditions"
    description = "Looks up current weather conditions for a destination or city."
    input_model = LiveConditionsRequest

    async def arun(
        self,
        payload: BaseModel,
        context: ToolContext,
    ) -> LiveConditionsResponse:
        if not isinstance(payload, LiveConditionsRequest):
            raise TypeError("LiveConditionsTool received an invalid payload type.")
        if context.http_client is None:
            raise RuntimeError("HTTP client is required for live conditions.")

        return await get_live_conditions(
            context.http_client,
            context.settings,
            payload,
        )

