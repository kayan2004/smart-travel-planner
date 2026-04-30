from pydantic import BaseModel, Field


class DiscordWebhookTestRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class DiscordWebhookTestResponse(BaseModel):
    delivered: bool
    channel: str = "discord"
    message_preview: str

