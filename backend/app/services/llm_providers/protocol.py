from typing import Literal, Protocol, TypedDict

ModelTier = Literal["fast", "strong"]


class Message(TypedDict):
    role: Literal["system", "user"]
    content: str


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        model_tier: ModelTier,
        **opts: object,
    ) -> str: ...


def split_system_and_user(messages: list[Message]) -> tuple[str, str]:
    """Flatten a messages list into (system_text, user_text) for providers
    whose wire format keeps the system instruction separate from user turns.
    """
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user_content = "\n".join(m["content"] for m in messages if m["role"] == "user")
    return system, user_content
