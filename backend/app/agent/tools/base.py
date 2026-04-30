from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings


@dataclass(slots=True)
class ToolContext:
    settings: Settings
    resources: Mapping[str, Any]
    session: AsyncSession | None = None
    http_client: httpx.AsyncClient | None = None


class BaseTool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]

    @abstractmethod
    async def arun(self, payload: BaseModel, context: ToolContext) -> BaseModel:
        raise NotImplementedError

