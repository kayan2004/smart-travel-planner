from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.base import Base
from app.db.models import agent_run  # noqa: F401
from app.db.models import tool_log  # noqa: F401
from app.db.models import user  # noqa: F401


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
