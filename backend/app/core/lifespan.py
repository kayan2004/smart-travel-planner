from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.init_db import init_db
from .config import get_settings
from app.db.session import create_db_engine, create_session_factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    db_engine = create_db_engine(settings)
    db_session_factory = create_session_factory(db_engine)

    app.state.settings = settings

    app.state.resources = {
        "db_engine": db_engine,
        "db_session_factory": db_session_factory,
        "http_client": None,
        "travel_style_model": None,
    }

    await init_db(db_engine)

    yield

    await db_engine.dispose()
    app.state.resources.clear()
