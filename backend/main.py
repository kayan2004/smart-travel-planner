from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.routes.agent_runs import router as agent_runs_router
from app.api.routes.anthropic import router as anthropic_router
from app.api.routes.auth import router as auth_router
from app.api.routes.claude import router as claude_router
from app.api.routes.classifier import router as classifier_router
from app.api.routes.discord_webhook import router as discord_webhook_router
from app.api.routes.feedback import router as feedback_router
from app.api.routes.health import router as health_router
from app.api.routes.live_conditions import router as live_conditions_router
from app.api.routes.rag_retrieval import router as rag_retrieval_router
from app.api.routes.recommendations import router as recommendations_router
from app.core.config import get_settings
from app.core.lifespan import lifespan
from app.core.logging_config import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        debug=settings.app_debug,
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(agent_runs_router)
    application.include_router(anthropic_router)
    application.include_router(auth_router)
    application.include_router(claude_router)
    application.include_router(classifier_router)
    application.include_router(discord_webhook_router)
    application.include_router(feedback_router)
    application.include_router(health_router)
    application.include_router(live_conditions_router)
    application.include_router(rag_retrieval_router)
    application.include_router(recommendations_router)
    return application

app = create_app()


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
