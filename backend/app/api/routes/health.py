from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict[str, str | bool]:
    # Deliberately unauthenticated (Docker's HEALTHCHECK and any uptime
    # monitor need to hit this without a JWT) but deliberately minimal -
    # this used to also return app.debug, app.env, and *_configured
    # booleans for database/discord, which told an unauthenticated caller
    # exactly what security posture and integrations this deployment has.
    # None of that is needed to answer "is the process up and did startup
    # succeed" - the only two things this endpoint actually needs to say.
    return {
        "status": "ok",
        "app_name": request.app.state.settings.app.name,
        "tool_registry_loaded": (
            request.app.state.resources.get("tool_registry") is not None
        ),
    }
