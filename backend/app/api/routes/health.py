from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(request: Request) -> dict[str, str | bool]:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "debug": settings.app_debug,
        "database_configured": bool(settings.database_url),
        "travel_style_model_loaded": (
            request.app.state.resources.get("travel_style_model") is not None
        ),
        "destination_catalog_loaded": (
            request.app.state.resources.get("destination_catalog") is not None
        ),
        "tool_registry_loaded": (
            request.app.state.resources.get("tool_registry") is not None
        ),
        "weather_provider": "open-meteo",
    }
