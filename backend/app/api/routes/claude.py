from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.schemas.claude import (
    ClaudeTestRequest,
    ClaudeTestResponse,
    ExtractionTestRequest,
    ExtractionTestResponse,
)
from app.services.llm import (
    extract_request_fields,
    model_name,
    synthesize_trip_response,
)

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post(
    "/test-claude",
    response_model=ClaudeTestResponse,
    status_code=status.HTTP_200_OK,
)
async def test_claude_route(
    payload: ClaudeTestRequest,
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> ClaudeTestResponse:
    http_client = request.app.state.resources.get("http_client")
    if http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shared HTTP client is not available.",
        )

    settings = request.app.state.settings
    selected_model = model_name(settings)
    generated_text = await synthesize_trip_response(
        http_client,
        settings,
        prompt=payload.prompt,
        predicted_style=payload.predicted_style,
        destination_name=payload.destination_name,
        response_sections=payload.response_sections,
        tool_logs=payload.tool_logs,
    )

    return ClaudeTestResponse(
        selected_model=selected_model,
        generated_text=generated_text,
    )


@router.post(
    "/test-extraction",
    response_model=ExtractionTestResponse,
    status_code=status.HTTP_200_OK,
)
async def test_extraction_route(
    payload: ExtractionTestRequest,
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> ExtractionTestResponse:
    http_client = request.app.state.resources.get("http_client")
    if http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Shared HTTP client is not available.",
        )

    settings = request.app.state.settings
    extracted_fields = await extract_request_fields(
        http_client,
        settings,
        prompt=payload.prompt,
    )

    return ExtractionTestResponse(
        selected_model=model_name(settings),
        extracted_fields=extracted_fields,
    )
