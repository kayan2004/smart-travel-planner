import asyncio
import math
from typing import Any

import httpx

from app.core.config import Settings


async def embed_texts(
    http_client: httpx.AsyncClient,
    settings: Settings,
    texts: list[str],
    *,
    input_type: str,
) -> list[list[float]]:
    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY is not configured.")
    if not texts:
        return []

    payload = await _post_embeddings_request(
        http_client,
        settings,
        texts,
        input_type=input_type,
    )

    embeddings = payload.get("data")
    if embeddings is None:
        raise ValueError("Voyage API response did not include a 'data' field.")

    return [item["embedding"] for item in embeddings]


def build_text_batches(
    texts: list[str],
    *,
    max_batch_size: int,
    max_request_tokens: int,
    estimated_chars_per_token: int,
) -> list[list[str]]:
    if not texts:
        return []

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        estimated_tokens = estimate_text_tokens(
            text,
            estimated_chars_per_token=estimated_chars_per_token,
        )
        if estimated_tokens > max_request_tokens:
            raise ValueError(
                "A single chunk exceeds the configured Voyage token budget. "
                "Reduce chunk size before embedding."
            )

        would_exceed_batch_size = len(current_batch) >= max_batch_size
        would_exceed_token_budget = current_tokens + estimated_tokens > max_request_tokens

        if current_batch and (would_exceed_batch_size or would_exceed_token_budget):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += estimated_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def estimate_text_tokens(
    text: str,
    *,
    estimated_chars_per_token: int,
) -> int:
    safe_divisor = max(estimated_chars_per_token, 1)
    return max(1, math.ceil(len(text) / safe_divisor))


async def _post_embeddings_request(
    http_client: httpx.AsyncClient,
    settings: Settings,
    texts: list[str],
    *,
    input_type: str,
) -> dict[str, Any]:
    last_error: httpx.HTTPStatusError | None = None

    for attempt in range(settings.voyage_max_retries + 1):
        response = await http_client.post(
            f"{settings.voyage_api_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.voyage_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "input": texts,
                "model": settings.voyage_embedding_model,
                "input_type": input_type,
                "output_dimension": settings.voyage_embedding_dimension,
            },
            timeout=settings.voyage_timeout_seconds,
        )

        if response.status_code == 429 and attempt < settings.voyage_max_retries:
            await asyncio.sleep(_get_retry_delay_seconds(response, settings, attempt))
            continue

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            break

        return response.json()

    if last_error is not None:
        raise last_error
    raise RuntimeError("Voyage embeddings request failed without a captured HTTP error.")


def _get_retry_delay_seconds(
    response: httpx.Response,
    settings: Settings,
    attempt: int,
) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass

    base_interval = max(60.0 / max(settings.voyage_requests_per_minute, 1), 1.0)
    return base_interval * (attempt + 1)
