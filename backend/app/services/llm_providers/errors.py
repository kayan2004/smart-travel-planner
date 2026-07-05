import httpx


def raise_for_status_with_body(response: httpx.Response, *, context: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        raise RuntimeError(
            f"{context} failed with status {response.status_code}. Response body: {body}"
        ) from exc
