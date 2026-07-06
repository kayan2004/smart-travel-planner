import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.classifier import TravelStylePredictionRequest
from app.schemas.claude import ExtractedRequestFields
from app.schemas.clustering import ClusterNamingProposal
from app.services.llm_providers import Message, get_llm_provider, raise_for_status_with_body
import httpx

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
REQUEST_FIELD_EXTRACTION_PROMPT_PATH = (
    PROMPTS_DIR / "request_field_extraction_prompt.txt"
)


async def list_anthropic_models(
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> dict:
    if not settings.anthropic_api_key:
        raise RuntimeError("Anthropic API key is not configured.")

    response = await http_client.get(
        f"{settings.anthropic_api_base_url}/v1/models",
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": settings.anthropic_api_version,
        },
        timeout=settings.weather_request_timeout_seconds,
    )
    raise_for_status_with_body(response, context="Anthropic model listing")
    return response.json()


def model_name(settings: Settings) -> str:
    """The single configured model for the active provider - no more
    fast/strong tiers (removed 2026-07-06 in favor of always using the free
    Gemma 4 model for Gemini, at least for now)."""
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    return settings.anthropic_model


async def extract_request_fields(
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    prompt: str,
) -> ExtractedRequestFields:
    provider = get_llm_provider(settings, http_client=http_client)
    messages: list[Message] = [
        {
            "role": "system",
            "content": (
                "You extract structured travel-planning fields from user prompts. "
                "Return strict JSON only. "
                "Do not invent a destination if one is not clearly implied. "
                "Follow the extraction spec exactly."
            ),
        },
        {"role": "user", "content": _build_request_field_extraction_prompt(prompt)},
    ]
    # max_tokens well above the actual JSON payload size needed - Gemma 4
    # (the current default model) spends a substantial token budget on
    # internal "thinking" before the real answer; too low a ceiling here
    # truncates mid-thought and returns an empty response (confirmed live).
    final_text = await provider.complete(messages, max_tokens=2048, temperature=0.0)

    if not final_text:
        raise RuntimeError("The LLM returned an empty extraction response.")

    try:
        extracted_payload = _extract_json_payload(final_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM extraction returned non-JSON output: {final_text}"
        ) from exc

    extracted_payload = _normalize_extracted_payload(extracted_payload)
    return _coerce_extracted_fields(extracted_payload, prompt=prompt)


async def synthesize_trip_response(
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    prompt: str,
    predicted_style: str | None,
    destination_name: str | None,
    response_sections: list[str],
    tool_logs: list[dict[str, str]],
) -> str:
    context_lines = [
        "Trip planning context gathered by backend tools:",
        *[f"- {section}" for section in response_sections],
        "",
        "Tool execution log:",
        *[
            (
                f"- {tool_log['tool_name']} [{tool_log['status']}]: "
                f"input={tool_log['input_payload']} | output={tool_log['output_payload']}"
            )
            for tool_log in tool_logs
        ],
    ]

    if predicted_style is not None:
        context_lines.append(f"\nPredicted travel style: {predicted_style}")
    if destination_name is not None:
        context_lines.append(f"Destination under discussion: {destination_name}")

    provider = get_llm_provider(settings, http_client=http_client)
    messages: list[Message] = [
        {
            "role": "system",
            "content": (
                "You are a concise travel-planning assistant. "
                "Use the provided tool outputs to produce a helpful recommendation. "
                "If some tool output failed, acknowledge uncertainty briefly and continue."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User prompt: {prompt}\n\n"
                + "\n".join(context_lines)
                + "\n\nWrite one polished travel-planning answer in 1-3 short paragraphs."
            ),
        },
    ]
    final_text = await provider.complete(messages)

    if not final_text:
        raise RuntimeError("The LLM returned an empty response.")

    return final_text


async def propose_cluster_tag(
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    cluster_id: int,
    example_destinations: list[dict[str, object]],
    quality_metrics: dict[str, object],
) -> ClusterNamingProposal:
    """Ask the configured LLM provider to name one HDBSCAN destination cluster.

    Used offline by scripts/cluster_destinations.py, never on the request
    path. Uses the provider's single configured model, same as every other
    call site - no more fast/strong tiers (removed 2026-07-06).
    """
    examples_block = "\n".join(
        f"- {entry.get('name')}, {entry.get('country')} "
        f"(region={entry.get('region')}, budget={entry.get('budget_level')}, "
        f"membership={entry.get('membership'):.2f}, "
        f"top POIs={entry.get('poi_kinds')})"
        for entry in example_destinations
    )

    provider = get_llm_provider(settings, http_client=http_client)
    messages: list[Message] = [
        {
            "role": "system",
            "content": (
                "You label clusters of travel destinations for a recommendation "
                "system. Given representative destinations and clustering "
                "quality metrics for one cluster, propose a short, specific "
                "travel-style tag name (2-4 words, title case, no generic "
                "words like 'Destinations' or 'Places') and a one-paragraph "
                "description of what unifies this cluster. Return strict "
                'JSON only: {"tag_name": "...", "description": "..."}. '
                "Do not invent facts not supported by the examples."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Cluster {cluster_id} - {len(example_destinations)} "
                f"representative destinations (highest soft-membership "
                f"weight first):\n{examples_block}\n\n"
                f"Clustering quality metrics for this cluster: "
                f"{json.dumps(quality_metrics)}"
            ),
        },
    ]
    # Same reasoning as extract_request_fields's max_tokens - Gemma 4's
    # thinking overhead needs headroom above the actual JSON payload size.
    final_text = await provider.complete(messages, max_tokens=2048, temperature=0.2)

    if not final_text:
        raise RuntimeError(f"The LLM returned an empty naming response for cluster {cluster_id}.")

    try:
        naming_payload = _extract_json_payload(final_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"LLM naming for cluster {cluster_id} returned non-JSON output: {final_text}"
        ) from exc

    try:
        return ClusterNamingProposal.model_validate(naming_payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"LLM naming for cluster {cluster_id} returned an invalid shape: {exc}"
        ) from exc


def _extract_json_payload(final_text: str) -> dict:
    cleaned_text = final_text.strip()
    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        cleaned_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        cleaned_text = fenced_match.group(1).strip()

    start = cleaned_text.find("{")
    end = cleaned_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned_text = cleaned_text[start : end + 1]

    return json.loads(cleaned_text)


def _normalize_extracted_payload(payload: dict) -> dict:
    normalized = dict(payload)
    travel_profile = normalized.get("travel_profile")

    if not isinstance(travel_profile, dict):
        return normalized

    normalized_profile = dict(travel_profile)
    score_fields = (
        "culture_score",
        "luxury_score",
        "family_friendly",
        "nightlife_level",
    )

    for field_name in score_fields:
        field_value = normalized_profile.get(field_name)
        if isinstance(field_value, bool):
            normalized_profile[field_name] = 8.0 if field_value else 2.0
        elif isinstance(field_value, int | float):
            normalized_profile[field_name] = float(field_value)

    avg_temp_peak = normalized_profile.get("avg_temp_peak")
    if isinstance(avg_temp_peak, int | float):
        normalized_profile["avg_temp_peak"] = float(avg_temp_peak)

    normalized["travel_profile"] = normalized_profile
    return normalized


def _coerce_extracted_fields(
    payload: dict,
    *,
    prompt: str,
) -> ExtractedRequestFields:
    safe_payload: dict[str, object | None] = {
        "destination_name": _coerce_optional_string(payload.get("destination_name")),
        "location_query": _coerce_optional_string(payload.get("location_query")),
        "location_country_code": _coerce_country_code(
            payload.get("location_country_code")
        ),
        "travel_profile": None,
    }

    raw_travel_profile = payload.get("travel_profile")
    if isinstance(raw_travel_profile, dict):
        try:
            safe_payload["travel_profile"] = TravelStylePredictionRequest.model_validate(
                raw_travel_profile
            )
        except ValidationError:
            safe_payload["travel_profile"] = _merge_with_inferred_travel_profile(
                raw_travel_profile,
                prompt=prompt,
            )
    else:
        safe_payload["travel_profile"] = _infer_travel_profile_from_prompt(prompt)

    try:
        return ExtractedRequestFields.model_validate(safe_payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"Extractor coercion still produced invalid structured fields: {exc}"
        ) from exc


def _merge_with_inferred_travel_profile(
    raw_profile: dict,
    *,
    prompt: str,
) -> TravelStylePredictionRequest:
    inferred = _infer_travel_profile_from_prompt(prompt)
    merged_profile = inferred.model_dump()

    for key, value in raw_profile.items():
        if key not in merged_profile or value is None:
            continue
        merged_profile[key] = value

    normalized_profile = _normalize_extracted_payload(
        {"travel_profile": merged_profile}
    )["travel_profile"]
    try:
        return TravelStylePredictionRequest.model_validate(normalized_profile)
    except ValidationError:
        return inferred


def _coerce_optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    return cleaned or None


def _coerce_country_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = value.strip().upper()
    if len(cleaned) != 2 or not cleaned.isalpha():
        return None
    return cleaned


def _infer_travel_profile_from_prompt(
    prompt: str,
) -> TravelStylePredictionRequest | None:
    lowered = prompt.casefold()

    budget_level = _infer_budget_level(lowered)
    tourism_level = _infer_tourism_level(lowered)
    has_hiking = _contains_any(
        lowered,
        ("hiking", "hike", "trek", "trekking", "mountain", "trail", "outdoors"),
    )
    has_beach = _contains_any(
        lowered,
        ("beach", "coast", "island", "seaside", "ocean", "sea"),
    )
    culture_score = _bounded_score(
        positive=_count_matches(
            lowered,
            ("culture", "cultural", "museum", "history", "historic", "temple", "art", "food", "local food"),
        ),
        negative=_count_matches(
            lowered,
            ("party", "nightlife", "clubbing"),
        ),
        base=5.0,
        step=1.25,
    )
    luxury_score = _bounded_score(
        positive=_count_matches(
            lowered,
            ("luxury", "resort", "spa", "boutique", "five-star", "upscale"),
        ),
        negative=_count_matches(
            lowered,
            ("budget", "cheap", "affordable", "hostel", "backpacking", "$"),
        ),
        base=4.0,
        step=1.5,
    )
    family_friendly = _bounded_score(
        positive=_count_matches(
            lowered,
            ("family", "kids", "kid-friendly", "child-friendly"),
        ),
        negative=_count_matches(
            lowered,
            ("party", "nightlife", "clubbing"),
        ),
        base=5.0,
        step=1.5,
    )
    nightlife_level = _bounded_score(
        positive=_count_matches(
            lowered,
            ("nightlife", "party", "bars", "bar scene", "clubs", "late night"),
        ),
        negative=_count_matches(
            lowered,
            ("quiet", "calm", "not too touristy", "relaxing", "peaceful"),
        ),
        base=4.0,
        step=1.75,
    )
    avg_temp_peak = _infer_temperature_preference(lowered)
    region = _infer_region(lowered)

    return TravelStylePredictionRequest(
        region=region,
        budget_level=budget_level,
        tourism_level=tourism_level,
        has_hiking=has_hiking,
        has_beach=has_beach,
        culture_score=culture_score,
        luxury_score=luxury_score,
        family_friendly=family_friendly,
        nightlife_level=nightlife_level,
        avg_temp_peak=avg_temp_peak,
    )


def _infer_budget_level(prompt: str) -> str:
    budget_amount = _extract_budget_amount(prompt)
    if budget_amount is not None:
        if budget_amount <= 1000:
            return "low"
        if budget_amount <= 2200:
            return "medium"
        return "high"

    if _contains_any(prompt, ("luxury", "upscale", "splurge", "five-star", "resort")):
        return "high"
    if _contains_any(prompt, ("budget", "cheap", "affordable", "backpacking", "hostel")):
        return "low"
    return "medium"


def _infer_tourism_level(prompt: str) -> str:
    if _contains_any(
        prompt,
        ("not too touristy", "not touristy", "quiet", "peaceful", "off the beaten path"),
    ):
        return "low"
    if _contains_any(prompt, ("moderately touristy", "somewhat touristy", "balanced")):
        return "medium"
    if _contains_any(prompt, ("touristy", "popular", "famous", "iconic", "buzzy")):
        return "high"
    return "medium"


def _infer_temperature_preference(prompt: str) -> float:
    if _contains_any(prompt, ("hot", "tropical", "very warm")):
        return 31.0
    if _contains_any(prompt, ("warm", "sunny")):
        return 27.0
    if _contains_any(prompt, ("mild", "temperate")):
        return 22.0
    if _contains_any(prompt, ("cool", "cold", "ski", "snow")):
        return 12.0
    return 24.0


def _infer_region(prompt: str) -> str:
    region_keywords = {
        "Europe": ("europe", "mediterranean", "italy", "france", "portugal", "spain", "greece"),
        "Asia": ("asia", "japan", "thailand", "vietnam", "indonesia", "bali", "nepal"),
        "North America": ("north america", "canada", "usa", "united states", "mexico"),
        "South America": ("south america", "peru", "argentina", "chile", "colombia"),
        "Oceania": ("oceania", "new zealand", "australia"),
        "Africa": ("africa", "morocco", "south africa", "tanzania"),
        "Middle East": ("middle east", "jordan", "uae", "oman"),
        "Caribbean": ("caribbean", "jamaica", "bahamas", "barbados"),
        "Central America": ("central america", "costa rica", "guatemala", "panama"),
    }

    for region, keywords in region_keywords.items():
        if _contains_any(prompt, keywords):
            return region

    return "Flexible"


def _extract_budget_amount(prompt: str) -> int | None:
    amount_match = re.search(r"\$\s?(\d{1,3}(?:,\d{3})+|\d{3,5})", prompt)
    if amount_match is None:
        return None

    try:
        return int(amount_match.group(1).replace(",", ""))
    except ValueError:
        return None


def _count_matches(prompt: str, keywords: tuple[str, ...]) -> int:
    return sum(1 for keyword in keywords if keyword in prompt)


def _contains_any(prompt: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in prompt for keyword in keywords)


def _bounded_score(*, positive: int, negative: int, base: float, step: float) -> float:
    return max(0.0, min(10.0, base + (positive - negative) * step))


@lru_cache(maxsize=1)
def _load_request_field_extraction_prompt_template() -> str:
    return REQUEST_FIELD_EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8")


def _build_request_field_extraction_prompt(prompt: str) -> str:
    template = _load_request_field_extraction_prompt_template()
    return template.replace("{{USER_PROMPT}}", prompt.strip())
