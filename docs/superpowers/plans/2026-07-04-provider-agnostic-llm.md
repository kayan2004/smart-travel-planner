# Provider-Agnostic LLM Layer + Gemini/Gemma 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the three LLM call sites (field extraction, trip synthesis, cluster naming)
from Anthropic's specific REST shape, and add Gemini/Gemma 4 as a second, config-selectable
provider.

**Architecture:** A new `app/services/llm_providers.py` defines a small `LLMProvider` protocol
with `AnthropicProvider`/`GeminiProvider` adapters, each translating a provider-agnostic
`generate(system, user_content, model, max_tokens, temperature)` call into their own REST shape.
`app/services/claude.py` is renamed to `app/services/llm.py`; its three call-site functions stop
building Anthropic-specific request bodies and instead call `get_llm_provider(settings).generate(...)`.

**Tech Stack:** Python 3.14, httpx (existing dependency, no new SDK), pydantic-settings.

## Global Constraints

- No new third-party LLM SDK - raw `httpx.AsyncClient` calls only, matching this project's
  existing convention (see `CLAUDE.md`: "No `requests`... shared `httpx.AsyncClient` instance").
- One global `llm_provider` setting controls all three call sites - no per-call-site override, no
  automatic fallback between providers.
- `app/api/routes/anthropic.py` (`list_anthropic_models`, the `/tools/anthropic-models` debug
  route) is explicitly out of scope and stays Anthropic-only.
- No automated test suite exists in this project (documented known gap) - this plan does not
  introduce one. Verification uses `httpx.MockTransport` for deterministic, network-free checks
  run via `uv run python -` heredocs, matching the ad-hoc verification pattern already used
  elsewhere in this project's scripts.
- Gemini fast/strong models: `gemma-4-26b-a4b-it` / `gemma-4-31b-it`, served via the Gemini
  Developer API (`generativelanguage.googleapis.com`), authenticated with an `x-goog-api-key`
  header.
- Full design context: `docs/superpowers/specs/2026-07-04-provider-agnostic-llm-design.md`.

---

### Task 1: Add provider-selection and Gemini settings to config

**Files:**
- Modify: `backend/app/core/config.py:36-43`
- Modify: `backend/.env.example:25-31`

**Interfaces:**
- Produces: `Settings.llm_provider: str` (default `"anthropic"`), `Settings.gemini_api_key: str`,
  `Settings.gemini_api_base_url: str`, `Settings.gemini_api_version: str`,
  `Settings.gemini_fast_model: str`, `Settings.gemini_strong_model: str`,
  `Settings.gemini_max_tokens: int`, `Settings.gemini_temperature: float` - all consumed by
  Task 2's `GeminiProvider` and Task 3's `llm.py`.

- [ ] **Step 1: Add the new settings fields**

In `backend/app/core/config.py`, find this exact block:

```python
    anthropic_api_key: str = ""
    anthropic_api_base_url: str = "https://api.anthropic.com"
    anthropic_api_version: str = "2023-06-01"
    anthropic_fast_model: str = "claude-3-5-haiku-latest"
    anthropic_strong_model: str = "claude-sonnet-4-5"
    anthropic_max_tokens: int = 700
    anthropic_temperature: float = 0.2
    discord_webhook_url: str = ""
```

Replace it with:

```python
    llm_provider: str = "anthropic"  # "anthropic" | "gemini" - selects the provider for all
    # three LLM call sites (extraction, synthesis, cluster naming). See app/services/llm_providers.py.
    anthropic_api_key: str = ""
    anthropic_api_base_url: str = "https://api.anthropic.com"
    anthropic_api_version: str = "2023-06-01"
    anthropic_fast_model: str = "claude-3-5-haiku-latest"
    anthropic_strong_model: str = "claude-sonnet-4-5"
    anthropic_max_tokens: int = 700
    anthropic_temperature: float = 0.2
    gemini_api_key: str = ""
    gemini_api_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_api_version: str = "v1beta"
    gemini_fast_model: str = "gemma-4-26b-a4b-it"
    gemini_strong_model: str = "gemma-4-31b-it"
    gemini_max_tokens: int = 700
    gemini_temperature: float = 0.2
    discord_webhook_url: str = ""
```

- [ ] **Step 2: Mirror the same settings in `.env.example`**

In `backend/.env.example`, find this exact block:

```
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_API_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_VERSION=2023-06-01
ANTHROPIC_FAST_MODEL=claude-3-5-haiku-latest
ANTHROPIC_STRONG_MODEL=claude-sonnet-4-5
ANTHROPIC_MAX_TOKENS=700
ANTHROPIC_TEMPERATURE=0.2
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your/webhook
```

Replace it with:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_API_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_VERSION=2023-06-01
ANTHROPIC_FAST_MODEL=claude-3-5-haiku-latest
ANTHROPIC_STRONG_MODEL=claude-sonnet-4-5
ANTHROPIC_MAX_TOKENS=700
ANTHROPIC_TEMPERATURE=0.2
# Generate a *restricted* API key (scoped to the Generative Language API) at
# https://aistudio.google.com/apikey - Google phases out unrestricted keys during 2026.
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com
GEMINI_API_VERSION=v1beta
GEMINI_FAST_MODEL=gemma-4-26b-a4b-it
GEMINI_STRONG_MODEL=gemma-4-31b-it
GEMINI_MAX_TOKENS=700
GEMINI_TEMPERATURE=0.2
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your/webhook
```

- [ ] **Step 3: Verify settings load correctly**

Run:
```powershell
cd backend
uv run python -c "from app.core.config import Settings; s = Settings(); print(s.llm_provider, s.gemini_fast_model, s.gemini_strong_model)"
```
Expected output: `anthropic gemma-4-26b-a4b-it gemma-4-31b-it`

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/config.py backend/.env.example
git commit -m "feat(config): add llm_provider switch and Gemini/Gemma settings"
```

---

### Task 2: Create the provider adapter module

**Files:**
- Create: `backend/app/services/llm_providers.py`

**Interfaces:**
- Consumes: `Settings` fields from Task 1 (`llm_provider`, `anthropic_*`, `gemini_*`).
- Produces: `LLMProvider` (Protocol), `AnthropicProvider`, `GeminiProvider` (both implement
  `async def generate(self, http_client: httpx.AsyncClient, settings: Settings, *, system: str,
  user_content: str, model: str, max_tokens: int, temperature: float) -> str`),
  `get_llm_provider(settings: Settings) -> LLMProvider`, and
  `_raise_for_status_with_body(response: httpx.Response, *, context: str) -> None` - all consumed
  by Task 3's `llm.py`.

- [ ] **Step 1: Write the module**

Create `backend/app/services/llm_providers.py`:

```python
from typing import Protocol

import httpx

from app.core.config import Settings


class LLMProvider(Protocol):
    async def generate(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
        *,
        system: str,
        user_content: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str: ...


class AnthropicProvider:
    async def generate(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
        *,
        system: str,
        user_content: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not settings.anthropic_api_key:
            raise RuntimeError("Anthropic API key is not configured.")

        response = await http_client.post(
            f"{settings.anthropic_api_base_url}/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": settings.anthropic_api_version,
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user_content}],
            },
            timeout=settings.weather_request_timeout_seconds,
        )
        _raise_for_status_with_body(
            response, context=f"Anthropic generation using model '{model}'"
        )
        payload = response.json()
        content_blocks = payload.get("content") or []
        text_parts = [
            block.get("text", "").strip()
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n\n".join(part for part in text_parts if part)


class GeminiProvider:
    async def generate(
        self,
        http_client: httpx.AsyncClient,
        settings: Settings,
        *,
        system: str,
        user_content: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if not settings.gemini_api_key:
            raise RuntimeError("Gemini API key is not configured.")

        response = await http_client.post(
            f"{settings.gemini_api_base_url}/{settings.gemini_api_version}"
            f"/models/{model}:generateContent",
            headers={
                "x-goog-api-key": settings.gemini_api_key,
                "content-type": "application/json",
            },
            json={
                "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=settings.weather_request_timeout_seconds,
        )
        _raise_for_status_with_body(
            response, context=f"Gemini generation using model '{model}'"
        )
        payload = response.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text_parts = [
            part.get("text", "").strip() for part in parts if isinstance(part, dict)
        ]
        return "\n\n".join(part for part in text_parts if part)


def get_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        return AnthropicProvider()
    if settings.llm_provider == "gemini":
        return GeminiProvider()
    raise RuntimeError(
        f"Unknown llm_provider '{settings.llm_provider}' - expected 'anthropic' or 'gemini'."
    )


def _raise_for_status_with_body(response: httpx.Response, *, context: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text.strip()
        raise RuntimeError(
            f"{context} failed with status {response.status_code}. Response body: {body}"
        ) from exc
```

- [ ] **Step 2: Verify both adapters against mocked HTTP responses (no real API calls/credentials needed)**

Run this from `backend/`:

```powershell
uv run python -c "
import asyncio
import httpx
from app.core.config import Settings
from app.services.llm_providers import AnthropicProvider, GeminiProvider, get_llm_provider

def anthropic_handler(request):
    assert request.headers['x-api-key'] == 'test-anthropic-key'
    return httpx.Response(200, json={'content': [{'type': 'text', 'text': 'hello from claude'}]})

def gemini_handler(request):
    assert request.headers['x-goog-api-key'] == 'test-gemini-key'
    assert request.url.path.endswith(':generateContent')
    return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'hello from gemini'}]}}]})

async def main():
    settings = Settings(anthropic_api_key='test-anthropic-key', gemini_api_key='test-gemini-key')

    async with httpx.AsyncClient(transport=httpx.MockTransport(anthropic_handler)) as client:
        text = await AnthropicProvider().generate(client, settings, system='sys', user_content='hi', model='claude-x', max_tokens=10, temperature=0.0)
        assert text == 'hello from claude', text

    async with httpx.AsyncClient(transport=httpx.MockTransport(gemini_handler)) as client:
        text = await GeminiProvider().generate(client, settings, system='sys', user_content='hi', model='gemma-x', max_tokens=10, temperature=0.0)
        assert text == 'hello from gemini', text

    settings.llm_provider = 'gemini'
    assert isinstance(get_llm_provider(settings), GeminiProvider)
    settings.llm_provider = 'anthropic'
    assert isinstance(get_llm_provider(settings), AnthropicProvider)

    try:
        settings.llm_provider = 'ollama'
        get_llm_provider(settings)
        raise AssertionError('expected RuntimeError for unknown provider')
    except RuntimeError:
        pass

    print('all provider checks passed')

asyncio.run(main())
"
```

Expected output: `all provider checks passed` (no assertion errors, no traceback).

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/llm_providers.py
git commit -m "feat(llm): add AnthropicProvider/GeminiProvider adapters"
```

---

### Task 3: Rename `claude.py` to `llm.py` and route call sites through the provider abstraction

**Files:**
- Delete: `backend/app/services/claude.py`
- Create: `backend/app/services/llm.py`

**Interfaces:**
- Consumes: `get_llm_provider`, `_raise_for_status_with_body` from Task 2's
  `app/services/llm_providers.py`.
- Produces: `fast_model_name(settings) -> str`, `strong_model_name(settings) -> str`,
  `choose_model(settings, *, prompt, response_sections, tool_logs) -> str` (renamed from
  `choose_anthropic_model`), `extract_request_fields`, `synthesize_trip_response`,
  `propose_cluster_tag`, `list_anthropic_models` (unchanged) - all consumed by Task 4's call-site
  updates.

- [ ] **Step 1: Create `backend/app/services/llm.py` with this exact content**

```python
import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.classifier import TravelStylePredictionRequest
from app.schemas.claude import ExtractedRequestFields
from app.schemas.clustering import ClusterNamingProposal
from app.services.llm_providers import _raise_for_status_with_body, get_llm_provider
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
    _raise_for_status_with_body(response, context="Anthropic model listing")
    return response.json()


def fast_model_name(settings: Settings) -> str:
    if settings.llm_provider == "gemini":
        return settings.gemini_fast_model
    return settings.anthropic_fast_model


def strong_model_name(settings: Settings) -> str:
    if settings.llm_provider == "gemini":
        return settings.gemini_strong_model
    return settings.anthropic_strong_model


def _generation_settings(settings: Settings) -> tuple[int, float]:
    if settings.llm_provider == "gemini":
        return settings.gemini_max_tokens, settings.gemini_temperature
    return settings.anthropic_max_tokens, settings.anthropic_temperature


def choose_model(
    settings: Settings,
    *,
    prompt: str,
    response_sections: list[str],
    tool_logs: list[dict[str, str]],
) -> str:
    failed_tools = sum(1 for tool_log in tool_logs if tool_log["status"] == "failed")
    long_prompt = len(prompt) > 220
    rich_context = len(response_sections) >= 4
    verbose_tool_payloads = any(
        len(tool_log["output_payload"]) > 800 for tool_log in tool_logs
    )

    if failed_tools > 0 or long_prompt or rich_context or verbose_tool_payloads:
        return strong_model_name(settings)
    return fast_model_name(settings)


async def extract_request_fields(
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    prompt: str,
) -> ExtractedRequestFields:
    provider = get_llm_provider(settings)
    final_text = await provider.generate(
        http_client,
        settings,
        system=(
            "You extract structured travel-planning fields from user prompts. "
            "Return strict JSON only. "
            "Do not invent a destination if one is not clearly implied. "
            "Follow the extraction spec exactly."
        ),
        user_content=_build_request_field_extraction_prompt(prompt),
        model=fast_model_name(settings),
        max_tokens=500,
        temperature=0.0,
    )

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

    selected_model = choose_model(
        settings,
        prompt=prompt,
        response_sections=response_sections,
        tool_logs=tool_logs,
    )
    max_tokens, temperature = _generation_settings(settings)

    provider = get_llm_provider(settings)
    final_text = await provider.generate(
        http_client,
        settings,
        system=(
            "You are a concise travel-planning assistant. "
            "Use the provided tool outputs to produce a helpful recommendation. "
            "If some tool output failed, acknowledge uncertainty briefly and continue."
        ),
        user_content=(
            f"User prompt: {prompt}\n\n"
            + "\n".join(context_lines)
            + "\n\nWrite one polished travel-planning answer in 1-3 short paragraphs."
        ),
        model=selected_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

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
    path - always the strong model (naming quality matters more than
    latency/cost here, and this runs a handful of times per corpus, not
    per-request).
    """
    examples_block = "\n".join(
        f"- {entry.get('name')}, {entry.get('country')} "
        f"(region={entry.get('region')}, budget={entry.get('budget_level')}, "
        f"membership={entry.get('membership'):.2f}, "
        f"top POIs={entry.get('poi_kinds')})"
        for entry in example_destinations
    )

    provider = get_llm_provider(settings)
    final_text = await provider.generate(
        http_client,
        settings,
        system=(
            "You label clusters of travel destinations for a recommendation "
            "system. Given representative destinations and clustering "
            "quality metrics for one cluster, propose a short, specific "
            "travel-style tag name (2-4 words, title case, no generic "
            "words like 'Destinations' or 'Places') and a one-paragraph "
            "description of what unifies this cluster. Return strict "
            'JSON only: {"tag_name": "...", "description": "..."}. '
            "Do not invent facts not supported by the examples."
        ),
        user_content=(
            f"Cluster {cluster_id} - {len(example_destinations)} "
            f"representative destinations (highest soft-membership "
            f"weight first):\n{examples_block}\n\n"
            f"Clustering quality metrics for this cluster: "
            f"{json.dumps(quality_metrics)}"
        ),
        model=strong_model_name(settings),
        max_tokens=400,
        temperature=0.2,
    )

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
```

- [ ] **Step 2: Delete the old file**

```bash
git rm backend/app/services/claude.py
```

- [ ] **Step 3: Verify the full orchestration against mocked responses for BOTH providers**

Run this from `backend/`:

```powershell
uv run python -c "
import asyncio
import httpx
from app.core.config import Settings
from app.services.llm import extract_request_fields, propose_cluster_tag

def anthropic_extract_handler(request):
    return httpx.Response(200, json={'content': [{'type': 'text', 'text': '{\"destination_name\": \"Paris\", \"location_query\": null, \"location_country_code\": null, \"travel_profile\": null}'}]})

def gemini_name_handler(request):
    return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': '{\"tag_name\": \"Coastal Getaways\", \"description\": \"Beach-forward destinations.\"}'}]}}]})

async def main():
    settings = Settings(anthropic_api_key='test-key', llm_provider='anthropic')
    async with httpx.AsyncClient(transport=httpx.MockTransport(anthropic_extract_handler)) as client:
        result = await extract_request_fields(client, settings, prompt='I want to visit Paris')
        assert result.destination_name == 'Paris', result

    settings2 = Settings(gemini_api_key='test-key', llm_provider='gemini')
    async with httpx.AsyncClient(transport=httpx.MockTransport(gemini_name_handler)) as client:
        proposal = await propose_cluster_tag(client, settings2, cluster_id=0, example_destinations=[{'name': 'Nice', 'country': 'France', 'region': 'Europe', 'budget_level': 'high', 'membership': 0.9, 'poi_kinds': 'beaches (5)'}], quality_metrics={'noise_ratio': 0.04})
        assert proposal.tag_name == 'Coastal Getaways', proposal

    print('orchestration checks passed for both providers')

asyncio.run(main())
"
```

Expected output: `orchestration checks passed for both providers`

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/llm.py
git commit -m "refactor(llm): rename services/claude.py to services/llm.py, route through provider abstraction"
```

---

### Task 4: Update call sites to import from `services/llm`

**Files:**
- Modify: `backend/app/agent/graph.py:13`
- Modify: `backend/app/services/clustering.py:29`
- Modify: `backend/app/api/routes/claude.py`
- Modify: `backend/app/api/routes/anthropic.py:5`

**Interfaces:**
- Consumes: `fast_model_name`, `choose_model`, `extract_request_fields`,
  `synthesize_trip_response`, `propose_cluster_tag`, `list_anthropic_models` from Task 3's
  `app/services/llm.py`.

- [ ] **Step 1: Update `graph.py`**

In `backend/app/agent/graph.py`, change:
```python
from app.services.claude import extract_request_fields, synthesize_trip_response
```
to:
```python
from app.services.llm import extract_request_fields, synthesize_trip_response
```

- [ ] **Step 2: Update `clustering.py`**

In `backend/app/services/clustering.py`, change:
```python
from app.services.claude import propose_cluster_tag
```
to:
```python
from app.services.llm import propose_cluster_tag
```

- [ ] **Step 3: Update `routes/anthropic.py`**

In `backend/app/api/routes/anthropic.py`, change:
```python
from app.services.claude import list_anthropic_models
```
to:
```python
from app.services.llm import list_anthropic_models
```

- [ ] **Step 4: Update `routes/claude.py`**

In `backend/app/api/routes/claude.py`, change:
```python
from app.services.claude import (
    choose_anthropic_model,
    extract_request_fields,
    synthesize_trip_response,
)
```
to:
```python
from app.services.llm import (
    choose_model,
    extract_request_fields,
    fast_model_name,
    synthesize_trip_response,
)
```

Then change:
```python
    selected_model = choose_anthropic_model(
        settings,
        prompt=payload.prompt,
        response_sections=payload.response_sections,
        tool_logs=payload.tool_logs,
    )
```
to:
```python
    selected_model = choose_model(
        settings,
        prompt=payload.prompt,
        response_sections=payload.response_sections,
        tool_logs=payload.tool_logs,
    )
```

Then change:
```python
    return ExtractionTestResponse(
        selected_model=settings.anthropic_fast_model,
        extracted_fields=extracted_fields,
    )
```
to:
```python
    return ExtractionTestResponse(
        selected_model=fast_model_name(settings),
        extracted_fields=extracted_fields,
    )
```

- [ ] **Step 5: Verify no references to the old module/function names remain**

Run from `backend/`:
```powershell
grep -rn "services\.claude\|services import claude\|choose_anthropic_model" --include="*.py" .
```
Expected output: no matches (empty output).

- [ ] **Step 6: Verify the app still imports cleanly end-to-end**

Run from `backend/`:
```powershell
uv run python -c "
import app.agent.graph
import app.services.clustering
import app.api.routes.claude
import app.api.routes.anthropic
print('all call sites import OK')
"
```
Expected output: `all call sites import OK`

- [ ] **Step 7: Commit**

```bash
git add backend/app/agent/graph.py backend/app/services/clustering.py backend/app/api/routes/claude.py backend/app/api/routes/anthropic.py
git commit -m "refactor(llm): update call sites to import from services/llm"
```

---

### Task 5: Documentation

**Files:**
- Modify: `backend/README.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: nothing (documentation only).

- [ ] **Step 1: Add a "Provider-Agnostic LLM Layer" section to `backend/README.md`**

`backend/README.md` has no existing section documenting `services/claude.py`/model routing and no
"Known Gaps" section - its last section is "Destination Clustering", ending with a "### When to
re-cluster" subsection. Add the new section immediately after that subsection's final paragraph
(find the exact text `already-clustered corpus means re-fitting from scratch, not assigning the
new rows to existing\nclusters.` and insert after it):

```markdown
## Provider-Agnostic LLM Layer

All three LLM call sites (field extraction, trip synthesis, offline cluster naming) go through
`app/services/llm_providers.py`'s `LLMProvider` interface rather than hardcoding Anthropic's REST
shape. `LLM_PROVIDER` (`anthropic` or `gemini`, default `anthropic`) is one global switch - it is
not a per-call-site setting and there is no automatic fallback between providers. Set both
providers' credentials if you want to be able to switch without restarting with different env
vars, or just the one you're using.

### Gemini / Gemma 4

Gemma 4 (Google's open-weight model family, Apache 2.0, released April 2026) is served through
the same Gemini Developer API used for proprietary Gemini models. Configure:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_FAST_MODEL=gemma-4-26b-a4b-it   # 26B MoE, 4B active - extraction
GEMINI_STRONG_MODEL=gemma-4-31b-it     # 31B dense - synthesis, cluster naming
```

Generate a **restricted** API key (scoped to the Generative Language API) at
[Google AI Studio](https://aistudio.google.com/apikey) - Google is phasing out unrestricted
Gemini API keys during 2026 (restricted keys work until September 2026; after that, only
service-account-bound auth keys are accepted).

### Adding a third provider

Implement the `LLMProvider` protocol in `app/services/llm_providers.py` (one `generate()` method
translating the provider-agnostic call into that provider's REST shape) and add it to
`get_llm_provider()`'s dispatch. `app/services/llm.py`'s three orchestration functions
(`extract_request_fields`, `synthesize_trip_response`, `propose_cluster_tag`) need no changes -
they only depend on the `LLMProvider` interface, not any specific provider.
```

- [ ] **Step 2: Update `CLAUDE.md`'s architecture tree**

In `CLAUDE.md`, change:
```
├── services/   # Business logic: classifier, claude (extraction+synthesis+model routing+cluster
│               #   naming), clustering (offline UMAP+HDBSCAN, scripts/cluster_destinations.py
│               #   only), discord_webhook, live_conditions, rag_ingestion, rag_retrieval,
│               #   recommendations, voyage_embeddings
```
to:
```
├── services/   # Business logic: classifier, llm (extraction+synthesis+model routing+cluster
│               #   naming, provider-agnostic via llm_providers.py's LLMProvider interface -
│               #   Anthropic/Gemini), clustering (offline UMAP+HDBSCAN,
│               #   scripts/cluster_destinations.py only), discord_webhook, live_conditions,
│               #   rag_ingestion, rag_retrieval, recommendations, voyage_embeddings
```

Also update the "Two-model routing" section (search for `choose_anthropic_model`):
```
**Two-model routing** (`services/claude.py`): `choose_anthropic_model()` picks
`anthropic_fast_model` (Haiku) vs `anthropic_strong_model` (Sonnet) based on prompt length,
number of failed tools, and response richness. Fast model does field extraction; strong model
does final synthesis.
```
to:
```
**Two-model routing** (`services/llm.py`): `choose_model()` picks the fast vs. strong model of
whichever provider is configured (`LLM_PROVIDER`) based on prompt length, number of failed tools,
and response richness. Fast model does field extraction; strong model does final synthesis and
cluster naming. Provider dispatch lives in `services/llm_providers.py`'s `LLMProvider` interface
(`AnthropicProvider`, `GeminiProvider`) - see backend/README.md's "Provider-Agnostic LLM Layer".
```

- [ ] **Step 3: Commit**

```bash
git add backend/README.md CLAUDE.md
git commit -m "docs: document the provider-agnostic LLM layer and Gemini/Gemma 4 setup"
```

---

### Task 6: Final verification and live-call follow-up

**Files:** none (verification only).

- [ ] **Step 1: Confirm the whole backend still imports cleanly**

Run from `backend/`:
```powershell
uv run python -c "import main; print('app imports OK')"
```
Expected output: `app imports OK` (no traceback).

- [ ] **Step 2: Re-run Task 2 and Task 3's mocked verification scripts once more**

(Same commands as Task 2 Step 2 and Task 3 Step 3.) This confirms nothing in Tasks 4-5 broke the
provider layer.

- [ ] **Step 3: Note the live-call verification gap**

Live end-to-end verification (an actual network call to each provider) is **not possible right
now**: the Anthropic account is out of credit (`400: credit balance too low`, confirmed this
session) and no `GEMINI_API_KEY` has been provided yet. This is expected and does not block
completion - the mocked verification in Tasks 2-3 exercises the exact same code paths (request
building, response parsing, error handling) that a live call would.

When either becomes available:
- Anthropic: `POST /tools/test-extraction` (with `LLM_PROVIDER=anthropic`) via the running app,
  or re-run `scripts/cluster_destinations.py name`.
- Gemini: set `GEMINI_API_KEY` and `LLM_PROVIDER=gemini` in `backend/.env`, then the same checks.

- [ ] **Step 4: Update session memory**

Append a dated entry to `.claude/memory/sessions/` and update `.claude/memory/state.md` per this
project's convention (see `.claude/memory/README.md`), noting: the provider abstraction landed,
`services/claude.py` no longer exists, and live-call verification is pending credentials.
