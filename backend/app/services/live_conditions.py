from fastapi import HTTPException, status
import httpx

from app.core.config import Settings
from app.schemas.live_conditions import (
    CurrentConditions,
    LiveConditionsRequest,
    LiveConditionsResponse,
    ResolvedLocation,
)

GEOCODING_PATH = "/v1/search"
FORECAST_PATH = "/v1/forecast"
CURRENT_WEATHER_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "is_day",
    "precipitation",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]
WEATHER_CODE_SUMMARIES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def get_live_conditions(
    http_client: httpx.AsyncClient,
    settings: Settings,
    payload: LiveConditionsRequest,
) -> LiveConditionsResponse:
    location = await _resolve_location(http_client, settings, payload)
    current = await _fetch_current_conditions(http_client, settings, location)

    return LiveConditionsResponse(
        provider="open-meteo",
        location=location,
        current=current,
    )


async def _resolve_location(
    http_client: httpx.AsyncClient,
    settings: Settings,
    payload: LiveConditionsRequest,
) -> ResolvedLocation:
    params = {
        "name": payload.location_query,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    if payload.country_code is not None:
        params["countryCode"] = payload.country_code.upper()

    response = await http_client.get(
        f"{settings.open_meteo_geocoding_base_url}{GEOCODING_PATH}",
        params=params,
        timeout=settings.weather_request_timeout_seconds,
    )
    response.raise_for_status()
    response_payload = response.json()
    results = response_payload.get("results") or []

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No matching location was found.",
        )

    best_match = results[0]
    return ResolvedLocation(
        name=str(best_match["name"]),
        country=str(best_match["country"]),
        country_code=str(best_match["country_code"]),
        admin1=best_match.get("admin1"),
        latitude=float(best_match["latitude"]),
        longitude=float(best_match["longitude"]),
        timezone=str(best_match["timezone"]),
    )


async def _fetch_current_conditions(
    http_client: httpx.AsyncClient,
    settings: Settings,
    location: ResolvedLocation,
) -> CurrentConditions:
    response = await http_client.get(
        f"{settings.open_meteo_forecast_base_url}{FORECAST_PATH}",
        params={
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": ",".join(CURRENT_WEATHER_FIELDS),
            "timezone": location.timezone,
        },
        timeout=settings.weather_request_timeout_seconds,
    )
    response.raise_for_status()
    response_payload = response.json()
    current = response_payload.get("current")

    if not isinstance(current, dict):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Weather provider returned an invalid current-conditions response.",
        )

    weather_code = int(current["weather_code"])
    return CurrentConditions(
        observed_at=str(current["time"]),
        temperature_c=float(current["temperature_2m"]),
        apparent_temperature_c=float(current["apparent_temperature"]),
        relative_humidity_percent=float(current["relative_humidity_2m"]),
        precipitation_mm=float(current["precipitation"]),
        cloud_cover_percent=float(current["cloud_cover"]),
        wind_speed_kmh=float(current["wind_speed_10m"]),
        wind_direction_degrees=float(current["wind_direction_10m"]),
        wind_gusts_kmh=float(current["wind_gusts_10m"]),
        weather_code=weather_code,
        weather_summary=WEATHER_CODE_SUMMARIES.get(weather_code, "Unknown conditions"),
        is_day=bool(current["is_day"]),
    )

