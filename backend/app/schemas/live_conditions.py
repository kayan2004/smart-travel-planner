from typing import Literal

from pydantic import BaseModel, Field


class LiveConditionsRequest(BaseModel):
    location_query: str = Field(min_length=2, max_length=120)
    country_code: str | None = Field(default=None, min_length=2, max_length=2)


class ResolvedLocation(BaseModel):
    name: str
    country: str
    country_code: str
    admin1: str | None = None
    latitude: float
    longitude: float
    timezone: str


class CurrentConditions(BaseModel):
    observed_at: str
    temperature_c: float
    apparent_temperature_c: float
    relative_humidity_percent: float
    precipitation_mm: float
    cloud_cover_percent: float
    wind_speed_kmh: float
    wind_direction_degrees: float
    wind_gusts_kmh: float
    weather_code: int
    weather_summary: str
    is_day: bool


class LiveConditionsResponse(BaseModel):
    provider: Literal["open-meteo"]
    location: ResolvedLocation
    current: CurrentConditions

