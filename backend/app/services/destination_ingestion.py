import asyncio
import hashlib
import json
import re
import statistics
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models.destination import Destination
from app.schemas.destination_ingestion import DestinationSeedEntry, DestinationSeedManifest
from app.services.rag_ingestion import _extract_main_text
from app.services.voyage_embeddings import build_text_batches, embed_texts

GEOCODING_PATH = "/v1/search"
WIKIVOYAGE_SUMMARY_CHAR_LIMIT = 1500
POI_SUMMARY_TOP_KINDS = 5

_T = TypeVar("_T")


@dataclass(slots=True)
class DestinationRecord:
    """Working state for one destination as it moves through the pipeline."""

    name: str
    country: str
    region: str | None
    budget_level: str | None
    details: str
    raw_sources: dict[str, Any]
    source_provenance: dict[str, str]
    fetched_at: datetime
    content_hash: str
    needs_embedding: bool
    sources_failed: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass(slots=True)
class DestinationIngestionSummary:
    timestamp: str
    total_destinations: int
    region_counts: dict[str, int]
    missing_field_rates: dict[str, float]
    details_length_stats: dict[str, float]
    sources_failed_counts: dict[str, int]
    embedded_count: int
    skipped_embedding_count: int
    numbeo_lookup_available: bool
    opentripmap_configured: bool
    embedding_provider_failed: bool


def load_seed_manifest(manifest_path: str) -> DestinationSeedManifest:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path

    if not path.exists():
        raise FileNotFoundError(f"Destination seed manifest was not found at {path}.")

    raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    return DestinationSeedManifest.model_validate(raw_manifest)


async def ingest_destinations(
    session: AsyncSession,
    http_client: httpx.AsyncClient,
    settings: Settings,
    *,
    manifest: DestinationSeedManifest | None = None,
) -> DestinationIngestionSummary:
    if manifest is None:
        manifest = load_seed_manifest(settings.destination_seed_manifest_path)

    numbeo_index, numbeo_thresholds, numbeo_available = await _load_numbeo_budget_index(
        http_client, settings
    )
    existing_hashes = await _load_existing_content_hashes(session)

    records: list[DestinationRecord] = []
    for seed in manifest.destinations:
        record = await _process_destination_seed(
            http_client,
            settings,
            seed,
            numbeo_index=numbeo_index,
            numbeo_thresholds=numbeo_thresholds,
            existing_hashes=existing_hashes,
        )
        records.append(record)

    to_embed = [record for record in records if record.needs_embedding]
    embedding_failed = False
    if to_embed:
        try:
            embeddings = await _embed_records_in_batches(http_client, settings, to_embed)
            for record, embedding in zip(to_embed, embeddings, strict=True):
                record.embedding = embedding
        except Exception:  # noqa: BLE001 - an embedding-provider outage must not
            # discard the fetched/composed details for every destination in
            # this run; rows are upserted with embedding=null and picked up
            # by the content-hash cache on the next successful re-run.
            embedding_failed = True
            for record in to_embed:
                record.sources_failed.append("embedding")

    for record in records:
        await _upsert_destination(session, record, settings)
    await session.commit()

    return _build_summary(
        records,
        numbeo_available=numbeo_available,
        opentripmap_configured=bool(settings.opentripmap_api_key),
        embedding_failed=embedding_failed,
    )


async def _process_destination_seed(
    http_client: httpx.AsyncClient,
    settings: Settings,
    seed: DestinationSeedEntry,
    *,
    numbeo_index: dict[str, float],
    numbeo_thresholds: tuple[float, float] | None,
    existing_hashes: dict[tuple[str, str], tuple[str, bool]],
) -> DestinationRecord:
    raw_sources: dict[str, Any] = {}
    source_provenance: dict[str, str] = {}
    sources_failed: list[str] = []

    # 1. Geocode (Open-Meteo - free, already used by live_conditions; stands
    # in for the spec's "optional GeoNames" for canonical lat/lon + region).
    coordinates = None
    try:
        coordinates = await _run_with_retry(
            lambda: _geocode_destination(http_client, settings, seed.name, seed.country),
            max_retries=settings.destination_max_retries,
            backoff_seconds=settings.destination_retry_backoff_seconds,
        )
        source_provenance["coordinates"] = "open-meteo-geocoding"
    except Exception as exc:  # noqa: BLE001 - isolate one source's failure from the rest
        sources_failed.append("geocoding")
        source_provenance["coordinates"] = f"failed: {type(exc).__name__}"

    # 2. Wikivoyage prose.
    wikivoyage_summary: str | None = None
    try:
        wikivoyage_text = await _run_with_retry(
            lambda: _fetch_wikivoyage_text(http_client, settings, seed.wikivoyage_url),
            max_retries=settings.destination_max_retries,
            backoff_seconds=settings.destination_retry_backoff_seconds,
        )
        raw_sources["wikivoyage"] = wikivoyage_text
        wikivoyage_summary = _truncate_at_sentence_boundary(
            wikivoyage_text, WIKIVOYAGE_SUMMARY_CHAR_LIMIT
        )
        source_provenance["details.wikivoyage_summary"] = "wikivoyage"
    except Exception as exc:  # noqa: BLE001
        sources_failed.append("wikivoyage")
        source_provenance["details.wikivoyage_summary"] = f"failed: {type(exc).__name__}"

    # 3. OpenTripMap POI aggregation (requires both a configured key and
    # resolved coordinates; skips - not fails - when either is absent).
    poi_summary: str | None = None
    if not settings.opentripmap_api_key:
        source_provenance["details.poi_summary"] = "skipped: opentripmap_api_key not configured"
    elif coordinates is None:
        source_provenance["details.poi_summary"] = "skipped: no coordinates resolved"
    else:
        lat, lon, _admin1 = coordinates
        try:
            kind_counts = await _run_with_retry(
                lambda: _fetch_opentripmap_pois(http_client, settings, lat, lon),
                max_retries=settings.destination_max_retries,
                backoff_seconds=settings.destination_retry_backoff_seconds,
            )
            raw_sources["opentripmap_kind_counts"] = kind_counts
            poi_summary = _compose_poi_summary(kind_counts)
            source_provenance["details.poi_summary"] = "opentripmap"
        except Exception as exc:  # noqa: BLE001
            sources_failed.append("opentripmap")
            source_provenance["details.poi_summary"] = f"failed: {type(exc).__name__}"

    # 4. Numbeo budget bucketing (looked up against the one shared fetch).
    budget_level: str | None = None
    if numbeo_thresholds is not None:
        budget_level = _lookup_budget_level(
            seed.name, seed.country, numbeo_index, numbeo_thresholds
        )
    source_provenance["budget_level"] = "numbeo" if budget_level is not None else "unmatched"

    details = _compose_details(
        name=seed.name,
        country=seed.country,
        region=seed.region,
        wikivoyage_summary=wikivoyage_summary,
        poi_summary=poi_summary,
    )
    content_hash = _content_hash(details)

    key = (seed.name.casefold(), seed.country.casefold())
    existing = existing_hashes.get(key)
    needs_embedding = existing is None or existing[0] != content_hash or not existing[1]

    return DestinationRecord(
        name=seed.name,
        country=seed.country,
        region=seed.region,
        budget_level=budget_level,
        details=details,
        raw_sources=raw_sources,
        source_provenance=source_provenance,
        fetched_at=datetime.now(timezone.utc),
        content_hash=content_hash,
        needs_embedding=needs_embedding,
        sources_failed=sources_failed,
    )


async def _run_with_retry(
    factory: Callable[[], Awaitable[_T]],
    *,
    max_retries: int,
    backoff_seconds: float,
) -> _T:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await factory()
        except Exception as exc:  # noqa: BLE001 - retried broadly, re-raised to the caller
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(backoff_seconds * (2**attempt))
    assert last_exc is not None
    raise last_exc


async def _geocode_destination(
    http_client: httpx.AsyncClient,
    settings: Settings,
    name: str,
    country: str,
) -> tuple[float, float, str | None] | None:
    response = await http_client.get(
        f"{settings.open_meteo_geocoding_base_url}{GEOCODING_PATH}",
        params={"name": name, "count": 5, "language": "en", "format": "json"},
        timeout=settings.destination_fetch_timeout_seconds,
    )
    response.raise_for_status()
    results = (response.json()).get("results") or []
    if not results:
        return None

    country_cf = country.casefold()
    best = next(
        (r for r in results if str(r.get("country", "")).casefold() == country_cf),
        results[0],
    )
    return float(best["latitude"]), float(best["longitude"]), best.get("admin1")


async def _fetch_wikivoyage_text(
    http_client: httpx.AsyncClient,
    settings: Settings,
    url: str,
) -> str:
    response = await http_client.get(
        url,
        headers={"User-Agent": settings.destination_user_agent},
        timeout=settings.destination_fetch_timeout_seconds,
    )
    response.raise_for_status()
    return _extract_main_text(response.text)


def _truncate_at_sentence_boundary(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    window = text[:limit]
    boundary = window.rfind(". ")
    return window[: boundary + 1] if boundary != -1 else window


async def _fetch_opentripmap_pois(
    http_client: httpx.AsyncClient,
    settings: Settings,
    lat: float,
    lon: float,
) -> dict[str, int]:
    response = await http_client.get(
        f"{settings.opentripmap_base_url}/places/radius",
        params={
            "radius": settings.opentripmap_radius_meters,
            "lon": lon,
            "lat": lat,
            "limit": settings.opentripmap_poi_limit,
            # Explicit "json" avoids the API's default GeoJSON response, whose
            # published schema nests kinds under a doubled properties.properties
            # key - the flat SimpleFeature list (xid/name/kinds/dist/point) is
            # unambiguous and all this pipeline needs is `kinds`.
            "format": "json",
            "apikey": settings.opentripmap_api_key,
        },
        timeout=settings.destination_fetch_timeout_seconds,
    )
    response.raise_for_status()
    places = response.json() or []

    kind_counts: Counter[str] = Counter()
    for place in places:
        for kind in str(place.get("kinds") or "").split(","):
            kind = kind.strip()
            if kind:
                kind_counts[kind] += 1
    return dict(kind_counts)


def _compose_poi_summary(kind_counts: dict[str, int]) -> str | None:
    if not kind_counts:
        return None
    top_kinds = sorted(kind_counts.items(), key=lambda item: -item[1])[:POI_SUMMARY_TOP_KINDS]
    formatted = ", ".join(f"{kind.replace('_', ' ')} ({count})" for kind, count in top_kinds)
    return f"Notable points of interest nearby include: {formatted}."


async def _load_numbeo_budget_index(
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> tuple[dict[str, float], tuple[float, float] | None, bool]:
    """Fetch Numbeo's public cost-of-living ranking table once per run.

    Numbeo has no free API; this is a single request for one public page
    (not a per-city scrape), matched against destinations by "city, country".
    Budget bucketing is never a hard dependency - any failure here degrades
    every destination's budget_level to null rather than aborting the run.
    """
    try:
        index_by_city_country = await _run_with_retry(
            lambda: _fetch_numbeo_rankings_table(http_client, settings),
            max_retries=settings.destination_max_retries,
            backoff_seconds=settings.destination_retry_backoff_seconds,
        )
    except Exception:  # noqa: BLE001
        return {}, None, False

    values = sorted(index_by_city_country.values())
    if not values:
        return {}, None, False

    q1, _median, q3 = statistics.quantiles(values, n=4)
    return index_by_city_country, (q1, q3), True


async def _fetch_numbeo_rankings_table(
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> dict[str, float]:
    response = await http_client.get(
        settings.numbeo_rankings_url,
        headers={"User-Agent": settings.destination_user_agent},
        timeout=settings.destination_fetch_timeout_seconds,
    )
    response.raise_for_status()

    table_match = re.search(r'<table[^>]*id="t2"[^>]*>.*?</table>', response.text, flags=re.DOTALL)
    if table_match is None:
        raise ValueError("Numbeo rankings table (id=t2) was not found in the response.")

    rows = re.findall(r"<tr[^>]*>.*?</tr>", table_match.group(0), flags=re.DOTALL)
    index_by_city_country: dict[str, float] = {}
    for row in rows[1:]:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL)
        if len(cells) < 3:
            continue
        city_country = re.sub(r"<[^>]+>", "", cells[1]).strip()
        try:
            index_value = float(re.sub(r"<[^>]+>", "", cells[2]).strip())
        except ValueError:
            continue
        index_by_city_country[city_country.casefold()] = index_value

    if not index_by_city_country:
        raise ValueError("Numbeo rankings table had no parseable rows.")
    return index_by_city_country


def _lookup_budget_level(
    name: str,
    country: str,
    index_by_city_country: dict[str, float],
    thresholds: tuple[float, float],
) -> str | None:
    value = index_by_city_country.get(f"{name}, {country}".casefold())

    if value is None:
        name_cf = name.casefold()
        candidates = [
            v for k, v in index_by_city_country.items() if k.split(",")[0].strip() == name_cf
        ]
        if len(candidates) == 1:
            value = candidates[0]

    if value is None:
        return None

    q1, q3 = thresholds
    if value < q1:
        return "low"
    if value > q3:
        return "high"
    return "medium"


def _compose_details(
    *,
    name: str,
    country: str,
    region: str | None,
    wikivoyage_summary: str | None,
    poi_summary: str | None,
) -> str:
    region_line = f"{name}, {country}" + (f" ({region})" if region else "") + "."
    parts = [region_line]
    if wikivoyage_summary:
        parts.append(wikivoyage_summary)
    if poi_summary:
        parts.append(poi_summary)
    return "\n\n".join(parts)


def _content_hash(details: str) -> str:
    return hashlib.sha256(details.encode("utf-8")).hexdigest()


async def _load_existing_content_hashes(
    session: AsyncSession,
) -> dict[tuple[str, str], tuple[str, bool]]:
    try:
        result = await session.execute(
            select(
                Destination.name,
                Destination.country,
                Destination.content_hash,
                Destination.embedding,
            )
        )
    except ProgrammingError as exc:
        await session.rollback()
        raise RuntimeError(
            "The `destinations` table does not exist yet. Run "
            "`uv run alembic upgrade head` before running ingestion."
        ) from exc

    return {
        (name.casefold(), country.casefold()): (content_hash, embedding is not None)
        for name, country, content_hash, embedding in result.all()
        if content_hash is not None
    }


async def _embed_records_in_batches(
    http_client: httpx.AsyncClient,
    settings: Settings,
    records: Sequence[DestinationRecord],
) -> list[list[float]]:
    texts = [record.details for record in records]
    text_batches = build_text_batches(
        texts,
        max_batch_size=settings.rag_embedding_batch_size,
        max_request_tokens=settings.rag_embedding_max_request_tokens,
        estimated_chars_per_token=settings.rag_estimated_chars_per_token,
    )
    min_request_interval_seconds = max(60.0 / max(settings.voyage_requests_per_minute, 1), 1.0)

    all_embeddings: list[list[float]] = []
    for batch_index, batch in enumerate(text_batches):
        if batch_index > 0:
            await asyncio.sleep(min_request_interval_seconds)
        batch_embeddings = await embed_texts(http_client, settings, batch, input_type="document")
        all_embeddings.extend(batch_embeddings)
    return all_embeddings


async def _upsert_destination(
    session: AsyncSession,
    record: DestinationRecord,
    settings: Settings,
) -> None:
    values: dict[str, Any] = {
        "name": record.name,
        "country": record.country,
        "region": record.region,
        "budget_level": record.budget_level,
        "details": record.details,
        "raw_sources": record.raw_sources,
        "source_provenance": record.source_provenance,
        "fetched_at": record.fetched_at,
        "content_hash": record.content_hash,
        "embedding": record.embedding,
        "embedding_model": settings.voyage_embedding_model if record.embedding is not None else None,
        "embedding_version": (
            settings.destination_embedding_version if record.embedding is not None else None
        ),
    }

    update_fields: dict[str, Any] = {
        "region": record.region,
        "budget_level": record.budget_level,
        "details": record.details,
        "raw_sources": record.raw_sources,
        "source_provenance": record.source_provenance,
        "fetched_at": record.fetched_at,
        "content_hash": record.content_hash,
    }
    if record.embedding is not None:
        # Only touch embedding columns when we actually computed a new one
        # this run, so a failed re-embed never clobbers a prior good vector.
        update_fields["embedding"] = record.embedding
        update_fields["embedding_model"] = settings.voyage_embedding_model
        update_fields["embedding_version"] = settings.destination_embedding_version

    statement = pg_insert(Destination).values(**values)
    statement = statement.on_conflict_do_update(
        constraint="uq_destinations_name_country",
        set_=update_fields,
    )
    await session.execute(statement)


def _build_summary(
    records: Sequence[DestinationRecord],
    *,
    numbeo_available: bool,
    opentripmap_configured: bool,
    embedding_failed: bool,
) -> DestinationIngestionSummary:
    total = len(records)
    region_counts = Counter(record.region or "Unknown" for record in records)

    missing_budget = sum(1 for record in records if record.budget_level is None)
    missing_poi = sum(
        1 for record in records if record.source_provenance.get("details.poi_summary") != "opentripmap"
    )
    missing_wikivoyage = sum(
        1
        for record in records
        if record.source_provenance.get("details.wikivoyage_summary") != "wikivoyage"
    )
    missing_embedding = sum(1 for record in records if record.embedding is None and record.needs_embedding)

    lengths = sorted(len(record.details) for record in records)
    details_length_stats = {
        "min": float(lengths[0]) if lengths else 0.0,
        "max": float(lengths[-1]) if lengths else 0.0,
        "mean": statistics.fmean(lengths) if lengths else 0.0,
        "median": float(statistics.median(lengths)) if lengths else 0.0,
    }

    sources_failed_counts: Counter[str] = Counter()
    for record in records:
        sources_failed_counts.update(record.sources_failed)
    if not numbeo_available:
        sources_failed_counts["numbeo"] = total

    return DestinationIngestionSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_destinations=total,
        region_counts=dict(region_counts),
        missing_field_rates={
            "budget_level": missing_budget / total if total else 0.0,
            "poi_summary": missing_poi / total if total else 0.0,
            "wikivoyage_summary": missing_wikivoyage / total if total else 0.0,
            "embedding": missing_embedding / total if total else 0.0,
        },
        details_length_stats=details_length_stats,
        sources_failed_counts=dict(sources_failed_counts),
        embedded_count=sum(1 for record in records if record.needs_embedding and record.embedding is not None),
        skipped_embedding_count=sum(1 for record in records if not record.needs_embedding),
        numbeo_lookup_available=numbeo_available,
        opentripmap_configured=opentripmap_configured,
        embedding_provider_failed=embedding_failed,
    )
