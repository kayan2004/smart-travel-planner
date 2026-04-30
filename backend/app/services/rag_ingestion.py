import json
import re
import asyncio
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models.destination_document import DestinationDocument
from app.schemas.rag import RagDocumentChunk, RagFetchedDocument, RagSourceDocument
from app.services.voyage_embeddings import (
    build_text_batches,
    embed_texts,
    estimate_text_tokens,
)


def load_source_manifest(source_manifest_path: str) -> list[RagSourceDocument]:
    path = Path(source_manifest_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path

    if not path.exists():
        raise FileNotFoundError(f"RAG source manifest file was not found at {path}.")

    raw_sources = json.loads(path.read_text(encoding="utf-8"))
    return [RagSourceDocument.model_validate(item) for item in raw_sources]


async def fetch_source_documents(
    http_client: httpx.AsyncClient,
    sources: list[RagSourceDocument],
    *,
    timeout_seconds: float,
    user_agent: str,
) -> list[RagFetchedDocument]:
    documents: list[RagFetchedDocument] = []
    for source in sources:
        documents.append(
            await _fetch_source_document(
                http_client,
                source,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
            )
        )
    return documents


async def ingest_destination_documents(
    session: AsyncSession,
    http_client: httpx.AsyncClient,
    settings: Settings,
) -> dict[str, int]:
    _print_progress(5, "Loading source manifest")
    sources = load_source_manifest(settings.rag_source_manifest_path)

    _print_progress(15, f"Fetching {len(sources)} source pages")
    fetched_documents = await fetch_source_documents(
        http_client,
        sources,
        timeout_seconds=settings.rag_fetch_timeout_seconds,
        user_agent=settings.rag_user_agent,
    )

    _print_progress(35, f"Chunking {len(fetched_documents)} fetched documents")
    chunks = chunk_source_documents(
        fetched_documents,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )

    _print_progress(45, f"Preparing embeddings for {len(chunks)} chunks")
    embeddings = await _embed_chunks_in_batches(http_client, settings, chunks)

    _print_progress(90, "Writing embedded chunks to Postgres")
    await session.execute(delete(DestinationDocument))
    session.add_all(
        [
            DestinationDocument(
                destination_name=chunk.destination_name,
                source_type=chunk.source_type,
                source_title=chunk.source_title,
                source_url=chunk.source_url,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
    )
    await session.commit()

    _print_progress(100, "RAG ingestion complete")

    return {
        "sources": len(sources),
        "documents": len(fetched_documents),
        "chunks": len(chunks),
        "embeddings": len(embeddings),
    }


def chunk_source_documents(
    documents: list[RagFetchedDocument],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RagDocumentChunk]:
    chunks: list[RagDocumentChunk] = []
    for document in documents:
        chunks.extend(
            _chunk_source_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def _chunk_source_document(
    document: RagFetchedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RagDocumentChunk]:
    normalized_text = _normalize_text(document.content)
    if len(normalized_text) <= chunk_size:
        return [
            RagDocumentChunk(
                destination_name=document.destination_name,
                source_type=document.source_type,
                source_title=document.source_title,
                source_url=document.source_url,
                chunk_index=0,
                content=normalized_text,
            )
        ]

    paragraphs = [part.strip() for part in normalized_text.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [normalized_text]

    chunks: list[RagDocumentChunk] = []
    current_text = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current_text else f"{current_text}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            current_text = candidate
            continue

        if current_text:
            chunks.append(
                _build_chunk(
                    document=document,
                    chunk_index=len(chunks),
                    content=current_text,
                )
            )
            overlap_text = current_text[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            current_text = (
                f"{overlap_text}\n\n{paragraph}".strip()
                if overlap_text
                else paragraph
            )
        else:
            slices = _slice_long_text(paragraph, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for slice_text in slices[:-1]:
                chunks.append(
                    _build_chunk(
                        document=document,
                        chunk_index=len(chunks),
                        content=slice_text,
                    )
                )
            current_text = slices[-1]

    if current_text:
        chunks.append(
            _build_chunk(
                document=document,
                chunk_index=len(chunks),
                content=current_text,
            )
        )

    return chunks


def _build_chunk(
    *,
    document: RagFetchedDocument,
    chunk_index: int,
    content: str,
) -> RagDocumentChunk:
    return RagDocumentChunk(
        destination_name=document.destination_name,
        source_type=document.source_type,
        source_title=document.source_title,
        source_url=document.source_url,
        chunk_index=chunk_index,
        content=content.strip(),
    )


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned_lines: list[str] = []
    blank_streak = 0
    for line in lines:
        if not line:
            blank_streak += 1
            if blank_streak <= 1:
                cleaned_lines.append("")
            continue
        blank_streak = 0
        cleaned_lines.append(" ".join(line.split()))
    return "\n".join(cleaned_lines).strip()


def _slice_long_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    step = max(chunk_size - chunk_overlap, 1)
    slices: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        slices.append(text[start:end].strip())
        if end >= len(text):
            break
        start += step
    return slices


async def _fetch_source_document(
    client: httpx.AsyncClient,
    source: RagSourceDocument,
    *,
    timeout_seconds: float,
    user_agent: str,
) -> RagFetchedDocument:
    response = await client.get(
        source.source_url,
        headers={"User-Agent": user_agent},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    extracted_text = _extract_main_text(response.text)
    return RagFetchedDocument(
        destination_name=source.destination_name,
        source_type=source.source_type,
        source_title=source.source_title,
        source_url=source.source_url,
        content=extracted_text,
    )


def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    main_node = (
        soup.find("main")
        or soup.find(id="mw-content-text")
        or soup.select_one(".mw-parser-output")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )
    if main_node is None:
        raise ValueError("Could not extract main HTML content from the source page.")

    for element in main_node(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "img",
            "header",
            "footer",
            "nav",
            "aside",
            "form",
            "table",
            "sup",
        ]
    ):
        element.decompose()

    for selector in [
        "[id*='toc']",
        "[class*='toc']",
        "[class*='navbox']",
        "[class*='metadata']",
        "[class*='reference']",
        "[class*='sidebar']",
        "[class*='banner']",
        "[class*='editsection']",
        ".thumb",
        ".vector-page-toolbar",
        ".vector-page-titlebar",
        ".language-list",
        ".mw-portlet",
        ".mw-footer",
        ".sistersitebox",
        ".hlist",
        ".plainlinks",
        ".mw-jump-link",
        ".mw-editsection",
        ".nomobile",
        ".noprint",
        ".printfooter",
        ".catlinks",
    ]:
        for element in main_node.select(selector):
            element.decompose()

    blocks = [
        node.get_text(" ", strip=True)
        for node in main_node.find_all(["h1", "h2", "h3", "p", "li"])
    ]
    text = "\n\n".join(block for block in blocks if block)
    if not text.strip():
        text = main_node.get_text("\n", strip=True)
    normalized_text = _normalize_text(text)
    if not normalized_text:
        normalized_text = _extract_metadata_fallback(soup)
    if not normalized_text:
        normalized_text = _extract_raw_html_metadata_fallback(html)
    if not normalized_text:
        raise ValueError("Extracted source content was empty after normalization.")
    return normalized_text


def _extract_metadata_fallback(soup: BeautifulSoup) -> str:
    candidates: list[str] = []

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title:
        candidates.append(title)

    for attrs in [
        {"name": "description"},
        {"property": "og:description"},
        {"name": "twitter:description"},
    ]:
        tag = soup.find("meta", attrs=attrs)
        if tag is not None:
            content = tag.get("content", "").strip()
            if content:
                candidates.append(content)

    return _normalize_text("\n\n".join(candidates))


def _extract_raw_html_metadata_fallback(html: str) -> str:
    candidates: list[str] = []

    title_match = re.search(
        r"<title[^>]*>(.*?)</title>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if title_match:
        candidates.append(_strip_html_whitespace(title_match.group(1)))

    for pattern in [
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+name=["\']twitter:description["\'][^>]+content=["\'](.*?)["\']',
    ]:
        match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            candidates.append(_strip_html_whitespace(match.group(1)))

    return _normalize_text("\n\n".join(candidate for candidate in candidates if candidate))


def _strip_html_whitespace(value: str) -> str:
    return " ".join(value.replace("&nbsp;", " ").split())


async def _embed_chunks_in_batches(
    http_client: httpx.AsyncClient,
    settings: Settings,
    chunks: list[RagDocumentChunk],
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    texts = [chunk.content for chunk in chunks]
    text_batches = build_text_batches(
        texts,
        max_batch_size=settings.rag_embedding_batch_size,
        max_request_tokens=settings.rag_embedding_max_request_tokens,
        estimated_chars_per_token=settings.rag_estimated_chars_per_token,
    )
    min_request_interval_seconds = max(
        60.0 / max(settings.voyage_requests_per_minute, 1),
        1.0,
    )

    if not text_batches:
        _print_progress(85, "No chunks to embed")
        return all_embeddings

    for batch_index, batch in enumerate(text_batches):
        batch_number = batch_index + 1
        percentage = 45 + int((batch_number / len(text_batches)) * 40)
        estimated_tokens = sum(
            estimate_text_tokens(
                text,
                estimated_chars_per_token=settings.rag_estimated_chars_per_token,
            )
            for text in batch
        )
        _print_progress(
            percentage,
            (
                f"Embedding batch {batch_number}/{len(text_batches)} "
                f"({len(batch)} chunks, ~{estimated_tokens} est. tokens)"
            ),
        )
        if batch_index > 0:
            print(
                f"Waiting {int(min_request_interval_seconds)}s to respect Voyage rate limits..."
            )
            await asyncio.sleep(min_request_interval_seconds)
        batch_embeddings = await embed_texts(
            http_client,
            settings,
            batch,
            input_type="document",
        )
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _print_progress(percentage: int, message: str) -> None:
    print(f"[{percentage:>3}%] {message}")
