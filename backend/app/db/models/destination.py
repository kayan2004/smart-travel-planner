import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Must match app.core.config.Settings.voyage_embedding_dimension. Changing
# this requires a new migration (ALTER COLUMN ... TYPE vector(N)), so it is
# a fixed schema constant rather than something read from settings at import
# time.
EMBEDDING_DIMENSIONS = 1024


class DestinationCorpusBase(DeclarativeBase):
    """Independent declarative base for the destination corpus.

    Deliberately not the same base as app.db.base.Base: this represents a
    conceptually separate corpus from destination_documents (the original
    RAG table), not a create_all()-avoidance workaround - all tables are
    Alembic-managed now (app/db/init_db.py and Base.metadata.create_all()
    no longer exist). alembic/env.py includes both bases' metadata via a
    list so autogenerate still sees the full schema.
    """


class Destination(DestinationCorpusBase):
    __tablename__ = "destinations"
    __table_args__ = (
        UniqueConstraint("name", "country", name="uq_destinations_name_country"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    country: Mapped[str] = mapped_column(Text, nullable=False)
    region: Mapped[str | None] = mapped_column(Text, nullable=True)
    budget_level: Mapped[str | None] = mapped_column(String(10), nullable=True)
    details: Mapped[str] = mapped_column(Text, nullable=False)
    raw_sources: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    source_provenance: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Weighted multi-tags from offline HDBSCAN soft clustering (see
    # scripts/cluster_destinations.py). Keyed by cluster_id until naming is
    # approved (Phase 2), then rewritten to tag_name keys. {} until the
    # corpus has been clustered.
    tags: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
