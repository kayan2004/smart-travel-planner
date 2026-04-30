from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

EMBEDDING_DIMENSIONS = 1024


class DestinationDocument(Base):
    __tablename__ = "destination_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    destination_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    travel_style: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    source_type: Mapped[str] = mapped_column(String(100), nullable=False)
    source_title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
