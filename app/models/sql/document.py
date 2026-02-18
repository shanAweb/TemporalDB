import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.postgres import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    source: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    filename: Mapped[str | None] = mapped_column(String(512))
    content_hash: Mapped[str | None] = mapped_column(
        String(64), unique=True, comment="SHA-256 hash for deduplication"
    )
    raw_text: Mapped[str | None] = mapped_column(Text)
    mime_type: Mapped[str | None] = mapped_column(String(128))
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", type_=Text, comment="JSON-encoded document metadata"
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
        index=True,
    )

    # Relationships
    events: Mapped[list["Event"]] = relationship(  # noqa: F821
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, source='{self.source}')>"
