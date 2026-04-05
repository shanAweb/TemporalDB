import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.database.postgres import Base


class Event(Base):
    __tablename__ = "events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    event_type: Mapped[str | None] = mapped_column(
        String(64), index=True,
        comment="action, state_change, declaration, occurrence",
    )
    ts_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True,
        comment="Event start timestamp (UTC)",
    )
    ts_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        comment="Event end timestamp (UTC), null if point-in-time",
    )
    confidence: Mapped[float] = mapped_column(
        Float, default=1.0, server_default=text("1.0"),
        comment="NLP extraction confidence score 0.0-1.0",
    )
    source_sentence: Mapped[str | None] = mapped_column(
        Text, comment="Original sentence from which event was extracted"
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimension),
        comment="Dense vector embedding for semantic search",
    )

    # Foreign keys
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
    )

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="events")  # noqa: F821
    entities: Mapped[list["Entity"]] = relationship(  # noqa: F821
        secondary="event_entities", back_populates="events"
    )
    caused_by: Mapped[list["CausalRelation"]] = relationship(  # noqa: F821
        foreign_keys="CausalRelation.effect_event_id", back_populates="effect_event"
    )
    causes: Mapped[list["CausalRelation"]] = relationship(  # noqa: F821
        foreign_keys="CausalRelation.cause_event_id", back_populates="cause_event"
    )

    def __repr__(self) -> str:
        return f"<Event(id={self.id}, description='{self.description[:50]}...')>"
