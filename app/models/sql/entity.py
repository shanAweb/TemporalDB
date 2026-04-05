import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.postgres import Base


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    canonical_name: Mapped[str] = mapped_column(
        String(512), nullable=False, index=True,
        comment="Resolved canonical name for entity linking",
    )
    type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment="Entity type: PERSON, ORG, GPE, DATE, EVENT, etc.",
    )
    description: Mapped[str | None] = mapped_column(Text)
    aliases: Mapped[str | None] = mapped_column(
        Text, comment="JSON array of known aliases"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=text("now()"),
    )

    # Relationships
    events: Mapped[list["Event"]] = relationship(  # noqa: F821
        secondary="event_entities", back_populates="entities"
    )

    def __repr__(self) -> str:
        return f"<Entity(id={self.id}, name='{self.name}', type='{self.type}')>"
